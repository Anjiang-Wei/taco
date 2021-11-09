#include <default_mapper.h>
#include "hdf5_utils.h"
#include "task_ids.h"
#include "legion_tensor.h"

using namespace Legion;

void generateCoordListHDF5(std::string filename, size_t order, size_t nnz) {
  // Open up the HDF5 file.
  hid_t fileID = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  assert(fileID >= 0);

  // Create a data space for the dimensions.
  hsize_t dims[1];
  dims[0] = order;
  hid_t dimensionsDataspaceID = H5Screate_simple(1, dims, NULL);
  assert(dimensionsDataspaceID >= 0);
  // Create a data space for the coordinates and values.
  dims[0] = nnz;
  hid_t coordSpaceID = H5Screate_simple(1, dims, NULL);
  assert(coordSpaceID >= 0);

  std::vector<hid_t> datasets;
  auto createDataset = [&](std::string name, hid_t size, hid_t dataspaceID) {
    hid_t dataset = H5Dcreate2(fileID, name.c_str(), size, dataspaceID, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dataset >= 0);
    datasets.push_back(dataset);
  };

  // Create a dataset for each of the coordinates.
  for (size_t i = 0; i < order; i++) {
    // We'll use int32_t's for the coordinates.
    createDataset(COOCoordsFields[i], H5T_NATIVE_INT32_g, coordSpaceID);
  }
  // Create a dataset for the values.
  createDataset(COOValsField, H5T_IEEE_F64LE_g, coordSpaceID);
  // Create a dataset for the dimensions.
  createDataset(COODimsField, H5T_NATIVE_INT32_g, dimensionsDataspaceID);

  // Close up everything now.
  for (auto id : datasets) {
    H5Dclose(id);
  }
  H5Sclose(coordSpaceID);
  H5Sclose(dimensionsDataspaceID);
  H5Fclose(fileID);
}

void getCoordListHDF5Meta(std::string filename, size_t& order, size_t& nnz) {
  auto hdf5file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  auto getDataSetDim = [&](const char* dataSetName) {
    auto dataset = H5Dopen1(hdf5file, dataSetName);
    auto dataSpace = H5Dget_space(dataset);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(dataSpace, dims, NULL);
    auto result = size_t(dims[0]);
    H5Sclose(dataSpace);
    H5Dclose(dataset);
    return result;
  };
  order = getDataSetDim(COODimsField);
  nnz = getDataSetDim(COOValsField);
  H5Fclose(hdf5file);
}

PhysicalRegion attachHDF5(Context ctx, Runtime *runtime, LogicalRegion region, std::map<FieldID, const char *> fieldMap, std::string filename, Legion::LegionFileMode mode) {
  AttachLauncher al(LEGION_EXTERNAL_HDF5_FILE, region, region);
  al.attach_hdf5(filename.c_str(), fieldMap, mode);
  return runtime->attach_external_resource(ctx, al);
}

LegionTensor loadCOOFromHDF5(Context ctx, Runtime* runtime, std::string& filename, FieldID coordField, size_t coordSize, FieldID valsField, size_t valsSize) {
  size_t order, nnz;
  getCoordListHDF5Meta(filename, order, nnz);

  auto fispace = runtime->create_field_space(ctx);
  auto fvspace = runtime->create_field_space(ctx);
  {
    FieldAllocator fa = runtime->create_field_allocator(ctx, fispace);
    fa.allocate_field(coordSize, coordField);
  }
  {
    FieldAllocator fa = runtime->create_field_allocator(ctx, fvspace);
    fa.allocate_field(valsSize, valsField);
  }
  auto dimIspace = runtime->create_index_space(ctx, Rect<1>(0, order - 1));
  auto nnzIspace = runtime->create_index_space(ctx, Rect<1>(0, nnz - 1));

  // Create regions -- 1 for the dimensions, 1 for each coordinate, 1 for the values.
  std::vector<LogicalRegion> regions;
  regions.push_back(runtime->create_logical_region(ctx, dimIspace, fispace));
  for (size_t i = 0; i < order; i++) {
    regions.push_back(runtime->create_logical_region(ctx, nnzIspace, fispace));
  }
  regions.push_back(runtime->create_logical_region(ctx, nnzIspace, fvspace));

  // Launch a task to attach to all of the regions.
  {
    TaskLauncher launcher(TID_ATTACH_COO_REGIONS, TaskArgument(filename.c_str(), filename.length()));
    for (size_t i = 0; i < regions.size(); i++) {
      auto reg = regions[i];
      auto fid = coordField;
      if (i == regions.size() - 1) {
        fid = valsField;
      }
      launcher.add_region_requirement(
          RegionRequirement(reg, READ_WRITE, EXCLUSIVE, reg, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(fid));
    }
    runtime->execute_task(ctx, launcher).wait();
  }

  // Copy the dimensions into CPU memory to access.
  auto dimsMem = runtime->create_logical_region(ctx, dimIspace, fispace);
  {
    CopyLauncher cl;
    cl.add_copy_requirements(
        RegionRequirement(regions[0], READ_ONLY, EXCLUSIVE, regions[0]),
        RegionRequirement(dimsMem, WRITE_DISCARD, EXCLUSIVE, dimsMem)
    );
    cl.add_src_field(0, coordField); cl.add_dst_field(0, coordField);
    runtime->issue_copy_operation(ctx, cl);
  }
  std::vector<int32_t> dims;
  {
    // Map the CPU region directly now.
    auto preg = runtime->map_region(
        ctx,
        RegionRequirement(dimsMem, READ_ONLY, EXCLUSIVE, dimsMem).add_field(coordField)
    );
    FieldAccessor<READ_ONLY,int32_t,1,coord_t, Realm::AffineAccessor<int32_t, 1, coord_t>> acc(preg, coordField);
    dims = std::vector<int32_t>(order);
    for (size_t i = 0; i < order; i++) {
      dims[i] = acc[i];
    }
    runtime->unmap_region(ctx, preg);
  }

  // Slice out the coordinate regions.
  std::vector<std::vector<LogicalRegion>> coordRegions;
  for (size_t i = 0; i < order; i++) {
    coordRegions.push_back({regions[i + 1]});
  }

  return LegionTensor {
    .order = int32_t(order),
    .dims = dims,
    .indices = coordRegions,
    .indicesParents = coordRegions,
    .vals = regions.back(),
    .valsParent = regions.back(),
    .denseLevelRuns = {},
  };
}

// Corresponds to TID_ATTACH_COO_REGIONS.
void attachCOORegionsTask(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  std::string filename((char*)task->args, task->arglen);
  std::vector<FieldID> fieldsAccessor;
  auto numRegs = regions.size();
  // Construct a field map for each of the regions we expect to see in the HDF5 file.
  std::vector<std::map<FieldID, const char*>> fieldMaps(numRegs);
  // fieldMap for the dimensions.
  runtime->get_field_space_fields(ctx, regions[0].get_logical_region().get_field_space(), fieldsAccessor);
  fieldMaps[0] = {{fieldsAccessor.front(), COODimsField}};
  for (size_t i = 1; i < numRegs - 1; i++) {
    fieldMaps[i - 1] = {{fieldsAccessor.front(), COOCoordsFields[i - 1]}};
  }
  runtime->get_field_space_fields(ctx, regions[numRegs - 1].get_logical_region().get_field_space(), fieldsAccessor);
  fieldMaps[numRegs - 1] = {{fieldsAccessor.front(), COOValsField}};

  std::vector<PhysicalRegion> physRegs;
  std::vector<LogicalRegion> logRegs;
  CopyLauncher cl;
  for (size_t i = 0; i < regions.size(); i++) {
    auto rg = regions[i].get_logical_region();
    auto copy = runtime->create_logical_region(ctx, rg.get_index_space(), rg.get_field_space());
    logRegs.push_back(copy);
    physRegs.push_back(attachHDF5(ctx, runtime, copy, fieldMaps[i], filename));
    cl.add_copy_requirements(
      RegionRequirement(copy, READ_ONLY, EXCLUSIVE, copy).add_field(fieldMaps[i].begin()->first),
      RegionRequirement(rg, READ_WRITE, EXCLUSIVE, rg).add_field(fieldMaps[i].begin()->first)
    );
  }
  runtime->issue_copy_operation(ctx, cl);
  // Clean up after ourselves after allocating these intermediate regions.
  for (const auto& reg : physRegs) {
    runtime->detach_external_resource(ctx, reg).wait();
  }
  for (auto& reg : logRegs) {
    runtime->destroy_logical_region(ctx, reg);
  }
}

// This needs to be called in whatever main functions want to use HDF5 utility functions.
void registerHDF5UtilTasks() {
  {
    TaskVariantRegistrar registrar(TID_ATTACH_COO_REGIONS, "attachCOORegions");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<attachCOORegionsTask>(registrar, "attachCOORegions");
  }
}
