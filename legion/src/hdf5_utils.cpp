#include <cstring>
#include <default_mapper.h>
#include "hdf5_utils.h"
#include "task_ids.h"
#include "legion_tensor.h"
#include "string_utils.h"
#include "taco_legion_header.h"
#include "legion/legion_utilities.h"
#include "error.h"
#include "dummy_read.h"

using namespace Legion;

void generateCoordListHDF5(std::string filename, size_t order, size_t nnz) {
  // Open up the HDF5 file.
  hid_t fileID = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  taco_iassert(fileID >= 0);

  // Create a data space for the dimensions.
  hsize_t dims[1];
  dims[0] = order;
  hid_t dimensionsDataspaceID = H5Screate_simple(1, dims, NULL);
  taco_iassert(dimensionsDataspaceID >= 0);
  // Create a data space for the coordinates and values.
  dims[0] = nnz;
  hid_t coordSpaceID = H5Screate_simple(1, dims, NULL);
  taco_iassert(coordSpaceID >= 0);

  std::vector<hid_t> datasets;
  auto createDataset = [&](std::string name, hid_t size, hid_t dataspaceID) {
    hid_t dataset = H5Dcreate2(fileID, name.c_str(), size, dataspaceID, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    taco_iassert(dataset >= 0);
    datasets.push_back(dataset);
  };

  // Create a dataset for each of the coordinates.
  for (size_t i = 0; i < order; i++) {
    // We'll use int32_t's for the coordinates.
    createDataset(COOCoordsFields[i], H5T_NATIVE_INT32_g, coordSpaceID);
  }
  // Create a dataset for the values.
  // TODO (rohany): This needs to be templated over the type.
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

Legion::PhysicalRegion attachHDF5RW(Legion::Context ctx, Legion::Runtime *runtime, Legion::LogicalRegion region,
                                    std::map<Legion::FieldID, const char *> fieldMap, std::string filename) {
  AttachLauncher al(LEGION_EXTERNAL_HDF5_FILE, region, region);
  al.attach_hdf5(filename.c_str(), fieldMap, LEGION_FILE_READ_WRITE);
  return runtime->attach_external_resource(ctx, al);
}
Legion::PhysicalRegion attachHDF5RO(Legion::Context ctx, Legion::Runtime *runtime, Legion::LogicalRegion region,
                                    std::map<Legion::FieldID, const char *> fieldMap, std::string filename) {
  AttachLauncher al(LEGION_EXTERNAL_HDF5_FILE, region, region, false /* restricted */, false /* mapped */);
  al.attach_hdf5(filename.c_str(), fieldMap, LEGION_FILE_READ_ONLY);
  return runtime->attach_external_resource(ctx, al);
}

LegionTensor loadCOOFromHDF5(Context ctx, Runtime* runtime, std::string& filename, FieldID rectFieldID, FieldID coordField, size_t coordSize, FieldID valsField, size_t valsSize) {
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

  // The first level of a COO tensor is actually a compressed level, so we need a pos
  // and crd region set for it. The pos region has size 1.
  auto posIndexSpace = runtime->create_index_space(ctx, Rect<1>(0, 0));
  auto posFspace = runtime->create_field_space(ctx);
  {
    FieldAllocator fa = runtime->create_field_allocator(ctx, posFspace);
    fa.allocate_field(sizeof(Rect<1>), rectFieldID);
  }
  auto posReg = runtime->create_logical_region(ctx, posIndexSpace, posFspace);
  {
    auto preg = runtime->map_region(ctx, RegionRequirement(posReg, WRITE_ONLY, EXCLUSIVE, posReg).add_field(rectFieldID));
    FieldAccessor<WRITE_ONLY,Rect<1>,1,coord_t, Realm::AffineAccessor<Rect<1>, 1, coord_t>> acc(preg, rectFieldID);
    acc[0] = Rect<1>(0, nnz - 1);
    runtime->unmap_region(ctx, preg);
  }

  // Now, make the set of indices and indicesParents.
  std::vector<std::vector<LogicalRegion>> indices(order), indicesParents(order);
  // Construct the first compressed level.
  indices[0].push_back(posReg); indicesParents[0].push_back(posReg);
  indices[0].push_back(regions[1]); indicesParents[0].push_back(regions[1]);
  // Now construct the remaining COO levels.
  for (size_t i = 1; i < order; i++) {
    indices[i].push_back(regions[i + 1]);
    indicesParents[i].push_back(regions[i + 1]);
  }

  LegionTensorFormat format;
  // COO Tensors are one compressed level and then a bunch of singleton levels.
  format.push_back(Sparse);
  for (size_t i = 1; i < order; i++) {
    format.push_back(Singleton);
  }

  return LegionTensor(
    format,
    order,
    dims,
    indices,
    indicesParents,
    regions.back() /* vals */,
    regions.back() /* valsParent */,
    {} /* denseLevelRuns */
  );
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
    fieldMaps[i] = {{fieldsAccessor.front(), COOCoordsFields[i - 1]}};
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
    physRegs.push_back(attachHDF5RO(ctx, runtime, copy, fieldMaps[i], filename));
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

// getTensorLevelFormatName returns a const char* that must be `delete`ed after use.
const char* getTensorLevelFormatName(LegionTensorLevelFormat format, int mode, int idx) {
  auto toCharStar = [](std::string&& s) {
    char* tmp = new char [s.size() + 1];
    std::strcpy(tmp, s.c_str());
    return tmp;
  };
  switch (format) {
    case Dense:
      // We shouldn't be trying to get the tensor names from a Dense level.
      taco_iassert(false);
      break;
    case Sparse: {
      // The name depends on what the value of idx is.
      std::stringstream nameSS;
      nameSS << (idx == 0 ? "pos" : "crd") << "_" << mode;
      return toCharStar(nameSS.str());
    }
    case Singleton: {
      taco_iassert(idx == 0);
      std::stringstream nameSS;
      nameSS << "crd_" << mode;
      return toCharStar(nameSS.str());
    }
    default:
      taco_iassert(false);
  }
  return nullptr; // Keep the compiler happy.
}

// createHDF5RectType returns a tuple of {Point, Rect} HDF5 types.
template<int DIM>
std::pair<hid_t, hid_t> createHDF5RectType() {
  // First create a type for Point<T>'s.
  auto pointTy = H5Tcreate(H5T_COMPOUND, sizeof(Point<DIM>));
  // Add all coordinates in the point.
  auto offset = 0;
  for (int i = 0; i < DIM; i++) {
    std::string fname = "coord" + toString(i);
    taco_iassert(H5Tinsert(pointTy, fname.c_str(), offset, H5T_NATIVE_INT64_g) >= 0);
    offset += sizeof(long long);
  }

  // Now add lo and hi to the Rect type.
  auto rectTy = H5Tcreate(H5T_COMPOUND, sizeof(Rect<DIM>));
  std::string rectLo = "lo";
  std::string rectHi = "hi";
  offset = 0;
  taco_iassert(H5Tinsert(rectTy, rectLo.c_str(), offset, pointTy) >= 0);
  offset += sizeof(Point<DIM>);
  taco_iassert(H5Tinsert(rectTy, rectHi.c_str(), offset, pointTy) >= 0);

  return std::make_pair(pointTy, rectTy);
}

// ResourceCollector is a helper widget to perform automatic collection of various
// objects used in the lifetime of loading and dumping LegionTensors to HDF5.
struct ResourceCollector {
  ~ResourceCollector() {
    for (auto c : chars) { delete c; }
    for (auto id : HDFdatasets) { H5Dclose(id); }
    for (auto id : HDFdataspaces) { H5Sclose(id); }
    for (auto id : HDFfiles) { H5Fclose(id); }
  }
  void add(const char* c) { this->chars.push_back(c); }
  void addHDFfile(hid_t id) { taco_iassert(id >= 0); this->HDFfiles.push_back(id); }
  void addHDFdataspace(hid_t id) { taco_iassert(id >= 0); this->HDFdataspaces.push_back(id); }
  void addHDFdataset(hid_t id) { taco_iassert(id >= 0); this->HDFdatasets.push_back(id); }

  std::vector<const char*> chars;
  std::vector<hid_t> HDFdataspaces;
  std::vector<hid_t> HDFdatasets;
  std::vector<hid_t> HDFfiles;
};

void dumpLegionTensorToHDF5File(Legion::Context ctx, Legion::Runtime *runtime, LegionTensor &t, std::string &filename) {
  // Declare HDF5 types.
  hid_t pointTy, rectTy;
  std::tie(pointTy, rectTy) = createHDF5RectType<1>();
  auto format = t.format;

  // First, we must create the HDF5 file with all the right datasets and data spaces.
  {
    ResourceCollector col;

    // Open up the HDF5 file.
    hid_t fileID = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    col.addHDFfile(fileID);

    // Add the dimensions to the HDF5 file.
    hsize_t dimsDim[1];
    dimsDim[0] = format.size();
    auto dimsDataspace = H5Screate_simple(1, dimsDim, NULL);
    col.addHDFdataspace(dimsDataspace);
    auto dimsDataset = H5Dcreate2(fileID, LegionTensorDimsField, H5T_NATIVE_INT32_g, dimsDataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    col.addHDFdataset(dimsDataset);

    for (size_t i = 0; i < format.size(); i++) {
      switch (format[i]) {
        case Dense: {
          // We don't need to do anything to serialize a dense level.
          break;
        }
        case Sparse: {
          // Sparse levels are where most of the work is getting done.
          // Create a dataspace for the pos array.
          auto posDom = runtime->get_index_space_domain(ctx, t.indices[i][0].get_index_space());
          std::vector<hsize_t> dims(posDom.dim);
          for (int dim = 0; dim < posDom.dim; dim++) {
            dims[dim] = posDom.hi()[dim] + 1;
          }
          hid_t posDataspaceID = H5Screate_simple(posDom.dim, dims.data(), NULL);
          col.addHDFdataspace(posDataspaceID);
          // Now create a dataset in this dataspace.
          auto posName = getTensorLevelFormatName(format[i], i, 0);
          col.add(posName);
          hid_t posDataset = H5Dcreate2(fileID, posName, rectTy, posDataspaceID, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          col.addHDFdataset(posDataset);

          // Do the same for the crd array.
          auto crdDom = runtime->get_index_space_domain(ctx, t.indices[i][1].get_index_space());
          taco_iassert(crdDom.dim == 1);
          hsize_t crdDims[1];
          crdDims[0] = crdDom.hi()[0] + 1;
          auto crdName = getTensorLevelFormatName(format[i], i, 1);
          col.add(crdName);
          hid_t crdDataspaceID = H5Screate_simple(1, crdDims, NULL);
          col.addHDFdataspace(crdDataspaceID);
          // TODO (rohany): Should the crd region hold int32's or int64s?
          hid_t crdDataset = H5Dcreate2(fileID, crdName, H5T_NATIVE_INT32_g, crdDataspaceID, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          col.addHDFdataset(crdDataset);

          break;
        }
        case Singleton: {
          // TODO (rohany): Support this case. I'm a bit lazy and don't want to consider
          //  it right now since I won't actually be doing parallel computations on COO matrices.
          taco_iassert(false);
          break;
        }
        default:
          taco_iassert(false);
      }
    }

    // Dump out the values region.
    auto valsDom = runtime->get_index_space_domain(ctx, t.vals.get_index_space());
    std::vector<hsize_t> dims(valsDom.dim);
    for (int dim = 0; dim < valsDom.dim; dim++) {
      dims[dim] = valsDom.hi()[dim] + 1;
    }
    auto valsDataSpace = H5Screate_simple(valsDom.dim, dims.data(), NULL);
    col.addHDFdataspace(valsDataSpace);
    // TODO (rohany): Template this dataset creation on the value type.
    auto valDataset = H5Dcreate2(fileID, LegionTensorValsField, H5T_IEEE_F64LE_g, valsDataSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    col.addHDFdataset(valDataset);
  }

  // Next, we actually have to attach regions to the HDF5 file.

  // Create and map a region for the dimensions array, then attach it to the HDF5 file.
  {
    auto ispace = runtime->create_index_space(ctx, Rect<1>(0, t.order - 1));
    auto fspace = runtime->create_field_space(ctx);
    auto alloc = runtime->create_field_allocator(ctx, fspace);
    alloc.allocate_field(sizeof(int32_t), FID_COORD);
    auto dims = runtime->create_logical_region(ctx, ispace, fspace);
    runtime->fill_field(ctx, dims, dims, FID_COORD, int32_t(0));
    auto dimsCopy = runtime->create_logical_region(ctx, ispace, fspace);
    {
      auto dimsMem = legionMalloc(ctx, runtime, dims, dims, FID_COORD, READ_WRITE);
      FieldAccessor<READ_WRITE,int32_t,1,coord_t, Realm::AffineAccessor<int32_t, 1, coord_t>> acc(dimsMem, FID_COORD);
      for (int i = 0; i < t.order; i++) {
        acc[i] = t.dims[i];
      }
      runtime->unmap_region(ctx, dimsMem);
    }
    auto dimsDisk = attachHDF5RW(ctx, runtime, dimsCopy, {{FID_COORD, LegionTensorDimsField}}, filename);
    {
      CopyLauncher cl;
      cl.add_copy_requirements(RegionRequirement(dims, READ_ONLY, EXCLUSIVE, dims),
                               RegionRequirement(dimsCopy, WRITE_DISCARD, EXCLUSIVE, dimsCopy));
      cl.add_src_field(0, FID_COORD); cl.add_dst_field(0, FID_COORD);
      runtime->issue_copy_operation(ctx, cl);
    }
    runtime->detach_external_resource(ctx, dimsDisk).wait();
    runtime->destroy_logical_region(ctx, dims);
    runtime->destroy_logical_region(ctx, dimsCopy);
  }

  {
    ResourceCollector col;
    for (size_t i = 0; i < format.size(); i++) {
      switch (format[i]) {
        case Dense:
          // There's nothing to do for dense arrays.
          break;
        case Sparse: {
          // Create a new region for the pos and crd regions, and attach the HDF5 file to each.
          auto pos = t.indices[i][0];
          auto posParent = t.indicesParents[i][0];
          auto crd = t.indices[i][1];
          auto crdParent = t.indicesParents[i][1];
          auto posName = getTensorLevelFormatName(format[i], i, 0);
          auto crdName = getTensorLevelFormatName(format[i], i, 1);
          col.add(posName); col.add(crdName);

          auto posCopy = runtime->create_logical_region(ctx, pos.get_index_space(), pos.get_field_space());
          // TODO (rohany): Does it make sense to allow for direct access to the fields here?
          auto posDisk = attachHDF5RW(ctx, runtime, posCopy, {{FID_RECT_1, posName}}, filename);
          {
            CopyLauncher cl;
            cl.add_copy_requirements(RegionRequirement(pos, READ_ONLY, EXCLUSIVE, posParent),
                                     RegionRequirement(posCopy, WRITE_DISCARD, EXCLUSIVE, posCopy));
            cl.add_src_field(0, FID_RECT_1); cl.add_dst_field(0, FID_RECT_1);
            runtime->issue_copy_operation(ctx, cl);
            runtime->detach_external_resource(ctx, posDisk).wait();
          }

          auto crdCopy = runtime->create_logical_region(ctx, crd.get_index_space(), crd.get_field_space());
          auto crdDisk = attachHDF5RW(ctx, runtime, crdCopy, {{FID_COORD, crdName}}, filename);
          {
            CopyLauncher cl;
            cl.add_copy_requirements(RegionRequirement(crd, READ_ONLY, EXCLUSIVE, crdParent),
                                     RegionRequirement(crdCopy, WRITE_DISCARD, EXCLUSIVE, crdCopy));
            cl.add_src_field(0, FID_COORD); cl.add_dst_field(0, FID_COORD);
            runtime->issue_copy_operation(ctx, cl);
            runtime->detach_external_resource(ctx, crdDisk).wait();
          }

          runtime->destroy_logical_region(ctx, posCopy);
          runtime->destroy_logical_region(ctx, crdCopy);
          break;
        }
        case Singleton:
          // TODO (rohany): Support singleton.
          taco_iassert(false);
          break;
        default:
          taco_iassert(false);
      }
    }

    // Dump out the values array.
    auto valsCopy = runtime->create_logical_region(ctx, t.vals.get_index_space(), t.vals.get_field_space());
    auto valsDisk = attachHDF5RW(ctx, runtime, valsCopy, {{FID_VAL, LegionTensorValsField}}, filename);
    {
      CopyLauncher cl;
      cl.add_copy_requirements(RegionRequirement(t.vals, READ_ONLY, EXCLUSIVE, t.valsParent),
                               RegionRequirement(valsCopy, WRITE_DISCARD, EXCLUSIVE, valsCopy));
      cl.add_src_field(0, FID_VAL); cl.add_dst_field(0, FID_VAL);
      runtime->issue_copy_operation(ctx, cl);
      runtime->detach_external_resource(ctx, valsDisk).wait();
    }
    runtime->destroy_logical_region(ctx, valsCopy);
  }

  // Final cleanup operations.
  H5Tclose(pointTy);
  H5Tclose(rectTy);
}

// Corresponds to TID_ATTACH_SPECIFIC_REGION.
struct AttachSpecificRegion {
  void run(Context ctx, Runtime* runtime, std::string& filename, LogicalRegion reg, FieldID fid, std::string fieldName) {
    Serializer ser;
    ser.serialize(filename.length());
    ser.serialize(filename.data(), filename.length());
    ser.serialize(fieldName.length());
    ser.serialize(fieldName.data(), fieldName.length());
    ser.serialize(fid);
    TaskLauncher launcher(AttachSpecificRegion::taskID, TaskArgument(ser.get_buffer(), ser.get_used_bytes()));
    launcher.add_region_requirement(RegionRequirement(reg, READ_WRITE, EXCLUSIVE, reg, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(fid));
    runtime->execute_task(ctx, launcher).wait();
  }

  static void task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
    // Unpack data from our task arguments.
    std::string filename, fieldName;
    FieldID fid;
    size_t strLen;
    Deserializer derez(task->args, task->arglen);
    derez.deserialize(strLen); filename.resize(strLen);
    derez.deserialize(filename.data(), strLen);
    derez.deserialize(strLen); fieldName.resize(strLen);
    derez.deserialize(fieldName.data(), strLen);
    derez.deserialize(fid);

    std::map<FieldID, const char*> fieldMap({{fid, fieldName.c_str()}});
    auto reg = regions[0].get_logical_region();
    auto copy = runtime->create_logical_region(ctx, reg.get_index_space(), reg.get_field_space());
    auto regPhys = attachHDF5RO(ctx, runtime, copy, fieldMap, filename);
    CopyLauncher cl;
    cl.add_copy_requirements(
        RegionRequirement(copy, READ_ONLY, EXCLUSIVE, copy).add_field(fid),
        RegionRequirement(reg, READ_WRITE, EXCLUSIVE, reg).add_field(fid)
    );
    runtime->issue_copy_operation(ctx, cl);
    runtime->detach_external_resource(ctx, regPhys).wait();
    runtime->destroy_logical_region(ctx, copy);
  }

  static const int taskID = TID_ATTACH_SPECIFIC_REGION;
  static void registerTask() {
    TaskVariantRegistrar registrar(AttachSpecificRegion::taskID, "attachDimRegion");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<AttachSpecificRegion::task>(registrar, "attachSpecificRegion");
  }
};

std::pair<LegionTensor, ExternalHDF5LegionTensor>
loadLegionTensorFromHDF5File(Legion::Context ctx, Legion::Runtime *runtime, std::string &filename,
                             std::vector<LegionTensorLevelFormat> format) {
  hid_t pointTy, rectTy;
  std::tie(pointTy, rectTy) = createHDF5RectType<1>();

  ResourceCollector col;
  ExternalHDF5LegionTensor ex;

  auto rectSpace = runtime->create_field_space(ctx);
  {
    auto fa = runtime->create_field_allocator(ctx, rectSpace);
    fa.allocate_field(sizeof(Rect<1>), FID_RECT_1);
  }
  auto coordSpace = runtime->create_field_space(ctx);
  {
    auto fa = runtime->create_field_allocator(ctx, coordSpace);
    // TODO (rohany): Should the coord region be int64_t's?
    fa.allocate_field(sizeof(int32_t), FID_COORD);
  }
  auto valSpace = runtime->create_field_space(ctx);
  {
    auto fa = runtime->create_field_allocator(ctx, valSpace);
    fa.allocate_field(sizeof(double), FID_VAL);
  }

  auto getDatasetBounds = [&](const char* dataSetName, int dim) {
    auto f = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    auto dset = H5Dopen1(f, dataSetName);
    auto dspace = H5Dget_space(dset);
    std::vector<hsize_t> dims(dim);
    H5Sget_simple_extent_dims(dspace, dims.data(), NULL);
    DomainPoint lo, hi;
    lo.dim = dim;
    hi.dim = dim;
    for (int i = 0; i < dim; i++) {
      lo[i] = 0;
      hi[i] = dims[i] - 1;
    }
    H5Sclose(dspace);
    H5Dclose(dset);
    H5Fclose(f);
    return Domain(lo, hi);
  };

  // Load the dimensions region.
  std::vector<int32_t> dims(format.size());
  {
    auto dimsISpace = runtime->create_index_space(ctx, Rect<1>(0, format.size() - 1));
    auto dimsReg = runtime->create_logical_region(ctx, dimsISpace, coordSpace);
    AttachSpecificRegion().run(ctx, runtime, filename, dimsReg, FID_COORD, LegionTensorDimsField);
    // Now copy the values into the dims vector in the output.
    {
      auto dimsMem = legionMalloc(ctx, runtime, dimsReg, dimsReg, FID_COORD, READ_WRITE);
      FieldAccessor<READ_WRITE,int32_t,1,coord_t, Realm::AffineAccessor<int32_t, 1, coord_t>> acc(dimsMem, FID_COORD);
      for (size_t i = 0; i < format.size(); i++) {
        dims[i] = acc[i];
      }
      runtime->unmap_region(ctx, dimsMem);
    }
  }

  LegionTensor result(format, dims);

  // Loop through the level formats to query for the expected data sizes and shapes.

  // dimensionality is a counter that manages the runs of dense formats to understand
  // what dimension each tensor level has.
  int dimensionality = 0;
  for (size_t i = 0; i < format.size(); i++) {
    switch (format[i]) {
      case Dense: {
        // We don't have to do anything for dense levels except increment the
        // dimensionality counter.
        dimensionality++;
        break;
      }
      case Sparse: {
        auto posName = getTensorLevelFormatName(format[i], i, 0);
        auto crdName = getTensorLevelFormatName(format[i], i, 1);
        col.add(posName); col.add(crdName);

        // Get the expected bounds of each region.
        auto posBounds = getDatasetBounds(posName, dimensionality == 0 ? 1 : dimensionality);
        auto crdBounds = getDatasetBounds(crdName, 1);
        auto posIspace = runtime->create_index_space(ctx, posBounds);
        auto crdIspace = runtime->create_index_space(ctx, crdBounds);

        // If we were in a dense run, then we need to append the dense run.
        if (dimensionality >= 1 && format[i - 1] == Dense) {
          result.denseLevelRuns.push_back(posIspace);
        }

        // Create target regions as well as regions to attach to the instances.
        auto posReg = runtime->create_logical_region(ctx, posIspace, rectSpace);
        auto crdReg = runtime->create_logical_region(ctx, crdIspace, coordSpace);
        // Record these regions in the output.
        result.indices[i].push_back(posReg); result.indicesParents[i].push_back(posReg);
        result.indices[i].push_back(crdReg); result.indicesParents[i].push_back(crdReg);

        // Attach data to the regions.
        ex.addExternalAllocation(attachHDF5RO(ctx, runtime, posReg, {{FID_RECT_1, posName}}, filename));
        ex.addExternalAllocation(attachHDF5RO(ctx, runtime, crdReg, {{FID_COORD, crdName}}, filename));

        // Additionally, launch a dummy task to pull this region from disk into equal sizes in
        // CPU memories across the machine. We additionally mark this operation to allow the
        // instances to be considered ready for garbage collection (so that the actual instances
        // used when doing the computation can take over the needed space). We also mark this
        // operation as CPU-only, which forces these equal portions of memory to live in CPU memories
        // instead of being duplicated in the frame-buffer (in case we're running with GPUs).
        result.indicesEqPartitions[i].push_back(launchDummyRead(ctx, runtime, posReg, FID_RECT_1, true /* wait */, true /* untrack */, true /* cpuOnly */));
        result.indicesEqPartitions[i].push_back(launchDummyRead(ctx, runtime, crdReg, FID_COORD, true /* wait */, true /* untrack */, true /* cpuOnly */));

        // We reset dimensionality back to 1 for sparse levels. This allows us to cleanly
        // encode the fact that we need one more dimension for dense levels after a sparse level.
        dimensionality = 1;
        break;
      }
      case Singleton:
        // TODO (rohany): Support singleton.
        taco_iassert(false);
        break;
      default:
        taco_iassert(false);
    }
  }

  // Perform similar logic as in the sparse case to load the values.
  auto valsBounds = getDatasetBounds(LegionTensorValsField, dimensionality);
  auto valsIspace = runtime->create_index_space(ctx, valsBounds);
  auto vals = runtime->create_logical_region(ctx, valsIspace, valSpace);
  result.vals = vals;
  result.valsParent = vals;

  if (dimensionality >= 1 && format[format.size() - 1] == Dense) {
    result.denseLevelRuns.push_back(valsIspace);
  }
  ex.addExternalAllocation(attachHDF5RO(ctx, runtime, vals, {{FID_VAL, LegionTensorValsField}}, filename));
  // Do the same read operation for the values array.
  result.valsEqPartition = launchDummyRead(ctx, runtime, vals, FID_VAL, true /* wait */, true /* untrack */, true /* cpuOnly */);

  // Final cleanup operations.
  H5Tclose(pointTy);
  H5Tclose(rectTy);

  return {result, ex};
}

// This needs to be called in whatever main functions want to use HDF5 utility functions.
void registerHDF5UtilTasks() {
  {
    TaskVariantRegistrar registrar(TID_ATTACH_COO_REGIONS, "attachCOORegions");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<attachCOORegionsTask>(registrar, "attachCOORegions");
  }
  AttachSpecificRegion::registerTask();
  // We're going to use the dummy read tasks here as well.
  registerDummyReadTasks();
}
