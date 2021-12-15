#include "legion.h"
#include "legion_tensor.h"
#include "string_utils.h"

using namespace Legion;

LegionTensor::LegionTensor(LegionTensorFormat format, std::vector<int32_t> dims) :
  order(int(format.size())),
  dims(dims),
  indices(std::vector<std::vector<LogicalRegion>>(format.size())),
  indicesParents(std::vector<std::vector<LogicalRegion>>(format.size())),
  format(format),
  indicesEqPartitions(std::vector<std::vector<LogicalPartition>>(format.size()))
  {}

LegionTensor::LegionTensor(LegionTensorFormat format, int32_t order, std::vector<int32_t> dims,
                           std::vector<std::vector<Legion::LogicalRegion>> indices,
                           std::vector<std::vector<Legion::LogicalRegion>> indicesParents, Legion::LogicalRegion vals,
                           Legion::LogicalRegion valsParent, std::vector<Legion::IndexSpace> denseLevelRuns) :
    order(order),
    dims(dims),
    indices(indices),
    indicesParents(indicesParents),
    vals(vals),
    valsParent(valsParent),
    denseLevelRuns(denseLevelRuns),
    format(format),
    indicesEqPartitions(std::vector<std::vector<LogicalPartition>>(order)) {}

std::string LegionTensor::toString(Legion::Context ctx, Legion::Runtime *runtime) {
  // TODO (rohany): This method does not display information about region parents yet.
  std::stringstream out;
  out << "Order: " << this->order << std::endl;
  out << "Dimensions: " << join(this->dims) << std::endl;
  out << "Indices: " << std::endl;
  for (size_t i = 0; i < this->indices.size(); i++) {
    out << "\t" << i << ": ";
    for (auto reg : this->indices[i]) {
      out << runtime->get_index_space_domain(ctx, reg.get_index_space()) << " ";
    }
    out << std::endl;
  }
  out << "DenseLevelRuns: " << std::endl;
  for (size_t i = 0; i < this->denseLevelRuns.size(); i++) {
    out << "\t" << i << ": " << runtime->get_index_space_domain(ctx, this->denseLevelRuns[i]) << std::endl;
  }
  out << "Values: " << runtime->get_index_space_domain(ctx, this->vals.get_index_space()) << std::endl;
  return out.str();
}

void ExternalHDF5LegionTensor::addExternalAllocation(Legion::PhysicalRegion r) {
  this->attachedRegions.push_back(r);
}

void ExternalHDF5LegionTensor::destroy(Legion::Context ctx, Legion::Runtime* runtime) {
  for (auto& r : this->attachedRegions) {
    runtime->detach_external_resource(ctx, r);
  }
}

LegionTensor copyNonZeroStructure(Context ctx, Runtime* runtime, LegionTensorFormat format, LegionTensor src) {
  // Double check that the result format is a prefix of the source format.
  taco_uassert(format.size() <= src.format.size());
  std::vector<int32_t> dims;
  for (size_t i = 0; i < format.size(); i++) {
    dims.push_back(src.dims[i]);
    taco_uassert(format[i] == src.format[i]);
  }

  auto createLogicalPart = [&](LogicalRegion part, IndexSpace domain) {
    auto ipart = runtime->create_equal_partition(ctx, part.get_index_space(), domain);
    return runtime->get_logical_partition(ctx, part, ipart);
  };

  auto getField = [&](LogicalRegion reg) {
    std::vector<FieldID> fields;
    runtime->get_field_space_fields(ctx, reg.get_field_space(), fields);
    taco_uassert(fields.size() == 1);
    return fields[0];
  };

  // TODO (rohany): We'll assume for now that once we hit a sparse level,
  //  we can't run into any more dense levels. This is just an implementation
  //  limitation because I'm too lazy to implement some of the logic needed
  //  for sparse-dense style formats.

  bool seenSparse = false;
  Domain currentValsDomain;
  LegionTensor result(format, dims);
  // TODO (rohany): This is probably not correct, but I'll run with it for now.
  result.denseLevelRuns = src.denseLevelRuns;
  for (size_t level = 0; level < format.size(); level++) {
    switch (format[level]) {
      case Dense: {
        taco_uassert(!seenSparse) << "currently not supporting sparse-dense formats";
        break; // Nothing to do here.
      }
      case Sparse: {
        seenSparse = true;
        // Copy the pos and crd arrays over from the source tensor.
        auto srcPosReg = src.indices[level][0];
        auto srcPosParent = src.indicesParents[level][0];
        auto srcCrdReg = src.indices[level][1];
        auto srcCrdParent = src.indicesParents[level][1];
        auto posField = getField(srcPosReg);
        auto crdField = getField(srcCrdReg);
        auto posReg = runtime->create_logical_region(ctx, srcPosReg.get_index_space(), srcPosReg.get_field_space());
        auto crdReg = runtime->create_logical_region(ctx, srcCrdReg.get_index_space(), srcCrdReg.get_field_space());

        // Add the regions to the result tensor.
        result.indices[level].push_back(posReg);
        result.indicesParents[level].push_back(posReg);
        result.indices[level].push_back(crdReg);
        result.indicesParents[level].push_back(crdReg);

        // Now copy the regions over from the source into the destination.
        // For simplicity, we'll assert that some partitions of the source tensor
        // have already been created.
        taco_uassert(src.indicesEqPartitions[level].size() == 2);
        auto srcPosPart = src.indicesEqPartitions[level][0];
        auto srcCrdPart = src.indicesEqPartitions[level][1];
        taco_uassert(srcPosPart.exists());
        taco_uassert(srcCrdPart.exists());

        // Create some equal partitions of the destination arrays as well.
        auto posPart = createLogicalPart(posReg, runtime->get_index_partition_color_space_name(srcPosPart.get_index_partition()));
        auto crdPart = createLogicalPart(crdReg, runtime->get_index_partition_color_space_name(srcCrdPart.get_index_partition()));
        result.indicesEqPartitions[level].push_back(posPart);
        result.indicesEqPartitions[level].push_back(crdPart);

        // Launch an IndexCopy over these partitions.
        IndexCopyLauncher launcher(runtime->get_index_partition_color_space_name(srcPosPart.get_index_partition()));
        launcher.add_copy_requirements(
          RegionRequirement(srcPosPart, 0, READ_ONLY, EXCLUSIVE, srcPosParent).add_field(posField),
          RegionRequirement(posPart, 0, WRITE_ONLY, EXCLUSIVE, posReg).add_field(posField)
        );
        launcher.add_copy_requirements(
          RegionRequirement(srcCrdPart, 0, READ_ONLY, EXCLUSIVE, srcCrdParent).add_field(crdField),
          RegionRequirement(crdPart, 0, WRITE_ONLY, EXCLUSIVE, crdReg).add_field(crdField)
        );
        runtime->issue_copy_operation(ctx, launcher);

        // Finally, remember that this is the dimensionality of the vals region.
        currentValsDomain = runtime->get_index_space_domain(srcCrdReg.get_index_space());
        break;
      }
      case Singleton: {
        taco_iassert(false) << "not handling the Singleton case here yet";
        break;
      }
    }
  }

  // Perform a similar operation as above but for the values array.
  // However, we only need to construct the values, not copy anything into them.
  taco_uassert(currentValsDomain.exists()) << "cannot copy prefix to dense tensor";
  auto valsIspace = runtime->create_index_space(ctx, currentValsDomain);
  auto vals = runtime->create_logical_region(ctx, valsIspace, src.vals.get_field_space());
  result.vals = vals;
  result.valsParent = vals;
  // TODO (rohany): This method should be templated over the result
  //  tensor type, or the LegionTensor should hold onto what type the
  //  elements actually are (like TACO) so that we can safely do these
  //  fill operations.
  runtime->fill_field(ctx, vals, vals, getField(vals), (double)0);
  return result;
}
