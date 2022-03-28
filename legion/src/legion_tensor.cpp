#include "legion.h"
#include "legion_tensor.h"
#include "string_utils.h"

using namespace Legion;

LegionTensor::LegionTensor(LegionTensorFormat format, std::vector<int32_t> dims) :
  order(int(format.size())),
  dims(dims),
  indices(std::vector<std::vector<LogicalRegion>>(format.size())),
  indicesParents(std::vector<std::vector<LogicalRegion>>(format.size())),
  indicesFieldIDs(std::vector<std::vector<FieldID>>(format.size())),
  format(format),
  indicesEqPartitions(std::vector<std::vector<LogicalPartition>>(format.size()))
  {}

LegionTensor::LegionTensor(LegionTensorFormat format, int32_t order, std::vector<int32_t> dims,
                           std::vector<std::vector<Legion::LogicalRegion>> indices,
                           std::vector<std::vector<Legion::LogicalRegion>> indicesParents,
                           std::vector<std::vector<Legion::FieldID>> indicesFieldIDs,
                           Legion::LogicalRegion vals,
                           Legion::LogicalRegion valsParent,
                           Legion::FieldID valsFieldID,
                           std::vector<Legion::IndexSpace> denseLevelRuns) :
    order(order),
    dims(dims),
    indices(indices),
    indicesParents(indicesParents),
    indicesFieldIDs(indicesFieldIDs),
    vals(vals),
    valsParent(valsParent),
    valsFieldID(valsFieldID),
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
