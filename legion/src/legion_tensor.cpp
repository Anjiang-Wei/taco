#include "legion.h"
#include "legion_tensor.h"
#include "string_utils.h"

using namespace Legion;

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
