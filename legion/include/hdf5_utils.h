#ifndef TACO_LEGION_HDF5_UTILS_H
#define TACO_LEGION_HDF5_UTILS_H
#include <hdf5.h>
#include <string>
#include <vector>

#include "legion.h"
#include "legion_tensor.h"

// Constants for coordinate list HDF5 storage.
const char* const COODimsField = "dims";
const char* const COOValsField = "vals";
const char* const COOCoordsFields[] = {
    "coord0",
    "coord1",
    "coord2",
    "coord3",
    "coord4",
    "coord5",
    "coord6",
    "coord7",
    // TODO (rohany): Add more if needed.
};
// generateCoordListHDF5 creates a serialized coordinate list representation of a
// tensor in an HDF5 file. It uses int32_t's to store coordinates and dimension
// information and double's to store the values. The generated HDF5 file has
// the following structure:
//   dims: ...
//   vals: ...
//   coord0: ...
//   coord1: ...
//   ...
void generateCoordListHDF5(std::string filename, size_t order, size_t nnz);
// getCoordListHDF5Meta extracts metadata from an HDF5 file created by generateCoordListHDF5.
void getCoordListHDF5Meta(std::string filename, size_t& order, size_t& nnz);

// Helpers to attach a HDF5 file to a logical region. The returned PhysicalRegion
// must be explicitly deallocated with runtime->detach_external_resource. We use
// separate functions for each as RO operations must be attached in a different manner
// to play nicely with control replication.
Legion::PhysicalRegion attachHDF5RW(Legion::Context ctx, Legion::Runtime *runtime, Legion::LogicalRegion region,
                                  std::map<Legion::FieldID, const char *> fieldMap, std::string filename);
Legion::PhysicalRegion attachHDF5RO(Legion::Context ctx, Legion::Runtime *runtime, Legion::LogicalRegion region,
                                  std::map<Legion::FieldID, const char *> fieldMap, std::string filename);

// Load a COO tensor from a HDF5 file into a LegionTensor. The COO HDF5 tensor
// must have been created by the tns_to_hdf5 utility.
LegionTensor loadCOOFromHDF5(Legion::Context ctx, Legion::Runtime *runtime, std::string &filename,
                             Legion::FieldID rectFieldID, Legion::FieldID coordField, size_t coordSize,
                             Legion::FieldID valsField, size_t valsSize);

// Constants for general tensor in HDF5 format storage.
const char* const LegionTensorDimsField = "dims";
const char* const LegionTensorValsField = "vals";

// TODO (rohany): Template this over the value type?
void dumpLegionTensorToHDF5File(Legion::Context ctx, Legion::Runtime *runtime, LegionTensor &t, std::string &filename);
// TODO (rohany): Template this over the value type?
std::pair<LegionTensor, ExternalHDF5LegionTensor>
loadLegionTensorFromHDF5File(Legion::Context ctx, Legion::Runtime *runtime, std::string &filename, LegionTensorFormat format);

// Registration function that must be called during initialization if any hdf5_utils task are to be used.
void registerHDF5UtilTasks();

#endif // TACO_LEGION_HDF5_UTILS_H
