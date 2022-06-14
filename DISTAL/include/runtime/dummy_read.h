#ifndef TACO_DUMMY_READ_H
#define TACO_DUMMY_READ_H

#include "legion.h"

// Utility functions that perform a naive dummy read operation over a partition
// in order to force reduction fills to be executed. launchDummyRead creates an
// equal partition to read over using heuristics in the mapper for the number
// of tasks to create. We also allow a knob for controlling whether these instances
// should be tracked by the runtime as valid or allow for being garbage collected.
// It returns the partition it created to perform the dummy read over.
Legion::LogicalPartition
launchDummyRead(Legion::Context ctx, Legion::Runtime *runtime, Legion::LogicalRegion reg, Legion::FieldID fid,
                bool wait = false, bool untrack = false, bool cpuOnly = false);

void launchDummyReadOverPartition(Legion::Context ctx, Legion::Runtime *runtime, Legion::LogicalRegion reg,
                                  Legion::LogicalPartition part, Legion::FieldID fid, Legion::Domain launchDim,
                                  bool wait = false, bool untrack = false, bool cpuOnly = false, bool sparse = false);

void registerDummyReadTasks();

#endif
