#ifndef TACO_GENERATED_H
#define TACO_GENERATED_H
#include "legion.h"
#include "legion_tensor.h"


void packLegionCOOToCSR(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* T, LegionTensor* TCOO);

void packLegionCOOToSSS(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* T, LegionTensor* TCOO);

void packLegionCOOToDSS(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* T, LegionTensor* TCOO);

void packLegionCOOToDDS(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* T, LegionTensor* TCOO);

void packLegionCOOToSDS(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* T, LegionTensor* TCOO);

void packLegionCOOToDCSR(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* T, LegionTensor* TCOO);

void packLegionCOOToSD(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* T, LegionTensor* TCOO);

void packLegionCOOToCSC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* T, LegionTensor* TCOO);

void packLegionCOOToVec(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* T, LegionTensor* TCOO);
void registerTacoTasks();
#endif // TACO_GENERATED_H
