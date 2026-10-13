#ifndef DECISION_FOREST_MODEL_H
#define DECISION_FOREST_MODEL_H

#include "common.h"

class DecisionForestModel {
public:
    double PredictCPU(const TaskDescriptor* task);
    double PredictGPU(const TaskDescriptor* task);
};

#endif
