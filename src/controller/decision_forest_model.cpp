#include "decision_forest_model.h"
#include <cmath>

double DecisionForestModel::PredictCPU(const TaskDescriptor* task) {
    switch (task->type) {
        case TaskType::QUICKSORT:
            return static_cast<double>(task->dim_m) * 0.05;

        case TaskType::SGEMM: {
            double n = static_cast<double>(task->dim_m);
            return (2.0 * n * n * n) / 3.5e9 * 1e6;
        }

        case TaskType::CONV2D: {
            double pixels = static_cast<double>(task->dim_m * task->dim_n);
            double ksize = static_cast<double>(task->dim_k);
            return (pixels * ksize * ksize * 2.0) / 3.5e9 * 1e6;
        }

        case TaskType::SPMV: {
            double n = static_cast<double>(task->dim_m);
            double nnz = std::min(static_cast<double>(task->dim_n), 10.0);
            return (n * nnz * 2.0) / 3.5e9 * 1e6;
        }

        case TaskType::STRING_TOKEN:
            return static_cast<double>(task->dim_m) * 0.002;
    }
    return 1000.0;
}

double DecisionForestModel::PredictGPU(const TaskDescriptor* task) {
    switch (task->type) {
        case TaskType::QUICKSORT:
            return 50000.0;

        case TaskType::SGEMM: {
            double n = static_cast<double>(task->dim_m);
            return (2.0 * n * n * n) / 15.0e12 * 1e6;
        }

        case TaskType::CONV2D: {
            double pixels = static_cast<double>(task->dim_m * task->dim_n);
            double ksize = static_cast<double>(task->dim_k);
            return (pixels * ksize * ksize * 2.0) / 15.0e12 * 1e6;
        }

        case TaskType::SPMV: {
            double n = static_cast<double>(task->dim_m);
            double nnz = std::min(static_cast<double>(task->dim_n), 10.0);
            return (n * nnz * 2.0) / 15.0e12 * 1e6;
        }

        case TaskType::STRING_TOKEN:
            return 50000.0;
    }
    return 500.0;
}
