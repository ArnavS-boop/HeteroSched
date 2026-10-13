#ifndef WORKLOAD_TYPES_H
#define WORKLOAD_TYPES_H

#include "common.h"
#include <string>

inline const char* TaskTypeToString(TaskType t) {
    switch (t) {
        case TaskType::SGEMM:        return "SGEMM";
        case TaskType::CONV2D:       return "CONV2D";
        case TaskType::SPMV:         return "SPMV";
        case TaskType::QUICKSORT:    return "QUICKSORT";
        case TaskType::STRING_TOKEN: return "STRING_TOKEN";
    }
    return "UNKNOWN";
}

inline TaskType StringToTaskType(const std::string& s) {
    if (s == "SGEMM")        return TaskType::SGEMM;
    if (s == "CONV2D")       return TaskType::CONV2D;
    if (s == "SPMV")         return TaskType::SPMV;
    if (s == "QUICKSORT")    return TaskType::QUICKSORT;
    if (s == "STRING_TOKEN") return TaskType::STRING_TOKEN;
    return TaskType::SGEMM;
}

inline const char* DeviceToString(ExecutionDevice d) {
    return (d == ExecutionDevice::DEVICE_GPU) ? "GPU" : "CPU";
}

#endif
