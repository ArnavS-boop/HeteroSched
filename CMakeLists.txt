cmake_minimum_required(VERSION 3.18)
project(HeteroSched LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

include_directories(
    include
    ${CUDAToolkit_INCLUDE_DIRS}
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --extended-lambda")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread -Wall")

add_library(heterosched_core STATIC
    src/cpu/cpu_scheduler.cpp
    src/cpu/cpu_kernels.cpp
    src/gpu/gpu_executor.cu
    src/gpu/gpu_kernels.cu
    src/controller/gated_controller.cpp
    src/controller/decision_forest_model.cpp
    src/controller/telemetry_monitor.cpp
)

target_link_libraries(heterosched_core
    PUBLIC
    CUDA::cudart
    Threads::Threads
)

add_executable(heterosched_bench src/main.cpp)
target_link_libraries(heterosched_bench
    PRIVATE
    heterosched_core
)

add_executable(test_phase_a tests/test_phase_a.cpp)
target_link_libraries(test_phase_a
    PRIVATE
    heterosched_core
)
target_compile_options(test_phase_a PRIVATE -UNDEBUG)

add_executable(test_phase_b tests/test_phase_b.cpp)
target_link_libraries(test_phase_b
    PRIVATE
    heterosched_core
)
target_compile_options(test_phase_b PRIVATE -UNDEBUG)
