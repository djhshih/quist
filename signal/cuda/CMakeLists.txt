cmake_minimum_required(VERSION 2.8)
project(emd_cuda)

include(FindOpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


include(FindCUDA)

set(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20")

CUDA_ADD_EXECUTABLE(emd_interl emd_interl.cu)
CUDA_ADD_EXECUTABLE(emd_strat emd_strat.cu)
CUDA_ADD_EXECUTABLE(dsemd dsemd.cu)

CUDA_ADD_EXECUTABLE(scan scan/example/inclusive_scan.cu)
CUDA_ADD_EXECUTABLE(spline spline/example/spline.cu)
CUDA_ADD_EXECUTABLE(spline_benchmark spline/example/spline_benchmark.cu)
CUDA_ADD_EXECUTABLE(tridiag tridiag/example/tridiag.cu)