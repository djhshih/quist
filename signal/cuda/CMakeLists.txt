cmake_minimum_required(VERSION 2.8)
project(emd_cuda)

INCLUDE(FindCUDA)

set(CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=sm_20")

CUDA_ADD_EXECUTABLE(emd emd.cu)
CUDA_ADD_EXECUTABLE(emd_strat emd_strat.cu)
CUDA_ADD_EXECUTABLE(dsemd dsemd.cu)
CUDA_ADD_EXECUTABLE(spline spline.cu)
CUDA_ADD_EXECUTABLE(scan scan.cu)
CUDA_ADD_EXECUTABLE(tridiag tridiag.cu)