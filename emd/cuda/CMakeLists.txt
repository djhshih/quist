cmake_minimum_required(VERSION 2.8)
project(emd_cuda)

INCLUDE(FindCUDA)

CUDA_ADD_EXECUTABLE(emd emd.cu)
