cmake_minimum_required(VERSION 3.6)
project(quist)

find_package(Boost REQUIRED)

include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR} ${BOOST_INCLUDES})

add_executable(quist main.cpp)
target_link_libraries(quist boost_program_options)


# optional dependencies

if (NOT WIN32)
    find_package(HDF5)
    if (HDF5_FOUND)
        add_subdirectory(numeric/test)
        add_subdirectory(signal/test)
    endif (HDF5_FOUND)
endif (NOT WIN32)

find_package(CUDA 4.2)
if (CUDA_FOUND)
    add_subdirectory(signal/cuda)
endif (CUDA_FOUND)
