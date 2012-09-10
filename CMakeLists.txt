cmake_minimum_required(VERSION 2.8)
project(quist)

find_package(Boost REQUIRED)
find_package(CUDA 4.2 REQUIRED)

include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR} ${BOOST_INCLUDES})

add_executable(quist main.cpp)
target_link_libraries(quist boost_program_options)

add_subdirectory(numeric/test)
add_subdirectory(signal/test)
add_subdirectory(signal/cuda)