cmake_minimum_required(VERSION 3.7)
project(quist)

find_package(Boost REQUIRED)

include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR} ${BOOST_INCLUDES})

add_executable(quist main.cpp)
target_link_libraries(quist boost_program_options)

if (NOT WIN32)
    add_subdirectory(numeric/test)
    add_subdirectory(signal/test)
endif (NOT WIN32)

#find_package(CUDA 4.2 REQUIRED)
#add_subdirectory(signal/cuda)
