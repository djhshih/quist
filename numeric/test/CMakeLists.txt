cmake_minimum_required(VERSION 2.8)
project(quist)

find_package(Boost REQUIRED)
find_package(HDF5 REQUIRED)

add_executable(bla_test bla_test.cpp)
target_link_libraries(bla_test hdf5 hdf5_hl boost_unit_test_framework)

# Copy data files from source to build tree
file(COPY data/ DESTINATION . FILES_MATCHING REGEX "/.*\\.(h5)$")
