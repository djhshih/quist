project(quist)

find_package(Boost REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${PROJECT_BINARY_DIR} ${BOOST_INCLUDES})

add_executable(quist main.cpp)
target_link_libraries(quist boost_program_options)
