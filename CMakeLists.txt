cmake_minimum_required(VERSION 3.5)

PROJECT(AlgorithmTest C CXX)

add_compile_options(-O3)

# required packages
find_package(ROOT 6.0 REQUIRED CONFIG)
include(${ROOT_USE_FILE})

find_package(Eigen3 REQUIRED)

set(exe test)
add_executable(${exe}
    test.cpp
    FuzzyKClusters.cpp
)

target_include_directories(${exe}
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
    ${ROOT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(${exe}
LINK_PUBLIC
    ${ROOT_LIBRARIES}
)

install(TARGETS ${exe} DESTINATION ${CMAKE_INSTALL_BINDIR})

