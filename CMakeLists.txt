cmake_minimum_required(VERSION 2.8)

project(PlaneExtr)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)


set (CMAKE_CXX_FLAGS "-Wall -O0")

set(CMAKE_BUILD_TYPE Release)

include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(planeExtraction
    main.cpp
    DepthOperations.cpp
    PlaneDetection.cpp
    PlaneSegment.cpp
	CylinderSegment.cpp
	Histogram.cpp)

target_link_libraries(planeExtraction ${OpenCV_LIBS} ${Eigen_LIBRARIES})


