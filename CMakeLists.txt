cmake_minimum_required(VERSION 2.8)

project(PlaneExtr)

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)


set (CMAKE_CXX_FLAGS "-O2")

set(CMAKE_BUILD_TYPE Release)
set(SRC src)
set(THIRD_PARTY third_party)

include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(${SRC})
include_directories(${THIRD_PARTY})

add_executable(planeExtraction
    main.cpp
    ${SRC}/DepthOperations.cpp
    ${SRC}/PlaneDetection.cpp
    ${SRC}/PlaneSegment.cpp
    ${SRC}/CylinderSegment.cpp
    ${SRC}/Histogram.cpp
    ${SRC}/MonocularDepthMap.cpp

    ${THIRD_PARTY}/LineSegmentDetector.cpp
    ${THIRD_PARTY}/dsyevh3.cpp
    )

target_link_libraries(planeExtraction ${OpenCV_LIBS} ${Eigen_LIBRARIES})


