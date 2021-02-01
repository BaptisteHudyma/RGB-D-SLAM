cmake_minimum_required(VERSION 3.1)

project(rgbSLAM)

set (CMAKE_CXX_STANDARD 20)

find_package(OpenCV 4 REQUIRED)
list(APPEND LINK_LIBS ${OpenCV_LIBS})
find_package(Eigen3 REQUIRED)
list(APPEND LINK_LIBS ${Eigen_LIBRARIES})
find_package(g2o)
list(APPEND LINK_LIBS ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_TYPES_SBA} ${G2O_TYPES_SLAM3D} ${G2O_SOLVER_PCG})

include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${G20_INCLUDE_DIR})

set (CMAKE_CXX_FLAGS "-O2")
set(CMAKE_BUILD_TYPE Release)


set(SRC src)
set(THIRD_PARTY third_party)
set(PRIM ${SRC}/primitives_detection)
set(POSE ${SRC}/pose_estimation)

include_directories(${SRC} ${PRIM} ${POSE})
include_directories(${THIRD_PARTY})


add_executable(rgbslam 
    main.cpp
    )

add_library(primitivesDetection
    ${PRIM}/DepthOperations.cpp
    ${PRIM}/PrimitiveDetection.cpp
    ${PRIM}/PlaneSegment.cpp
    ${PRIM}/CylinderSegment.cpp
    ${PRIM}/Histogram.cpp
    )
add_library(poseEstimation
    ${POSE}/Image_Features_Handler.cpp
    ${POSE}/Image_Features_Struct.cpp
    ${POSE}/LocalMap.cpp
    ${POSE}/MotionModel.cpp
    ${POSE}/Parameters.cpp
    ${POSE}/PNPSolver.cpp
    ${POSE}/PoseUtils.cpp
    )
add_library(thirdParty
    ${SRC}/MonocularDepthMap.cpp

    ${THIRD_PARTY}/LineSegmentDetector.cpp
    ${THIRD_PARTY}/dsyevh3.cpp
    )

target_link_libraries(rgbslam 
    primitivesDetection
    poseEstimation
    thirdParty
    ${LINK_LIBS} 
    )


