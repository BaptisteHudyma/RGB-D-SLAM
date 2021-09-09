cmake_minimum_required(VERSION 3.1)

project(rgbSLAM)
set(CMAKE_BUILD_TYPE Debug)

set (CMAKE_CXX_STANDARD 20)
# -g for valgrind line of origin
set (CMAKE_CXX_FLAGS "-O4 -g -pedantic -pedantic-errors -Wall -Wextra")# -Wconversion -Wsign-conversion")

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

#add special cmakes (here for g2o)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

#Find opencv
find_package(OpenCV 4 REQUIRED)
list(APPEND INC_DIRS ${OpenCV_INCLUDE_DIRS})
list(APPEND LINK_LIBS ${OpenCV_LIBS})

#Find eigen
find_package(Eigen3 REQUIRED)
list(APPEND INC_DIRS ${EIGEN3_INCLUDE_DIR} )
list(APPEND LINK_LIBS ${Eigen_LIBRARIES})

#Find g2o
# Find the header files
FIND_PATH(G2O_INCLUDE_DIR g2o/core/base_vertex.h)
find_package(G2O REQUIRED)
set(G2O_LIBRARIES ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_TYPES_SBA} ${G2O_TYPES_SLAM3D} ${G2O_SOLVER_PCG})
list(APPEND INC_DIRS ${G2O_INCLUDE_DIR})
list(APPEND LINK_LIBS ${G2O_LIBRARIES})

MESSAGE(${LINK_LIBS})


set(SRC src)
set(THIRD_PARTY third_party)
set(PRIM    ${SRC}/primitives_detection)
set(POSE    ${SRC}/pose_estimation)
set(MAP     ${SRC}/map_management)
set(UTILS   ${SRC}/utils)

include_directories(SYSTEM
    ${INC_DIRS}
    )

include_directories(
    ${SRC}
    ${PRIM}
    ${POSE}  
    ${MAP}
    ${THIRD_PARTY}
    ${UTILS}
    )

add_executable(rgbslam 
    main.cpp
    )

add_library(primitivesDetection
    ${PRIM}/DepthOperations.cpp
    ${PRIM}/PrimitiveDetection.cpp
    ${PRIM}/PlaneSegment.cpp
    ${PRIM}/CylinderSegment.cpp
    ${PRIM}/Histogram.cpp
    ${PRIM}/DepthMapSegmentation.cpp
    ${PRIM}/ShapePrimitives.cpp
    )

add_library(poseEstimation
    ${POSE}/Pose.cpp
    ${POSE}/PoseUtils.cpp
    ${POSE}/Image_Features_Handler.cpp
    ${POSE}/Image_Features_Struct.cpp
    ${POSE}/LocalMap.cpp
    ${POSE}/MotionModel.cpp
    ${POSE}/Parameters.cpp
    ${POSE}/PNPSolver.cpp

    ${POSE}/RGB_Slam.cpp
    )

add_library(mapManagement
    ${MAP}/map_point.cpp
    ${MAP}/local_map.cpp
    )

add_library(thirdParty
    ${SRC}/MonocularDepthMap.cpp

    ${THIRD_PARTY}/LineSegmentDetector.cpp
    ${THIRD_PARTY}/GeodesicOperations.cpp
    ${THIRD_PARTY}/dsyevh3.cpp
    )

add_library(utils
    ${UTILS}/parameters.cpp
    ${UTILS}/utils.cpp
    ${UTILS}/KeyPointDetection.cpp
    ${UTILS}/LevenbergMarquardFunctors.cpp
    ${UTILS}/PoseOptimization.cpp
    )

add_library(slam
    ${SRC}/RGBD_SLAM.cpp
    )

target_link_libraries(rgbslam 
    slam
    primitivesDetection
    poseEstimation
    mapManagement
    thirdParty
    utils
    ${LINK_LIBS} 
    )


