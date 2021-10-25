cmake_minimum_required(VERSION 3.1)

project(rgbSLAM)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

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
set(FEATS       ${SRC}/features)
set(POSE_OPTI   ${SRC}/pose_optimization)
set(MAP         ${SRC}/map_management)
set(UTILS       ${SRC}/utils)
set(EXAMPLES    examples)

include_directories(SYSTEM
    ${INC_DIRS}
    )

include_directories(
    ${SRC}
    ${FEATS}
    ${POSE_OPTI}
    ${MAP}
    ${THIRD_PARTY}
    ${UTILS}
    ${EXAMPLES}
    )

add_library(features SHARED
    ${FEATS}/DepthOperations.cpp
    ${FEATS}/PrimitiveDetection.cpp
    ${FEATS}/PlaneSegment.cpp
    ${FEATS}/CylinderSegment.cpp
    ${FEATS}/Histogram.cpp
    ${FEATS}/ShapePrimitives.cpp

    ${FEATS}/KeyPointDetection.cpp
    )

add_library(mapManagement SHARED
    ${MAP}/map_point.cpp
    ${MAP}/local_map.cpp
    )

add_library(thirdParty SHARED
    ${THIRD_PARTY}/LineSegmentDetector.cpp
    ${THIRD_PARTY}/GeodesicOperations.cpp
    ${THIRD_PARTY}/dsyevh3.cpp
    )

add_library(poseOptimization SHARED
    ${POSE_OPTI}/LevenbergMarquardFunctors.cpp
    ${POSE_OPTI}/PoseOptimization.cpp
    )

add_library(utils SHARED
    ${UTILS}/parameters.cpp
    ${UTILS}/utils.cpp
    ${UTILS}/Pose.cpp
    ${UTILS}/MotionModel.cpp
    )

add_library(${PROJECT_NAME} SHARED
    ${SRC}/RGBD_SLAM.cpp
    )

target_link_libraries(${PROJECT_NAME}
    features 
    mapManagement
    poseOptimization
    thirdParty
    utils
    ${LINK_LIBS} 
    )


# Examples of use
add_executable(slam_yoga_mat
    ${EXAMPLES}/main_yoga_mat_dataset.cpp)
target_link_libraries(slam_yoga_mat ${PROJECT_NAME})

add_executable(slam_tunnel_mat
    ${EXAMPLES}/main_tunnel_dataset.cpp)
target_link_libraries(slam_tunnel_mat ${PROJECT_NAME})

add_executable(slam_freiburg_xyz
    ${EXAMPLES}/main_xyz_dataset.cpp)
target_link_libraries(slam_freiburg_xyz ${PROJECT_NAME})


