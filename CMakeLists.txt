cmake_minimum_required(VERSION 3.1)

project(rgbSLAM)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

set(CMAKE_CXX_STANDARD 20)
# -g for valgrind line of origin
set(CMAKE_CXX_FLAGS "-O4 -g -lgtest -pedantic -pedantic-errors -Wall -Wextra -Wfloat-equal")# -Wconversion -Wsign-conversion")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__FILENAME__='\"$(subst ${CMAKE_SOURCE_DIR}/,,$(abspath $<))\"'")

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

# Load GTEST
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
## For Windows: Prevent overriding the parent project's compiler/linker settings
#set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
#FetchContent_MakeAvailable(googletest)
#
set(TESTS tests)
enable_testing()



# Compile main program
set(SRC src)
set(THIRD_PARTY third_party)
set(FEATS       ${SRC}/features)
set(KEYPOINTS   ${FEATS}/keypoints)
set(PRIMITIVES  ${FEATS}/primitives)
set(POSE_OPTI   ${SRC}/pose_optimization)
set(MAP         ${SRC}/map_management)
set(UTILS       ${SRC}/utils)
set(EXAMPLES    examples)

include_directories(SYSTEM
    ${INC_DIRS}
    )

include_directories(
    ${SRC}
    ${KEYPOINTS}
    ${PRIMITIVES}
    ${POSE_OPTI}
    ${MAP}
    ${THIRD_PARTY}
    ${UTILS}
    ${EXAMPLES}
    )

add_library(keypoints SHARED
    ${KEYPOINTS}/keypoint_handler.cpp
    ${KEYPOINTS}/keypoint_detection.cpp
    )

add_library(primitives SHARED
    ${PRIMITIVES}/depth_map_transformation.cpp
    ${PRIMITIVES}/primitive_detection.cpp
    ${PRIMITIVES}/plane_segment.cpp
    ${PRIMITIVES}/cylinder_segment.cpp
    ${PRIMITIVES}/histogram.cpp
    ${PRIMITIVES}/shape_primitives.cpp
    )

add_library(mapManagement SHARED
    ${MAP}/map_point.cpp
    ${MAP}/local_map.cpp
    )

add_library(thirdParty SHARED
    ${THIRD_PARTY}/line_segment_detector.cpp
    ${THIRD_PARTY}/geodesic_operations.cpp
    ${THIRD_PARTY}/dsyevh3.cpp
    ${THIRD_PARTY}/p3p.cpp
    )

add_library(poseOptimization SHARED
    ${POSE_OPTI}/levenberg_marquard_functors.cpp
    ${POSE_OPTI}/pose_optimization.cpp
    )

add_library(utils SHARED
    ${UTILS}/angle_utils.cpp
    ${UTILS}/distance_utils.cpp
    ${UTILS}/covariances.cpp
    ${UTILS}/camera_transformation.cpp
    ${UTILS}/logger.cpp
    ${UTILS}/pose.cpp
    ${UTILS}/motion_model.cpp
    ${UTILS}/triangulation.cpp
    ${UTILS}/map_writer.cpp
    ${UTILS}/kalman_filter.cpp
    )

add_library(${PROJECT_NAME} SHARED
    ${SRC}/parameters.cpp
    ${SRC}/rgbd_slam.cpp
    )

target_link_libraries(${PROJECT_NAME}
    keypoints
    primitives
    mapManagement
    poseOptimization
    thirdParty
    utils
    ${LINK_LIBS} 
    )


# Examples of use
add_executable(slam_CAPE
    ${EXAMPLES}/main_CAPE.cpp)
target_link_libraries(slam_CAPE ${PROJECT_NAME})

add_executable(slam_freiburg1
    ${EXAMPLES}/main_freiburg1.cpp)
target_link_libraries(slam_freiburg1 ${PROJECT_NAME})

add_executable(slam_freiburg2
    ${EXAMPLES}/main_freiburg2.cpp)
target_link_libraries(slam_freiburg2 ${PROJECT_NAME})

add_executable(test_p3p
    ${THIRD_PARTY}/p3p_test.cpp)
target_link_libraries(test_p3p ${PROJECT_NAME})


# Include test dir
include(${CMAKE_MODULE_PATH}/runTests.cmake) 



