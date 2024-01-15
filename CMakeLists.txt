cmake_minimum_required(VERSION 3.5)

project(rgbSLAM)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

# debug symbols for programing tasks
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

set(CMAKE_CXX_STANDARD 20)
# -g for valgrind line of origin
set(CMAKE_CXX_FLAGS "-O4 -g -lgtest -pedantic -pedantic-errors -Wall -Wextra -Wfloat-equal -lflann_cpp -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DGLEW_STATIC")# -Wconversion -Wsign-conversion")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__FILENAME__='\"$(subst ${CMAKE_SOURCE_DIR}/,,$(abspath $<))\"'")
# if this is set, the progam will be deterministic, and loose some perfs
# add_compile_definitions(MAKE_DETERMINISTIC)
# reduce performances (not usefull for now)
# add_compile_definitions(USE_ORB_DETECTOR_AND_MATCHING)

# ORB detector/descriptor as a different implementation than classical detector in opencv
#add_compile_definitions(USE_ORB_DETECTOR_AND_MATCHING)

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

#Find boost
find_package(Boost REQUIRED)
list(APPEND INC_DIRS ${Boost_INCLUDE_DIRS} )
list(APPEND LINK_LIBS ${Boost_LIBS})

find_package(flann REQUIRED)
list(APPEND INC_DIRS "/usr/include/flann" )

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

set(COORDINATES ${SRC}/coordinates)

set(FEATS       ${SRC}/features)
set(KEYPOINTS   ${FEATS}/keypoints)
set(LINES       ${FEATS}/lines)
set(PRIMITIVES  ${FEATS}/primitives)

set(POSE_OPTI   ${SRC}/pose_optimization)
set(MAP         ${SRC}/map_management)
set(TRACKING    ${SRC}/tracking)
set(OUTPUTS     ${SRC}/outputs)
set(UTILS       ${SRC}/utils)

set(EXAMPLES    examples)

include_directories(SYSTEM
    ${INC_DIRS}
    )

include_directories(
    ${SRC}
    ${KEYPOINTS}
    ${LINES}
    ${PRIMITIVES}
    ${POSE_OPTI}
    ${MAP}
    ${THIRD_PARTY}
    ${TRACKING}
    ${OUTPUTS}
    ${UTILS}
    ${EXAMPLES}
    )

add_library(coordinates SHARED
    ${COORDINATES}/basis_changes.cpp
    ${COORDINATES}/inverse_depth_coordinates.cpp
    ${COORDINATES}/point_coordinates.cpp
    ${COORDINATES}/plane_coordinates.cpp
    ${COORDINATES}/polygon_coordinates.cpp
    )

add_library(keypoints SHARED
    ${KEYPOINTS}/keypoint_handler.cpp
    ${KEYPOINTS}/keypoint_detection.cpp
    )

add_library(lines SHARED
    ${LINES}/line_detection.cpp
    )

add_library(primitives SHARED
    ${PRIMITIVES}/depth_map_transformation.cpp
    ${PRIMITIVES}/primitive_detection.cpp
    ${PRIMITIVES}/plane_segment.cpp
    ${PRIMITIVES}/cylinder_segment.cpp
    ${PRIMITIVES}/shape_primitives.cpp
    )

add_library(mapManagement SHARED
    ${MAP}/local_map.cpp
    ${MAP}/map_point.cpp
    ${MAP}/map_point2d.cpp
    ${MAP}/map_primitive.cpp
    )

add_library(outputs SHARED
    ${OUTPUTS}/map_writer.cpp
    ${OUTPUTS}/logger.cpp
    )

add_library(poseOptimization SHARED
    ${POSE_OPTI}/levenberg_marquardt_functors.cpp
    ${POSE_OPTI}/pose_optimization.cpp
    )

add_library(tracking SHARED
${TRACKING}/inverse_depth_with_tracking.cpp
    ${TRACKING}/motion_model.cpp
    ${TRACKING}/plane_with_tracking.cpp
    ${TRACKING}/point_with_tracking.cpp
    ${TRACKING}/triangulation.cpp
)

add_library(utils SHARED
    ${UTILS}/angle_utils.cpp
    ${UTILS}/camera_transformation.cpp
    ${UTILS}/covariances.cpp
    ${UTILS}/distance_utils.cpp
    ${UTILS}/line.cpp
    ${UTILS}/polygon.cpp
    ${UTILS}/pose.cpp
    )

add_library(${PROJECT_NAME} SHARED
    ${SRC}/parameters.cpp
    ${SRC}/rgbd_slam.cpp
    )

add_library(thirdParty SHARED
    ${THIRD_PARTY}/concave_fitting.cpp
    ${THIRD_PARTY}/line_segment_detector.cpp
    ${THIRD_PARTY}/geodesic_operations.cpp
    ${THIRD_PARTY}/p3p.cpp
    )

target_link_libraries(${PROJECT_NAME}
    coordinates
    keypoints
    lines
    primitives
    mapManagement
    poseOptimization
    thirdParty
    tracking
    outputs
    utils
    ${LINK_LIBS} 
    )


# Examples of use
add_executable(slam_CAPE
    ${EXAMPLES}/main_CAPE.cpp)
target_link_libraries(slam_CAPE ${PROJECT_NAME})

add_executable(slam_TUM
    ${EXAMPLES}/main_TUM.cpp)
target_link_libraries(slam_TUM ${PROJECT_NAME})

add_executable(test_p3p
    ${THIRD_PARTY}/p3p_test.cpp)
target_link_libraries(test_p3p ${PROJECT_NAME})

# Include test dir
include(${CMAKE_MODULE_PATH}/runTests.cmake) 



