

# Run pose optimizations, with error tolerance
add_executable(testCoordinateSystems
    ${TESTS}/test_coordinate_systems.cpp
    )
add_executable(testPoseOptimization
    ${TESTS}/test_pose_optimization.cpp
    )
add_executable(testKalmanFiltering
    ${TESTS}/test_kalman_filtering.cpp
    )
add_executable(testPolygons
    ${TESTS}/test_polygons.cpp
    )
add_executable(testMotionModel
    ${TESTS}/test_motion_model.cpp
    )

target_link_libraries(testCoordinateSystems
    gtest_main
    ${PROJECT_NAME}
    )
target_link_libraries(testPoseOptimization
    gtest_main
    ${PROJECT_NAME}
    )
target_link_libraries(testKalmanFiltering
    gtest_main
    ${PROJECT_NAME}
    )
target_link_libraries(testPolygons
    gtest_main
    ${PROJECT_NAME}
    )
target_link_libraries(testMotionModel
    gtest_main
    ${PROJECT_NAME}
    )

include(GoogleTest)
gtest_discover_tests(testCoordinateSystems)
gtest_discover_tests(testPoseOptimization)
gtest_discover_tests(testKalmanFiltering)
gtest_discover_tests(testPolygons)
gtest_discover_tests(testMotionModel)
