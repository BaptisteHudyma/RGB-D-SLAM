

# Run pose optimizations, with error tolerance
add_executable(testPoseOptimization
    ${TESTS}/test_pose_optimization.cpp
    )
add_executable(testKalmanFiltering
    ${TESTS}/test_kalman_filtering.cpp
    )

target_link_libraries(testPoseOptimization
    gtest_main
    ${PROJECT_NAME}
    )
target_link_libraries(testKalmanFiltering
    gtest_main
    ${PROJECT_NAME}
    )


include(GoogleTest)
gtest_discover_tests(testPoseOptimization)
gtest_discover_tests(testKalmanFiltering)
