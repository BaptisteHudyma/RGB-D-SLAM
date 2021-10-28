

# Run pose optimizations, with error tolerance
add_executable(testPoseOptimization
    ${TESTS}/test_pose_optimization.cpp
    )
target_link_libraries(testPoseOptimization
    gtest_main
    ${PROJECT_NAME}
    )



include(GoogleTest)
gtest_discover_tests(testPoseOptimization)
