

# Run pose optimizations, with error tolerance
add_executable(poseOptimizationTest
    ${TESTS}/pose_optimization_test.cpp
    )
target_link_libraries(poseOptimizationTest
    gtest_main
    ${PROJECT_NAME}
    )



include(GoogleTest)
gtest_discover_tests(poseOptimizationTest)
