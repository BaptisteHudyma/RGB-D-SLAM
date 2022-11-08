# Tests

Those tests are based on gtest, and are high level unitary tests.

## test_coordinates_systems
This files contains tests for the coordinate system switching.
Transforms screen to camera to world points and back.
Transforms camera to world planes and back.

## test_kalman_filtering
This file launches a set of unit tests for the Kalman filtering process.
Those examples include basic free falling objects, vehicule position tracking, etc

## test_pose_optimization
This file launches a set of unit tests for the frame by frame pose optimization process.
We generate a point cloud, apply random translations and rotations, with some noise, and feed the original and transformed clouds to the optimization process.

We evaluate the ability of the pose optimization process by comparing the original pose to the pose found by the optimization process.