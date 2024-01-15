# Sources

The program source files are organized as so:
- **coordinates** define the diverse coordinates used in this program (screen, camera and world point, inverse depth points, camera plane, world plane, ...)
- **features**: Handles the feature detection and matching.
    - **keypoints**: detect and match keypoints, using their descriptors and short term optical flow
    - **lines**: detect and match lines
    - **primitives**: Handle the detection and tracking of planes and cylinders, based on a connected graph technique described in [Fast Cylinder and Plane Extraction from Depth Cameras for Visua Odometry](https://arxiv.org/pdf/1803.02380.pdf).
- **map_management** The "Mapping" part of SLAM. It's a local map implementation, with no loop closures
- **outputs** All the outputs that this program produces are stored here EXCEPT the images, that are handled by their classes (the map display is handled by the map class)
- **pose_optimization** Contains all the pose optimization functions. The optimization process uses a Levenberg-Marquardt algorithm for frame to frame optimization
- **tracking** The tracking functions, as Kalman filters and observers motion model. it also handle the tracking of the feature used in the program (points, planes, inverse depth points)
- **utils** Math and cameras utils functions. Also contains the main types (eg: Pose)


- **matches_containes**: Containers for feature matches
- **parameters**: Contains all parameters (magic numbers !) for this program.
- **rgbd_slam**: Main class of this program
- **types**: Define all base types used in this program (specially vectors and matrices) 