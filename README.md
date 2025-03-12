# RGB-D-SLAM

My take on a SLAM, based on shape primitive recognition with a RGB-D camera.
The system extracts primitive shapes for the depth image, and use those primitives with 3D points and lines to estimate the observer position in space.
A map is created with those points, lines and primitive shapes.
See the Doxygen documentation for this program on [GitHub Pages](https://baptistehudyma.github.io/RGB-D-SLAM/html/index.html).

Each map feature is tracked using independent Kalman filter instead of bundle adjustment.

Point detection is based on [ORB detection and descriptors](https://ieeexplore.ieee.org/document/6126544), as well as optical flow for short term tracking.
Line detection uses [LSD - Line Segment Detector](http://www.ipol.im/pub/art/2012/gjmr-lsd/).
The primitive detection is based on [Fast Cylinder and Plane Extraction from Depth Cameras for Visual Odometry](https://arxiv.org/pdf/1803.02380.pdf).


For now, the program can do basic visual odometry with local mapping, and do not uses loop closures.
The program runs above real time (200 - 400 FPS), the feature extraction process being the most time consumming part (50%).


## Run example
Here is a run example on the dataset icl_living_room_kt2, from TUM.

![SLAM run example](/Medias/SLAM_run_2.gif)

The left image is the input RGB image.
It has an associated depth image, not shown here for clarity.

The right image is the global map, seen from above.


The colored lines are the map plane boundaries, and the points with a colored dot.
The top bar indicates the average FPS (excluding the visualisation), and the map content as "FeatureType: NumberInLocalMap: NumberInGlobalMap".

P2D corresponds to 2D points, and stays to zero because this dataset as almost no depth discontinuity.

## Examples
Some examples are provided, but their datasets must be downloaded separatly.
A configuration file must be provided for every example, based on the model given in **configuration_example.yaml**.
This configuration file must be placed in the **data/my_dataset** folder.

- **CAPE** : The CAPE dataset contains RGB-D (size: 640x480) sequences of low textured environments, that are challenging for point-based SLAMs. It contains planes and cylinders, initially to test the CAPE process.
The groundtrut trajectory are given for some sequences. 
- **TUM-RGBD** : The TUM RGBD datasets are sequences of raw RGB-D (size: 640x480), with accelerometers data and groundtruth trajectories. There is a lot of different videos, with different conditions (translation/rotations only, dynamic environments, low texture, ...).
The RGB and depth images must be synchronised by the user. The example program given here synchronises them with a greedy method.

## packages
```
opencv
Eigen
boost
flann
pugixml
vtk
fmt
```

## Build and Run
```
mkdir build && cd build
cmake ..
make
```

### Run the tests
```
./test_p3p
./testPoseOptimization
./testKalmanFiltering
```

When all the tests are validated, you can run the main SLAM algorithm

### How to use
Use the provided example programs with the dataset at your disposal.
The dataset should be located next to the src folder, in a data folder.
Ex: For TUM fr1_xyz, you should place it in data/TUM/fr1_xyz

Running this program will produce a map file (format is .xyz for now) at the location of the executable.

While the program is running, an OpenCV windows displays the current frame with the tracked features in it.
Each feature is assigned to a random color, that will never change during the mapping process.
A good tracking/SLAM process will keep the same feature's colors during the whole process.

The top bar displays the number of points in the local map, as well as the planes and their colors.


At the end of the process, a file out.obj is produced.
The map this program produced is in the physical convention coordinate system (x forward, y left, z up).


#### CAPE
CAPE provides the yoga and tunnel dataset, composed of low textured environment with cylinder primitives
```
./slam_CAPE tunnel
```

#### TUM
TUM contains many sequences, but it's best to start with the fr1_xyz and fr1_rpy (respectivly pure translations and pure rotations).
```
./slam_TUM fr1_xyz
```

#### Launch parameters
```
-h Display the help
-d displays the staged features (not yet in local map)
-f path to the file containing the data (depth, rgb, cam parameters)
-c use cylinder detection 
-j Drop j frames between slam process
-i index of starting frame (> 0)
-l compute line features
-s Save the trajectory in an output file
-r FPS limiter, to debug sequences in real time
```

Check memory errors
```
valgrind --suppressions=/usr/share/opencv4/valgrind.supp --suppressions=/usr/share/opencv4/valgrind_3rdparty.supp ./slam_TUM desk
```

Most of the program's parameters are stored in the parameters.cpp file.
They can be modified as needed, and basic checks are launched at startup to detect erroneous parameters.
The provided configuration should work for most of the use cases.

The user can also choose to run the program with deterministic results, by activating the `MAKE_DETERMINISTIC` option in the CMakeList.txt file.
The maping process will be a bit slower but the result will always be the same between two sequences, allowing for reproductibility and debugging.

## Detailed process
### feature detection & matching
The system starts by splitting the depth image as a 2D connected graph, and run a primitive analysis on it.
This analysis extract local planar features, that are merged into bigger planes.
Using an iterative RANSAC algorithm, those planar features are fitted to cylinders if possible, providing two kind of high level features.
Those high level features are matched on one frame to another by comparing their normals and Inter-Over-Union score.

The system also extract points and lines, to improve reliability of pose extraction.
The lines are extracted using LSD.
The feature points are tracked by optical flow and matched by their ORB descriptors when needed, and every N frames to prevent optical flow drift.

### local map
Every features are maintained in a local map, that keep tracks of the reliable local features.
Those features parameters (positions, uncertainties, speed, ...) are updated if a map feature is matched, using an individual Kalman filter per feature (no bundle adjustment yet).

New features are maintained in a staged feature container, that behaves exactly like the local map, but as a lesser priority during the matching step.

### pose optimization
The final pose for the frame is computed using the local map to detected feature matches.
Outliers are filtered out using a simple RANSAC, and the final pose is computed using Levenberg Marquardt algorithm, with custom weighting functions (based on per feature covariance).

The optimized pose is used to update the local map and decaying motion model, in case the features are lost for a moment.

The complete systems runs in real time, between 300FPS and 600FPS for images of 640x480 pixels (depth and RGB images).

## To be implemented soon
- Advanced camera parameter model
- Point descriptors based on custom neural network
- Usage of keyframe to separate local maps
- Loop closing based on multiscale neural network 
- Use a graph between keyframe to represent the trajectory, and close the loop easily (using bundle adjustment)
- Create a new separate trajectory when loosing track of features, and merge it to the main trajectory when possible


# References:
## Primitive extraction
- [Fast Cylinder and Plane Extraction from Depth Cameras for Visual Odometry](https://arxiv.org/pdf/1803.02380.pdf)
- [Fast Sampling Plane Filtering, Polygon Construction and Merging from Depth Images](http://www.cs.cmu.edu/~mmv/papers/11rssw-BiswasVeloso2.pdf)
- [Fast Plane Extraction in Organized Point Clouds Using Agglomerative HierarchicalClustering](https://merl.com/publications/docs/TR2014-066.pdf)
- [Depth image-based plane detection](https://www.researchgate.net/publication/328822338_Depth_image-based_plane_detection)

## Pose Optimization
- [Using Quaternions for Parametrizing 3D Rotations in Unconstrained Nonlinear Optimization](https://www.yumpu.com/en/document/read/30132260/using-quaternions-for-parametrizing-3-d-rotations-in-) - 2001
- [At All Costs: A Comparison of Robust Cost Functions for Camera Correspondence Outliers]() - 2015
- [An Eﬀicient Solution to the Homography-Based Relative Pose Problem With a Common Reference Direction]() - 2019
- [A Quaternion-based Certifiably Optimal Solution to the Wahba Problem with Outliers]()
- [A General and Adaptive Robust Loss Function]() - 2019

## Visual odometry
- [LVT: Lightweight Visual Odometry for Autonomous Mobile Robots](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6165120/)
- [Dense RGB-D visual odometry using inverse depth](https://www.researchgate.net/publication/283806535_Dense_RGB-D_visual_odometry_using_inverse_depth)
- [SPLODE: Semi-Probabilistic Point and Line Odometry with Depth Estimation from RGB-D Camera Motion](http://epubs.surrey.ac.uk/846020/1/SPLODE.pdf)
- [PL-SVO: Semi-Direct Monocular Visual Odometry byCombining Points and Line Segments](http://mapir.isa.uma.es/rgomez/publications/iros16plsvo.pdf)
- [Probabilistic Combination of Noisy Points andPlanes for RGB-D Odometry](https://arxiv.org/pdf/1705.06516v1.pdf) and [Probabilistic RGB-D Odometry based on Points, Linesand Planes Under Depth Uncertainty](https://arxiv.org/pdf/1706.04034.pdf) - 2017
- [Robust Stereo Visual Inertial Navigation System Based on Multi-Stage Outlier Removal in Dynamic Environments](https://www.mdpi.com/1424-8220/20/10/2922/htm)
- [RP-VIO: Robust Plane-based Visual-Inertial Odometry for Dynamic Environments]() - 2021

## SLAM
- [ORB-SLAM2:  an  Open-Source  SLAM  System  for Monocular,  Stereo  and  RGB-D  Cameras](https://arxiv.org/pdf/1610.06475.pdf)
- [3D SLAM in texture-less environments using rank order statistics](https://www.researchgate.net/publication/283273992_3D_SLAM_in_texture-less_environments_using_rank_order_statistics)
- [3D Mapping with an RGB-D Camera](http://www2.informatik.uni-freiburg.de/~endres/files/publications/felix-endres-phd-thesis.pdf)
- [Online 3D SLAM by Registration of Large PlanarSurface Segments and Closed Form Pose-GraphRelaxation](http://robotics.jacobs-university.de/publicationData/JFR-3D-PlaneSLAM.pdf)
- [LIPS: LiDAR-Inertial 3D Plane SLAM](http://udel.edu/~yuyang/downloads/geneva_iros2018.pdf)
- [Pop-up SLAM: Semantic Monocular Plane SLAM for Low-texture Environments](https://arxiv.org/pdf/1703.07334.pdf)
- [Monocular Object and Plane SLAM in Structured Environments](https://arxiv.org/pdf/1809.03415.pdf)
- [Point-Plane SLAM Using Supposed Planes for Indoor Environments](https://www.mdpi.com/1424-8220/19/17/3795/htm)
- [Stereo Plane SLAM Based on Intersecting Lines]() - 2020
