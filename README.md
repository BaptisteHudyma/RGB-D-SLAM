# RGB-D-SLAM

My take on a SLAM, based on shape primitive recognition with a RGB-D camera.
The system uses depth informations as a connected 2D graph to extract primitive shapes, and use them with 3D points and lines to estimate it's position in space.
A map is created with those points, lines and primitive shapes.

The primitive detection is based on [Fast Cylinder and Plane Extraction from Depth Cameras for Visua Odometry](https://arxiv.org/pdf/1803.02380.pdf).
See the Doxygen documentation on [GitHub Pages](https://baptistehudyma.github.io/RGB-D-SLAM/html/index.html).

For now, we are at the state of visual odometry with local mapping.

## To be implemented soon
- Map point error model
- Advanced camera parameter model
- Point descriptors based on custom neural network
- Loop closing based on multiscale neural network 


## packages
```
opencv
Eigen
g2o
```

How to build Build
```
mkdir build && cd build
cmake ..
make
```

How to use
```
./rgbdslam
```
Parameters
```
-h Display the help
-f path to the file containing the data (depth, rgb, cam parameters)
-c use cylinder detection 
-i index of starting frame (> 0)
```

Check mem errors
```
valgrind --suppressions=/usr/share/opencv4/valgrind.supp --suppressions=/usr/share/opencv4/valgrind_3rdparty.supp ./rgbdslam
```



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