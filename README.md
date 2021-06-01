# RGB-D-SLAM

My take on a SLAM, based on shape primitive recognition with a RGB-D camera.
The system uses depth informations as a connected 2D graph to extract primitive shapes, and use them with 3D points and lines to estimate it's position in space.
A map is created with those points, lines and primitive shapes.

The primitive detection is based on [Fast Cylinder and Plane Extraction from Depth Cameras for Visua Odometry](https://arxiv.org/pdf/1803.02380.pdf).
The pose estimation is based on [Lightweight Visual Odometry for Autonomous Mobile Robots](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6165120/).


packages
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



