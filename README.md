# RGB-D-SLAM


Build
```
mkdir build && cd build
cmake ..
make
```


Check mem errors
```
valgrind --suppressions=/usr/share/opencv4/valgrind.supp --suppressions=/usr/share/opencv4/valgrind_3rdparty.supp ./planeExtraction
```
