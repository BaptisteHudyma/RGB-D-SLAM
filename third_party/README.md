# Third party programs

## concave_fitting
Computes the concave hull of a point set using the Moreira-Santos algorithm.
Shady implementation that often returns invalid hulls with multiple intersections

## correct_boost_polygon
Implementation to correct the wrong polygon hulls. Used to compensate the shady concave hull fitting.

## geodesic_operations
Geodesic erosion and dilation, with optional masking

## line_segment_detector
Implementation of the LSD, by opencv.

## p3p
A p3P implementation. 
Given two 3 points sets in 3D, with a pose transformation between the two, compute a set of possible transformations leading to this pose.