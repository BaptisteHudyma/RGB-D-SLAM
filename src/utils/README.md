# Sources: utils

- **angle_utils**: Euler to quaternion and quaternion to euler. nothing much
- **camera_transformation**: Define the camera transformation matrices
- **covariances**: Define the covariance models for points and planes. ideally, all of this will be exploded in other dedicated classes
- **distance_utils**: handle some distance computation. ideally, all of this will be exploded in other dedicated classes
- **line**: Define line operations (intersections, distance, etc)
- **polygon**: Define a polygon by it's boundary points, and it's operations (intersections, union, etc)
- **pose**: Define a 6D pose class, with pose covariance
- **random**: All random generation (random numbers, shuffling, etc) should be based on this