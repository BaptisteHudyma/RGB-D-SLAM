# Sources: Coordinates

- **basis_changes** : Define some coordinate basic chantges : Cartesian, spherical, cylindrical, as well as the jacobian of those transformations
- **inverse_depth_coordinates**: Define the 2d point class, implemented with inverse depth. (in World coordinates)
- **plane_coordinates**: Define a plane class. planes are represented in hessian form (normal vector and distance from origin). Each plane also handle a boundary polygon. (in Camera and World coordinates)
- **point_coordinates**: Define points coordinates (in Screen, Screen2D, Camera, Camera2d, and World coordinates)
- **polygon_coordinates**: Define a polygon (in Camera and World coordinates)