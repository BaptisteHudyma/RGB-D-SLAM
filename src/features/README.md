# Sources: Features

- **keypoints**
    - **keypoint_detection**: Detection of keypoints, and optical flow tracking
    - **keypoint_handler**: handle the matching of keypoints, with internal states for already matched points

- **lines**
    - **line_detection**: WIP: Detection of lines 

- **primitives**
    - **cylinder_segments**: Store a cylinder segment, defined as a composition of plane segments. This also handle the cylinder fitting from plane segments
    - **depth_map_transformation**: Transform a depth image into a depth matrix
    - **histogram**: Define a 2D histogram
    - **plane_segment**: Store a plane segment, defined as a Principal Component Analysis
    - **primitive_detection**: Main detection class, takes in a depth matrix and return the planes and cylinders.
    - **shape_primitives**: The types returnes by the PrimitiveDetection class