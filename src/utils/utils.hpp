#ifndef UTILS_FUNCTIONS_HPP
#define UTILS_FUNCTIONS_HPP

namespace utils {

    const double cx = 3.1649e+02;
    const double cy = 2.2923e+02;
    const double fx = 5.4886e+02;
    const double fy = 5.4959e+02;


    /*
     * \brief Transform a screen point with a depth value to a 3D point
     *
     * \param[in] screenX X coordinates of the 2D point
     * \param[in] screenY Y coordinates of the 2D point
     * \param[in] measuredZ Measured z depth of the point, in meters
     * \param[in] cameraToWorldMatrix Matrix to transform local to world coordinates
     *
     * \return A 3D point in frame coordinates
     */
    const vector3 screen_to_3D_coordinates(const unsigned int screenX, const unsigned int screenY, const double measuredZ, const matrix34& cameraToWorldMatrix) 
    {
        const double x = (static_cast<double>(screenX) - cx) * measuredZ / fx;
        const double y = (static_cast<double>(screenY) - cy) * measuredZ / fy;

        vector4 worldPoint;
        worldPoint << x, y, measuredZ, 1.0;
        return cameraToWorldMatrix * worldPoint;
    }


    /**
     * \brief Transform a point from world to screen coordinate system
     *
     * \param[in] position3D Coordinates of the detected point (world coordinates)
     * \param[in] worldToCameraMatrix Matrix to transform the world to a local coordinate system
     *
     * \return The position of the point in screen coordinates
     */
    const vector2 world_to_screen_coordinates(const vector3& position3D, const matrix34& worldToCameraMatrix)
    {
        vector4 ptH;
        ptH << position3D, 1.0;
        const vector3& point3D = worldToCameraMatrix * ptH; 

        if (point3D[2] == 0) {
            return vector2(0.0, 0.0);
        }

        const double inverseDepth  = 1.0 / point3D[2];
        const double screenX = fx * point3D[0] * inverseDepth + cx;
        const double screenY = fy * point3D[1] * inverseDepth + cy;

        return vector2(screenX, screenY);
    }

    /**
      * \brief Given a camera pose, returns a transformation matrix to convert a world point to camera point
      *
      * \param[in] cameraPose
      *
      */
    const matrix34 compute_world_to_camera_transform(const poseEstimation::Pose& cameraPose)
    {
        const matrix33& worldToCamRotMtrx = (cameraPose.get_orientation_matrix()).transpose();
        const vector3& worldToCamTranslation = (-worldToCamRotMtrx) * cameraPose.get_position();
        matrix34 worldToCamMtrx;
        worldToCamMtrx << worldToCamRotMtrx, worldToCamTranslation;
        return worldToCamMtrx;
    }


}




#endif
