#ifndef RGBDSLAM_TYPES_HPP
#define RGBDSLAM_TYPES_HPP 


#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <list>

namespace rgbd_slam {
    /*
     *        Declare the most common types used in this program
     */

    const double EulerToRadian = M_PI/180.0;

    typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> Matrixb;
    typedef Eigen::Vector2d vector2;
    typedef Eigen::Matrix<double, 3, 1> vector3;
    typedef Eigen::Vector4d vector4;
    typedef Eigen::Matrix3d matrix33;
    typedef Eigen::Matrix<double, 3, 4> matrix34;
    typedef Eigen::Matrix<double, 4, 3> matrix43;
    typedef Eigen::Matrix4d matrix44;
    typedef Eigen::Quaternion<double> quaternion;

    
    //define new classes to not mix types
    class worldToCameraMatrix : public matrix44 {};
    class cameraToWorldMatrix : public matrix44 {};

    struct EulerAngles
    {
        double yaw;
        double pitch;
        double roll;

        EulerAngles(): yaw(0.0), pitch(0.0), roll(0.0) {};

        EulerAngles(const double y, const double p, const double r):
            yaw(y),
            pitch(p),
            roll(r)
        {};
    };

    typedef std::vector<vector2, Eigen::aligned_allocator<vector2>> vector2_vector;
    typedef std::vector<vector3, Eigen::aligned_allocator<vector3>> vector3_vector;
    typedef std::vector<vector4, Eigen::aligned_allocator<vector4>> vector4_vector;
    typedef std::vector<matrix33, Eigen::aligned_allocator<matrix33>> matrix33_vector;
    typedef std::vector<matrix34, Eigen::aligned_allocator<matrix34>> matrix34_vector;
    typedef std::vector<matrix44, Eigen::aligned_allocator<matrix44>> matrix44_vector;
    typedef std::vector<quaternion, Eigen::aligned_allocator<quaternion>> quaternion_vector;
}

#endif
