#ifndef TYPE_DEFINITION_HPP
#define TYPE_DEFINITION_HPP


#include <Eigen/StdVector>
#include <map>
#include <list>

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

struct EulerAngles
{
    double yaw;
    double pitch;
    double roll;

    EulerAngles() {};

    EulerAngles(const double y, const double p, const double r):
        yaw(y),
        pitch(p),
        roll(r)
    {};
};

typedef std::vector<vector2, Eigen::aligned_allocator<vector2>> vector2_array;
typedef std::vector<vector3, Eigen::aligned_allocator<vector3>> vector3_array;
typedef std::vector<vector4, Eigen::aligned_allocator<vector4>> vector4_array;
typedef std::vector<matrix33, Eigen::aligned_allocator<matrix33>> matrix33_array;
typedef std::vector<matrix34, Eigen::aligned_allocator<matrix34>> matrix34_array;
typedef std::vector<matrix44, Eigen::aligned_allocator<matrix44>> matrix44_array;
typedef std::vector<quaternion, Eigen::aligned_allocator<quaternion>> quaternion_array;

// KeyPoint matching
typedef std::pair<vector3, vector3> point_pair;
typedef std::list<point_pair> match_point_container;


#endif
