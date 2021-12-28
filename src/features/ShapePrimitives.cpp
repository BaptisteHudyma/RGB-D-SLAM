#include "ShapePrimitives.hpp"
#include "utils.hpp"

namespace rgbd_slam {
namespace features {
namespace primitives {


    /*
     *
     *      PRIMITIVE
     *
     */
    Primitive::Primitive(const uint id, const cv::Mat& shapeMask) :
        _id(id)
    {
        _shapeMask = shapeMask.clone();
    }

    double Primitive::get_IOU(const std::unique_ptr<Primitive>& prim) const {
        //get union of masks
        cv::Mat unionMat = (_shapeMask | prim->_shapeMask);

        int IOU = cv::countNonZero(unionMat);
        if(IOU <= 0)
            return 0.0;

        //get inter of masks
        cv::Mat interMat = (_shapeMask & prim->_shapeMask);
        return static_cast<double>(cv::countNonZero(interMat)) / static_cast<double>(IOU);
    }

    /*
     *
     *      CYLINDER
     *
     */
    Cylinder::Cylinder(const std::unique_ptr<Cylinder_Segment>& cylinderSeg, const uint id, const cv::Mat& shapeMask) :
        Primitive(id, shapeMask)
    {
        _radius = 0;
        for(uint i = 0; i < cylinderSeg->get_segment_count(); ++i) {
            _radius += cylinderSeg->get_radius(i);
        }
        _radius /= cylinderSeg->get_segment_count();
        _normal = cylinderSeg->get_normal();
    }

    bool Cylinder::is_similar(const std::unique_ptr<Primitive>& prim) {
        if(get_IOU(prim) < 0.2)
            return false;

        const Cylinder* cylinder = dynamic_cast<const Cylinder*>(prim.get());
        if(cylinder != nullptr) {
            return std::abs( _normal.dot( cylinder->_normal ) ) > 0.95;
        }
        else    //plane overlaps cylinder
            return true;
        return false;
    }

    double Cylinder::get_distance(const Eigen::Vector3d& point) {
        //TODO implement
        utils::log_error("Error: get_point_distance is not implemented for Cylinder objects");
        return 0;
    }

    /*
     *
     *        PLANE
     *
     */
    Plane::Plane(const std::unique_ptr<Plane_Segment>& planeSeg, const uint id, const cv::Mat& shapeMask) :
        Primitive(id, shapeMask)
    {
        _mean = planeSeg->get_mean();
        _normal = planeSeg->get_normal();
        _d = planeSeg->get_plane_d();
    }

    bool Plane::is_similar(const std::unique_ptr<Primitive>& prim) {
        if(get_IOU(prim) < 0.2)
            return false;

        const Plane* plane = dynamic_cast<const Plane*>(prim.get());
        if(plane != nullptr) {
            return (_normal.dot(plane->_normal) + 1.0) / 2.0 > 0.95;
        }
        return false;
    }

    double Plane::get_distance(const Eigen::Vector3d& point) {
        return 
            _normal[0] * (point[0] - _mean[0]) + 
            _normal[1] * (point[1] - _mean[1]) + 
            _normal[2] * (point[2] - _mean[2]); 
    }


}
}
}
