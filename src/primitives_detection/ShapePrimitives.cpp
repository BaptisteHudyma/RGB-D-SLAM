#include "ShapePrimitives.hpp"
#include <iostream>

namespace primitiveDetection {

    /*
     *
     *      CYLINDER
     *
     */
    Cylinder::Cylinder(const std::unique_ptr<Cylinder_Segment>& cylinderSeg, unsigned int id) {
        _id = id;

        _radius = 0;
        for(unsigned int i = 0; i < cylinderSeg->get_segment_count(); ++i) {
            _radius += cylinderSeg->get_radius(i);
        }
        _radius /= cylinderSeg->get_segment_count();
        _normal = cylinderSeg->get_normal();
    }

    double Cylinder::get_similarity(const std::unique_ptr<IPrimitive>& prim) {
        const Cylinder* cylinder = dynamic_cast<const Cylinder*>(prim.get());
        if(cylinder == nullptr)
            return 0.0;
        return std::abs( _normal.dot( cylinder->_normal ) );
    }

    double Cylinder::get_distance(const Eigen::Vector3d& point) {
        return 0;
    }

    /*
     *
     *        PLANE
     *
     */
    Plane::Plane(const std::unique_ptr<Plane_Segment>& planeSeg, unsigned int id) {
        _id = id;

        _mean = planeSeg->get_mean();
        _normal = planeSeg->get_normal();
        _d = planeSeg->get_plane_d();
    }

    double Plane::get_similarity(const std::unique_ptr<IPrimitive>& prim) {
        const Plane* plane = dynamic_cast<const Plane*>(prim.get());
        if(plane == nullptr)
            return 0.0;
        return (_normal.dot(plane->_normal) + 1.0) / 2.0;
    }

    double Plane::get_distance(const Eigen::Vector3d& point) {
        return 
            _normal[0] * (point[0] - _mean[0]) + 
            _normal[1] * (point[1] - _mean[1]) + 
            _normal[2] * (point[2] - _mean[2]); 
    }


}
