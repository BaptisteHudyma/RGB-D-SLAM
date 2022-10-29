#include "shape_primitives.hpp"

#include "../../outputs/logger.hpp"
#include "../../parameters.hpp"
#include "cylinder_segment.hpp"

namespace rgbd_slam {
    namespace features {
        namespace primitives {

            /*
             *
             *      PRIMITIVE
             *
             */
            IPrimitive::IPrimitive(const uint id, const cv::Mat& shapeMask) :
                _id(id),
                _shapeMask(shapeMask.clone())
            {
                assert(not shapeMask.empty());
            }

            double IPrimitive::get_IOU(const IPrimitive& prim) const {
                assert(not _shapeMask.empty());
                assert(not prim._shapeMask.empty());
                assert(_shapeMask.size == prim._shapeMask.size);

                //get union of masks
                const cv::Mat unionMat = (_shapeMask | prim._shapeMask);
                const int IOU = cv::countNonZero(unionMat);
                if (IOU <= 0)
                    // Union is empty, quit
                    return 0;

                //get inter of masks
                const cv::Mat interMat = (_shapeMask & prim._shapeMask);
                return static_cast<double>(cv::countNonZero(interMat)) / static_cast<double>(IOU);
            }

            /*
             *
             *      CYLINDER
             *
             */
            Cylinder::Cylinder(const Cylinder_Segment& cylinderSeg, const uint id, const cv::Mat& shapeMask) :
                IPrimitive(id, shapeMask)
            {
                _radius = 0;
                for(uint i = 0; i < cylinderSeg.get_segment_count(); ++i) {
                    _radius += cylinderSeg.get_radius(i);
                }
                _radius /= cylinderSeg.get_segment_count();
                _normal = cylinderSeg.get_normal();
            }

            bool Cylinder::is_similar(const Cylinder& cylinder) {
                if(get_IOU(cylinder) < Parameters::get_minimum_iou_for_match())
                    return false;

                return std::abs(_normal.dot(cylinder._normal)) > Parameters::get_minimum_normals_dot_difference();
            }

            double Cylinder::get_distance(const vector3& point) {
                //TODO implement
                outputs::log_error("Error: get_point_distance is not implemented for Cylinder objects");
                return 0;
            }

            /*
             *
             *        PLANE
             *
             */
            Plane::Plane(const Plane_Segment& planeSeg, const uint id, const cv::Mat& shapeMask) :
                IPrimitive(id, shapeMask),

                _normal(planeSeg.get_normal()),
                _d(planeSeg.get_plane_d()),
                _mean(planeSeg.get_mean())
            {
            }

            bool Plane::is_similar(const Plane& plane) {
                if(get_IOU(plane) < Parameters::get_minimum_iou_for_match())
                    return false;

                return (_normal.dot(plane._normal) + 1.0) / 2.0 > Parameters::get_minimum_normals_dot_difference();
            }

            bool Plane::is_similar(const Cylinder& cylinder) {
                if(get_IOU(cylinder) < Parameters::get_minimum_iou_for_match())
                    return false;

                // TODO: not implemented
                return false;
            }

            double Plane::get_distance(const vector3& point) {
                return _normal.dot(point - _mean); 
            }

        }
    }
}
