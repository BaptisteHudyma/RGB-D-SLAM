#include "shape_primitives.hpp"

#include "../../outputs/logger.hpp"
#include "../../parameters.hpp"
#include "cylinder_segment.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/VectorBlock.h>

namespace rgbd_slam {
    namespace features {
        namespace primitives {

            /*
             *
             *      PRIMITIVE
             *
             */
            IPrimitive::IPrimitive(const cv::Mat& shapeMask) :
                _shapeMask(shapeMask.clone())
            {
                assert(not shapeMask.empty());
            }

            double IPrimitive::get_IOU(const IPrimitive& prim) const {
                assert(not _shapeMask.empty());
                assert(not prim._shapeMask.empty());
                assert(_shapeMask.size == prim._shapeMask.size);

                return get_IOU(prim._shapeMask);
            }

            double IPrimitive::get_IOU(const cv::Mat& mask) const {
                //get union of masks
                const cv::Mat unionMat = (_shapeMask | mask);
                const int IOU = cv::countNonZero(unionMat);
                if (IOU <= 0)
                    // Union is empty, quit
                    return 0;

                //get inter of masks
                const cv::Mat interMat = (_shapeMask & mask);
                return static_cast<double>(cv::countNonZero(interMat)) / static_cast<double>(IOU);
            }

            /*
             *
             *      CYLINDER
             *
             */
            Cylinder::Cylinder(const Cylinder_Segment& cylinderSeg, const cv::Mat& shapeMask) :
                IPrimitive(shapeMask)
            {
                _radius = 0;
                for(uint i = 0; i < cylinderSeg.get_segment_count(); ++i) {
                    _radius += cylinderSeg.get_radius(i);
                }
                _radius /= cylinderSeg.get_segment_count();
                _normal = cylinderSeg.get_normal();
            }

            bool Cylinder::is_similar(const Cylinder& cylinder) const {
                const static double minimumIOUForMatch = Parameters::get_minimum_iou_for_match();
                const static double minimumNormalDotDiff = Parameters::get_maximum_plane_normals_angle_for_match();
                if(get_IOU(cylinder) < minimumIOUForMatch)
                    return false;

                return std::abs(_normal.dot(cylinder._normal)) > minimumNormalDotDiff;
            }

            double Cylinder::get_distance(const vector3& point) const {
                //TODO implement
                outputs::log_error("Error: get_point_distance is not implemented for Cylinder objects");
                return 0;
            }

            /*
             *
             *        PLANE
             *
             */
            Plane::Plane(const Plane_Segment& planeSeg, const cv::Mat& shapeMask) :
                IPrimitive(shapeMask),

                _parametrization(
                    planeSeg.get_normal().x(),
                    planeSeg.get_normal().y(),
                    planeSeg.get_normal().z(),
                    planeSeg.get_plane_d()
                ),
                _centroid(planeSeg.get_centroid()),
                _descriptor(compute_descriptor())
            {
            }

            bool Plane::is_similar(const Plane& plane) const {
                return is_similar(plane._shapeMask, plane._parametrization);
            }

            bool Plane::is_similar(const cv::Mat& mask, const utils::PlaneCameraCoordinates& planeParametrization) const
            {
                const static double minimumIOUForMatch = Parameters::get_minimum_iou_for_match();
                const static double minimumNormalDotDiff = cos(Parameters::get_maximum_plane_normals_angle_for_match() * M_PI/180.0);
                if(get_IOU(mask) < minimumIOUForMatch)
                    return false;
                return abs(get_normal().dot(planeParametrization.head(3))) > minimumNormalDotDiff;
            }

            bool Plane::is_similar(const Cylinder& cylinder) const {
                const double minimumIOUForMatch = Parameters::get_minimum_iou_for_match();
                if(get_IOU(cylinder) < minimumIOUForMatch)
                    return false;

                // TODO: not implemented
                return false;
            }

            double Plane::get_distance(const vector3& point) const {
                return get_normal().dot(point - _centroid.base()); 
            }

        }
    }
}
