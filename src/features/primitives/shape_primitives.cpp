#include "shape_primitives.hpp"

#include "../../outputs/logger.hpp"
#include "../../parameters.hpp"

namespace rgbd_slam {
    namespace features {
        namespace primitives {

            /*
             *
             *      PRIMITIVE
             *
             */
            Primitive::Primitive(const uint id, const cv::Mat& shapeMask) :
                _id(id),
                _shapeMask(shapeMask.clone()),
                _primitiveType(PrimitiveType::Invalid)
            {
                assert(not shapeMask.empty());
            }

            double Primitive::get_IOU(const Primitive& prim) const {
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
                Primitive(id, shapeMask)
            {
                _primitiveType = PrimitiveType::Cylinder;
                _radius = 0;
                for(uint i = 0; i < cylinderSeg.get_segment_count(); ++i) {
                    _radius += cylinderSeg.get_radius(i);
                }
                _radius /= cylinderSeg.get_segment_count();
                _normal = cylinderSeg.get_normal();
            }

            bool Cylinder::is_similar(const Primitive& prim) {
                const PrimitiveType& primitiveType = prim.get_primitive_type();
                assert(primitiveType != PrimitiveType::Invalid);

                if(get_IOU(prim) < Parameters::get_minimum_iou_for_match())
                    return false;

                switch(primitiveType)
                {
                    case PrimitiveType::Plane:
                    {
                        // Not implemented. Maybe some day ?
                        break;
                    }
                    case PrimitiveType::Cylinder:
                    {
                        const Cylinder* cylinder = dynamic_cast<const Cylinder*>(&prim);
                        if(cylinder != nullptr)
                        {
                            return std::abs( _normal.dot( cylinder->_normal ) ) > Parameters::get_minimum_normals_dot_difference();
                        }
                        outputs::log_error("Failed attempt to convert a primitive indicated as a cylinder to a cylinder");
                        break;
                    }
                    default:
                    {
                        outputs::log_error("Unknown primitive type");
                        break;
                    }
                }
                return false;
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
                Primitive(id, shapeMask),

                _d(planeSeg.get_plane_d()),
                _mean(planeSeg.get_mean())
            {
                _primitiveType = PrimitiveType::Plane;
                _normal = planeSeg.get_normal();
            }

            bool Plane::is_similar(const Primitive& prim) {
                const PrimitiveType& primitiveType = prim.get_primitive_type();
                assert(primitiveType != PrimitiveType::Invalid);

                if(get_IOU(prim) < Parameters::get_minimum_iou_for_match())
                    return false;

                switch(primitiveType)
                {
                    case PrimitiveType::Cylinder:
                    {
                        // TODO: Not implemented. Maybe some day ?
                        break;
                    }
                    case PrimitiveType::Plane:
                    {
                        // Check the conversion
                        const Plane* plane = dynamic_cast<const Plane*>(&prim);
                        if(plane != nullptr)
                        {
                            return (_normal.dot(plane->_normal) + 1.0) / 2.0 > Parameters::get_minimum_normals_dot_difference();
                        }
                        outputs::log_error("Failed attempt to convert a primitive indicated as a plane to a plane");
                        break;
                    }
                    default:
                    {
                        outputs::log_error("Unknown primitive type");
                        break;
                    }
                }
                return false;
            }

            double Plane::get_distance(const vector3& point) {
                return _normal.dot(point - _mean); 
            }

        }
    }
}
