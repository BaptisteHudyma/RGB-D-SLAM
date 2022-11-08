#ifndef RGBDSLAM_FEATURES_PRIMITIVES_PRIMITIVES_HPP
#define RGBDSLAM_FEATURES_PRIMITIVES_PRIMITIVES_HPP 

//cv:Mat
#include <opencv2/opencv.hpp>

#include "plane_segment.hpp"
#include "cylinder_segment.hpp"

#include "../../types.hpp"
#include "../../utils/coordinates.hpp"

namespace rgbd_slam {
    namespace features {
        namespace primitives {

            /**
             * \brief A base class used to compute the tracking analysis.
             * It is a pure virtual class.
             */
            class IPrimitive 
            {
                public:
                    virtual ~IPrimitive() {};

                    /**
                     * \brief Return this shape assigned id
                     */
                    uint get_id() const { return _id; };
                    void set_id(const uint id) { _id = id; };

                    cv::Mat get_shape_mask() const { return _shapeMask; };
                    void set_shape_mask(const cv::Mat& mask) { _shapeMask = mask.clone(); };

                protected:
                    /**
                     * \brief Hidden constructor, to set _id and shape
                     */
                    IPrimitive(const uint id, const cv::Mat& shapeMask);

                    /**
                     * \brief Compute the Inter over Union factor of two masks
                     *
                     * \param[in] prim Another primitive object
                     *
                     * \return A number between 0 and 1, indicating the IoU
                     */
                    double get_IOU(const IPrimitive& prim) const;
                    double get_IOU(const cv::Mat& mask) const;

                    //members
                    uint _id;
                    cv::Mat _shapeMask;

                private:
                    //remove copy functions
                    //IPrimitive(const IPrimitive&) = delete;
                    IPrimitive& operator=(const IPrimitive&) = delete;
            };

            /**
             * \brief Handles cylinder primitives.
             */
            class Cylinder : public IPrimitive
            {
                public:
                    /**
                     * \brief Construct a cylinder object
                     *
                     * \param[in] cylinderSeg Cylinder segment to copy
                     * \param[in] id ID assigned to this shape (for tracking and debug)
                     * \param[in] shapeMask Mask of the shape in the reference image
                     */
                    Cylinder(const Cylinder_Segment& cylinderSeg, uint id, const cv::Mat& shapeMask);

                    /**
                     * \brief Get the similarity of two cylinders, based on normal direction and radius
                     * 
                     * \param[in] prim Another cylinder to compare to
                     * 
                     * \return A double between 0 and 1, with 1 indicating identical cylinders
                     */
                    virtual bool is_similar(const Cylinder& prim) const;

                    /**
                     * \brief Get the distance of a point to the surface of the cylinder
                     *
                     * \return The signed distance of the point to the surface, 0 if the point is on the surface, and < 0 if the point is inside the cylinder
                     */
                    virtual double get_distance(const vector3& point) const;

                    vector3 _normal;
                    double _radius;
            };

            /**
             * \brief Handles planes.
             */
            class Plane : public IPrimitive 
            {
                public:

                    /**
                     * \brief Construct a plane object
                     *
                     * \param[in] planeSeg Plane to copy
                     * \param[in] id ID assigned to this shape (for tracking and debug)
                     * \param[in] shapeMask Mask of the shape in the reference image
                     */
                    Plane(const Plane_Segment& planeSeg, uint id, const cv::Mat& shapeMask);

                    /**
                     * \brief Get the similarity of two planes, based on normal direction
                     *
                     * \param[in] prim Another primitive to compare to
                     * 
                     * \return A double between 0 and 1, with 1 indicating identical planes
                     */
                    bool is_similar(const Plane& prim) const;
                    bool is_similar(const cv::Mat& mask, const utils::PlaneCameraCoordinates& planeParametrization) const;
                    bool is_similar(const Cylinder& prim) const;

                    vector3 get_plane_normal() const { return _parametrization.head(3); };

                    /**
                     * Return the distance of this primitive to a point
                     */
                    double get_distance(const vector3& point) const;

                    utils::PlaneCameraCoordinates _parametrization;     // infinite plane representation
                    vector3 _mean;      // mean center point of the plane; in camera coordinates
            };
        }
    }
}

#endif
