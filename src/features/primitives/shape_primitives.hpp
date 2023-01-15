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

                    cv::Mat get_shape_mask() const { return _shapeMask; };
                    void set_shape_mask(const cv::Mat& mask) { _shapeMask = mask.clone(); };

                    /**
                    * \brief Return the number of pixels in this plane mask
                    */
                    uint get_contained_pixels() const
                    {
                        const static uint cellSize = Parameters::get_depth_map_patch_size();
                        const static uint pixelPerCell = cellSize * cellSize;
                        return cv::countNonZero(_shapeMask) * pixelPerCell;
                    }

                    bool can_add_to_map() const
                    {
                        // TODO
                        return true;
                    }

                protected:
                    /**
                     * \brief Hidden constructor, to set shape
                     */
                    IPrimitive(const cv::Mat& shapeMask);

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
                     * \param[in] shapeMask Mask of the shape in the reference image
                     */
                    Cylinder(const Cylinder_Segment& cylinderSeg, const cv::Mat& shapeMask);

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
                     * \param[in] shapeMask Mask of the shape in the reference image
                     */
                    Plane(const Plane_Segment& planeSeg, const cv::Mat& shapeMask);

                    static vector6 compute_descriptor(const utils::PlaneCameraCoordinates& parametrization, const utils::CameraCoordinate& planeCentroid, const uint pixelCount) {
                        const vector3& normal = parametrization.head(3);
                        vector6 descriptor(
                            {
                            normal.x(), normal.y(), normal.z(),
                            abs(planeCentroid.x()/pixelCount), abs(planeCentroid.y()/pixelCount), abs(planeCentroid.z()/pixelCount)
                            }
                        );
                        return descriptor;
                    };

                    double get_similarity(const vector6& descriptor) const
                    {
                        return (_descriptor - descriptor).norm();
                    };

                    /**
                     * \brief Get the similarity of two planes, based on normal direction
                     *
                     * \param[in] prim Another primitive to compare to
                     * 
                     * \return A true if those shapes are similar
                     */
                    bool is_similar(const Plane& prim) const;
                    bool is_similar(const cv::Mat& mask, const utils::PlaneCameraCoordinates& planeParametrization) const;
                    bool is_similar(const Cylinder& prim) const;

                    vector3 get_normal() const { return _parametrization.head(3); };
                    utils::PlaneCameraCoordinates get_parametrization() const { return _parametrization; };
                    utils::CameraCoordinate get_centroid() const { return _centroid; };

                private:
                    vector6 compute_descriptor() const
                    {
                        return compute_descriptor(get_parametrization(), get_centroid(), get_contained_pixels());
                    };

                    /**
                     * Return the distance of this primitive to a point
                     */
                    double get_distance(const vector3& point) const;

                    utils::PlaneCameraCoordinates _parametrization;     // infinite plane representation
                    utils::CameraCoordinate _centroid;      // mean center point of the plane; in camera coordinates

                    vector6 _descriptor;
            };


            // types for detected primitives
            typedef std::vector<Cylinder> cylinder_container; 
            typedef std::vector<Plane> plane_container;
        }
    }
}

#endif
