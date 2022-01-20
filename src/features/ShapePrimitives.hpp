#ifndef RGBDSLAM_FEATURES_PRIMITIVES_PRIMITIVES_HPP
#define RGBDSLAM_FEATURES_PRIMITIVES_PRIMITIVES_HPP 

#include "PlaneSegment.hpp"
#include "CylinderSegment.hpp"

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>


namespace rgbd_slam {
namespace features {
namespace primitives {

    /**
      * \brief A base class used to compute the tracking analysis. It is a pure virtual class.
      */
    class Primitive {
        public:
            /**
             * \brief Get the similarity of two primitives
             * 
             * \return A double between 0 and 1, with 1 indicating identical primitives 
             */
            virtual bool is_similar(const std::unique_ptr<Primitive>& prim) = 0; 

            /**
             * \brief Get the distance of a point to the primitive
             *
             * \param[in] point 3D Point to compute distance to

             * \return The signed distance of the point to the shape, 0 if the point is on the shape
             */
            virtual double get_distance(const Eigen::Vector3d& point) = 0;

            virtual ~Primitive() {};

            /**
              * \brief Return this shape assigned id
              */
            uint get_id() const { return _id; };
            void set_id(const uint id) { _id = id; };

            Eigen::Vector3d _normal;

        protected:
            /**
              * \brief Hidden constructor, to set _id and shape
              */
            Primitive(const uint id, const cv::Mat& shapeMask);

            /**
              * \brief Compute the Inter over Union factor of two masks
              *
              * \param[in] prim Another primitive object
              *
              * \return A number between 0 and 1, indicating the IoU
              */
            double get_IOU(const std::unique_ptr<Primitive>& prim) const;

            //members
            uint _id;
            cv::Mat _shapeMask;

        private:
            //remove copy functions
            Primitive(const Primitive&) = delete;
            Primitive& operator=(const Primitive&) = delete;
    };

    /**
      * \brief Specification of the Primitive class. Handles cylinder primitives.
      */
    class Cylinder :
        public Primitive
    {
        public:
            /**
             * \brief Construct a cylinder object
             *
             * \param[in] cylinderSeg Cylinder segment to copy
             * \param[in] id ID assigned to this shape (for tracking and debug)
             * \param[in] shapeMask Mask of the shape in the reference image
             */
            Cylinder(const std::unique_ptr<Cylinder_Segment>& cylinderSeg, uint id, const cv::Mat& shapeMask);

            /**
             * \brief Get the similarity of two cylinders, based on normal direction and radius
             * 
             * \param[in] prim Another primitive to compare to
             * 
             * \return A double between 0 and 1, with 1 indicating identical cylinders
             */
            virtual bool is_similar(const std::unique_ptr<Primitive>& prim) override;

            /**
             * \brief Get the distance of a point to the surface of the cylinder
             *
             * \return The signed distance of the point to the surface, 0 if the point is on the surface, and < 0 if the point is inside the cylinder
             */
            virtual double get_distance(const Eigen::Vector3d& point) override;

        protected:

        private:
            double _radius;
    };

    /**
      * \brief Specification of the Primitive class. Handles planes.
      */
    class Plane :
        public Primitive 
    {
        public:

            /**
             * \brief Construct a plane object
             *
             * \param[in] planeSeg Plane to copy
             * \param[in] id ID assigned to this shape (for tracking and debug)
             * \param[in] shapeMask Mask of the shape in the reference image
             */
            Plane(const std::unique_ptr<Plane_Segment>& planeSeg, uint id, const cv::Mat& shapeMask);

            /**
             * \brief Get the similarity of two planes, based on normal direction
             *
             * \param[in] prim Another primitive to compare to
             * 
             * \return A double between 0 and 1, with 1 indicating identical planes
             */
            virtual bool is_similar(const std::unique_ptr<Primitive>& prim) override;

            virtual double get_distance(const Eigen::Vector3d& point) override;

        private:
            Eigen::Vector3d _mean;      //mean center point
            double _d;     //fourth component of the plane parameters
    };



    typedef std::unique_ptr<Primitive> primitive_uniq_ptr;

}
}
}

#endif
