#ifndef PRIMITIVE_DETECTION_PRIMITIVES_HPP
#define PRIMITIVE_DETECTION_PRIMITIVES_HPP 

#include "PlaneSegment.hpp"
#include "CylinderSegment.hpp"

#include <Eigen/Dense>


namespace primitiveDetection {

    class IPrimitive {
        public:
            /**
             * \brief Get the similarity of two primitives
             * 
             * \return A double between 0 and 1, with 1 indicating identical primitives 
             */
            virtual double get_similarity(const std::unique_ptr<IPrimitive>& prim) = 0;

            /**
             * \brief Get the distance of a point to the primitive
             */
            virtual double get_distance(const Eigen::Vector3d& point) = 0;

            virtual unsigned int get_id() const = 0;

            virtual ~IPrimitive() {};
            IPrimitive() {};

        private:
            IPrimitive(const IPrimitive&) = delete;
            IPrimitive& operator=(const IPrimitive&) = delete;
    };

    class Cylinder :
        public IPrimitive
    {
        public:
            /**
             * \brief Construct a cylinder object
             *
             * \param[in] cylinderSeg Cylinder segment to copy
             */
            Cylinder(const std::unique_ptr<Cylinder_Segment>& cylinderSeg, unsigned int id);

            /**
             * \brief Get the similarity of two cylinders, based on normal direction and radius
             * 
             * \return A double between 0 and 1, with 1 indicating identical cylinders
             */
            virtual double get_similarity(const std::unique_ptr<IPrimitive>& cylinder);

            /**
             * \brief Get the distance of a point to the surface of the cylinder
             *
             * \return The signed distance of the point to the surface, 0 if the point is on the surface, and < 0 if the point is inside the cylinder
             */
            virtual double get_distance(const Eigen::Vector3d& point);

            virtual unsigned int get_id() const { return _id; };

        protected:

        private:
            unsigned int _id;
            double _radius;
            Eigen::Vector3d _normal;
    };

    class Plane :
        public IPrimitive 
    {
        public:
            Plane(const std::unique_ptr<Plane_Segment>& planeSeg, unsigned int id);

            /**
             * \brief Get the similarity of two planes, based on normal direction
             * 
             * \return A double between 0 and 1, with 1 indicating identical planes
             */
            virtual double get_similarity(const std::unique_ptr<IPrimitive>& plane);

            /**
             * \brief Get the distance of a point to the plane 
             *
             * \return The signed distance of the point to the surface, 0 if the point is on the surface, and < 0 if the point is behind
             */
            virtual double get_distance(const Eigen::Vector3d& point);

            virtual unsigned int get_id() const { return _id; };

        protected:


        private:
            unsigned int _id;           //in frame unique ID
            Eigen::Vector3d _mean;      //mean center point
            Eigen::Vector3d _normal;    //normal of the plane
            double _d;     //fourth component of the plane parameters
    };


}



#endif
