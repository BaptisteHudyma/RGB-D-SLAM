#ifndef RGBDSLAM_UTILS_LINE_HPP
#define RGBDSLAM_UTILS_LINE_HPP

#include "types.hpp"
#include <Eigen/Dense>
#include <Eigen/src/Core/ArrayWrapper.h>
#include <Eigen/src/Core/Matrix.h>

namespace rgbd_slam::utils {

/**
 * \brief Base class for all line related stuff
 */
template<int Dim> class ILine
{
  public:
    ILine() { _startPoint.setZero(); }

    ILine(const Eigen::Vector<double, Dim>& startPoint) : _startPoint(startPoint) {}

    ILine(const ILine<Dim>& other) : _startPoint(other._startPoint) {}

    /**
     * \brief Find the closest point to a given point, that will be on the current line
     * \param[in] point the point to find the closest point to
     * \return the point on the line that is the closest to the given point
     */
    [[nodiscard]] Eigen::Vector<double, Dim> get_closest_point_on_line(
            const Eigen::Vector<double, Dim>& point) const noexcept
    {
        const Eigen::Vector<double, Dim>& normal = compute_normal();
        const double distance = (point - _startPoint).dot(normal);
        return _startPoint + normal * distance;
    }

    /**
     * \brief Compute the signed distance between this line and a point
     */
    [[nodiscard]] virtual Eigen::Vector<double, Dim> distance(const Eigen::Vector<double, Dim>& point) const noexcept
    {
        return point - get_closest_point_on_line(point);
    }

    /**
     * \brief get the normal vector from the start point
     */
    virtual Eigen::Vector<double, Dim> compute_normal() const noexcept = 0;

    [[nodiscard]] Eigen::Vector<double, Dim> get_start_point() const noexcept { return _startPoint; };
    void set_start_point(const Eigen::Vector<double, Dim>& startPoint) noexcept { _startPoint = startPoint; };

  protected:
    Eigen::Vector<double, Dim> _startPoint;
};

/**
 * \brief check the intersection of two lines (in 2d)
 * \param[in] firstLine
 * \param[in] secondLine
 * \param[in] point the point of intersection of the two lines, if the function returns true
 */
[[nodiscard]] bool intersects(const ILine<2>& firstLine,
                              const ILine<2>& secondLine,
                              Eigen::Vector<double, 2>& point) noexcept;

/**
 * \brief check the intersection of two lines (in 3d)
 * \param[in] firstLine
 * \param[in] secondLine
 * \param[in] point the point of intersection of the two lines, if the function returns true
 */
[[nodiscard]] bool intersects(const ILine<3>& firstLine,
                              const ILine<3>& secondLine,
                              Eigen::Vector<double, 3>& point) noexcept;

/**
 * \brief Segment class: a line defined by two points
 */
template<int Dim> class Segment : public ILine<Dim>
{
  public:
    Segment() { _endPoint.setZero(); }

    Segment(const Eigen::Vector<double, Dim>& startPoint, const Eigen::Vector<double, Dim>& endPoint) :
        ILine<Dim>(startPoint),
        _endPoint(endPoint)
    {
    }

    Segment(const Segment<Dim>& other) : ILine<Dim>(other._startPoint), _endPoint(other._endPoint) {}

    /**
     * \brief compute the normal of this line
     */
    [[nodiscard]] Eigen::Vector<double, Dim> compute_normal() const noexcept override
    {
        Eigen::Vector<double, Dim> normal = _endPoint - this->_startPoint;
        return normal.normalized();
    }

    /**
     * \brief Specialization of the 2D normal. Find the covariance of the normal from the variance of the two defining
     * points.
     * \param[in] cov The covariance of the start and end point
     */
    matrix22 get_normal_variance(const matrix44& cov) const noexcept
    {
        Eigen::Matrix<double, 2, 4> jac;
        jac.setZero();

        const vector2 res = this->_startPoint - _endPoint;

        const double theta6 = pow(SQR(res.x()) + SQR(res.y()), 3.0 / 2.0);
        const double theta5 = SQR(res.x()) / theta6;
        const double theta4 = res.x() * res.y() / theta6;
        const double theta3 = theta4;
        const double theta2 = SQR(res.y()) / theta6;
        const double theta1 = 1.0 / res.norm();

        jac(0, 0) = theta5 - theta1;
        jac(0, 1) = theta3;
        jac(0, 2) = theta1 - theta5;
        jac(0, 3) = -theta3;

        jac(1, 0) = theta4;
        jac(1, 1) = theta2 - theta1;
        jac(1, 2) = -theta4;
        jac(1, 3) = theta1 - theta2;

        return jac * cov * jac.transpose();
    }

    /**
     * \brief Specialization of the 2D. Find the covariance of distance fonction between point and line
     * \param[in] cov The covariance of the start and end point
     */
    matrix22 get_distance_covariance(const vector2& point, const matrix44& cov)
    {
        const double x0 = point.x();
        const double y0 = point.y();

        const double x1 = this->_startPoint.x();
        const double y1 = this->_startPoint.y();

        const double x2 = this->_endPoint.x();
        const double y2 = this->_endPoint.y();

        const double theta17 = SQR(x1 - x2) + SQR(y1 - y2);
        const double theta16 = 2.0 * (x1 - x2);
        const double theta15 = 2.0 * pow(theta17, 3.0 / 2.0);
        const double theta14 = 2.0 * (y1 - y2);
        const double theta13 = (x1 - x2) * (x1 - x0) / sqrt(theta17) + (y1 - y2) * (y1 - y0) / sqrt(theta17);
        const double theta12 = (x1 - x0) / sqrt(theta17);
        const double theta11 = theta16 * (x1 - x2) * (x1 - x0) / theta15;
        const double theta10 = theta16 * (y1 - y2) * (y1 - y0) / theta15;
        const double theta9 = (y1 - y0) / sqrt(theta17);
        const double theta8 = theta14 * (x1 - x2) * (x1 - x0) / theta15;
        const double theta7 = theta14 * (y1 - y2) * (y1 - y0) / theta15;
        const double theta6 = theta16 * theta13 * (x1 - x2) / theta15;
        const double theta5 = theta16 * theta13 * (y1 - y2) / theta15;
        const double theta4 = theta14 * theta13 * (x1 - x2) / theta15;
        const double theta3 = theta14 * theta13 * (y1 - y2) / theta15;
        const double theta2 = (x1 - x2) / sqrt(theta17) + theta12 - theta11 - theta10;
        const double theta1 = (y1 - y2) / sqrt(theta17) + theta9 - theta8 - theta7;

        Eigen::Matrix<double, 2, 4> jac;

        jac(0, 0) = theta6 - (x1 - x2) * theta2 / sqrt(theta17) - theta13 / sqrt(theta17) - 1.0;
        jac(0, 1) = theta4 - (x1 - x2) * theta1 / sqrt(theta17);
        jac(0, 2) = theta13 / sqrt(theta17) - (x1 - x2) * (theta11 - theta12 + theta10) / sqrt(theta17) - theta6;
        jac(0, 3) = -(x1 - x2) * (theta8 - theta9 + theta7) / sqrt(theta17) - theta4;

        jac(1, 0) = theta5 - (y1 - y2) * theta2 / sqrt(theta17);
        jac(1, 1) = theta3 - (y1 - y2) * theta1 / sqrt(theta17) - theta13 / sqrt(theta17) - 1.0;
        jac(1, 2) = -(y1 - y2) * (theta11 - theta12 + theta10) / sqrt(theta17) - theta5;
        jac(1, 3) = theta13 / sqrt(theta17) - (y1 - y2) * (theta8 - theta9 + theta7) / sqrt(theta17) - theta3;

        return jac * cov * jac.transpose();
    }

    [[nodiscard]] Eigen::Vector<double, Dim> get_end_point() const noexcept { return _endPoint; };
    void set_end_point(const Eigen::Vector<double, Dim>& endPoint) noexcept { _endPoint = endPoint; };
    void set_points(const Eigen::Vector<double, Dim>& startPoint, const Eigen::Vector<double, Dim>& endPoint) noexcept
    {
        this->set_start_point(startPoint);
        set_end_point(endPoint);
    };

  protected:
    Eigen::Vector<double, Dim> _endPoint;
};

/**
 * \brief clamp a 2D line to screen space
 * \param[in] line the line to clamp
 * \param[out] out the line clamped
 * \return true if the line is still in screen space, false if out is invalid
 */
[[nodiscard]] bool clamp_to_screen(const Segment<2>& line, Segment<2>& out) noexcept;

/**
 * \brief check the intersection of two lines (in 2d)
 * \param[in] firstLine
 * \param[in] secondLine
 * \param[in] point the point of intersection of the two lines, if the function returns true
 */
[[nodiscard]] bool intersects(const Segment<2>& firstLine,
                              const Segment<2>& secondLine,
                              Eigen::Vector<double, 2>& point) noexcept;

/**
 * \brief check the intersection of two lines (in 3d)
 * \param[in] firstLine
 * \param[in] secondLine
 * \param[in] point the point of intersection of the two lines, if the function returns true
 */
[[nodiscard]] bool intersects(const Segment<3>& firstLine,
                              const Segment<3>& secondLine,
                              Eigen::Vector<double, 3>& point) noexcept;

/**
 * \brief Segment class: a line defined by a start point and a normal vector
 */
template<int Dim> class Line : public ILine<Dim>
{
  public:
    Line() { _normal.setZero(); }

    Line(const Eigen::Vector<double, Dim>& startPoint, const Eigen::Vector<double, Dim>& normal) :
        ILine<Dim>(startPoint),
        _normal(normal.normalized())
    {
    }

    Line(const Line<Dim>& other) : ILine<Dim>(other._startPoint), _normal(other._normal) {}

    /**
     * \brief compute the normal of this line
     */
    [[nodiscard]] Eigen::Vector<double, Dim> compute_normal() const noexcept override { return _normal; }

    [[nodiscard]] Eigen::Vector<double, Dim> get_normal() const noexcept { return _normal; };
    void set_normal(const Eigen::Vector<double, Dim>& normal) noexcept { _normal = normal; };
    void set_point_and_normal(const Eigen::Vector<double, Dim>& startPoint,
                              const Eigen::Vector<double, Dim>& normal) noexcept
    {
        this->set_start_point(startPoint);
        set_normal(normal);
    };

  protected:
    Eigen::Vector<double, Dim> _normal;
};

} // namespace rgbd_slam::utils

#endif