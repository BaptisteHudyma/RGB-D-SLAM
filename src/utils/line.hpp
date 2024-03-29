#ifndef RGBDSLAM_UTILS_LINE_HPP
#define RGBDSLAM_UTILS_LINE_HPP

#include <Eigen/Dense>
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