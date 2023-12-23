#ifndef RGBDSLAM_OUTPUTS_MAP_WRITER_HPP
#define RGBDSLAM_OUTPUTS_MAP_WRITER_HPP

#include "../types.hpp"
#include <fstream>

namespace rgbd_slam::outputs {

/**
 * \brief Interface for the map writter classes
 */
class IMap_Writer
{
  public:
    IMap_Writer(const std::string& filename);
    virtual ~IMap_Writer();

    virtual void add_point(const vector3& pointCoordinates) noexcept = 0;

    virtual void add_line(const std::vector<vector3>& coordinates) noexcept = 0;

    virtual void add_polygon(const std::vector<vector3>& coordinates, const vector3& normal) noexcept = 0;

  protected:
    std::ofstream _file;
};

/**
 * Writes .xyz file formats.
 * Stores only 3D points
 */
class XYZ_Map_Writer : public IMap_Writer
{
  public:
    XYZ_Map_Writer(const std::string& filename);

    void add_point(const vector3& pointCoordinates) noexcept override;

    void add_line(const std::vector<vector3>& coordinates) noexcept override;

    void add_polygon(const std::vector<vector3>& coordinates, const vector3& normal) noexcept override;
};

/**
 * Writes .pcd file formats.
 * Point Cloud Data format
 */
class PCD_Map_Writer : public IMap_Writer
{
  public:
    PCD_Map_Writer(const std::string& filename);

    void add_point(const vector3& pointCoordinates) noexcept override;

    void add_line(const std::vector<vector3>& coordinates) noexcept override;

    void add_polygon(const std::vector<vector3>& coordinates, const vector3& normal) noexcept override;
};

/**
 * Writes .obg file formats.
 * Object file Format
 */
class OBJ_Map_Writer : public IMap_Writer
{
  public:
    OBJ_Map_Writer(const std::string& filename);

    void add_point(const vector3& pointCoordinates) noexcept override;

    void add_line(const std::vector<vector3>& coordinates) noexcept override;

    void add_polygon(const std::vector<vector3>& coordinates, const vector3& normal) noexcept override;

  protected:
    size_t _vertexIndex = 1;
    size_t _vectorIndex = 1;
};

} // namespace rgbd_slam::outputs

#endif
