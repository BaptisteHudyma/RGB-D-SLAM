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

    virtual void add_point(const vector3& pointCoordinates) = 0;

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

    void add_point(const vector3& pointCoordinates) override;
};

/**
 * Writes .pcd file formats.
 * Point Cloud Data format
 */
class PCD_Map_Writer : public IMap_Writer
{
  public:
    PCD_Map_Writer(const std::string& filename);

    void add_point(const vector3& pointCoordinates) override;
};

} // namespace rgbd_slam::outputs

#endif
