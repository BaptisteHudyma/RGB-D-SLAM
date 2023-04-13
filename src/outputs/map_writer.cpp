#include "map_writer.hpp"
#include "logger.hpp"
#include <string>

namespace rgbd_slam::outputs {

IMap_Writer::IMap_Writer(const std::string& filename)
{
    _file.open(filename, std::ios_base::trunc | std::ios_base::out);
    if (not _file.is_open())
    {
        log_error("Could not open out file " + filename);
        exit(-1);
    }
}

IMap_Writer::~IMap_Writer()
{
    if (_file.is_open())
        _file.close();
}

/**
 *     XYZ format
 */
XYZ_Map_Writer::XYZ_Map_Writer(const std::string& filename) : IMap_Writer(filename + ".xyz") {}

void XYZ_Map_Writer::add_point(const vector3& pointCoordinates)
{
    if (not _file.is_open())
    {
        log_error("File is not opened");
        exit(-1);
    }

    _file << pointCoordinates.x() << " " << pointCoordinates.y() << " " << pointCoordinates.z() << "\n";
}

void XYZ_Map_Writer::add_polygon(const std::vector<vector3>& coordinates)
{
    (void)coordinates;
    // not implemented for xyz
}

/**
 *     PCD format
 */

PCD_Map_Writer::PCD_Map_Writer(const std::string& filename) : IMap_Writer(filename + ".pcd")
{
    _file << "# .PCD v.7 - Point Cloud Data file format\n";
    _file << "VERSION .7\n";
    _file << "FIELDS x y z\n";
    _file << "SIZE 4 4 4\n";
    _file << "TYPE F F F\n";
    _file << "POINTS 0\n"; // Index 5
    _file << "DATA ascii\n";
}

void PCD_Map_Writer::add_point(const vector3& pointCoordinates)
{
    if (not _file.is_open())
    {
        log_error("File is not opened");
        exit(-1);
    }

    /*
    TODO: update point count
    const std::string searchString = "POINTS";
    uint indexOfLine = 0;
    while(std::getline(_file, line)) { // I changed this, see below
        ++indexOfLine;
        if (line.find(search, 0) != string::npos) {
            cout << "found: " << search << "line: " << curLine << endl;
        }
    }*/
    _file << pointCoordinates.x() << " " << pointCoordinates.y() << " " << pointCoordinates.z() << "\n";
}

void PCD_Map_Writer::add_polygon(const std::vector<vector3>& coordinates)
{
    (void)coordinates;
    // not implemented for pcd
}

/**
 *     OBJ format
 */

OBJ_Map_Writer::OBJ_Map_Writer(const std::string& filename) : IMap_Writer(filename + ".obj") {}

void OBJ_Map_Writer::add_point(const vector3& pointCoordinates)
{
    if (not _file.is_open())
    {
        log_error("File is not opened");
        exit(-1);
    }

    // points are not really visible
    _file << "v " << pointCoordinates.transpose() << "\n";
    _file << "p -1\n";
}

void OBJ_Map_Writer::add_polygon(const std::vector<vector3>& coordinates)
{
    if (not _file.is_open())
    {
        log_error("File is not opened");
        exit(-1);
    }

    // not implemented for pcd
    std::string vertices = "";
    for (uint i = 0; i < coordinates.size(); ++i)
    {
        const vector3& point = coordinates[i];
        _file << "v " << point.transpose() << "\n";
        vertices += " -" + std::to_string(i + 1);
    }
    _file << "f" << vertices << "\n";
}

} // namespace rgbd_slam::outputs
