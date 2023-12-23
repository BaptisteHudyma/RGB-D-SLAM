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

void XYZ_Map_Writer::add_point(const vector3& pointCoordinates) noexcept
{
    if (not _file.is_open())
    {
        log_error("File is not opened");
        exit(-1);
    }

    _file << pointCoordinates.x() << " " << pointCoordinates.y() << " " << pointCoordinates.z() << "\n";
}

void XYZ_Map_Writer::add_polygon(const std::vector<vector3>& coordinates, const vector3& normal) noexcept
{
    std::ignore = coordinates;
    std::ignore = normal;
}

void XYZ_Map_Writer::add_line(const std::vector<vector3>& coordinates) noexcept { std::ignore = coordinates; }

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

void PCD_Map_Writer::add_point(const vector3& pointCoordinates) noexcept
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

void PCD_Map_Writer::add_polygon(const std::vector<vector3>& coordinates, const vector3& normal) noexcept
{
    std::ignore = coordinates;
    std::ignore = normal;
}

void PCD_Map_Writer::add_line(const std::vector<vector3>& coordinates) noexcept { std::ignore = coordinates; }
/**
 *     OBJ format
 */

OBJ_Map_Writer::OBJ_Map_Writer(const std::string& filename) : IMap_Writer(filename + ".obj") {}

void OBJ_Map_Writer::add_point(const vector3& point) noexcept
{
    if (not _file.is_open())
    {
        log_error("File is not opened");
        exit(-1);
    }

    // points are not really visible
    _file << "v " << point.x() << " " << point.y() << " " << point.z() << "\n";
    _vertexIndex++;
    _file << "p " << _vertexIndex << "\n";
}

void OBJ_Map_Writer::add_line(const std::vector<vector3>& coordinates) noexcept
{
    if (not _file.is_open())
    {
        log_error("File is not opened");
        exit(-1);
    }

    std::string vertices = "";
    for (const vector3& point: coordinates)
    {
        _file << "v " << point.x() << " " << point.y() << " " << point.z() << "\n";
        vertices += " " + std::to_string(_vertexIndex);
        _vertexIndex++;
    }
    _file << "l" << vertices << "\n";
}

void OBJ_Map_Writer::add_polygon(const std::vector<vector3>& coordinates, const vector3& normal) noexcept
{
    if (not _file.is_open())
    {
        log_error("File is not opened");
        exit(-1);
    }

    // normal can be ignored for now (enforcer by the boundary order)
    std::ignore = normal;

    std::string vertices = "";
    // the order is reversed in .obj
    if (not coordinates.empty())
    {
        for (uint i = coordinates.size() - 1; i > 0; --i)
        {
            const auto& point = coordinates[i];
            _file << "v " << point.x() << " " << point.y() << " " << point.z() << "\n";
            vertices += " " + std::to_string(_vertexIndex); // + "//" + normalIndex;
            _vertexIndex++;
        }
        _file << "f" << vertices << "\n";
    }
    else
    {
        log_error("Cannot write empty polygon to file");
    }
}

} // namespace rgbd_slam::outputs
