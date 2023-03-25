#include "map_writer.hpp"
#include "logger.hpp"

namespace rgbd_slam {
namespace outputs {

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

} // namespace outputs
} // namespace rgbd_slam
