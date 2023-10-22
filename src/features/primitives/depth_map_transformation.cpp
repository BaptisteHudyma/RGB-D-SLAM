#include "depth_map_transformation.hpp"
#include "../../parameters.hpp"
#include "../../utils/angle_utils.hpp"
#include "../../utils/random.hpp"
#include "coordinates.hpp"
#include <cstddef>
#include <opencv2/core/eigen.hpp>
#include <tbb/parallel_for.h>

namespace rgbd_slam::features::primitives {

Depth_Map_Transformation::Depth_Map_Transformation(const uint width, const uint height, const uint cellSize) :
    _width(width),
    _height(height),
    _cellSize(cellSize),
    _Xpre(static_cast<int>(height), static_cast<int>(width)),
    _Ypre(static_cast<int>(height), static_cast<int>(width)),
    _cellMap(static_cast<int>(height), static_cast<int>(width))
{
    assert(height > 0 and width > 0);

    // init static matrices
    init_matrices();
}

bool Depth_Map_Transformation::rectify_depth(const cv::Mat_<float>& depthImage,
                                             cv::Mat_<float>& rectifiedDepth) noexcept
{
    // will contain the projected depth image to rgb space
    rectifiedDepth = cv::Mat_<float>::zeros(static_cast<int>(_height), static_cast<int>(_width));

#ifndef MAKE_DETERMINISTIC
    // parallel loop to speed up the process
    // USING THIS PARALLEL LOOP BREAKS THE RANDOM SEEDING
    tbb::parallel_for(
            uint(0),
            _height,
            [&](uint row) {
#else
    for (uint row = 0; row < _height; ++row)
    {
#endif
                const float* depthRow = depthImage.ptr<float>(static_cast<int>(row));
                const float* preXRow = _Xpre.ptr<float>(static_cast<int>(row));
                const float* preYRow = _Ypre.ptr<float>(static_cast<int>(row));

                for (uint column = 0; column < _width; ++column)
                {
                    const float originalZ = depthRow[column];
                    if (originalZ <= 0)
                    {
                        continue;
                    }
                    // undistord the depth image
                    const vector3 original(preXRow[column] * originalZ, preYRow[column] * originalZ, originalZ);

                    // project to camera1 space
                    const vector3 projected =
                            (Parameters::get_camera_2_to_camera_1_transformation() * original.homogeneous()).head<3>();

                    // distord to align with camera1 image
                    utils::ScreenCoordinate2D screenCoordinates;
                    if (utils::CameraCoordinate(projected).to_screen_coordinates(screenCoordinates))
                    {
                        const uint projCoordColumn = static_cast<uint>(floor(screenCoordinates.x()));
                        const uint projCoordRow = static_cast<uint>(floor(screenCoordinates.y()));

                        // keep projected coordinates that are in rgb image boundaries
                        if (projCoordColumn > 0 and projCoordRow > 0 and projCoordColumn < _width and
                            projCoordRow < _height)
                        {
                            // set transformed depth image
                            rectifiedDepth.at<float>(static_cast<int>(projCoordRow),
                                                     static_cast<int>(projCoordColumn)) =
                                    static_cast<float>(projected.z());
                        }
                    }
                    else
                    {
                        // impossible: we checked that z > 0 before
                        exit(-1);
                    }
                }
            }
#ifndef MAKE_DETERMINISTIC
    );
#endif

    return true;
}

bool Depth_Map_Transformation::get_organized_cloud_array(const cv::Mat_<float>& depthImage,
                                                         matrixf& organizedCloudArray) noexcept
{
    assert(depthImage.rows == static_cast<int>(_height));
    assert(depthImage.cols == static_cast<int>(_width));

    // will contain the projected depth image to rgb space
    organizedCloudArray = matrixf::Zero(static_cast<long>(_width) * _height, 3);

#ifndef MAKE_DETERMINISTIC
    // parallel loop to speed up the process
    // USING THIS PARALLEL LOOP BREAKS THE RANDOM SEEDING
    tbb::parallel_for(uint(0), _height, [&](uint row) {
        for (uint row = 0; row < _height; ++row)
        {
            const float* depthRow = depthImage.ptr<float>(static_cast<int>(row));
            for (uint column = 0; column < _width; ++column)
            {
                const float z = depthRow[column];
                if (z > 0)
                {
                    // set convertion matrix
                    const int id = _cellMap.at<int>(static_cast<int>(row), static_cast<int>(column));
                    assert(id >= 0 and id < organizedCloudArray.rows());

                    const utils::CameraCoordinate& cameraCoordinates =
                            utils::ScreenCoordinate(position[1], position[0], z).to_camera_coordinates();

                    // undistorded depth
                    organizedCloudArray(id, 0) = static_cast<float>(cameraCoordinates.x());
                    organizedCloudArray(id, 1) = static_cast<float>(cameraCoordinates.y());
                    organizedCloudArray(id, 2) = z;
                }
            }
        }
    });
#else
    // use opencv parallel foreach (TODO: maybe needs a mutex ? than not as efficient)
    depthImage.forEach([this, &organizedCloudArray](const float z, const int position[]) {
        if (z > 0)
        {
            // set convertion matrix
            const int id = _cellMap(position[0], position[1]);
            assert(id >= 0 and id < organizedCloudArray.rows());

            const utils::CameraCoordinate& cameraCoordinates =
                    utils::ScreenCoordinate(position[1], position[0], z).to_camera_coordinates();

            // undistorded depth
            organizedCloudArray(id, 0) = static_cast<float>(cameraCoordinates.x());
            organizedCloudArray(id, 1) = static_cast<float>(cameraCoordinates.y());
            organizedCloudArray(id, 2) = z;
        }
    });
#endif

    return true;
}

/*
 *  Called after loading parameters to init matrices
 */
void Depth_Map_Transformation::init_matrices() noexcept
{
    const uint horizontalCellsCount = _width / _cellSize;

    // Pre-computations for backprojection
    for (uint row = 0; row < _height; ++row)
    {
        const uint cellR = static_cast<uint>(floor(row / _cellSize));
        const uint localR = static_cast<uint>(floor(row % _cellSize));

        for (uint colum = 0; colum < _width; ++colum)
        {
            const utils::CameraCoordinate2D& cameraProjection =
                    utils::ScreenCoordinate2D(colum, row).to_camera_coordinates();

            // Not efficient but at this stage doesn t matter
            _Xpre.at<float>(static_cast<int>(row), static_cast<int>(colum)) = static_cast<float>(cameraProjection.x());
            _Ypre.at<float>(static_cast<int>(row), static_cast<int>(colum)) = static_cast<float>(cameraProjection.y());

            // Pre-computations for maping an image point cloud to a cache-friendly array where cell's local point
            // clouds are contiguous
            const uint cellC = static_cast<uint>(floor(colum / _cellSize));
            const uint localC = static_cast<uint>(floor(colum % _cellSize));
            _cellMap.at<int>(static_cast<int>(row), static_cast<int>(colum)) = static_cast<int>(
                    floor((cellR * horizontalCellsCount + cellC) * pow(_cellSize, 2) + localR * _cellSize + localC));
        }
    }
}

} // namespace rgbd_slam::features::primitives
