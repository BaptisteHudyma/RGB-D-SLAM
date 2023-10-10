#include "depth_map_transformation.hpp"
#include "../../parameters.hpp"
#include "../../utils/angle_utils.hpp"
#include "../../utils/random.hpp"
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

    _isOk = false;
    _isOk = load_parameters();
    if (this->is_ok())
        init_matrices();
}

bool Depth_Map_Transformation::rectify_depth(const cv::Mat_<float>& depthImage,
                                             cv::Mat_<float>& rectifiedDepth) noexcept
{
    if (not this->is_ok())
        return false;

    const static float x0RStereo = _Rstereo(0, 0);
    const static float x1RStereo = _Rstereo(1, 0);
    const static float x2RStereo = _Rstereo(2, 0);

    const static float y0RStereo = _Rstereo(0, 1);
    const static float y1RStereo = _Rstereo(1, 1);
    const static float y2RStereo = _Rstereo(2, 1);

    const static float z0RStereo = _Rstereo(0, 2);
    const static float z1RStereo = _Rstereo(1, 2);
    const static float z2RStereo = _Rstereo(2, 2);

    const static float xTStereo = _Tstereo(0);
    const static float yTStereo = _Tstereo(1);
    const static float zTStereo = _Tstereo(2);

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
                    const float originalX = preXRow[column] * originalZ;
                    const float originalY = preYRow[column] * originalZ;

                    // project to rgb space
                    const float x = originalX * x0RStereo + originalY * y0RStereo + originalZ * z0RStereo + xTStereo;
                    const float y = originalX * x1RStereo + originalY * y1RStereo + originalZ * z1RStereo + yTStereo;
                    const float z = originalX * x2RStereo + originalY * y2RStereo + originalZ * z2RStereo + zTStereo;

                    // distord to align with rgb image
                    const uint projCoordColumn = static_cast<uint>(floor(x * _fxRgb / z + _cxRgb));
                    const uint projCoordRow = static_cast<uint>(floor(y * _fyRgb / z + _cyRgb));

                    // keep projected coordinates that are in rgb image boundaries
                    if (projCoordColumn > 0 and projCoordRow > 0 and projCoordColumn < _width and
                        projCoordRow < _height)
                    {
                        // set transformed depth image
                        rectifiedDepth.at<float>(static_cast<int>(projCoordRow), static_cast<int>(projCoordColumn)) = z;
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
    if (not this->is_ok())
        return false;

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

                    // undistord depth
                    organizedCloudArray(id, 0) = (static_cast<float>(column) - _cxRgb) * z / _fxRgb;
                    organizedCloudArray(id, 1) = (static_cast<float>(row) - _cyRgb) * z / _fyRgb;
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

            // undistorded depth
            organizedCloudArray(id, 0) = (static_cast<float>(position[1]) - _cxRgb) * z / _fxRgb;
            organizedCloudArray(id, 1) = (static_cast<float>(position[0]) - _cyRgb) * z / _fyRgb;
            organizedCloudArray(id, 2) = z;
        }
    });
#endif

    return true;
}

bool Depth_Map_Transformation::load_parameters() noexcept
{
    // TODO check parameters
    _fxIr = static_cast<float>(Parameters::get_camera_2_focal_x());
    _fyIr = static_cast<float>(Parameters::get_camera_2_focal_y());
    _cxIr = static_cast<float>(Parameters::get_camera_2_center_x());
    _cyIr = static_cast<float>(Parameters::get_camera_2_center_y());

    _fxRgb = static_cast<float>(Parameters::get_camera_1_focal_x());
    _fyRgb = static_cast<float>(Parameters::get_camera_1_focal_y());
    _cxRgb = static_cast<float>(Parameters::get_camera_1_center_x());
    _cyRgb = static_cast<float>(Parameters::get_camera_1_center_y());

    _Tstereo = cv::Mat_<float>(3, 1);
    _Tstereo.at<float>(0) = static_cast<float>(Parameters::get_camera_2_translation_x());
    _Tstereo.at<float>(1) = static_cast<float>(Parameters::get_camera_2_translation_y());
    _Tstereo.at<float>(2) = static_cast<float>(Parameters::get_camera_2_translation_z());

    const EulerAngles rotationEuler(Parameters::get_camera_2_rotation_x(),
                                    Parameters::get_camera_2_rotation_y(),
                                    Parameters::get_camera_2_rotation_z());
    const matrix33 cameraRotation = utils::get_rotation_matrix_from_euler_angles(rotationEuler);

    cv::Mat_<double> rot(3, 3);
    cv::eigen2cv(cameraRotation, rot);
    _Rstereo = rot;

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
            // Not efficient but at this stage doesn t matter
            _Xpre.at<float>(static_cast<int>(row), static_cast<int>(colum)) =
                    (static_cast<float>(colum) - _cxIr) / _fxIr;
            _Ypre.at<float>(static_cast<int>(row), static_cast<int>(colum)) = (static_cast<float>(row) - _cyIr) / _fyIr;

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
