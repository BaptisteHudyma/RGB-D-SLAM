
#include "angle_utils.hpp"
#include "distance_utils.hpp"
#include "line.hpp"
#include "parameters.hpp"
#include "types.hpp"
#include "utils/camera_transformation.hpp"
#include "coordinates/inverse_depth_coordinates.hpp"
#include "coordinates/point_coordinates.hpp"
#include "coordinates/plane_coordinates.hpp"
#include "coordinates/basis_changes.hpp"
#include <gtest/gtest.h>

namespace rgbd_slam {

void estimate_point_error(const vector3& pointA, const vector3& pointB)
{
    EXPECT_NEAR(pointA.x(), pointB.x(), 0.001);
    EXPECT_NEAR(pointA.y(), pointB.y(), 0.001);
    EXPECT_NEAR(pointA.z(), pointB.z(), 0.001);
}

TEST(QuaternionFromEuler, BasicRotationsFromEuler)
{
    quaternion quat;

    quat = utils::get_quaternion_from_euler_angles(
            EulerAngles(0.0 * EulerToRadian, 0.0 * EulerToRadian, 0.0 * EulerToRadian));
    EXPECT_NEAR(quat.x(), 0.0, 1e-5);
    EXPECT_NEAR(quat.y(), 0.0, 1e-5);
    EXPECT_NEAR(quat.z(), 0.0, 1e-5);
    EXPECT_NEAR(quat.w(), 1.0, 1e-5);

    quat = utils::get_quaternion_from_euler_angles(
            EulerAngles(0.0 * EulerToRadian, 0.0 * EulerToRadian, 180.0 * EulerToRadian));
    EXPECT_NEAR(quat.x(), 1.0, 1e-5);
    EXPECT_NEAR(quat.y(), 0.0, 1e-5);
    EXPECT_NEAR(quat.z(), 0.0, 1e-5);
    EXPECT_NEAR(quat.w(), 0.0, 1e-5);

    quat = utils::get_quaternion_from_euler_angles(
            EulerAngles(0.0 * EulerToRadian, 180.0 * EulerToRadian, 0.0 * EulerToRadian));
    EXPECT_NEAR(quat.x(), 0.0, 1e-5);
    EXPECT_NEAR(quat.y(), 1.0, 1e-5);
    EXPECT_NEAR(quat.z(), 0.0, 1e-5);
    EXPECT_NEAR(quat.w(), 0.0, 1e-5);

    quat = utils::get_quaternion_from_euler_angles(
            EulerAngles(180.0 * EulerToRadian, 0.0 * EulerToRadian, 0.0 * EulerToRadian));
    EXPECT_NEAR(quat.x(), 0.0, 1e-5);
    EXPECT_NEAR(quat.y(), 0.0, 1e-5);
    EXPECT_NEAR(quat.z(), 1.0, 1e-5);
    EXPECT_NEAR(quat.w(), 0.0, 1e-5);

    // inverse rotations

    quat = utils::get_quaternion_from_euler_angles(
            EulerAngles(0.0 * EulerToRadian, 0.0 * EulerToRadian, -180.0 * EulerToRadian));
    EXPECT_NEAR(quat.x(), -1.0, 1e-5);
    EXPECT_NEAR(quat.y(), 0.0, 1e-5);
    EXPECT_NEAR(quat.z(), 0.0, 1e-5);
    EXPECT_NEAR(quat.w(), 0.0, 1e-5);

    quat = utils::get_quaternion_from_euler_angles(
            EulerAngles(0.0 * EulerToRadian, -180.0 * EulerToRadian, 0.0 * EulerToRadian));
    EXPECT_NEAR(quat.x(), 0.0, 1e-5);
    EXPECT_NEAR(quat.y(), -1.0, 1e-5);
    EXPECT_NEAR(quat.z(), 0.0, 1e-5);
    EXPECT_NEAR(quat.w(), 0.0, 1e-5);

    quat = utils::get_quaternion_from_euler_angles(
            EulerAngles(-180.0 * EulerToRadian, 0.0 * EulerToRadian, 0.0 * EulerToRadian));
    EXPECT_NEAR(quat.x(), 0.0, 1e-5);
    EXPECT_NEAR(quat.y(), 0.0, 1e-5);
    EXPECT_NEAR(quat.z(), -1.0, 1e-5);
    EXPECT_NEAR(quat.w(), 0.0, 1e-5);

    // half rotations

    quat = utils::get_quaternion_from_euler_angles(
            EulerAngles(0.0 * EulerToRadian, 0.0 * EulerToRadian, 90.0 * EulerToRadian));
    EXPECT_NEAR(quat.x(), 0.7071068, 1e-5);
    EXPECT_NEAR(quat.y(), 0.0, 1e-5);
    EXPECT_NEAR(quat.z(), 0.0, 1e-5);
    EXPECT_NEAR(quat.w(), 0.7071068, 1e-5);

    quat = utils::get_quaternion_from_euler_angles(
            EulerAngles(0.0 * EulerToRadian, 90.0 * EulerToRadian, 0.0 * EulerToRadian));
    EXPECT_NEAR(quat.x(), 0.0, 1e-5);
    EXPECT_NEAR(quat.y(), 0.7071068, 1e-5);
    EXPECT_NEAR(quat.z(), 0.0, 1e-5);
    EXPECT_NEAR(quat.w(), 0.7071068, 1e-5);

    quat = utils::get_quaternion_from_euler_angles(
            EulerAngles(90.0 * EulerToRadian, 0.0 * EulerToRadian, 0.0 * EulerToRadian));
    EXPECT_NEAR(quat.x(), 0.0, 1e-5);
    EXPECT_NEAR(quat.y(), 0.0, 1e-5);
    EXPECT_NEAR(quat.z(), 0.7071068, 1e-5);
    EXPECT_NEAR(quat.w(), 0.7071068, 1e-5);

    // inverse half rotations

    quat = utils::get_quaternion_from_euler_angles(
            EulerAngles(0.0 * EulerToRadian, 0.0 * EulerToRadian, -90.0 * EulerToRadian));
    EXPECT_NEAR(quat.x(), -0.7071068, 1e-5);
    EXPECT_NEAR(quat.y(), 0.0, 1e-5);
    EXPECT_NEAR(quat.z(), 0.0, 1e-5);
    EXPECT_NEAR(quat.w(), 0.7071068, 1e-5);

    quat = utils::get_quaternion_from_euler_angles(
            EulerAngles(0.0 * EulerToRadian, -90.0 * EulerToRadian, 0.0 * EulerToRadian));
    EXPECT_NEAR(quat.x(), 0.0, 1e-5);
    EXPECT_NEAR(quat.y(), -0.7071068, 1e-5);
    EXPECT_NEAR(quat.z(), 0.0, 1e-5);
    EXPECT_NEAR(quat.w(), 0.7071068, 1e-5);

    quat = utils::get_quaternion_from_euler_angles(
            EulerAngles(-90.0 * EulerToRadian, 0.0 * EulerToRadian, 0.0 * EulerToRadian));
    EXPECT_NEAR(quat.x(), 0.0, 1e-5);
    EXPECT_NEAR(quat.y(), 0.0, 1e-5);
    EXPECT_NEAR(quat.z(), -0.7071068, 1e-5);
    EXPECT_NEAR(quat.w(), 0.7071068, 1e-5);
}

TEST(QuaternionFromEuler, DoubleRotationsFromEuler)
{
    quaternion quat;

    quat = utils::get_quaternion_from_euler_angles(
            EulerAngles(0.0 * EulerToRadian, 90.0 * EulerToRadian, 90.0 * EulerToRadian));
    EXPECT_NEAR(quat.x(), 0.5, 1e-5);
    EXPECT_NEAR(quat.y(), 0.5, 1e-5);
    EXPECT_NEAR(quat.z(), 0.5, 1e-5);
    EXPECT_NEAR(quat.w(), 0.5, 1e-5);

    quat = utils::get_quaternion_from_euler_angles(
            EulerAngles(0.0 * EulerToRadian, -90.0 * EulerToRadian, 90.0 * EulerToRadian));
    EXPECT_NEAR(quat.x(), 0.5, 1e-5);
    EXPECT_NEAR(quat.y(), -0.5, 1e-5);
    EXPECT_NEAR(quat.z(), -0.5, 1e-5);
    EXPECT_NEAR(quat.w(), 0.5, 1e-5);

    quat = utils::get_quaternion_from_euler_angles(
            EulerAngles(0.0 * EulerToRadian, -90.0 * EulerToRadian, -90.0 * EulerToRadian));
    EXPECT_NEAR(quat.x(), -0.5, 1e-5);
    EXPECT_NEAR(quat.y(), -0.5, 1e-5);
    EXPECT_NEAR(quat.z(), 0.5, 1e-5);
    EXPECT_NEAR(quat.w(), 0.5, 1e-5);

    quat = utils::get_quaternion_from_euler_angles(
            EulerAngles(90.0 * EulerToRadian, 0.0 * EulerToRadian, 90.0 * EulerToRadian));
    EXPECT_NEAR(quat.x(), 0.5, 1e-5);
    EXPECT_NEAR(quat.y(), -0.5, 1e-5);
    EXPECT_NEAR(quat.z(), 0.5, 1e-5);
    EXPECT_NEAR(quat.w(), 0.5, 1e-5);

    quat = utils::get_quaternion_from_euler_angles(
            EulerAngles(90.0 * EulerToRadian, 0.0 * EulerToRadian, -90.0 * EulerToRadian));
    EXPECT_NEAR(quat.x(), -0.5, 1e-5);
    EXPECT_NEAR(quat.y(), 0.5, 1e-5);
    EXPECT_NEAR(quat.z(), 0.5, 1e-5);
    EXPECT_NEAR(quat.w(), 0.5, 1e-5);

    quat = utils::get_quaternion_from_euler_angles(
            EulerAngles(-90.0 * EulerToRadian, 0.0 * EulerToRadian, 90.0 * EulerToRadian));
    EXPECT_NEAR(quat.x(), 0.5, 1e-5);
    EXPECT_NEAR(quat.y(), 0.5, 1e-5);
    EXPECT_NEAR(quat.z(), -0.5, 1e-5);
    EXPECT_NEAR(quat.w(), 0.5, 1e-5);

    quat = utils::get_quaternion_from_euler_angles(
            EulerAngles(-90.0 * EulerToRadian, 0.0 * EulerToRadian, -90.0 * EulerToRadian));
    EXPECT_NEAR(quat.x(), -0.5, 1e-5);
    EXPECT_NEAR(quat.y(), -0.5, 1e-5);
    EXPECT_NEAR(quat.z(), -0.5, 1e-5);
    EXPECT_NEAR(quat.w(), 0.5, 1e-5);
}

TEST(CoordinateSystemChangeTests, CameraToWorldAtOrigin)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            utils::compute_camera_to_world_transform_no_correction(quaternion::Identity(), vector3(0, 0, 0));

    const matrix44& tr = get_transformation_matrix(
            vector3(1, 0, 0), vector3(0, 1, 0), vector3::Zero(), vector3(1, 0, 0), vector3(0, 1, 0), vector3::Zero());

    EXPECT_TRUE(cameraToWorld.isApprox(tr));
}

TEST(CoordinateSystemChangeTests, CameraToWorldFarFromOrigin)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            utils::compute_camera_to_world_transform_no_correction(quaternion::Identity(), vector3(-100, 100, 200));

    const matrix44& tr = get_transformation_matrix(vector3(1, 0, 0),
                                                   vector3(0, 1, 0),
                                                   vector3::Zero(),
                                                   vector3(1, 0, 0),
                                                   vector3(0, 1, 0),
                                                   vector3(-100, 100, 200));

    EXPECT_TRUE(cameraToWorld.isApprox(tr));
}

TEST(CoordinateSystemChangeTests, CameraToWorldAtOriginWithRotation)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            utils::compute_camera_to_world_transform_no_correction(quaternion(0.0, 1.0, 0.0, 0.0), vector3(0, 0, 0));

    const matrix44& tr = get_transformation_matrix(
            vector3(1, 0, 0), vector3(0, 1, 0), vector3::Zero(), vector3(1, 0, 0), vector3(0, -1, 0), vector3::Zero());

    EXPECT_TRUE(cameraToWorld.isApprox(tr));
}

TEST(CoordinateSystemChangeTests, CameraToWorldAtOriginWithRotation2)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            utils::compute_camera_to_world_transform_no_correction(quaternion(0.0, 0.0, 1.0, 0.0), vector3(0, 0, 0));

    const matrix44& tr = get_transformation_matrix(
            vector3(1, 0, 0), vector3(0, 1, 0), vector3::Zero(), vector3(-1, 0, 0), vector3(0, 1, 0), vector3::Zero());

    EXPECT_TRUE(cameraToWorld.isApprox(tr));
}

TEST(CoordinateSystemChangeTests, CameraToWorldAtOriginWithRotation3)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            utils::compute_camera_to_world_transform_no_correction(quaternion(0.0, 0, 0, 1.0), vector3(0, 0, 0));

    const matrix44& tr = get_transformation_matrix(
            vector3(1, 0, 0), vector3(0, 1, 0), vector3::Zero(), vector3(-1, 0, 0), vector3(0, -1, 0), vector3::Zero());

    EXPECT_TRUE(cameraToWorld.isApprox(tr));
}

TEST(CoordinateSystemChangeTests, CameraToWorldAtOriginWithRotationCombined)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            utils::compute_camera_to_world_transform_no_correction(quaternion(0.5, 0.5, 0.5, 0.5), vector3(0, 0, 0));

    const matrix44& tr = get_transformation_matrix(
            vector3(1, 0, 0), vector3(0, 1, 0), vector3::Zero(), vector3(0, 1, 0), vector3(0, 0, 1), vector3::Zero());

    EXPECT_TRUE(cameraToWorld.isApprox(tr));
}

TEST(CoordinateSystemChangeTests, CameraToWorldFarFromOriginWithRotation)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld = utils::compute_camera_to_world_transform_no_correction(
            quaternion(0.0, 1.0, 0.0, 0.0), vector3(-100, 100, 200));

    const matrix44& tr = get_transformation_matrix(vector3(1, 0, 0),
                                                   vector3(0, 1, 0),
                                                   vector3::Zero(),
                                                   vector3(1, 0, 0),
                                                   vector3(0, -1, 0),
                                                   vector3(-100, 100, 200));
    EXPECT_TRUE(cameraToWorld.isApprox(tr));
}

TEST(CoordinateSystemChangeTests, CameraToWorldFarFromOriginSameWithRotation)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            utils::compute_camera_to_world_transform_no_correction(quaternion(0.0, 1.0, 0.0, 0.0), vector3(0, 0, 0));

    const matrix44& tr = get_transformation_matrix(vector3(1, 0, 0),
                                                   vector3(0, 1, 0),
                                                   vector3(-100, 100, 200),
                                                   vector3(1, 0, 0),
                                                   vector3(0, -1, 0),
                                                   vector3(-100, 100, 200));
    EXPECT_TRUE(cameraToWorld.isApprox(tr));
}

/**
 *
 * CHECK CAMERA TO WORLD AND WORLD TO CAMERA BASIS CHANGES
 *
 */

TEST(CoordinateCameraToWorld, CenterToWorld)
{
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(
            utils::get_quaternion_from_euler_angles(
                    EulerAngles(0.0 * EulerToRadian, 0.0 * EulerToRadian, 0.0 * EulerToRadian)),
            vector3(0.0, 0.0, 0.0));
    /*
        ASSERT_NEAR(c2w.translation().x(), 0.0, 1e-10);
        ASSERT_NEAR(c2w.translation().y(), 0.0, 1e-10);
        ASSERT_NEAR(c2w.translation().z(), 0.0, 1e-10);
        ASSERT_NEAR(c2w.rotation().eulerAngles(0, 1, 2).x(), M_PI / 2.0, 1e-10);
        ASSERT_NEAR(c2w.rotation().eulerAngles(0, 1, 2).y(), M_PI / 2.0, 1e-10);
        ASSERT_NEAR(c2w.rotation().eulerAngles(0, 1, 2).z(), -M_PI, 1e-10);
    */
    WorldCoordinate wc;

    CameraCoordinate c1(0.0, 0.0, 0.0);
    wc = c1.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);

    // check axis

    CameraCoordinate c2(1.0, 0.0, 0.0);
    wc = c2.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), -1.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);

    CameraCoordinate c4(0.0, 1.0, 0.0);
    wc = c4.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), -1.0, 1e-10);

    CameraCoordinate c6(0.0, 0.0, 1.0);
    wc = c6.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 1.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);

    // check inverse

    CameraCoordinate c3(-1.0, 0.0, 0.0);
    wc = c3.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 1.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);

    CameraCoordinate c5(0.0, -1.0, 0.0);
    wc = c5.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 1.0, 1e-10);

    CameraCoordinate c7(0.0, 0.0, -1.0);
    wc = c7.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), -1.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);
}

TEST(CoordinateWorldToCamera, CenterToCamera)
{
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(
            utils::get_quaternion_from_euler_angles(
                    EulerAngles(0.0 * EulerToRadian, 0.0 * EulerToRadian, 0.0 * EulerToRadian)),
            vector3(0.0, 0.0, 0.0));

    const WorldToCameraMatrix& w2c = utils::compute_world_to_camera_transform(c2w);
    /*
        ASSERT_NEAR(w2c.translation().x(), 0.0, 1e-10);
        ASSERT_NEAR(w2c.translation().y(), 0.0, 1e-10);
        ASSERT_NEAR(w2c.translation().z(), 0.0, 1e-10);
        ASSERT_NEAR(w2c.rotation().eulerAngles(0, 1, 2).x(), M_PI / 2.0, 1e-10);
        ASSERT_NEAR(w2c.rotation().eulerAngles(0, 1, 2).y(), 0.0, 1e-10);
        ASSERT_NEAR(w2c.rotation().eulerAngles(0, 1, 2).z(), M_PI / 2.0, 1e-10);
    */
    CameraCoordinate cc;

    WorldCoordinate w1(0.0, 0.0, 0.0);
    cc = w1.to_camera_coordinates(w2c);
    EXPECT_NEAR(cc.x(), 0.0, 1e-10);
    EXPECT_NEAR(cc.y(), 0.0, 1e-10);
    EXPECT_NEAR(cc.z(), 0.0, 1e-10);

    // check axis

    WorldCoordinate w2(1.0, 0.0, 0.0);
    cc = w2.to_camera_coordinates(w2c);
    EXPECT_NEAR(cc.x(), 0.0, 1e-10);
    EXPECT_NEAR(cc.y(), 0.0, 1e-10);
    EXPECT_NEAR(cc.z(), 1.0, 1e-10);

    WorldCoordinate w4(0.0, 1.0, 0.0);
    cc = w4.to_camera_coordinates(w2c);
    EXPECT_NEAR(cc.x(), -1.0, 1e-10);
    EXPECT_NEAR(cc.y(), 0.0, 1e-10);
    EXPECT_NEAR(cc.z(), 0.0, 1e-10);

    WorldCoordinate w6(0.0, 0.0, 1.0);
    cc = w6.to_camera_coordinates(w2c);
    EXPECT_NEAR(cc.x(), 0.0, 1e-10);
    EXPECT_NEAR(cc.y(), -1.0, 1e-10);
    EXPECT_NEAR(cc.z(), 0.0, 1e-10);

    // check inverse

    WorldCoordinate w3(-1.0, 0.0, 0.0);
    cc = w3.to_camera_coordinates(w2c);
    EXPECT_NEAR(cc.x(), 0.0, 1e-10);
    EXPECT_NEAR(cc.y(), 0.0, 1e-10);
    EXPECT_NEAR(cc.z(), -1.0, 1e-10);

    WorldCoordinate w5(0.0, -1.0, 0.0);
    cc = w5.to_camera_coordinates(w2c);
    EXPECT_NEAR(cc.x(), 1.0, 1e-10);
    EXPECT_NEAR(cc.y(), 0.0, 1e-10);
    EXPECT_NEAR(cc.z(), 0.0, 1e-10);

    WorldCoordinate w7(0.0, 0.0, -1.0);
    cc = w7.to_camera_coordinates(w2c);
    EXPECT_NEAR(cc.x(), 0.0, 1e-10);
    EXPECT_NEAR(cc.y(), 1.0, 1e-10);
    EXPECT_NEAR(cc.z(), 0.0, 1e-10);
}

// check raw axis (sanity)

TEST(CoordinateCameraToWorld, RawPoseRotatedYawRight)
{
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform_no_correction(
            utils::get_quaternion_from_euler_angles(
                    EulerAngles(90.0 * EulerToRadian, 0.0 * EulerToRadian, 0.0 * EulerToRadian)),
            vector3(0.0, 0.0, 0.0));

    EXPECT_NEAR(c2w.translation().x(), 0.0, 1e-10);
    EXPECT_NEAR(c2w.translation().y(), 0.0, 1e-10);
    EXPECT_NEAR(c2w.translation().z(), 0.0, 1e-10);

    WorldCoordinate wc;

    CameraCoordinate c1(0.0, 0.0, 0.0);
    wc = c1.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);

    // check axis
    CameraCoordinate c2(1.0, 0.0, 0.0);
    wc = c2.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 1.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);

    CameraCoordinate c4(0.0, 1.0, 0.0);
    wc = c4.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), -1.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);

    CameraCoordinate c6(0.0, 0.0, 1.0);
    wc = c6.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 1.0, 1e-10);
}

TEST(CoordinateCameraToWorld, RawPoseRotatedYawLeft)
{
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform_no_correction(
            utils::get_quaternion_from_euler_angles(
                    EulerAngles(-90.0 * EulerToRadian, 0.0 * EulerToRadian, 0.0 * EulerToRadian)),
            vector3(0.0, 0.0, 0.0));

    EXPECT_NEAR(c2w.translation().x(), 0.0, 1e-10);
    EXPECT_NEAR(c2w.translation().y(), 0.0, 1e-10);
    EXPECT_NEAR(c2w.translation().z(), 0.0, 1e-10);

    WorldCoordinate wc;

    CameraCoordinate c1(0.0, 0.0, 0.0);
    wc = c1.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);

    // check axis
    CameraCoordinate c2(1.0, 0.0, 0.0);
    wc = c2.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), -1.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);

    CameraCoordinate c4(0.0, 1.0, 0.0);
    wc = c4.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 1.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);

    CameraCoordinate c6(0.0, 0.0, 1.0);
    wc = c6.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 1.0, 1e-10);
}

TEST(CoordinateCameraToWorld, RawPoseRotatedPitchRight)
{
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform_no_correction(
            utils::get_quaternion_from_euler_angles(EulerAngles(0.0, 90.0 * EulerToRadian, 0.0 * EulerToRadian)),
            vector3(0.0, 0.0, 0.0));

    EXPECT_NEAR(c2w.translation().x(), 0.0, 1e-10);
    EXPECT_NEAR(c2w.translation().y(), 0.0, 1e-10);
    EXPECT_NEAR(c2w.translation().z(), 0.0, 1e-10);

    WorldCoordinate wc;

    CameraCoordinate c1(0.0, 0.0, 0.0);
    wc = c1.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);

    // check axis
    CameraCoordinate c2(1.0, 0.0, 0.0);
    wc = c2.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), -1.0, 1e-10);

    CameraCoordinate c4(0.0, 1.0, 0.0);
    wc = c4.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 1.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);

    CameraCoordinate c6(0.0, 0.0, 1.0);
    wc = c6.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 1.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);
}

TEST(CoordinateCameraToWorld, RawPoseRotatedPitchLeft)
{
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform_no_correction(
            utils::get_quaternion_from_euler_angles(EulerAngles(0.0, -90.0 * EulerToRadian, 0.0 * EulerToRadian)),
            vector3(0.0, 0.0, 0.0));

    EXPECT_NEAR(c2w.translation().x(), 0.0, 1e-10);
    EXPECT_NEAR(c2w.translation().y(), 0.0, 1e-10);
    EXPECT_NEAR(c2w.translation().z(), 0.0, 1e-10);

    WorldCoordinate wc;

    CameraCoordinate c1(0.0, 0.0, 0.0);
    wc = c1.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);

    // check axis
    CameraCoordinate c2(1.0, 0.0, 0.0);
    wc = c2.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 1.0, 1e-10);

    CameraCoordinate c4(0.0, 1.0, 0.0);
    wc = c4.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 1.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);

    CameraCoordinate c6(0.0, 0.0, 1.0);
    wc = c6.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), -1.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);
}

TEST(CoordinateCameraToWorld, RawPoseRotatedRollRight)
{
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform_no_correction(
            utils::get_quaternion_from_euler_angles(EulerAngles(0.0, 0.0, 90.0 * EulerToRadian)),
            vector3(0.0, 0.0, 0.0));

    EXPECT_NEAR(c2w.translation().x(), 0.0, 1e-10);
    EXPECT_NEAR(c2w.translation().y(), 0.0, 1e-10);
    EXPECT_NEAR(c2w.translation().z(), 0.0, 1e-10);

    WorldCoordinate wc;

    CameraCoordinate c1(0.0, 0.0, 0.0);
    wc = c1.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);

    // check axis
    CameraCoordinate c2(1.0, 0.0, 0.0);
    wc = c2.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 1.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);

    CameraCoordinate c4(0.0, 1.0, 0.0);
    wc = c4.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 1.0, 1e-10);

    CameraCoordinate c6(0.0, 0.0, 1.0);
    wc = c6.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), -1.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);
}

TEST(CoordinateCameraToWorld, RawPoseRotatedRollLeft)
{
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform_no_correction(
            utils::get_quaternion_from_euler_angles(EulerAngles(0.0, 0.0, -90.0 * EulerToRadian)),
            vector3(0.0, 0.0, 0.0));

    EXPECT_NEAR(c2w.translation().x(), 0.0, 1e-10);
    EXPECT_NEAR(c2w.translation().y(), 0.0, 1e-10);
    EXPECT_NEAR(c2w.translation().z(), 0.0, 1e-10);

    WorldCoordinate wc;

    CameraCoordinate c1(0.0, 0.0, 0.0);
    wc = c1.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);

    // check axis
    CameraCoordinate c2(1.0, 0.0, 0.0);
    wc = c2.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 1.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);

    CameraCoordinate c4(0.0, 1.0, 0.0);
    wc = c4.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), .0, 1e-10);
    EXPECT_NEAR(wc.z(), -1.0, 1e-10);

    CameraCoordinate c6(0.0, 0.0, 1.0);
    wc = c6.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 1.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);
}

// check axis in camera space

TEST(CoordinateCameraToWorld, poseRotatedYawLeft)
{
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(
            utils::get_quaternion_from_euler_angles(
                    EulerAngles(90.0 * EulerToRadian, 0.0 * EulerToRadian, 0.0 * EulerToRadian)),
            vector3(0.0, 0.0, 0.0));

    EXPECT_NEAR(c2w.translation().x(), 0.0, 1e-10);
    EXPECT_NEAR(c2w.translation().y(), 0.0, 1e-10);
    EXPECT_NEAR(c2w.translation().z(), 0.0, 1e-10);

    WorldCoordinate wc;

    CameraCoordinate c1(0.0, 0.0, 0.0);
    wc = c1.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);

    // check axis
    CameraCoordinate c2(1.0, 0.0, 0.0);
    wc = c2.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), -1.0, 1e-10);

    CameraCoordinate c4(0.0, 1.0, 0.0);
    wc = c4.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 1.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);

    CameraCoordinate c6(0.0, 0.0, 1.0);
    wc = c6.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 1.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);
}

TEST(CoordinateCameraToWorld, poseRotatedYawRight)
{
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(
            utils::get_quaternion_from_euler_angles(
                    EulerAngles(-90.0 * EulerToRadian, 0.0 * EulerToRadian, 0.0 * EulerToRadian)),
            vector3(0.0, 0.0, 0.0));

    EXPECT_NEAR(c2w.translation().x(), 0.0, 1e-10);
    EXPECT_NEAR(c2w.translation().y(), 0.0, 1e-10);
    EXPECT_NEAR(c2w.translation().z(), 0.0, 1e-10);

    WorldCoordinate wc;

    CameraCoordinate c1(0.0, 0.0, 0.0);
    wc = c1.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);

    // check axis
    CameraCoordinate c2(1.0, 0.0, 0.0);
    wc = c2.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 1.0, 1e-10);

    CameraCoordinate c4(0.0, 1.0, 0.0);
    wc = c4.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), 0.0, 1e-10);
    EXPECT_NEAR(wc.y(), 1.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);

    CameraCoordinate c6(0.0, 0.0, 1.0);
    wc = c6.to_world_coordinates(c2w);
    EXPECT_NEAR(wc.x(), -1.0, 1e-10);
    EXPECT_NEAR(wc.y(), 0.0, 1e-10);
    EXPECT_NEAR(wc.z(), 0.0, 1e-10);
}

/**
 *
 * TEST CAMERA PROJECTIONS
 *
 */

TEST(PointCoordinateSystemTests, ScreenToCameraToScreen)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const double xRange = 640.0;
    const double yRange = 480.0;
    const double zRange = 50.0;
    const double xIncrement = 7.5;
    const double yIncrement = 5.5;
    const double zIncrement = 0.5;
    for (double x = 0; x < xRange; x += xIncrement)
    {
        for (double y = 0; y < yRange; y += yIncrement)
        {
            for (double z = zIncrement; z < zRange; z += zIncrement)
            {
                const ScreenCoordinate originalScreenCoordinates(x, y, z);
                const CameraCoordinate cameraCoordinates = originalScreenCoordinates.to_camera_coordinates();
                ScreenCoordinate newScreenCoordinates;
                if (cameraCoordinates.to_screen_coordinates(newScreenCoordinates))
                {
                    estimate_point_error(originalScreenCoordinates, newScreenCoordinates);
                }
                else
                {
                    FAIL();
                }
            }

            for (double z = -zRange; z < -zIncrement; z += zIncrement)
            {
                const ScreenCoordinate originalScreenCoordinates(x, y, z);
                const CameraCoordinate cameraCoordinates = originalScreenCoordinates.to_camera_coordinates();
                ScreenCoordinate newScreenCoordinates;
                if (cameraCoordinates.to_screen_coordinates(newScreenCoordinates))
                {
                    estimate_point_error(originalScreenCoordinates, newScreenCoordinates);
                }
                else
                {
                    FAIL();
                }
            }
        }
    }
}

void test_point_set_screen_to_world_to_screen(const CameraToWorldMatrix& cameraToWorld)
{
    const WorldToCameraMatrix worldToCamera = utils::compute_world_to_camera_transform(cameraToWorld);

    const double xRange = 640.0;
    const double yRange = 480.0;
    const double zRange = 50.0;
    const double xIncrement = 7.5;
    const double yIncrement = 5.5;
    const double zIncrement = 0.5;
    for (double x = 0; x < xRange; x += xIncrement)
    {
        for (double y = 0; y < yRange; y += yIncrement)
        {
            for (double z = zIncrement; z < zRange; z += zIncrement)
            {
                const ScreenCoordinate originalScreenCoordinates(x, y, z);
                const WorldCoordinate worldCoordinates = originalScreenCoordinates.to_world_coordinates(cameraToWorld);
                ScreenCoordinate newScreenCoordinates;
                if (worldCoordinates.to_screen_coordinates(worldToCamera, newScreenCoordinates))
                {
                    estimate_point_error(originalScreenCoordinates, newScreenCoordinates);
                }
                else
                {
                    FAIL();
                }
            }
        }
    }
}

TEST(PointCoordinateSystemTests, ScreenToWorldToScreenAtOrigin)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            utils::compute_camera_to_world_transform_no_correction(quaternion::Identity(), vector3(0, 0, 0));
    test_point_set_screen_to_world_to_screen(cameraToWorld);
}

TEST(PointCoordinateSystemTests, ScreenToWorldToScreenFarFromOrigin)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            utils::compute_camera_to_world_transform_no_correction(quaternion::Identity(), vector3(-100, 1000, 100));
    test_point_set_screen_to_world_to_screen(cameraToWorld);
}

TEST(PointCoordinateSystemTests, ScreenToWorldToScreenRotation1)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            utils::compute_camera_to_world_transform_no_correction(quaternion(0.3, 0.2, 0.1, 0.4), vector3(0, 0, 0));
    test_point_set_screen_to_world_to_screen(cameraToWorld);
}

TEST(PointCoordinateSystemTests, ScreenToWorldToScreenRotation2)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            utils::compute_camera_to_world_transform_no_correction(quaternion(0.6, 0.1, 0.2, 0.1), vector3(0, 0, 0));
    test_point_set_screen_to_world_to_screen(cameraToWorld);
}

TEST(PointCoordinateSystemTests, ScreenToWorldToScreenRotation3)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld = utils::compute_camera_to_world_transform_no_correction(
            quaternion(0.6, 0.1, 0.2, 0.1), vector3(100, -100, -100));
    test_point_set_screen_to_world_to_screen(cameraToWorld);
}

void assert_inverse_point_back_proj(const CameraToWorldMatrix& c2w, const ScreenCoordinate2D& observation)
{
    const auto w2c = utils::compute_world_to_camera_transform(c2w);

    // convert to inverse
    const InverseDepthWorldPoint inverseDepth(observation, c2w);

    const auto screenProjNoMovs = inverseDepth.get_projected_screen_estimation(w2c, 0.0);
    // 1 px error
    EXPECT_NEAR(screenProjNoMovs.x(), observation.x(), 0.1);
    EXPECT_NEAR(screenProjNoMovs.y(), observation.y(), 0.1);

    // retroproject to screen
    ScreenCoordinate2D screenCoordinates;
    EXPECT_TRUE(inverseDepth.to_world_coordinates().to_screen_coordinates(w2c, screenCoordinates));
    // should be the same
    EXPECT_NEAR(screenCoordinates.x(), observation.x(), 0.01);
    EXPECT_NEAR(screenCoordinates.y(), observation.y(), 0.01);
}

TEST(InverseDepthPoint, convertBackAndForthCenter)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthTopLeft)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(vector2::Zero());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthBottomRight)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const auto& imageSize = Parameters::get_camera_1_image_size();

    // observe the center of the camera
    const ScreenCoordinate2D observation(imageSize.x(), imageSize.y());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(quaternion::Identity(), vector3::Zero());

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthCenterWithTransfoX)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w =
            utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(2500, 0.0, 0.0));

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthCenterWithTransfoY)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w =
            utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(0.0, 2500, 0.0));

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthCenterWithTransfoZ)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w =
            utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(0.0, 0.0, 2500));

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthCenterWithTransfo)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w =
            utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(2500, 1500, 3000));

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthTopLeftWithTransfo)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(vector2::Zero());
    const CameraToWorldMatrix& c2w =
            utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(2500, 1500, 3000));

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthBottomRightWithTransfo)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const auto& imageSize = Parameters::get_camera_1_image_size();

    // observe the center of the camera
    const ScreenCoordinate2D observation(imageSize.x(), imageSize.y());
    const CameraToWorldMatrix& c2w =
            utils::compute_camera_to_world_transform(quaternion::Identity(), vector3(2500, 1500, 3000));

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthCenterWithRotation)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(Parameters::get_camera_1_center());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(
            quaternion(0.246242, -0.312924, -0.896867, 0.189256), vector3::Zero());

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthTopLeftWithRotation)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(vector2::Zero());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(
            quaternion(0.246242, -0.312924, -0.896867, 0.189256), vector3::Zero());

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthBottomRightWithRotation)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const auto& imageSize = Parameters::get_camera_1_image_size();

    // observe the center of the camera
    const ScreenCoordinate2D observation(imageSize.x(), imageSize.y());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(
            quaternion(0.246242, -0.312924, -0.896867, 0.189256), vector3::Zero());

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthTopLeftWithRotationTranslation)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // observe the center of the camera
    const ScreenCoordinate2D observation(0.0, 0.0);
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(
            quaternion(0.246242, -0.312924, -0.896867, 0.189256), vector3(2500, 1500, 3000));

    assert_inverse_point_back_proj(c2w, observation);
}

TEST(InverseDepthPoint, convertBackAndForthBottomRightWithRotationTranslation)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const auto& imageSize = Parameters::get_camera_1_image_size();

    // observe the center of the camera
    const ScreenCoordinate2D observation(imageSize.x(), imageSize.y());
    const CameraToWorldMatrix& c2w = utils::compute_camera_to_world_transform(
            quaternion(0.246242, -0.312924, -0.896867, 0.189256), vector3(2500, 1500, 3000));

    assert_inverse_point_back_proj(c2w, observation);
}

void estimate_plane_error(const PlaneCoordinates& planeA, const PlaneCoordinates& planeB)
{
    const vector3& normalA = planeA.get_normal();
    const vector3& normalB = planeB.get_normal();

    EXPECT_NEAR(normalA.x(), normalB.x(), 0.001);
    EXPECT_NEAR(normalA.y(), normalB.y(), 0.001);
    EXPECT_NEAR(normalA.z(), normalB.z(), 0.001);
    EXPECT_NEAR(planeA.get_d(), planeB.get_d(), 15); // renormalization error accumulation
}

void test_plane_set_camera_to_world_to_camera(const CameraToWorldMatrix& cameraToWorld)
{
    const PlaneCameraToWorldMatrix& planeCameraToWorld = utils::compute_plane_camera_to_world_matrix(cameraToWorld);
    const PlaneWorldToCameraMatrix& planeWorldToCamera =
            utils::compute_plane_world_to_camera_matrix(utils::compute_world_to_camera_transform(cameraToWorld));
    const double normalXIter = 0.3;
    const double normalYIter = 0.1;
    const double normalZIter = 0.1;

    for (double x = 1; x <= 1.0; x += normalXIter)
    {
        for (double y = -1; y < 1.0; y += normalYIter)
        {
            for (double z = -1; z < 1.0; z += normalZIter)
            {
                const vector3 planeNormal = vector3(x, y, z).normalized();
                for (double d = 1; d < 100; d += 5.5)
                {
                    const PlaneCameraCoordinates originalCameraPlane(planeNormal, d);
                    const PlaneWorldCoordinates worldPlane =
                            originalCameraPlane.to_world_coordinates(planeCameraToWorld);
                    const PlaneCameraCoordinates newCameraCoordinates =
                            worldPlane.to_camera_coordinates(planeWorldToCamera);

                    estimate_plane_error(originalCameraPlane, newCameraCoordinates);
                }
            }
        }
    }
}

TEST(PlaneCoordinateSystemTests, ScreenToWorldToScreenAtOrigin)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            utils::compute_camera_to_world_transform_no_correction(quaternion::Identity(), vector3(0, 0, 0));
    test_plane_set_camera_to_world_to_camera(cameraToWorld);
}

TEST(PlaneCoordinateSystemTests, CameraToWorldToCameraFarFromOrigin)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            utils::compute_camera_to_world_transform_no_correction(quaternion::Identity(), vector3(-100, 1000, 100));
    test_plane_set_camera_to_world_to_camera(cameraToWorld);
}

TEST(PlaneCoordinateSystemTests, CameraToWorldToCameraRotation1)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            utils::compute_camera_to_world_transform_no_correction(quaternion(0.3, 0.2, 0.1, 0.4), vector3(0, 0, 0));
    test_plane_set_camera_to_world_to_camera(cameraToWorld);
}

TEST(PlaneCoordinateSystemTests, CameraToWorldToCameraRotation2)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld =
            utils::compute_camera_to_world_transform_no_correction(quaternion(0.6, 0.1, 0.2, 0.1), vector3(0, 0, 0));
    test_plane_set_camera_to_world_to_camera(cameraToWorld);
}

TEST(PlaneCoordinateSystemTests, CameraToWorldToCameraRotation3)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const CameraToWorldMatrix& cameraToWorld = utils::compute_camera_to_world_transform_no_correction(
            quaternion(0.6, 0.1, 0.2, 0.1), vector3(100, -100, -100));
    test_plane_set_camera_to_world_to_camera(cameraToWorld);
}

/**
 *      Test the point to line distance function
 */

TEST(PointToLine2dDistances, LineDistancesAtZero)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // origin forward
    const vector2 point1(0.0, 0.0);
    const vector2 normal1(1.0, 0.0);

    utils::Line<2> line1(point1, normal1);

    ASSERT_NEAR(line1.distance(vector2(0.0, 0.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(0.0, 0.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(1000.0, 0.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(1000.0, 0.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(-1000.0, 0.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(-1000.0, 0.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(0.0, 1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(0.0, 1000.0)).y(), 1000.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(0.0, -1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(0.0, -1000.0)).y(), -1000.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(1000.0, -1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(1000.0, -1000.0)).y(), -1000.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(1000.0, 1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(1000.0, 1000.0)).y(), 1000.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(-1000.0, -1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(-1000.0, -1000.0)).y(), -1000.0, 0.0001);

    // origin backward
    const vector2 point2(0.0, 0.0);
    const vector2 normal2(-1.0, 0.0);

    utils::Line<2> line2(point2, normal2);

    ASSERT_NEAR(line2.distance(vector2(0.0, 0.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line2.distance(vector2(0.0, 0.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line2.distance(vector2(1000.0, 0.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line2.distance(vector2(1000.0, 0.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line2.distance(vector2(-1000.0, 0.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line2.distance(vector2(-1000.0, 0.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line2.distance(vector2(0.0, 1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line2.distance(vector2(0.0, 1000.0)).y(), 1000.0, 0.0001);

    ASSERT_NEAR(line2.distance(vector2(0.0, -1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line2.distance(vector2(0.0, -1000.0)).y(), -1000.0, 0.0001);

    ASSERT_NEAR(line2.distance(vector2(1000.0, -1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line2.distance(vector2(1000.0, -1000.0)).y(), -1000.0, 0.0001);

    ASSERT_NEAR(line2.distance(vector2(1000.0, 1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line2.distance(vector2(1000.0, 1000.0)).y(), 1000.0, 0.0001);

    ASSERT_NEAR(line2.distance(vector2(-1000.0, -1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line2.distance(vector2(-1000.0, -1000.0)).y(), -1000.0, 0.0001);

    // origin right
    const vector2 point3(0.0, 0.0);
    const vector2 normal3(0.0, 1.0);

    utils::Line<2> line3(point3, normal3);

    ASSERT_NEAR(line3.distance(vector2(0.0, 0.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line3.distance(vector2(0.0, 0.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line3.distance(vector2(1000.0, 0.0)).x(), 1000.0, 0.0001);
    ASSERT_NEAR(line3.distance(vector2(1000.0, 0.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line3.distance(vector2(-1000.0, 0.0)).x(), -1000.0, 0.0001);
    ASSERT_NEAR(line3.distance(vector2(-1000.0, 0.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line3.distance(vector2(0.0, 1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line3.distance(vector2(0.0, 1000.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line3.distance(vector2(0.0, -1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line3.distance(vector2(0.0, -1000.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line3.distance(vector2(1000.0, -1000.0)).x(), 1000.0, 0.0001);
    ASSERT_NEAR(line3.distance(vector2(1000.0, -1000.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line3.distance(vector2(1000.0, 1000.0)).x(), 1000.0, 0.0001);
    ASSERT_NEAR(line3.distance(vector2(1000.0, 1000.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line3.distance(vector2(-1000.0, -1000.0)).x(), -1000.0, 0.0001);
    ASSERT_NEAR(line3.distance(vector2(-1000.0, -1000.0)).y(), 0.0, 0.0001);
}

TEST(PointToLine2dDistances, LineDistancesRotated)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // origin diagonal
    const vector2 point1(100.0, 100.0);
    const vector2 normal1(1.0, 1.0); // pointing right at 45 degrees

    utils::Line<2> line1(point1, normal1);

    ASSERT_NEAR(line1.distance(vector2(0.0, 0.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(0.0, 0.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(1000.0, 0.0)).x(), 500.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(1000.0, 0.0)).y(), -500.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(-1000.0, 0.0)).x(), -500.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(-1000.0, 0.0)).y(), 500.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(0.0, 1000.0)).x(), -500.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(0.0, 1000.0)).y(), 500.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(0.0, -1000.0)).x(), 500.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(0.0, -1000.0)).y(), -500.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(1000.0, -1000.0)).x(), 1000.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(1000.0, -1000.0)).y(), -1000.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(1000.0, 1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(1000.0, 1000.0)).y(), 0.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector2(-1000.0, -1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector2(-1000.0, -1000.0)).y(), 0.0, 0.0001);
}

TEST(PointToLine3Distances, LineDistancesRotated)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // origin diagonal
    const vector3 point1(0.0, 0.0, 0.0);
    const vector3 normal1(1.0, 1.0, 1.0); // pointing right at 45 degrees

    utils::Line<3> line1(point1, normal1);

    ASSERT_NEAR(line1.distance(vector3(0.0, 0.0, 0.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, 0.0, 0.0)).y(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, 0.0, 0.0)).z(), 0.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector3(1000.0, 0.0, 0.0)).x(), 666.66666, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(1000.0, 0.0, 0.0)).y(), -333.33333, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(1000.0, 0.0, 0.0)).z(), -333.33333, 0.0001);

    ASSERT_NEAR(line1.distance(vector3(-1000.0, 0.0, 0.0)).x(), -666.66666, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(-1000.0, 0.0, 0.0)).y(), 333.33333, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(-1000.0, 0.0, 0.0)).z(), 333.33333, 0.0001);

    ASSERT_NEAR(line1.distance(vector3(0.0, 1000.0, 0.0)).x(), -333.33333, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, 1000.0, 0.0)).y(), 666.66666, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, 1000.0, 0.0)).z(), -333.33333, 0.0001);

    ASSERT_NEAR(line1.distance(vector3(0.0, -1000.0, 0.0)).x(), 333.33333, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, -1000.0, 0.0)).y(), -666.66666, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, -1000.0, 0.0)).z(), 333.33333, 0.0001);

    ASSERT_NEAR(line1.distance(vector3(0.0, 0.0, 1000.0)).x(), -333.33333, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, 0.0, 1000.0)).y(), -333.33333, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, 0.0, 1000.0)).z(), 666.66666, 0.0001);

    ASSERT_NEAR(line1.distance(vector3(0.0, 0.0, -1000.0)).x(), 333.33333, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, 0.0, -1000.0)).y(), 333.33333, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, 0.0, -1000.0)).z(), -666.66666, 0.0001);

    ASSERT_NEAR(line1.distance(vector3(0.0, 1000.0, 0.0)).x(), -333.33333, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, 1000.0, 0.0)).y(), 666.66666, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, 1000.0, 0.0)).z(), -333.33333, 0.0001);

    ASSERT_NEAR(line1.distance(vector3(0.0, -1000.0, 0.0)).x(), 333.33333, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, -1000.0, 0.0)).y(), -666.66666, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(0.0, -1000.0, 0.0)).z(), 333.33333, 0.0001);

    ASSERT_NEAR(line1.distance(vector3(1000.0, -1000.0, 0.0)).x(), 1000.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(1000.0, -1000.0, 0.0)).y(), -1000.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(1000.0, -1000.0, 0.0)).z(), 0.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector3(1000.0, 1000.0, 1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(1000.0, 1000.0, 1000.0)).y(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(1000.0, 1000.0, 1000.0)).z(), 0.0, 0.0001);

    ASSERT_NEAR(line1.distance(vector3(-1000.0, -1000.0, -1000.0)).x(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(-1000.0, -1000.0, -1000.0)).y(), 0.0, 0.0001);
    ASSERT_NEAR(line1.distance(vector3(-1000.0, -1000.0, -1000.0)).z(), 0.0, 0.0001);
}

/**
 *      Test the line to line distance function
 */

TEST(LineToLineDistances, LineDistancesAtZero)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // origin forward
    const vector3 point1(0.0, 0.0, 0.0);
    const vector3 normal1(1.0, 0.0, 0.0);

    // test forward and backward normal
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, normal1, point1, normal1).norm(), 0.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, normal1, point1, -normal1).norm(), 0.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, -normal1, point1, -normal1).norm(), 0.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, -normal1, point1, normal1).norm(), 0.0, 0.0001);

    // point further on x
    const vector3 point2(1000.0, 0.0, 0.0);
    const vector3 normal2(1.0, 0.0, 0.0);
    // test forward and backward normal, no difference
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, normal1, point2, normal2).norm(), 0.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, normal1, point2, -normal2).norm(), 0.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, -normal1, point2, -normal2).norm(), 0.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, -normal1, point2, normal2).norm(), 0.0, 0.0001);

    // point looking sideway
    const vector3 point3(1000.0, 0.0, 0.0);
    const vector3 normal3(0.0, 1.0, 0.0);
    // test forward and backward normal, no difference
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, normal1, point3, normal3).norm(), 0.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, normal1, point3, -normal3).norm(), 0.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, -normal1, point3, -normal3).norm(), 0.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, -normal1, point3, normal3).norm(), 0.0, 0.0001);

    // point looking down
    const vector3 point4(1000.0, 0.0, 0.0);
    const vector3 normal4(0.0, 0.0, 1.0);
    // test forward and backward normal, no difference
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, normal1, point4, normal4).norm(), 0.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, normal1, point4, -normal4).norm(), 0.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, -normal1, point4, -normal4).norm(), 0.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, -normal1, point4, normal4).norm(), 0.0, 0.0001);
}

TEST(LineToLineDistances, LineDistances)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    // origin forward
    const vector3 point1(0.0, 0.0, 0.0);
    const vector3 normal1(1.0, 0.0, 0.0);

    // high point, looking left
    const vector3 point2(0.0, 0.0, 1000.0);
    const vector3 normal2(0.0, 1.0, 0.0);

    // test forward and backward normal
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, normal1, point2, normal2).norm(), 1000.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, normal1, point2, -normal2).norm(), 1000.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, -normal1, point2, -normal2).norm(), 1000.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, -normal1, point2, normal2).norm(), 1000.0, 0.0001);

    // shifted parralel point
    const vector3 point3(0.0, 0.0, 1000.0);
    const vector3 normal3(1.0, 0.0, 0.0);

    // test forward and backward normal
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, normal1, point3, normal3).norm(), 1000.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, normal1, point3, -normal3).norm(), 1000.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, -normal1, point3, -normal3).norm(), 1000.0, 0.0001);
    ASSERT_NEAR(utils::signed_line_distance<3>(point1, -normal1, point3, normal3).norm(), 1000.0, 0.0001);
}

/**
 * Changes the base computations
 */

TEST(BasisChange, CartesianToSphericalZero)
{
    Cartesian c(0, 0, 0);

    ASSERT_NEAR(c.x, 0.0, 1e-10);
    ASSERT_NEAR(c.y, 0.0, 1e-10);
    ASSERT_NEAR(c.z, 0.0, 1e-10);

    const auto& s = Spherical::from(c);
    ASSERT_NEAR(s.p, 0.0, 1e-10);
    ASSERT_NEAR(s.polar_rad, 0.0, 1e-10);
    ASSERT_NEAR(s.azimuth_rad, 0.0, 1e-10);
}

TEST(BasisChange, CartesianToSphericalUnitX)
{
    Cartesian c(1.0, 0.0, 0.0);

    ASSERT_NEAR(c.x, 1.0, 1e-10);
    ASSERT_NEAR(c.y, 0.0, 1e-10);
    ASSERT_NEAR(c.z, 0.0, 1e-10);

    const auto& s = Spherical::from(c);
    ASSERT_NEAR(s.p, 1.0, 1e-10);
    ASSERT_NEAR(s.polar_rad, M_PI / 2.0, 1e-10);
    ASSERT_NEAR(s.azimuth_rad, 0.0, 1e-10);

    // inverse
    Cartesian c2(-1.0, 0.0, 0.0);
    ASSERT_NEAR(c2.x, -1.0, 1e-10);
    ASSERT_NEAR(c2.y, 0.0, 1e-10);
    ASSERT_NEAR(c2.z, 0.0, 1e-10);

    const auto& s2 = Spherical::from(c2);
    ASSERT_NEAR(s2.p, 1.0, 1e-10);
    ASSERT_NEAR(s2.polar_rad, M_PI / 2.0, 1e-10);
    ASSERT_NEAR(s2.azimuth_rad, M_PI, 1e-10);
}

TEST(BasisChange, CartesianToSphericalUnitY)
{
    Cartesian c(0.0, 1.0, 0.0);

    ASSERT_NEAR(c.x, 0.0, 1e-10);
    ASSERT_NEAR(c.y, 1.0, 1e-10);
    ASSERT_NEAR(c.z, 0.0, 1e-10);

    const auto& s = Spherical::from(c);
    ASSERT_NEAR(s.p, 1.0, 1e-10);
    ASSERT_NEAR(s.polar_rad, M_PI / 2.0, 1e-10);
    ASSERT_NEAR(s.azimuth_rad, M_PI / 2.0, 1e-10);

    Cartesian c2(0.0, -1.0, 0.0);

    ASSERT_NEAR(c2.x, 0.0, 1e-10);
    ASSERT_NEAR(c2.y, -1.0, 1e-10);
    ASSERT_NEAR(c2.z, 0.0, 1e-10);

    const auto& s2 = Spherical::from(c2);
    ASSERT_NEAR(s2.p, 1.0, 1e-10);
    ASSERT_NEAR(s2.polar_rad, M_PI / 2.0, 1e-10);
    ASSERT_NEAR(s2.azimuth_rad, -M_PI / 2.0, 1e-10);
}

TEST(BasisChange, CartesianToSphericalUnitZ)
{
    Cartesian c(0.0, 0.0, 1.0);

    ASSERT_NEAR(c.x, 0.0, 1e-10);
    ASSERT_NEAR(c.y, 0.0, 1e-10);
    ASSERT_NEAR(c.z, 1.0, 1e-10);

    const auto& s = Spherical::from(c);
    ASSERT_NEAR(s.p, 1.0, 1e-10);
    ASSERT_NEAR(s.polar_rad, 0.0, 1e-10);
    ASSERT_NEAR(s.azimuth_rad, 0.0, 1e-10);

    Cartesian c2(0.0, 0.0, -1.0);

    ASSERT_NEAR(c2.x, 0.0, 1e-10);
    ASSERT_NEAR(c2.y, 0.0, 1e-10);
    ASSERT_NEAR(c2.z, -1.0, 1e-10);

    const auto& s2 = Spherical::from(c2);
    ASSERT_NEAR(s2.p, 1.0, 1e-10);
    ASSERT_NEAR(s2.polar_rad, M_PI, 1e-10);
    ASSERT_NEAR(s2.azimuth_rad, 0.0, 1e-10);
}

TEST(BasisChange, SphericalToCartesianZero)
{
    Spherical s(0, 0, 0);

    ASSERT_NEAR(s.p, 0.0, 1e-10);
    ASSERT_NEAR(s.polar_rad, 0.0, 1e-10);
    ASSERT_NEAR(s.azimuth_rad, 0.0, 1e-10);

    const auto& c = Cartesian::from(s);
    ASSERT_NEAR(c.x, 0.0, 1e-10);
    ASSERT_NEAR(c.y, 0.0, 1e-10);
    ASSERT_NEAR(c.z, 0.0, 1e-10);
}

TEST(BasisChange, SphericalToCartesianUnitX)
{
    Spherical s(1.0, M_PI / 2.0, 0.0);

    ASSERT_NEAR(s.p, 1.0, 1e-10);
    ASSERT_NEAR(s.polar_rad, M_PI / 2.0, 1e-10);
    ASSERT_NEAR(s.azimuth_rad, 0.0, 1e-10);

    const auto& c = Cartesian::from(s);
    ASSERT_NEAR(c.x, 1.0, 1e-10);
    ASSERT_NEAR(c.y, 0.0, 1e-10);
    ASSERT_NEAR(c.z, 0.0, 1e-10);

    // inverse
    Spherical s2(1.0, M_PI / 2.0, M_PI);

    ASSERT_NEAR(s2.p, 1.0, 1e-10);
    ASSERT_NEAR(s2.polar_rad, M_PI / 2.0, 1e-10);
    ASSERT_NEAR(s2.azimuth_rad, M_PI, 1e-10);

    const auto& c2 = Cartesian::from(s2);
    ASSERT_NEAR(c2.x, -1.0, 1e-10);
    ASSERT_NEAR(c2.y, 0.0, 1e-10);
    ASSERT_NEAR(c2.z, 0.0, 1e-10);
}

TEST(BasisChange, SphericalToCartesianUnitY)
{
    Spherical s(1.0, M_PI / 2.0, M_PI / 2.0);

    ASSERT_NEAR(s.p, 1.0, 1e-10);
    ASSERT_NEAR(s.polar_rad, M_PI / 2.0, 1e-10);
    ASSERT_NEAR(s.azimuth_rad, M_PI / 2.0, 1e-10);

    const auto& c = Cartesian::from(s);
    ASSERT_NEAR(c.x, 0.0, 1e-10);
    ASSERT_NEAR(c.y, 1.0, 1e-10);
    ASSERT_NEAR(c.z, 0.0, 1e-10);

    // inverse
    Spherical s2(1.0, M_PI / 2.0, -M_PI / 2.0);

    ASSERT_NEAR(s2.p, 1.0, 1e-10);
    ASSERT_NEAR(s2.polar_rad, M_PI / 2.0, 1e-10);
    ASSERT_NEAR(s2.azimuth_rad, -M_PI / 2.0, 1e-10);

    const auto& c2 = Cartesian::from(s2);
    ASSERT_NEAR(c2.x, 0.0, 1e-10);
    ASSERT_NEAR(c2.y, -1.0, 1e-10);
    ASSERT_NEAR(c2.z, 0.0, 1e-10);
}

TEST(BasisChange, SphericalToCartesianUnitZ)
{
    Spherical s(1.0, 0.0, 0.0);

    ASSERT_NEAR(s.p, 1.0, 1e-10);
    ASSERT_NEAR(s.polar_rad, 0.0, 1e-10);
    ASSERT_NEAR(s.azimuth_rad, 0.0, 1e-10);

    const auto& c = Cartesian::from(s);
    ASSERT_NEAR(c.x, 0.0, 1e-10);
    ASSERT_NEAR(c.y, 0.0, 1e-10);
    ASSERT_NEAR(c.z, 1.0, 1e-10);

    // inverse
    Spherical s2(1.0, M_PI, 0.0);

    ASSERT_NEAR(s2.p, 1.0, 1e-10);
    ASSERT_NEAR(s2.polar_rad, M_PI, 1e-10);
    ASSERT_NEAR(s2.azimuth_rad, 0.0, 1e-10);

    const auto& c2 = Cartesian::from(s2);
    ASSERT_NEAR(c2.x, 0.0, 1e-10);
    ASSERT_NEAR(c2.y, 0.0, 1e-10);
    ASSERT_NEAR(c2.z, -1.0, 1e-10);
}

} // namespace rgbd_slam
