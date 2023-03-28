#include <gtest/gtest.h>
#include <iostream>
#include "motion_model.hpp"
#include "pose.hpp"
#include "types.hpp"

namespace rgbd_slam::tracking {

void compare_pose(const utils::Pose& toTest, const utils::Pose& reference, const double delta = 0.001)
{
    EXPECT_NEAR(toTest.get_position().x(), reference.get_position().x(), delta);
    EXPECT_NEAR(toTest.get_position().y(), reference.get_position().y(), delta);
    EXPECT_NEAR(toTest.get_position().z(), reference.get_position().z(), delta);

    EXPECT_NEAR(abs(toTest.get_orientation_quaternion().x()), abs(reference.get_orientation_quaternion().x()), delta);
    EXPECT_NEAR(abs(toTest.get_orientation_quaternion().y()), abs(reference.get_orientation_quaternion().y()), delta);
    EXPECT_NEAR(abs(toTest.get_orientation_quaternion().z()), abs(reference.get_orientation_quaternion().z()), delta);
    EXPECT_NEAR(abs(toTest.get_orientation_quaternion().w()), abs(reference.get_orientation_quaternion().w()), delta);
}

TEST(MotionModelTests, EmptyPoseTest)
{
    Motion_Model mm;
    utils::Pose emptyPose;

    utils::Pose predictedPose;
    for (uint i = 0; i < 10; ++i)
    {
        predictedPose = mm.predict_next_pose(emptyPose);
        mm.update_model(predictedPose);
    }

    compare_pose(predictedPose, emptyPose);
}

TEST(MotionModelTests, ConstantPosePositionUpdateTest)
{
    Motion_Model mm;
    utils::Pose pose(vector3::Random(), quaternion::Identity());

    utils::Pose predictedPose;
    for (uint i = 0; i < 10; ++i)
    {
        predictedPose = mm.predict_next_pose(pose);
        mm.update_model(predictedPose);
    }

    compare_pose(predictedPose, pose);
}

TEST(MotionModelTests, ConstantPoseOrientationTest)
{
    Motion_Model mm;
    utils::Pose pose(vector3::Zero(), quaternion::UnitRandom());

    utils::Pose predictedPose;
    for (uint i = 0; i < 10; ++i)
    {
        predictedPose = mm.predict_next_pose(pose);
        mm.update_model(predictedPose);
    }

    compare_pose(predictedPose, pose);
}

TEST(MotionModelTests, TrackEmptyPoseTest)
{
    Motion_Model mm;
    utils::Pose pose(vector3::Random(), quaternion::UnitRandom());

    // track a pose
    utils::Pose predictedPose;
    for (uint i = 0; i < 10; ++i)
    {
        predictedPose = mm.predict_next_pose(pose);
        mm.update_model(predictedPose);
    }

    // track the empty pose
    utils::Pose emptyPose;
    for (uint i = 0; i < 10; ++i)
    {
        mm.update_model(emptyPose);
    }

    compare_pose(mm.predict_next_pose(emptyPose), emptyPose);
}

TEST(MotionModelTests, DecayingModelTest)
{
    Motion_Model mm;
    utils::Pose pose(vector3::Random(), quaternion::UnitRandom());

    // decay model: give initial prediction and update with
    utils::Pose predictedPose = pose;
    for (uint i = 0; i < 100; ++i)
    {
        predictedPose = mm.predict_next_pose(predictedPose);
        mm.update_model(predictedPose);
    }

    EXPECT_NEAR(mm.get_angular_velocity().x(), 0.0, 0.0001);
    EXPECT_NEAR(mm.get_angular_velocity().y(), 0.0, 0.0001);
    EXPECT_NEAR(mm.get_angular_velocity().z(), 0.0, 0.0001);
    EXPECT_NEAR(mm.get_angular_velocity().w(), 1.0, 0.0001);

    EXPECT_NEAR(mm.get_position_velocity().x(), 0.0, 0.0001);
    EXPECT_NEAR(mm.get_position_velocity().y(), 0.0, 0.0001);
    EXPECT_NEAR(mm.get_position_velocity().z(), 0.0, 0.0001);
}

} // namespace rgbd_slam::tracking