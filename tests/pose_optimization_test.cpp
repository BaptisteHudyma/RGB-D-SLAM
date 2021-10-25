#include <gtest/gtest.h>

#include "pose_optimization/PoseOptimization.hpp"
#include "Pose.hpp"
#include "utils.hpp"
#include "parameters.hpp"

namespace rgbd_slam {

    const double MAP_POINTS[20][3] = {
        {37, 56, 24},   //1
        {42, 86, 20},   //2
        {25, 45, 8},    //3
        {0, 14, 28},    //4
        {37, 49, 11},   //5
        {49, 22, 21},   //6
        {52, 22, 22},   //7
        {82, 0.5, 1},   //8
        {49, 74, 92},   //9
        {21, 22, 21},   //10
        {40, 87, 2},    //11
        {79, 34, 56},   //12
        {01, 0.1, 29},  //13
        {32, 78, 92},  //14
        {10, 178, 18},  //15
        {04, 912, 2},  //16
        {179, 98, 12},  //17
        {31, 12, 4},  //18
        {157, 84, 9},  //19
        {29, 102, 2},  //20
    };

    const match_point_container get_matched_points(const utils::Pose& endPose, const double error)
    {
        const matrix34& transformationMatrix = utils::compute_world_to_camera_transform(endPose.get_orientation_quaternion(), endPose.get_position());

        const double divider = 100.0;
        match_point_container matchedPoints;
        for (const double* array : MAP_POINTS)
        {
            // world coordinates
            const vector3 pointStart(array[0] / divider, array[1] / divider, array[2] / divider);

            const vector2& transformedPoint = utils::world_to_screen_coordinates(pointStart, transformationMatrix);
            // screen coordinates
            const vector3 pointEnd(transformedPoint.x(), transformedPoint.y(), 0);

            const point_pair matched(pointEnd, pointStart);
            matchedPoints.push_back(matched);
            std::cout << pointStart.transpose() << " | " << pointEnd .transpose()<< std::endl;
        }
        return matchedPoints;
    }


    // Check pose optimization
    TEST(NoErrorNoMovePoseOptimization, OptimizationAssertions) {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // Initial pose
        const vector3 initialPosition(0, 0, 0);
        const quaternion initialQuaternion(1, 0, 0, 0);
        const utils::Pose initialPose(initialPosition, initialQuaternion);

        // True End pose
        const vector3 truePosition(0, 0, 0.5);
        const quaternion trueQuaternion(1, 0, 0, 0);
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const match_point_container& matchedPoints = get_matched_points(trueEndPose, 0);


        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0.4);
        const quaternion initialQuaternionGuess(1, 0, 0, 0);
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        // Compute end pose
        const utils::Pose endPose = pose_optimization::Pose_Optimization::compute_optimized_pose(initialPoseGuess, matchedPoints);

        const double approxPositionError = 0.1;  // 10 cm
        const double approxRotationError = 0.1;  //
        EXPECT_NEAR(trueEndPose.get_position().x(), endPose.get_position().x(), approxPositionError);
        EXPECT_NEAR(trueEndPose.get_position().y(), endPose.get_position().y(), approxPositionError);
        EXPECT_NEAR(trueEndPose.get_position().z(), endPose.get_position().z(), approxPositionError);

        EXPECT_NEAR(trueEndPose.get_orientation_quaternion().w(), endPose.get_orientation_quaternion().w(), approxRotationError);
        EXPECT_NEAR(trueEndPose.get_orientation_quaternion().x(), endPose.get_orientation_quaternion().x(), approxRotationError);
        EXPECT_NEAR(trueEndPose.get_orientation_quaternion().y(), endPose.get_orientation_quaternion().y(), approxRotationError);
        EXPECT_NEAR(trueEndPose.get_orientation_quaternion().z(), endPose.get_orientation_quaternion().z(), approxRotationError);
    }

}
