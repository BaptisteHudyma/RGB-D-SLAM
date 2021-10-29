#include <gtest/gtest.h>

#include "pose_optimization/PoseOptimization.hpp"
#include "Pose.hpp"
#include "utils.hpp"
#include "parameters.hpp"

namespace rgbd_slam {


    // CM
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
        const matrix34& W2CtransformationMatrix = utils::compute_world_to_camera_transform(endPose.get_orientation_quaternion(), endPose.get_position());

        match_point_container matchedPoints;
        for (const double* array : MAP_POINTS)
        {
            // world coordinates
            const vector3 worldPointStart(array[0] * 100, array[1] * 100, array[2] * 100);
            //
            const vector2& transformedPoint = utils::world_to_screen_coordinates(worldPointStart, W2CtransformationMatrix);
            // screen coordinates
            const vector3 screenPointEnd(transformedPoint.x(), transformedPoint.y(), worldPointStart.z());

            const point_pair matched(screenPointEnd, worldPointStart);
            matchedPoints.push_back(matched);
        }
        return matchedPoints;
    }

    void run_test_optimization(const match_point_container& matchedPoints, const utils::Pose& trueEndPose, const utils::Pose& initialPoseGuess)
    {
        // Compute end pose
        const utils::Pose endPose = pose_optimization::Pose_Optimization::compute_optimized_pose(initialPoseGuess, matchedPoints);

        const double approxPositionError = 0.05;  // cm
        const double approxRotationError = 1.0 * EulerToRadian;   // 1 degree
        EXPECT_NEAR(trueEndPose.get_position().x(), endPose.get_position().x(), approxPositionError);
        EXPECT_NEAR(trueEndPose.get_position().y(), endPose.get_position().y(), approxPositionError);
        EXPECT_NEAR(trueEndPose.get_position().z(), endPose.get_position().z(), approxPositionError);

        const EulerAngles trueEndEulerAngles = utils::get_euler_angles_from_quaternion(trueEndPose.get_orientation_quaternion());
        const EulerAngles endEulerAngles = utils::get_euler_angles_from_quaternion(endPose.get_orientation_quaternion());

        EXPECT_NEAR(trueEndEulerAngles.yaw, endEulerAngles.yaw, approxRotationError);
        EXPECT_NEAR(trueEndEulerAngles.pitch, endEulerAngles.pitch, approxRotationError);
        EXPECT_NEAR(trueEndEulerAngles.roll, endEulerAngles.roll, approxRotationError);
    }


    /*
     * Run a test with a no movements, with a perfect initial guess, in perfect environnement (no error on points)
     */
    TEST(no_rotation_no_translation, PoseOptimizationAssertions) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // Initial pose
        const vector3 initialPosition(0, 0, 0);
        const EulerAngles initialEulerAngles(0, 0, 0);
        const quaternion initialQuaternion(utils::get_quaternion_from_euler_angles(initialEulerAngles));
        const utils::Pose initialPose(initialPosition, initialQuaternion);

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(0, 0, 0);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const match_point_container& matchedPoints = get_matched_points(trueEndPose, 0);


        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(0, 0, 0);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, trueEndPose, initialPoseGuess);

    }

    /*
     * Run a test with a perfect initial guess, in perfect environnement (no error on points)
     */
    TEST(perfect_guess, PoseOptimizationAssertions) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // Initial pose
        const vector3 initialPosition(0, 0, 0);
        const EulerAngles initialEulerAngles(0, 0, 0);
        const quaternion initialQuaternion(utils::get_quaternion_from_euler_angles(initialEulerAngles));
        const utils::Pose initialPose(initialPosition, initialQuaternion);

        // True End pose
        const vector3 truePosition(1, 1, 1);
        const EulerAngles trueEulerAngles(1, 1, 1);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const match_point_container& matchedPoints = get_matched_points(trueEndPose, 0);


        // Estimated pose base
        const vector3 initialPositionGuess(1, 1, 1);
        const EulerAngles initialEulerAnglesGuess(1, 1, 1);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, trueEndPose, initialPoseGuess);
    }


    /**
     *          TRANSLATION TESTS
     */

    /*
     * Run a test with a very good initial guess, in perfect environnement (no error on points)
     */
    TEST(translation_good_guess, PoseOptimizationAssertions) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // Initial pose
        const vector3 initialPosition(0, 0, 0);
        const EulerAngles initialEulerAngles(0, 0, 0);
        const quaternion initialQuaternion(utils::get_quaternion_from_euler_angles(initialEulerAngles));
        const utils::Pose initialPose(initialPosition, initialQuaternion);

        // True End pose
        const vector3 truePosition(1, 1, 1);
        const EulerAngles trueEulerAngles(0, 0, 0);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const match_point_container& matchedPoints = get_matched_points(trueEndPose, 0);


        // Estimated pose base
        const vector3 initialPositionGuess(0.9, 0.9, 0.9);
        const EulerAngles initialEulerAnglesGuess(0, 0, 0);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, trueEndPose, initialPoseGuess);
    }

    /*
     * Run a test with a medium initial guess, in perfect environnement (no error on points)
     */
    TEST(translation_medium_guess, PoseOptimizationAssertions) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // Initial pose
        const vector3 initialPosition(0, 0, 0);
        const EulerAngles initialEulerAngles(0, 0, 0);
        const quaternion initialQuaternion(utils::get_quaternion_from_euler_angles(initialEulerAngles));
        const utils::Pose initialPose(initialPosition, initialQuaternion);

        // True End pose
        const vector3 truePosition(1, 1, 1);
        const EulerAngles trueEulerAngles(0, 0, 0);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const match_point_container& matchedPoints = get_matched_points(trueEndPose, 0);


        // Estimated pose base
        const vector3 initialPositionGuess(0.5, 0.5, 0.5);
        const EulerAngles initialEulerAnglesGuess(0, 0, 0);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, trueEndPose, initialPoseGuess);
    }

    /*
     * Run a test with a bad initial guess, in perfect environnement (no error on points)
     */
    TEST(translation_bad_guess, PoseOptimizationAssertions) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // Initial pose
        const vector3 initialPosition(0, 0, 0);
        const EulerAngles initialEulerAngles(0, 0, 0);
        const quaternion initialQuaternion(utils::get_quaternion_from_euler_angles(initialEulerAngles));
        const utils::Pose initialPose(initialPosition, initialQuaternion);

        // True End pose
        const vector3 truePosition(1, 1, 1);
        const EulerAngles trueEulerAngles(0, 0, 0);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const match_point_container& matchedPoints = get_matched_points(trueEndPose, 0);


        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(0, 0, 0);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, trueEndPose, initialPoseGuess);
    }


    /**
     *          ROTATION TESTS
     */

    /*
     * Run a test with a good initial guess (5 degrees apart), in perfect environnement, with a rotation (no error on points)
     */
    TEST(rotation_yaw_good_guess, PoseOptimizationAssertions) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // Initial pose
        const vector3 initialPosition(0, 0, 0);
        const EulerAngles initialEulerAngles(0, 0, 0);
        const quaternion initialQuaternion(utils::get_quaternion_from_euler_angles(initialEulerAngles));
        const utils::Pose initialPose(initialPosition, initialQuaternion);

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(1.4, 0, 0);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const match_point_container& matchedPoints = get_matched_points(trueEndPose, 0);


        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(1.3, 0, 0);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        std::cout << trueQuaternion << "  |   " << initialQuaternionGuess << std::endl;

        run_test_optimization(matchedPoints, trueEndPose, initialPoseGuess);
    }

    TEST(rotation_pitch_good_guess, PoseOptimizationAssertions) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // Initial pose
        const vector3 initialPosition(0, 0, 0);
        const EulerAngles initialEulerAngles(0, 0, 0);
        const quaternion initialQuaternion(utils::get_quaternion_from_euler_angles(initialEulerAngles));
        const utils::Pose initialPose(initialPosition, initialQuaternion);

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(0, 1.4, 0);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const match_point_container& matchedPoints = get_matched_points(trueEndPose, 0);


        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(0, 1.3, 0);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        std::cout << trueQuaternion << "  |   " << initialQuaternionGuess << std::endl;

        run_test_optimization(matchedPoints, trueEndPose, initialPoseGuess);
    }

    TEST(rotation_roll_good_guess, PoseOptimizationAssertions) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // Initial pose
        const vector3 initialPosition(0, 0, 0);
        const EulerAngles initialEulerAngles(0, 0, 0);
        const quaternion initialQuaternion(utils::get_quaternion_from_euler_angles(initialEulerAngles));
        const utils::Pose initialPose(initialPosition, initialQuaternion);

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(0, 0, 1.4);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const match_point_container& matchedPoints = get_matched_points(trueEndPose, 0);


        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(0, 0, 1.3);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        std::cout << trueQuaternion << "  |   " << initialQuaternionGuess << std::endl;

        run_test_optimization(matchedPoints, trueEndPose, initialPoseGuess);
    }

    TEST(rotation_good_guess, PoseOptimizationAssertions) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // Initial pose
        const vector3 initialPosition(0, 0, 0);
        const EulerAngles initialEulerAngles(0, 0, 0);
        const quaternion initialQuaternion(utils::get_quaternion_from_euler_angles(initialEulerAngles));
        const utils::Pose initialPose(initialPosition, initialQuaternion);

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(1.4, 1.4, 1.4);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const match_point_container& matchedPoints = get_matched_points(trueEndPose, 0);


        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(1.3, 1.3, 1.3);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        std::cout << trueQuaternion << "  |   " << initialQuaternionGuess << std::endl;

        run_test_optimization(matchedPoints, trueEndPose, initialPoseGuess);
    }

    /*
     * Run a test with a medium initial guess (20 degrees apart), in perfect environnement, with a rotation (no error on points)
     */
    TEST(rotation_yaw_medium_guess, PoseOptimizationAssertions) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // Initial pose
        const vector3 initialPosition(0, 0, 0);
        const EulerAngles initialEulerAngles(0, 0, 0);
        const quaternion initialQuaternion(utils::get_quaternion_from_euler_angles(initialEulerAngles));
        const utils::Pose initialPose(initialPosition, initialQuaternion);

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(1.4, 0, 0);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const match_point_container& matchedPoints = get_matched_points(trueEndPose, 0);


        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(0.7, 0, 0);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, trueEndPose, initialPoseGuess);
    }

    TEST(rotation_pitch_medium_guess, PoseOptimizationAssertions) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // Initial pose
        const vector3 initialPosition(0, 0, 0);
        const EulerAngles initialEulerAngles(0, 0, 0);
        const quaternion initialQuaternion(utils::get_quaternion_from_euler_angles(initialEulerAngles));
        const utils::Pose initialPose(initialPosition, initialQuaternion);

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(0, 1.4, 0);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const match_point_container& matchedPoints = get_matched_points(trueEndPose, 0);


        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(0, 0.7, 0);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, trueEndPose, initialPoseGuess);
    }

    TEST(rotation_roll_medium_guess, PoseOptimizationAssertions) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // Initial pose
        const vector3 initialPosition(0, 0, 0);
        const EulerAngles initialEulerAngles(0, 0, 0);
        const quaternion initialQuaternion(utils::get_quaternion_from_euler_angles(initialEulerAngles));
        const utils::Pose initialPose(initialPosition, initialQuaternion);

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(0, 0, 1.4);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const match_point_container& matchedPoints = get_matched_points(trueEndPose, 0);


        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(0, 0, 0.7);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, trueEndPose, initialPoseGuess);
    }

    TEST(rotation_medium_guess, PoseOptimizationAssertions) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // Initial pose
        const vector3 initialPosition(0, 0, 0);
        const EulerAngles initialEulerAngles(0, 0, 0);
        const quaternion initialQuaternion(utils::get_quaternion_from_euler_angles(initialEulerAngles));
        const utils::Pose initialPose(initialPosition, initialQuaternion);

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(1.4, 1.4, 1.4);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const match_point_container& matchedPoints = get_matched_points(trueEndPose, 0);


        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(0.7, 0.7, 0.7);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, trueEndPose, initialPoseGuess);
    }

    /*
     * Run a test with bad initial guess (90% apart), in perfect environnement, with a rotation (no error on points)
     */
    TEST(rotation_yaw_bad_guess, PoseOptimizationAssertions) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // Initial pose
        const vector3 initialPosition(0, 0, 0);
        const EulerAngles initialEulerAngles(0, 0, 0);
        const quaternion initialQuaternion(utils::get_quaternion_from_euler_angles(initialEulerAngles));
        const utils::Pose initialPose(initialPosition, initialQuaternion);

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(1.4, 0, 0);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const match_point_container& matchedPoints = get_matched_points(trueEndPose, 0);


        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(0, 0, 0);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, trueEndPose, initialPoseGuess);
    }

    TEST(rotation_pitch_bad_guess, PoseOptimizationAssertions) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // Initial pose
        const vector3 initialPosition(0, 0, 0);
        const EulerAngles initialEulerAngles(0, 0, 0);
        const quaternion initialQuaternion(utils::get_quaternion_from_euler_angles(initialEulerAngles));
        const utils::Pose initialPose(initialPosition, initialQuaternion);

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(0, 1.4, 0);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const match_point_container& matchedPoints = get_matched_points(trueEndPose, 0);


        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(0, 0, 0);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, trueEndPose, initialPoseGuess);
    }

    TEST(rotation_roll_bad_guess, PoseOptimizationAssertions) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // Initial pose
        const vector3 initialPosition(0, 0, 0);
        const EulerAngles initialEulerAngles(0, 0, 0);
        const quaternion initialQuaternion(utils::get_quaternion_from_euler_angles(initialEulerAngles));
        const utils::Pose initialPose(initialPosition, initialQuaternion);

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(0, 0, 1.4);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const match_point_container& matchedPoints = get_matched_points(trueEndPose, 0);


        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(0, 0, 0);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, trueEndPose, initialPoseGuess);
    }

    TEST(rotation_bad_guess, PoseOptimizationAssertions) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // Initial pose
        const vector3 initialPosition(0, 0, 0);
        const EulerAngles initialEulerAngles(0, 0, 0);
        const quaternion initialQuaternion(utils::get_quaternion_from_euler_angles(initialEulerAngles));
        const utils::Pose initialPose(initialPosition, initialQuaternion);

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(1.4, 1.4, 1.4);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const match_point_container& matchedPoints = get_matched_points(trueEndPose, 0);


        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(0, 0, 0);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, trueEndPose, initialPoseGuess);
    }


    /**
     *          ROTATION & TRANSLATION TESTS
     */

    /*
     * Run a test with a good initial guess (5 degrees apart), in perfect environnement, with a rotation (no error on points)
     */
    TEST(rotation_translation_good_guess, PoseOptimizationAssertions) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // Initial pose
        const vector3 initialPosition(0, 0, 0);
        const EulerAngles initialEulerAngles(0, 0, 0);
        const quaternion initialQuaternion(utils::get_quaternion_from_euler_angles(initialEulerAngles));
        const utils::Pose initialPose(initialPosition, initialQuaternion);

        // True End pose
        const vector3 truePosition(1, 1, 1);
        const EulerAngles trueEulerAngles(1.4, 1.4, 1.4);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const match_point_container& matchedPoints = get_matched_points(trueEndPose, 0);


        // Estimated pose base
        const vector3 initialPositionGuess(0.9, 0.9, 0.9);
        const EulerAngles initialEulerAnglesGuess(1.2, 1.2, 1.2);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, trueEndPose, initialPoseGuess);
    }

    /*
     * Run a test with a medium initial guess (20 degrees apart), in perfect environnement, with a rotation (no error on points)
     */
    TEST(rotation_translation_medium_guess, PoseOptimizationAssertions) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // Initial pose
        const vector3 initialPosition(0, 0, 0);
        const EulerAngles initialEulerAngles(0, 0, 0);
        const quaternion initialQuaternion(utils::get_quaternion_from_euler_angles(initialEulerAngles));
        const utils::Pose initialPose(initialPosition, initialQuaternion);

        // True End pose
        const vector3 truePosition(1, 1, 1);
        const EulerAngles trueEulerAngles(1.4, 1.4, 1.4);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const match_point_container& matchedPoints = get_matched_points(trueEndPose, 0);


        // Estimated pose base
        const vector3 initialPositionGuess(0.5, 0.5, 0.5);
        const EulerAngles initialEulerAnglesGuess(0.7, 0.7, 0.7);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, trueEndPose, initialPoseGuess);
    }

    /*
     * Run a test with bad initial guess (90% appart), in perfect environnement, with a rotation (no error on points)
     */
    TEST(rotation_translation_bad_guess, PoseOptimizationAssertions) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // Initial pose
        const vector3 initialPosition(0, 0, 0);
        const EulerAngles initialEulerAngles(0, 0, 0);
        const quaternion initialQuaternion(utils::get_quaternion_from_euler_angles(initialEulerAngles));
        const utils::Pose initialPose(initialPosition, initialQuaternion);

        // True End pose
        const vector3 truePosition(1, 1, 1);
        const EulerAngles trueEulerAngles(1.4, 1.4, 1.4);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const match_point_container& matchedPoints = get_matched_points(trueEndPose, 0);


        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(0, 0, 0);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, trueEndPose, initialPoseGuess);
    }


}
