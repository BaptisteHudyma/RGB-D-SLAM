#include <gtest/gtest.h>

#include "pose_optimization/PoseOptimization.hpp"
#include "Pose.hpp"
#include "utils.hpp"
#include "parameters.hpp"

namespace rgbd_slam {

    const unsigned int NUMBER_OF_POINTS_IN_CUBE = 4 * 4 * 4;
    const double CUBE_SIDE_SIZE = 3;   // Meters
    const double CUBE_START_X = 10; 
    const double CUBE_START_Y = 10;
    const double CUBE_START_Z = 10;


    struct Point {
        double x;
        double y;
        double z;
    };
    typedef std::vector<Point> point_container;
    const point_container get_cube_points(const unsigned int numberOfPoints)
    {
        const unsigned int numberOfPointsByLine = static_cast<unsigned int>(pow(static_cast<double>(numberOfPoints), 1.0 / 3.0));
        const double numberOfPointsByLineDouble = CUBE_SIDE_SIZE / static_cast<double>(numberOfPointsByLine - 1);

        point_container pointContainer;
        pointContainer.reserve(pow(numberOfPointsByLine, 3));

        for(unsigned int cubePlaneIndex = 0; cubePlaneIndex < numberOfPointsByLine; ++cubePlaneIndex)
        {
            for(unsigned int cubeLineIndex = 0; cubeLineIndex < numberOfPointsByLine; ++cubeLineIndex)
            {
                for(unsigned int cubeColumnIndex = 0; cubeColumnIndex < numberOfPointsByLine; ++cubeColumnIndex)
                {
                    Point cubePoint;
                    cubePoint.x = CUBE_START_X + cubePlaneIndex  * numberOfPointsByLineDouble;
                    cubePoint.y = CUBE_START_Y + cubeLineIndex   * numberOfPointsByLineDouble;
                    cubePoint.z = CUBE_START_Z + cubeColumnIndex * numberOfPointsByLineDouble;

                    pointContainer.push_back(cubePoint);
                }
            }
        }
        return pointContainer;
    }

    const match_point_container get_matched_points(const utils::Pose& endPose, const double error)
    {
        const matrix34& W2CtransformationMatrix = utils::compute_world_to_camera_transform(endPose.get_orientation_quaternion(), endPose.get_position());

        match_point_container matchedPoints;
        for (const Point point : get_cube_points(NUMBER_OF_POINTS_IN_CUBE))
        {
            // world coordinates
            const vector3 worldPointStart(point.x, point.y, point.z);
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
        const double approxRotationError = 0.2;   // 0.2 degree
        EXPECT_NEAR(trueEndPose.get_position().x(), endPose.get_position().x(), approxPositionError);
        EXPECT_NEAR(trueEndPose.get_position().y(), endPose.get_position().y(), approxPositionError);
        EXPECT_NEAR(trueEndPose.get_position().z(), endPose.get_position().z(), approxPositionError);

        const EulerAngles trueEndEulerAngles = utils::get_euler_angles_from_quaternion(trueEndPose.get_orientation_quaternion());
        const EulerAngles endEulerAngles = utils::get_euler_angles_from_quaternion(endPose.get_orientation_quaternion());

        EXPECT_NEAR(trueEndEulerAngles.yaw, endEulerAngles.yaw, approxRotationError);
        EXPECT_NEAR(trueEndEulerAngles.pitch, endEulerAngles.pitch, approxRotationError);
        EXPECT_NEAR(trueEndEulerAngles.roll, endEulerAngles.roll, approxRotationError);
    }


    /**
     *          ROTATION & TRANSLATION TESTS
     */

    /*
     * Run a test with a no movements, with a perfect initial guess, in perfect environnement (no error on points)
     */
    TEST(PoseOptimizationTests, noRotationNoTranslation) 
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
    TEST(PoseOptimizationTests, perfectGuess) 
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

    /*
     * Run a test with a good initial guess (5 degrees apart), in perfect environnement, with a rotation (no error on points)
     */
    TEST(PoseOptimizationTests, rotationTranslationGoodGuess) 
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
    TEST(PoseOptimizationTests, rotationTranslationMediumGuess) 
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
    TEST(PoseOptimizationTests, rotationTranslationBadGuess) 
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



    /**
     *          TRANSLATION TESTS
     */

    /*
     * Run a test with a very good initial guess, in perfect environnement (no error on points)
     */
    TEST(TranslationOptimizationTests, translationGoodGuess) 
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
    TEST(TranslationOptimizationTests, translationMediumGuess) 
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
    TEST(TranslationOptimizationTests, translationBadGuess) 
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
    TEST(RotationOptimizationTests, rotationYawGoodGuess) 
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

        run_test_optimization(matchedPoints, trueEndPose, initialPoseGuess);
    }

    TEST(RotationOptimizationTests, rotationPitchGoodGuess) 
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

        run_test_optimization(matchedPoints, trueEndPose, initialPoseGuess);
    }

    TEST(RotationOptimizationTests, rotationRollGoodGuess) 
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

        run_test_optimization(matchedPoints, trueEndPose, initialPoseGuess);
    }

    TEST(RotationOptimizationTests, rotationGoodGuess) 
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

        run_test_optimization(matchedPoints, trueEndPose, initialPoseGuess);
    }

    /*
     * Run a test with a medium initial guess (20 degrees apart), in perfect environnement, with a rotation (no error on points)
     */
    TEST(RotationOptimizationTests, rotationYawMediumGuess) 
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

    TEST(RotationOptimizationTests, rotationPitchMedium_guess) 
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

    TEST(RotationOptimizationTests, rotationRollMediumGuess) 
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

    TEST(RotationOptimizationTests, rotationMediumGuess) 
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
    TEST(RotationOptimizationTests, rotationYawBadGuess) 
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

    TEST(RotationOptimizationTests, rotationPitchBadGuess) 
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

    TEST(RotationOptimizationTests, rotationRollBadGuess) 
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

    TEST(RotationOptimizationTests, rotationBadGuess) 
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


}
