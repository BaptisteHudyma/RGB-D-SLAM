#include <gtest/gtest.h>
#include <random>

#include "parameters.hpp"
#include "pose_optimization/pose_optimization.hpp"
#include "outputs/logger.hpp"

#include "utils/coordinates.hpp"
#include "utils/pose.hpp"
#include "utils/camera_transformation.hpp"
#include "utils/angle_utils.hpp"

namespace rgbd_slam {

    const uint NUMBER_OF_POINTS_IN_CUBE = pow(4, 3);
    const double CUBE_SIDE_SIZE = 20;   // Millimeters
    const double CUBE_START_X = 100;
    const double CUBE_START_Y = 100;
    const double CUBE_START_Z = 100;

    const double END_POSITION = 10;
    const double END_ROTATION_YAW = 45 * EulerToRadian;    // [-180, 180] degrees
    const double END_ROTATION_PITCH = 45 * EulerToRadian;  // [-90, 90] degrees
    const double END_ROTATION_ROLL = 20 * EulerToRadian;   // [-180, 180] degrees

    const double GOOD_GUESS = 0.9;
    const double MEDIUM_GUESS = 0.5;
    const double BAD_GUESS = 0.1;

    // Error associated with each points of the cube
    const double POINTS_ERROR = 5; 

    // set random
    std::random_device randomDevice;
    std::mt19937 randomEngine(randomDevice());

    struct Point {
        double x;
        double y;
        double z;
    };
    typedef std::vector<Point> point_container;
    const point_container get_cube_points(const uint numberOfPoints, const double error)
    {
        assert(error >= 0);
        std::uniform_real_distribution<double> errorDistribution(-error, error);

        const uint numberOfPointsByLine = static_cast<uint>(pow(static_cast<double>(numberOfPoints), 1.0 / 3.0));
        const double numberOfPointsByLineDouble = CUBE_SIDE_SIZE / static_cast<double>(numberOfPointsByLine - 1);

        point_container pointContainer;
        pointContainer.reserve(pow(numberOfPointsByLine, 3));

        for(uint cubePlaneIndex = 0; cubePlaneIndex <= numberOfPointsByLine; ++cubePlaneIndex)
        {
            for(uint cubeLineIndex = 0; cubeLineIndex <= numberOfPointsByLine; ++cubeLineIndex)
            {
                for(uint cubeColumnIndex = 0; cubeColumnIndex <= numberOfPointsByLine; ++cubeColumnIndex)
                {
                    Point cubePoint;
                    cubePoint.x = CUBE_START_X + cubePlaneIndex  * numberOfPointsByLineDouble + errorDistribution(randomEngine);
                    cubePoint.y = CUBE_START_Y + cubeLineIndex   * numberOfPointsByLineDouble + errorDistribution(randomEngine);
                    cubePoint.z = CUBE_START_Z + cubeColumnIndex * numberOfPointsByLineDouble + errorDistribution(randomEngine);

                    pointContainer.push_back(cubePoint);
                }
            }
        }
        return pointContainer;
    }

    const matches_containers::match_point_container get_matched_points(const utils::Pose& endPose, const double error = 0.0)
    {
        assert(error >= 0);
        const worldToCameraMatrix& worldToCamera = utils::compute_world_to_camera_transform(endPose.get_orientation_quaternion(), endPose.get_position());
        uint invalidPointsCounter = 0;

        matches_containers::match_point_container matchedPoints;
        for (const Point point : get_cube_points(NUMBER_OF_POINTS_IN_CUBE, error))
        {
            // world coordinates
            const utils::WorldCoordinate worldPointStart(point.x, point.y, point.z);
            //
            utils::ScreenCoordinate transformedPoint; 
            const bool isScreenCoordinatesValid = worldPointStart.to_screen_coordinates(worldToCamera, transformedPoint);
            if (isScreenCoordinatesValid)
            {
                // Dont care about the map id
                matchedPoints.emplace(matchedPoints.end(), 
                    transformedPoint,   // screenPoint
                    worldPointStart,    // worldPoint
                    0                   // uniq map id
                );
            }
            else
            {
                ++invalidPointsCounter;
            }
        }
        if (invalidPointsCounter != 0)
        {
            outputs::log_error("The chosen transformation is not valid, as some points hand up behind the camera");
        }
        return matchedPoints;
    }

    /**
     * Return the distance between two angles, in radians
     */
    double get_angle_distance(const double angleA, const double angleB)
    {
        const double diff = std::fmod(abs(angleA - angleB), 2 * M_PI);
        return std::min(diff, abs(diff - 2.0 * M_PI));
    }

    void run_test_optimization(const matches_containers::match_point_container& matchedPoints, const matches_containers::match_plane_container& matchedPlanes, const utils::Pose& trueEndPose, const utils::Pose& initialPoseGuess)
    {
        // Compute end pose
        utils::Pose endPose; 
        matches_containers::match_point_container outlierMatchedPoints;
        const bool isPoseValid = pose_optimization::Pose_Optimization::compute_optimized_pose(initialPoseGuess, matchedPoints, matchedPlanes, endPose, outlierMatchedPoints);

        if (not isPoseValid)
            FAIL();

        const double approxPositionError = 1 + POINTS_ERROR;  // mm
        const double approxRotationError = 0.1 * EulerToRadian;   // 0.1 degree
        EXPECT_NEAR(trueEndPose.get_position().x(), endPose.get_position().x(), approxPositionError);
        EXPECT_NEAR(trueEndPose.get_position().y(), endPose.get_position().y(), approxPositionError);
        EXPECT_NEAR(trueEndPose.get_position().z(), endPose.get_position().z(), approxPositionError);

        const EulerAngles trueEndEulerAngles = utils::get_euler_angles_from_quaternion(trueEndPose.get_orientation_quaternion());
        const EulerAngles endEulerAngles = utils::get_euler_angles_from_quaternion(endPose.get_orientation_quaternion());

        EXPECT_LT( get_angle_distance(trueEndEulerAngles.yaw, endEulerAngles.yaw), approxRotationError);
        EXPECT_LT( get_angle_distance(trueEndEulerAngles.pitch, endEulerAngles.pitch), approxRotationError);
        EXPECT_LT( get_angle_distance(trueEndEulerAngles.roll, endEulerAngles.roll), approxRotationError);
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

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(0, 0, 0);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const matches_containers::match_point_container& matchedPoints = get_matched_points(trueEndPose, POINTS_ERROR);
        matches_containers::match_plane_container matchedPlanes;

        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(0, 0, 0);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, matchedPlanes, trueEndPose, initialPoseGuess);

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

        // True End pose
        const vector3 truePosition(END_POSITION, END_POSITION, END_POSITION);
        const EulerAngles trueEulerAngles(END_ROTATION_YAW, END_ROTATION_PITCH, END_ROTATION_ROLL);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const matches_containers::match_point_container& matchedPoints = get_matched_points(trueEndPose, POINTS_ERROR);
        matches_containers::match_plane_container matchedPlanes;

        // Estimated pose base
        const vector3 initialPositionGuess(END_POSITION, END_POSITION, END_POSITION);
        const EulerAngles initialEulerAnglesGuess(END_ROTATION_YAW, END_ROTATION_PITCH, END_ROTATION_ROLL);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, matchedPlanes, trueEndPose, initialPoseGuess);
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

        // True End pose
        const vector3 truePosition(END_POSITION, END_POSITION, END_POSITION);
        const EulerAngles trueEulerAngles(END_ROTATION_YAW, END_ROTATION_PITCH, END_ROTATION_ROLL);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const matches_containers::match_point_container& matchedPoints = get_matched_points(trueEndPose, POINTS_ERROR);
        matches_containers::match_plane_container matchedPlanes;

        // Estimated pose base
        const vector3 initialPositionGuess(END_POSITION * GOOD_GUESS, END_POSITION * GOOD_GUESS, END_POSITION * GOOD_GUESS);
        const EulerAngles initialEulerAnglesGuess(END_ROTATION_YAW * GOOD_GUESS, END_ROTATION_PITCH * GOOD_GUESS, END_ROTATION_ROLL * GOOD_GUESS);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, matchedPlanes, trueEndPose, initialPoseGuess);
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

        // True End pose
        const vector3 truePosition(END_POSITION, END_POSITION, END_POSITION);
        const EulerAngles trueEulerAngles(END_ROTATION_YAW, END_ROTATION_PITCH, END_ROTATION_ROLL);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const matches_containers::match_point_container& matchedPoints = get_matched_points(trueEndPose, POINTS_ERROR);
        matches_containers::match_plane_container matchedPlanes;

        // Estimated pose base
        const vector3 initialPositionGuess(END_POSITION * MEDIUM_GUESS, END_POSITION * MEDIUM_GUESS, END_POSITION * MEDIUM_GUESS);
        const EulerAngles initialEulerAnglesGuess(END_ROTATION_YAW * MEDIUM_GUESS, END_ROTATION_PITCH * MEDIUM_GUESS, END_ROTATION_ROLL * MEDIUM_GUESS);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, matchedPlanes, trueEndPose, initialPoseGuess);
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

        // True End pose
        const vector3 truePosition(END_POSITION, END_POSITION, END_POSITION);
        const EulerAngles trueEulerAngles(END_ROTATION_YAW, END_ROTATION_PITCH, END_ROTATION_ROLL);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const matches_containers::match_point_container& matchedPoints = get_matched_points(trueEndPose, POINTS_ERROR);
        matches_containers::match_plane_container matchedPlanes;

        // Estimated pose base
        const vector3 initialPositionGuess(END_POSITION * BAD_GUESS, END_POSITION * BAD_GUESS, END_POSITION * BAD_GUESS);
        const EulerAngles initialEulerAnglesGuess(END_ROTATION_YAW * BAD_GUESS, END_ROTATION_PITCH * BAD_GUESS, END_ROTATION_ROLL * BAD_GUESS);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, matchedPlanes, trueEndPose, initialPoseGuess);
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

        // True End pose
        const vector3 truePosition(END_POSITION, END_POSITION, END_POSITION);
        const EulerAngles trueEulerAngles(0, 0, 0);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const matches_containers::match_point_container& matchedPoints = get_matched_points(trueEndPose, POINTS_ERROR);
        matches_containers::match_plane_container matchedPlanes;

        // Estimated pose base
        const vector3 initialPositionGuess(END_POSITION * GOOD_GUESS, END_POSITION * GOOD_GUESS, END_POSITION * GOOD_GUESS);
        const EulerAngles initialEulerAnglesGuess(0, 0, 0);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, matchedPlanes, trueEndPose, initialPoseGuess);
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

        // True End pose
        const vector3 truePosition(END_POSITION, END_POSITION, END_POSITION);
        const EulerAngles trueEulerAngles(0, 0, 0);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const matches_containers::match_point_container& matchedPoints = get_matched_points(trueEndPose, POINTS_ERROR);
        matches_containers::match_plane_container matchedPlanes;

        // Estimated pose base
        const vector3 initialPositionGuess(END_POSITION* MEDIUM_GUESS, END_POSITION * MEDIUM_GUESS, END_POSITION * MEDIUM_GUESS);
        const EulerAngles initialEulerAnglesGuess(0, 0, 0);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, matchedPlanes, trueEndPose, initialPoseGuess);
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

        // True End pose
        const vector3 truePosition(END_POSITION, END_POSITION, END_POSITION);
        const EulerAngles trueEulerAngles(0, 0, 0);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const matches_containers::match_point_container& matchedPoints = get_matched_points(trueEndPose, POINTS_ERROR);
        matches_containers::match_plane_container matchedPlanes;

        // Estimated pose base
        const vector3 initialPositionGuess(END_POSITION * BAD_GUESS, END_POSITION * BAD_GUESS, END_POSITION * BAD_GUESS);
        const EulerAngles initialEulerAnglesGuess(0, 0, 0);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, matchedPlanes, trueEndPose, initialPoseGuess);
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

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(END_ROTATION_YAW, 0, 0);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const matches_containers::match_point_container& matchedPoints = get_matched_points(trueEndPose, POINTS_ERROR);
        matches_containers::match_plane_container matchedPlanes;

        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(END_ROTATION_YAW * GOOD_GUESS, 0, 0);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, matchedPlanes, trueEndPose, initialPoseGuess);
    }

    TEST(RotationOptimizationTests, rotationPitchGoodGuess) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(0, END_ROTATION_PITCH, 0);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const matches_containers::match_point_container& matchedPoints = get_matched_points(trueEndPose, POINTS_ERROR);
        matches_containers::match_plane_container matchedPlanes;

        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(0, END_ROTATION_PITCH * GOOD_GUESS, 0);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, matchedPlanes, trueEndPose, initialPoseGuess);
    }

    TEST(RotationOptimizationTests, rotationRollGoodGuess) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(0, 0, END_ROTATION_ROLL);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const matches_containers::match_point_container& matchedPoints = get_matched_points(trueEndPose, POINTS_ERROR);
        matches_containers::match_plane_container matchedPlanes;

        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(0, 0, END_ROTATION_ROLL * GOOD_GUESS);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, matchedPlanes, trueEndPose, initialPoseGuess);
    }

    TEST(RotationOptimizationTests, rotationGoodGuess) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(END_ROTATION_YAW, END_ROTATION_PITCH, END_ROTATION_ROLL);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const matches_containers::match_point_container& matchedPoints = get_matched_points(trueEndPose, POINTS_ERROR);
        matches_containers::match_plane_container matchedPlanes;

        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(END_ROTATION_YAW * GOOD_GUESS, END_ROTATION_PITCH * GOOD_GUESS, END_ROTATION_ROLL * GOOD_GUESS);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, matchedPlanes, trueEndPose, initialPoseGuess);
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

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(END_ROTATION_YAW, 0, 0);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const matches_containers::match_point_container& matchedPoints = get_matched_points(trueEndPose, POINTS_ERROR);
        matches_containers::match_plane_container matchedPlanes;

        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(END_ROTATION_YAW * MEDIUM_GUESS, 0, 0);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, matchedPlanes, trueEndPose, initialPoseGuess);
    }

    TEST(RotationOptimizationTests, rotationPitchMediumguess) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(0, END_ROTATION_PITCH, 0);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const matches_containers::match_point_container& matchedPoints = get_matched_points(trueEndPose, POINTS_ERROR);
        matches_containers::match_plane_container matchedPlanes;

        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(0, END_ROTATION_PITCH * MEDIUM_GUESS, 0);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, matchedPlanes, trueEndPose, initialPoseGuess);
    }

    TEST(RotationOptimizationTests, rotationRollMediumGuess) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(0, 0, END_ROTATION_ROLL);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const matches_containers::match_point_container& matchedPoints = get_matched_points(trueEndPose, POINTS_ERROR);
        matches_containers::match_plane_container matchedPlanes;

        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(0, 0, END_ROTATION_ROLL * MEDIUM_GUESS);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, matchedPlanes, trueEndPose, initialPoseGuess);
    }

    TEST(RotationOptimizationTests, rotationMediumGuess) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(END_ROTATION_YAW, END_ROTATION_PITCH, END_ROTATION_ROLL);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const matches_containers::match_point_container& matchedPoints = get_matched_points(trueEndPose, POINTS_ERROR);
        matches_containers::match_plane_container matchedPlanes;

        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(END_ROTATION_YAW * MEDIUM_GUESS, END_ROTATION_PITCH * MEDIUM_GUESS, END_ROTATION_ROLL * MEDIUM_GUESS);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, matchedPlanes, trueEndPose, initialPoseGuess);
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

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(END_ROTATION_YAW, 0, 0);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const matches_containers::match_point_container& matchedPoints = get_matched_points(trueEndPose, POINTS_ERROR);
        matches_containers::match_plane_container matchedPlanes;

        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(END_ROTATION_YAW * BAD_GUESS, 0, 0);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, matchedPlanes, trueEndPose, initialPoseGuess);
    }

    TEST(RotationOptimizationTests, rotationPitchBadGuess) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(0, END_ROTATION_PITCH, 0);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const matches_containers::match_point_container& matchedPoints = get_matched_points(trueEndPose, POINTS_ERROR);
        matches_containers::match_plane_container matchedPlanes;

        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(0, END_ROTATION_PITCH * BAD_GUESS, 0);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, matchedPlanes, trueEndPose, initialPoseGuess);
    }

    TEST(RotationOptimizationTests, rotationRollBadGuess) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(0, 0, END_ROTATION_ROLL);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const matches_containers::match_point_container& matchedPoints = get_matched_points(trueEndPose, POINTS_ERROR);
        matches_containers::match_plane_container matchedPlanes;

        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(0, 0, END_ROTATION_ROLL * BAD_GUESS);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, matchedPlanes, trueEndPose, initialPoseGuess);
    }

    TEST(RotationOptimizationTests, rotationBadGuess) 
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        // True End pose
        const vector3 truePosition(0, 0, 0);
        const EulerAngles trueEulerAngles(END_ROTATION_YAW, END_ROTATION_PITCH, END_ROTATION_ROLL);
        const quaternion trueQuaternion(utils::get_quaternion_from_euler_angles(trueEulerAngles));
        const utils::Pose trueEndPose(truePosition, trueQuaternion);

        const matches_containers::match_point_container& matchedPoints = get_matched_points(trueEndPose, POINTS_ERROR);
        matches_containers::match_plane_container matchedPlanes;

        // Estimated pose base
        const vector3 initialPositionGuess(0, 0, 0);
        const EulerAngles initialEulerAnglesGuess(END_ROTATION_YAW * BAD_GUESS, END_ROTATION_PITCH * BAD_GUESS, END_ROTATION_ROLL * BAD_GUESS);
        const quaternion initialQuaternionGuess(utils::get_quaternion_from_euler_angles(initialEulerAnglesGuess));
        const utils::Pose initialPoseGuess(initialPositionGuess, initialQuaternionGuess);

        run_test_optimization(matchedPoints, matchedPlanes, trueEndPose, initialPoseGuess);
    }


}
