#include <gtest/gtest.h>

#include "parameters.hpp"
#include "types.hpp"
#include "utils/coordinates.hpp"
#include "utils/camera_transformation.hpp"

namespace rgbd_slam::utils {

    void estimate_point_error(const vector3& pointA, const vector3& pointB)
    {
        EXPECT_NEAR(pointA.x(), pointB.x(), 0.001);
        EXPECT_NEAR(pointA.y(), pointB.y(), 0.001);
        EXPECT_NEAR(pointA.z(), pointB.z(), 0.001);
    }


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
        for(double x = 0; x < xRange; x += xIncrement) 
        {
            for(double y = 0; y < yRange; y += yIncrement) 
            {
                for(double z = zIncrement; z < zRange; z += zIncrement) 
                {
                    const ScreenCoordinate originalScreenCoordinates(x, y, z);
                    const CameraCoordinate cameraCoordinates = originalScreenCoordinates.to_camera_coordinates();
                    ScreenCoordinate newScreenCoordinates;
                    if (cameraCoordinates.to_screen_coordinates(newScreenCoordinates))
                    {
                        estimate_point_error(originalScreenCoordinates.vector(), newScreenCoordinates.vector());
                    }
                    else {
                        FAIL();
                    }
                }

                for(double z = -zRange; z < -zIncrement; z += zIncrement) 
                {
                    const ScreenCoordinate originalScreenCoordinates(x, y, z);
                    const CameraCoordinate cameraCoordinates = originalScreenCoordinates.to_camera_coordinates();
                    ScreenCoordinate newScreenCoordinates;
                    if (cameraCoordinates.to_screen_coordinates(newScreenCoordinates))
                    {
                        estimate_point_error(originalScreenCoordinates.vector(), newScreenCoordinates.vector());
                    }
                    else {
                        FAIL();
                    }
                }
            }
        }
    }


    void test_point_set_screen_to_world_to_screen(const cameraToWorldMatrix& cameraToWorld)
    {
        const worldToCameraMatrix worldToCamera = compute_world_to_camera_transform(cameraToWorld);

        const double xRange = 640.0;
        const double yRange = 480.0;
        const double zRange = 50.0;
        const double xIncrement = 7.5;
        const double yIncrement = 5.5;
        const double zIncrement = 0.5;
        for(double x = 0; x < xRange; x += xIncrement) 
        {
            for(double y = 0; y < yRange; y += yIncrement) 
            {
                for(double z = zIncrement; z < zRange; z += zIncrement) 
                {
                    const ScreenCoordinate originalScreenCoordinates(x, y, z);
                    const WorldCoordinate worldCoordinates = originalScreenCoordinates.to_world_coordinates(cameraToWorld);
                    ScreenCoordinate newScreenCoordinates;
                    if (worldCoordinates.to_screen_coordinates(worldToCamera, newScreenCoordinates))
                    {
                        estimate_point_error(originalScreenCoordinates.vector(), newScreenCoordinates.vector());
                    }
                    else {
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

        const cameraToWorldMatrix& cameraToWorld = compute_camera_to_world_transform(
            quaternion(0, 0, 0, 1),
            vector3(0, 0, 0)
        );
        test_point_set_screen_to_world_to_screen(cameraToWorld);
    }

    TEST(PointCoordinateSystemTests, ScreenToWorldToScreenFarFromOrigin)
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        const cameraToWorldMatrix& cameraToWorld = compute_camera_to_world_transform(
            quaternion(0, 0, 0, 1),
            vector3(-100, 1000, 100)
        );
        test_point_set_screen_to_world_to_screen(cameraToWorld);
    }

    TEST(PointCoordinateSystemTests, ScreenToWorldToScreenRotation1)
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        const cameraToWorldMatrix& cameraToWorld = compute_camera_to_world_transform(
            quaternion(0.3, 0.2, 0.1, 0.4),
            vector3(0, 0, 0)
        );
        test_point_set_screen_to_world_to_screen(cameraToWorld);
    }

    TEST(PointCoordinateSystemTests, ScreenToWorldToScreenRotation2)
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        const cameraToWorldMatrix& cameraToWorld = compute_camera_to_world_transform(
            quaternion(0.6, 0.1, 0.2, 0.1),
            vector3(0, 0, 0)
        );
        test_point_set_screen_to_world_to_screen(cameraToWorld);
    }

    TEST(PointCoordinateSystemTests, ScreenToWorldToScreenRotation3)
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        const cameraToWorldMatrix& cameraToWorld = compute_camera_to_world_transform(
            quaternion(0.6, 0.1, 0.2, 0.1),
            vector3(100, -100, -100)
        );
        test_point_set_screen_to_world_to_screen(cameraToWorld);
    }




    void estimate_plane_error(const vector4& pointA, const vector4& pointB)
    {
        EXPECT_NEAR(pointA.x(), pointB.x(), 0.001);
        EXPECT_NEAR(pointA.y(), pointB.y(), 0.001);
        EXPECT_NEAR(pointA.z(), pointB.z(), 0.001);
        EXPECT_NEAR(pointA.w(), pointB.w(), 0.001);
    }

    void test_plane_set_camera_to_world_to_camera(const cameraToWorldMatrix& cameraToWorld)
    {
        const worldToCameraMatrix worldToCamera = compute_world_to_camera_transform(cameraToWorld);
        const double normalXIter = 0.1;
        const double normalYIter = 0.1;
        const double normalZIter = 0.1;

        for(double x = 0; x < 1.0; x += normalXIter)
        {
            for(double y = 0; y < 1.0; y += normalYIter)
            {
                for(double z = 0; z < 1.0; z += normalZIter)
                {
                    for(double d = 0; d < 100; d += 5.5)
                    {
                        const vector3 planeNormal = vector3(x, y, z);
                        const PlaneCameraCoordinates originalCameraPlane(vector4(planeNormal.x(), planeNormal.y(), planeNormal.z(), d));
                        const PlaneWorldCoordinates worldPlane = originalCameraPlane.to_world_coordinates(cameraToWorld);
                        const PlaneCameraCoordinates newCameraCoordinates = worldPlane.to_camera_coordinates(worldToCamera);

                        estimate_plane_error(originalCameraPlane.vector(), newCameraCoordinates.vector());
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

        const cameraToWorldMatrix& cameraToWorld = compute_camera_to_world_transform(
            quaternion(0, 0, 0, 1),
            vector3(0, 0, 0)
        );
        test_plane_set_camera_to_world_to_camera(cameraToWorld);
    }

    TEST(PlaneCoordinateSystemTests, CameraToWorldToCameraFarFromOrigin)
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        const cameraToWorldMatrix& cameraToWorld = compute_camera_to_world_transform(
            quaternion(0, 0, 0, 1),
            vector3(-100, 1000, 100)
        );
        test_plane_set_camera_to_world_to_camera(cameraToWorld);
    }

    TEST(PlaneCoordinateSystemTests, CameraToWorldToCameraRotation1)
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        const cameraToWorldMatrix& cameraToWorld = compute_camera_to_world_transform(
            quaternion(0.3, 0.2, 0.1, 0.4),
            vector3(0, 0, 0)
        );
        test_plane_set_camera_to_world_to_camera(cameraToWorld);
    }

    TEST(PlaneCoordinateSystemTests, CameraToWorldToCameraRotation2)
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        const cameraToWorldMatrix& cameraToWorld = compute_camera_to_world_transform(
            quaternion(0.6, 0.1, 0.2, 0.1),
            vector3(0, 0, 0)
        );
        test_plane_set_camera_to_world_to_camera(cameraToWorld);
    }

    TEST(PlaneCoordinateSystemTests, CameraToWorldToCameraRotation3)
    {
        if (not Parameters::is_valid())
        {
            Parameters::load_defaut();
        }

        const cameraToWorldMatrix& cameraToWorld = compute_camera_to_world_transform(
            quaternion(0.6, 0.1, 0.2, 0.1),
            vector3(100, -100, -100)
        );
        test_plane_set_camera_to_world_to_camera(cameraToWorld);
    }

}