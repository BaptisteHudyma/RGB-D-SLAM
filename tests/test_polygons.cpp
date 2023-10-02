#include <gtest/gtest.h>
#include "polygon.hpp"

namespace rgbd_slam::utils {

TEST(SquareTests, SimpleFitting)
{
    // rectangle
    const std::vector<vector3> points({vector3(-1000.0, 1000.0, 0.0),
                                       vector3(1000.0, 1000.0, 0.0),
                                       vector3(-1000.0, -1000.0, 0.0),
                                       vector3(1000.0, -1000.0, 0.0)});
    const vector3 normal = vector3(0.0, 0.0, 1.0);
    const vector3 center = vector3::Zero();

    Polygon polygon(points, normal, center);

    // check parameters
    EXPECT_EQ(polygon.get_center(), center);
    EXPECT_EQ(polygon.get_normal(), normal);
    EXPECT_EQ(polygon.get_x_axis().cross(polygon.get_y_axis()), normal);

    EXPECT_EQ(polygon.boundary_length(), 4);
    // area: 2000x2000
    EXPECT_NEAR(polygon.get_area(), 4e6, 0.1);

    // should contain the boundary points and center
    EXPECT_TRUE(polygon.contains(vector2(0.0, 0.0)));
    EXPECT_TRUE(polygon.contains(vector2(-999.99, 999.99)));
    EXPECT_TRUE(polygon.contains(vector2(-999.99, -999.99)));
    EXPECT_TRUE(polygon.contains(vector2(999.99, -999.99)));
    EXPECT_TRUE(polygon.contains(vector2(999.99, 999.99)));
    // should contain other points in the square boundary
    EXPECT_TRUE(polygon.contains(vector2(0.0, 999.99)));
    EXPECT_TRUE(polygon.contains(vector2(0.0, -999.99)));
    EXPECT_TRUE(polygon.contains(vector2(999.99, 0.0)));
    EXPECT_TRUE(polygon.contains(vector2(-999.99, 0.0)));

    // check area of union/inter with itself
    EXPECT_NEAR(polygon.get_area(), polygon.union_area(polygon), 0.1);
    EXPECT_NEAR(polygon.get_area(), polygon.inter_area(polygon), 0.1);

    // same polygon with flipped normal
    Polygon polygonInverse(points, -normal, center);
    EXPECT_EQ(polygonInverse.get_normal(), -normal);

    // check area of union/inter with itself
    EXPECT_NEAR(polygon.get_area(), polygon.union_area(polygonInverse), 0.1);
    EXPECT_NEAR(polygon.get_area(), polygon.inter_area(polygonInverse), 0.1);

    // transformation tests
    const auto& inversedPolygon = polygon.transform(-normal, center);
    EXPECT_EQ(inversedPolygon.get_normal(), -normal);
    EXPECT_EQ(inversedPolygon.get_center(), center);
    // EXPECT_EQ(inversedPolygon.get_area(), polygon.get_area());

    // transform with different center
    const auto shiftedPolygon = polygon.transform(normal, vector3(500.0, 500.0, 0.0));
    EXPECT_EQ(shiftedPolygon.get_normal(), normal);
    EXPECT_EQ(shiftedPolygon.get_center(), vector3(500.0, 500.0, 0.0));
    EXPECT_EQ(shiftedPolygon.get_area(), polygon.get_area());

    // transform with different normal
    const auto turnedPolygon = polygon.transform(vector3(1.0, 0.0, 0.0), center);
    EXPECT_EQ(turnedPolygon.get_normal(), vector3(1.0, 0.0, 0.0));
    EXPECT_EQ(turnedPolygon.get_center(), center);
    // EXPECT_NEAR(turnedPolygon.get_area(), polygon.get_area(), 0.1);
    EXPECT_EQ(turnedPolygon.boundary_length(), 4);

    const auto shiftedProjectedPolygon = polygon.project(normal, vector3(500.0, 500.0, 0.0));
    EXPECT_EQ(shiftedProjectedPolygon.get_normal(), normal);
    EXPECT_EQ(shiftedProjectedPolygon.get_center(), vector3(500.0, 500.0, 0.0));
    EXPECT_EQ(shiftedProjectedPolygon.get_area(), polygon.get_area());

    // project with different normal, and different center
    const auto turnedProjectPolygon = polygon.project(vector3(1.0, 0.0, 0.0), vector3(1000, 0, 0));
    EXPECT_EQ(turnedProjectPolygon.get_normal(), vector3(1.0, 0.0, 0.0));
    EXPECT_EQ(turnedProjectPolygon.get_center(), vector3(1000, 0, 0));
    EXPECT_EQ(turnedProjectPolygon.boundary_length(), 4);
    EXPECT_EQ(turnedProjectPolygon.get_area(), 0.0); // area will be 0, we projected on a perpendicular plane

    const auto semiturnedProjectPolygon = polygon.project(vector3(0.5, 0.0, 0.5).normalized(), vector3(1000, 0, 0));
    EXPECT_TRUE(semiturnedProjectPolygon.get_normal().isApprox(vector3(0.5, 0.0, 0.5).normalized()));
    EXPECT_EQ(semiturnedProjectPolygon.get_center(), vector3(1000, 0, 0));
    EXPECT_EQ(semiturnedProjectPolygon.boundary_length(), 4);
    // area should be smaller but non zero
    EXPECT_LT(semiturnedProjectPolygon.get_area(), polygon.get_area());
    EXPECT_GT(semiturnedProjectPolygon.get_area(), 0.0);
}

TEST(SquareTests, Unions)
{
    // rectangle
    const std::vector<vector3> pointsSquare({vector3(-1000.0, 1000.0, 0.0),
                                             vector3(1000.0, 1000.0, 0.0),
                                             vector3(-1000.0, -1000.0, 0.0),
                                             vector3(1000.0, -1000.0, 0.0)});
    const vector3 normal = vector3(0.0, 0.0, 1.0);
    const vector3 center = vector3::Zero();

    Polygon rectangle(pointsSquare, normal, center);

    // diamond contained in rectangle
    const std::vector<vector3> pointsDiamond({vector3(-1000.0, 0.0, 0.0),
                                              vector3(1000.0, 0.0, 0.0),
                                              vector3(0.0, -1000.0, 0.0),
                                              vector3(0.0, 1000.0, 0.0)});

    Polygon diamond(pointsDiamond, normal, center);
    EXPECT_EQ(diamond.area(), 2e6);

    rectangle.merge_union(diamond);

    EXPECT_EQ(rectangle.area(), 4e6);
    EXPECT_EQ(rectangle.boundary_length(), 4);

    // shift the shape by half a length
    diamond = diamond.transform(normal, vector3(1000.0, 0.0, 0.0));
    EXPECT_EQ(diamond.area(), 2e6);

    rectangle.merge_union(diamond);

    EXPECT_EQ(rectangle.boundary_length(), 5);
    EXPECT_EQ(rectangle.area(), 4e6 + 1e6);

    // set a diamond offsetted to the left
    diamond = diamond.transform(normal, vector3(-1000.0, 0.0, 0.0));
    EXPECT_EQ(diamond.area(), 2e6);

    rectangle.merge_union(diamond);

    EXPECT_EQ(rectangle.boundary_length(), 6);
    EXPECT_EQ(rectangle.area(), 4e6 + 2e6);

    // set a diamond offsetted to the top
    diamond = diamond.transform(normal, vector3(0.0, 1000.0, 0.0));
    EXPECT_EQ(diamond.area(), 2e6);

    rectangle.merge_union(diamond);

    EXPECT_EQ(rectangle.boundary_length(), 5); // the boundary length should be reduced by simplification process
    EXPECT_EQ(rectangle.area(), 4e6 + 3e6);

    // set a diamond offsetted to the top
    diamond = diamond.transform(normal, vector3(0.0, -1000.0, 0.0));
    EXPECT_EQ(diamond.area(), 2e6);

    rectangle.merge_union(diamond);

    EXPECT_EQ(rectangle.boundary_length(), 4); // the boundary length should be reduced by simplification process
    EXPECT_EQ(rectangle.area(), 4e6 + 4e6);
}

} // namespace rgbd_slam::utils