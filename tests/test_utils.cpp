#include "coordinates/point_coordinates.hpp"
#include "parameters.hpp"
#include "utils/line.hpp"

#include <gtest/gtest.h>

#include <random>

namespace rgbd_slam {

TEST(ClampToScreen, pointsInScreen)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const auto imageSize = Parameters::get_camera_1_image_size();

    std::mt19937 randomEngine(1000);
    std::uniform_real_distribution<double> errorDistributionX(1.0, imageSize.x() - 1);
    std::uniform_real_distribution<double> errorDistributionY(1.0, imageSize.y() - 1);

    for (size_t i = 0; i < 100; i++)
    {
        utils::Segment<2> inSegment(vector2(errorDistributionX(randomEngine), errorDistributionY(randomEngine)),
                                    vector2(errorDistributionX(randomEngine), errorDistributionY(randomEngine)));

        utils::Segment<2> outSegment;
        EXPECT_TRUE(utils::clamp_to_screen(inSegment, outSegment));

        EXPECT_NEAR(inSegment.get_start_point().x(), outSegment.get_start_point().x(), 1e-3);
        EXPECT_NEAR(inSegment.get_start_point().y(), outSegment.get_start_point().y(), 1e-3);
        EXPECT_NEAR(inSegment.get_end_point().x(), outSegment.get_end_point().x(), 1e-3);
        EXPECT_NEAR(inSegment.get_end_point().y(), outSegment.get_end_point().y(), 1e-3);

        EXPECT_TRUE(ScreenCoordinate2D(outSegment.get_start_point()).is_in_screen_boundaries());
        EXPECT_TRUE(ScreenCoordinate2D(outSegment.get_end_point()).is_in_screen_boundaries());
    }
}

TEST(ClampToScreen, onePointInsideScreen)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const auto& imageSize = Parameters::get_camera_1_image_size();

    vector2 centerPoint(imageSize.x() / 2.0, imageSize.y() / 2.0);

    std::mt19937 randomEngine(1000);
    std::uniform_real_distribution<double> errorDistributionX(imageSize.x(), 2.0 * imageSize.x());
    std::uniform_real_distribution<double> errorDistributionY(imageSize.y(), 2.0 * imageSize.y());

    for (size_t i = 0; i < 100; i++)
    {
        vector2 randomVec(errorDistributionX(randomEngine), errorDistributionY(randomEngine));

        utils::Segment<2> inSegment(centerPoint, randomVec);

        utils::Segment<2> outSegment;
        EXPECT_TRUE(utils::clamp_to_screen(inSegment, outSegment));

        EXPECT_NEAR(inSegment.get_start_point().x(), outSegment.get_start_point().x(), 1e-3);
        EXPECT_NEAR(inSegment.get_start_point().y(), outSegment.get_start_point().y(), 1e-3);

        EXPECT_TRUE(ScreenCoordinate2D(outSegment.get_start_point()).is_in_screen_boundaries());
        EXPECT_TRUE(ScreenCoordinate2D(outSegment.get_end_point()).is_in_screen_boundaries());

        // test the other side

        inSegment = utils::Segment<2>(randomVec, centerPoint);

        EXPECT_TRUE(utils::clamp_to_screen(inSegment, outSegment));

        EXPECT_NEAR(inSegment.get_end_point().x(), outSegment.get_end_point().x(), 1e-3);
        EXPECT_NEAR(inSegment.get_end_point().y(), outSegment.get_end_point().y(), 1e-3);

        EXPECT_TRUE(ScreenCoordinate2D(outSegment.get_start_point()).is_in_screen_boundaries());
        EXPECT_TRUE(ScreenCoordinate2D(outSegment.get_end_point()).is_in_screen_boundaries());
        // test the other side
    }
}

TEST(ClampToScreen, twoPointOutsideScreen)
{
    if (not Parameters::is_valid())
    {
        Parameters::load_defaut();
    }

    const auto& imageSize = Parameters::get_camera_1_image_size();

    // center point X

    vector2 startPoint = vector2(-150.0, imageSize.y() / 2.0);
    vector2 endPoint = vector2(imageSize.x() * 2.0, imageSize.y() / 2.0);

    utils::Segment<2> inSegment(startPoint, endPoint);

    utils::Segment<2> outSegment;
    EXPECT_TRUE(utils::clamp_to_screen(inSegment, outSegment));

    EXPECT_NEAR(inSegment.get_end_point().y(), outSegment.get_end_point().y(), 1e-3);
    EXPECT_NEAR(inSegment.get_start_point().y(), outSegment.get_start_point().y(), 1e-3);

    EXPECT_NEAR(imageSize.x() - 1.0, outSegment.get_end_point().x(), 1e-3);
    EXPECT_NEAR(1.0, outSegment.get_start_point().x(), 1e-3);

    EXPECT_TRUE(ScreenCoordinate2D(outSegment.get_start_point()).is_in_screen_boundaries());
    EXPECT_TRUE(ScreenCoordinate2D(outSegment.get_end_point()).is_in_screen_boundaries());

    // Center point Y

    startPoint = vector2(imageSize.x() / 2.0, -150.0);
    endPoint = vector2(imageSize.x() / 2.0, imageSize.y() * 2.0);

    inSegment = utils::Segment<2>(startPoint, endPoint);

    EXPECT_TRUE(utils::clamp_to_screen(inSegment, outSegment));

    EXPECT_NEAR(inSegment.get_end_point().x(), outSegment.get_end_point().x(), 1e-3);
    EXPECT_NEAR(inSegment.get_start_point().x(), outSegment.get_start_point().x(), 1e-3);

    EXPECT_NEAR(imageSize.y() - 1.0, outSegment.get_start_point().y(), 1e-3);
    EXPECT_NEAR(1.0, outSegment.get_end_point().y(), 1e-3);

    EXPECT_TRUE(ScreenCoordinate2D(outSegment.get_start_point()).is_in_screen_boundaries());
    EXPECT_TRUE(ScreenCoordinate2D(outSegment.get_end_point()).is_in_screen_boundaries());
}

} // namespace rgbd_slam
