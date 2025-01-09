#include "keypoint_handler.hpp"
#include "../../outputs/logger.hpp"
#include "../../parameters.hpp"
#include "../../types.hpp"
#include "coordinates/point_coordinates.hpp"
#include "line.hpp"
#include <cstddef>
#include <opencv2/features2d.hpp>

namespace rgbd_slam::features::keypoints {

// max match search per keypoint
constexpr uint MAX_POINT_MATCH = 2;

bool is_in_border(const cv::Point2f& pt, const cv::Mat& im, const double borderSize) noexcept
{
    assert(borderSize >= 0);
    return borderSize <= pt.x and borderSize <= pt.y and pt.x < static_cast<double>(im.cols) - borderSize and
           pt.y < static_cast<double>(im.rows) - borderSize;
}

double get_depth(const cv::Mat_<float>& depthImage, const cv::Point2f& depthCoordinates) noexcept
{
    const double border = BORDER_SIZE;
    assert(border > 0);
    if (is_in_border(depthCoordinates, depthImage, border))
    {
        return depthImage.at<float>(std::round(depthCoordinates.y), std::round(depthCoordinates.x));
    }
    return 0.0;
}

Keypoint_Handler::Keypoint_Handler(const uint depthImageCols, const uint depthImageRows)
{
    // knn matcher
    _featuresMatcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
    assert(_featuresMatcher->isMaskSupported()); // TODO: find another way to handle matchers without masks

    constexpr double cellSize = parameters::matching::matchSearchRadius_px + 1.0;
    static_assert(cellSize > 0);

    _cellCountX = static_cast<uint>(std::ceil(depthImageCols / cellSize));
    _cellCountY = static_cast<uint>(std::ceil(depthImageRows / cellSize));
    assert(_cellCountX > 0 and _cellCountY > 0);

    _searchSpaceIndexContainer.resize(static_cast<size_t>(_cellCountY) * _cellCountX);
    for (index_container& indexContainer: _searchSpaceIndexContainer)
    {
        indexContainer.reserve(5);
    }
}

void Keypoint_Handler::clear() noexcept
{
    _keypoints.clear();
    _uniqueIdsToKeypointIndex.clear();

    for (index_container& indexContainer: _searchSpaceIndexContainer)
    {
        indexContainer.clear();
    }

    _featuresMatcher->clear();
    _descriptors.release();
}

void Keypoint_Handler::set(std::vector<cv::Point2f>& inKeypoints,
                           const cv::Mat& inDescriptors,
                           const KeypointsWithIdStruct& lastKeypointsWithIds,
                           const cv::Mat_<float>& depthImage) noexcept
{
    // clear last state
    clear();

    _featuresMatcher->add(inDescriptors);
    _featuresMatcher->train();

    _descriptors = inDescriptors;

    // Fill depth values, add points to image boxes
    const size_t allKeypointSize = inKeypoints.size() + lastKeypointsWithIds.size();
    _keypoints = std::vector<ScreenCoordinate>(allKeypointSize);

    // Add detected keypoints first
    const uint keypointIndexOffset = static_cast<uint>(inKeypoints.size());
    for (uint pointIndex = 0; pointIndex < keypointIndexOffset; ++pointIndex)
    {
        const cv::Point2f& pt = inKeypoints[pointIndex];
        // Depths are in millimeters, will be 0 if coordinates are invalid
        const ScreenCoordinate vectorKeypoint(pt.x, pt.y, get_depth(depthImage, pt));

        _keypoints[pointIndex] = vectorKeypoint;

        const uint searchSpaceIndex = get_search_space_index(get_search_space_coordinates(vectorKeypoint.get_2D()));
        assert(searchSpaceIndex < _searchSpaceIndexContainer.size());

        _searchSpaceIndexContainer[searchSpaceIndex].push_back(pointIndex);
    }

    // Add optical flow keypoints then
    const size_t opticalPointSize = lastKeypointsWithIds.size();
    for (size_t pointIndex = 0; pointIndex < opticalPointSize; ++pointIndex)
    {
        const size_t newKeypointIndex = pointIndex + keypointIndexOffset;

        // fill in unique point index
        const KeypointsWithIdStruct::keypointWithId& kp = lastKeypointsWithIds.at(pointIndex);
        const size_t uniqueIndex = kp._id;
        const cv::Point2f& pt = kp._point;

        // TODO: >= 0 ?
        if (uniqueIndex > 0)
        {
            _uniqueIdsToKeypointIndex[uniqueIndex] = newKeypointIndex;
        }
        else
        {
            outputs::log_error("A keypoint detected by optical flow does not have a valid keypoint id");
        }

        // Depths are in millimeters, will be 0 if coordinates are invalid
        // no search space: already matched
        _keypoints[newKeypointIndex] = ScreenCoordinate(pt.x, pt.y, get_depth(depthImage, pt));
    }
}

uint Keypoint_Handler::get_search_space_index(const uint_pair& searchSpaceIndex) const noexcept
{
    return get_search_space_index(searchSpaceIndex.second, searchSpaceIndex.first);
}
uint Keypoint_Handler::get_search_space_index(const uint x, const uint y) const noexcept { return y * _cellCountY + x; }

Keypoint_Handler::uint_pair Keypoint_Handler::get_search_space_coordinates(
        const ScreenCoordinate2D& pointToPlace) const noexcept
{
    constexpr double cellSize = parameters::matching::matchSearchRadius_px + 1.0;
    const uint_pair cellCoordinates(std::clamp(floor(pointToPlace.y() / cellSize), 0.0, _cellCountY - 1.0),
                                    std::clamp(floor(pointToPlace.x() / cellSize), 0.0, _cellCountX - 1.0));
    return cellCoordinates;
}

cv::Mat_<uchar> Keypoint_Handler::compute_key_point_mask(const ScreenCoordinate2D& pointToSearch,
                                                         const uint searchSpaceCellRadius) const noexcept
{
    const auto [searchSpaceCoordinatesY, searchSpaceCoordinatesX] = get_search_space_coordinates(pointToSearch);
    // compute a search zone for the potential matches of this point
    const uint startY = std::max(0U, searchSpaceCoordinatesY - searchSpaceCellRadius);
    const uint startX = std::max(0U, searchSpaceCoordinatesX - searchSpaceCellRadius);

    const uint endY = std::min(_cellCountY, searchSpaceCoordinatesY + searchSpaceCellRadius + 1);
    const uint endX = std::min(_cellCountX, searchSpaceCoordinatesX + searchSpaceCellRadius + 1);

    // set a mask of the size of the keypoints, with everything at zero (nothing can be matched)
    cv::Mat_<uchar> keyPointMask = cv::Mat_<float>::zeros(1, _descriptors.rows);
    for (uint i = startY; i < endY; ++i)
    {
        for (uint j = startX; j < endX; ++j)
        {
            const size_t searchSpaceIndex = get_search_space_index(j, i);
            assert(searchSpaceIndex < _searchSpaceIndexContainer.size());

            // get all keypoints in this area
            const index_container& keypointIndexContainer = _searchSpaceIndexContainer[searchSpaceIndex];
            fill_keypoint_mask(pointToSearch, keypointIndexContainer, keyPointMask);
        }
    }

    return keyPointMask;
}

void Keypoint_Handler::fill_keypoint_mask(const ScreenCoordinate2D& pointToSearch,
                                          const index_container& keypointIndexContainer,
                                          cv::Mat_<uchar>& keyPointMask) const noexcept
{
    // Squared search diameter, to compare distance without sqrt
    constexpr float squaredSearchDiameter = static_cast<float>(SQR(parameters::matching::matchSearchRadius_px));
    for (const uint keypointIndex: keypointIndexContainer)
    {
        const ScreenCoordinate2D& keypoint = get_keypoint(keypointIndex).get_2D();
        const double squarredDistance = (keypoint - pointToSearch).squaredNorm();

        // keypoint is in a circle around the target keypoints, allow a potential match
        if (squarredDistance <= squaredSearchDiameter)
            keyPointMask(0, static_cast<int>(keypointIndex)) = 1;
    }
}

void Keypoint_Handler::fill_keypoint_mask(const utils::Segment<2>& pointToSearch,
                                          const index_container& keypointIndexContainer,
                                          cv::Mat_<uchar>& keyPointMask) const noexcept
{
    // Squared search diameter, to compare distance without sqrt
    constexpr float squaredSearchDiameter = static_cast<float>(SQR(parameters::matching::matchSearchRadius_px));
    for (const uint keypointIndex: keypointIndexContainer)
    {
        const ScreenCoordinate2D& keypoint = get_keypoint(keypointIndex).get_2D();
        const double squarredDistance = pointToSearch.distance(keypoint).squaredNorm();

        // keypoint is in a circle around the target keypoints, allow a potential match
        if (squarredDistance <= squaredSearchDiameter)
            keyPointMask(0, static_cast<int>(keypointIndex)) = 1;
    }
}

int Keypoint_Handler::get_tracking_match_index(const size_t mapPointId) const noexcept
{
    if (_keypoints.empty())
        return INVALID_MATCH_INDEX;

    // search if the keypoint id is in the detected points
    if (mapPointId != INVALID_MAP_POINT_ID)
    {
        // return the match if it's the case
        uintToUintContainer::const_iterator uniqueIndexIterator = _uniqueIdsToKeypointIndex.find(mapPointId);
        if (uniqueIndexIterator != _uniqueIdsToKeypointIndex.cend())
        {
            return static_cast<int>(uniqueIndexIterator->second);
        }
    }
    return INVALID_MATCH_INDEX;
}

Keypoint_Handler::matchIndexSet Keypoint_Handler::get_match_index(const ScreenCoordinate2D& projectedMapPoint,
                                                                  const cv::Mat& mapPointDescriptor,
                                                                  const double searchSpaceRadius) const noexcept
{
    assert(_featuresMatcher != nullptr);
    Keypoint_Handler::matchIndexSet matchSet;

    // cannot compute matches without a match or descriptors
    if (_keypoints.empty() or _descriptors.empty())
        return matchSet;

    constexpr double cellSize = parameters::matching::matchSearchRadius_px + 1.0;
    static_assert(cellSize > 0);
    const uint searchSpaceCellRadius = static_cast<uint>(std::ceil(searchSpaceRadius / cellSize));
    assert(searchSpaceCellRadius > 0);

    // check descriptor dimensions
    assert(!mapPointDescriptor.empty());
    assert(mapPointDescriptor.cols == _descriptors.cols);

    const cv::Mat_<uchar>& keyPointMask = compute_key_point_mask(projectedMapPoint, searchSpaceCellRadius);

    std::vector<std::vector<cv::DMatch>> knnMatches;
    _featuresMatcher->knnMatch(mapPointDescriptor, _descriptors, knnMatches, MAX_POINT_MATCH, keyPointMask, true);

    if (knnMatches.empty())
        return matchSet;

    // check the neighbors
    const std::vector<cv::DMatch>& firstMatch = knnMatches[0];
    for (const auto& match: firstMatch)
    {
        int id = match.trainIdx;
        matchSet.emplace(id);
    }
    return matchSet;
}

Keypoint_Handler::matchIndexSet Keypoint_Handler::get_match_index(const utils::Segment<2>& projectedMapPoint,
                                                                  const cv::Mat& mapPointDescriptor,
                                                                  const double searchSpaceRadius) const noexcept
{
    assert(_featuresMatcher != nullptr);
    Keypoint_Handler::matchIndexSet matchSet;

    // cannot compute matches without a match or descriptors
    if (_keypoints.empty() or _descriptors.empty())
        return matchSet;

    constexpr double cellSize = parameters::matching::matchSearchRadius_px + 1.0;
    static_assert(cellSize > 0);
    const uint searchSpaceCellRadius = static_cast<uint>(std::ceil(searchSpaceRadius / cellSize));
    assert(searchSpaceCellRadius > 0);

    // check descriptor dimensions
    assert(!mapPointDescriptor.empty());
    assert(mapPointDescriptor.cols == _descriptors.cols);

    // set a mask of the size of the keypoints, with everything at zero (nothing can be matched)
    cv::Mat_<uchar> keyPointMask = cv::Mat_<float>::zeros(1, _descriptors.rows);
    for (uint i = 0; i < _cellCountX; ++i)
    {
        for (uint j = 0; j < _cellCountY; ++j)
        {
            const size_t searchSpaceIndex = get_search_space_index(j, i);
            assert(searchSpaceIndex < _searchSpaceIndexContainer.size());

            // get all keypoints in this area
            const index_container& keypointIndexContainer = _searchSpaceIndexContainer[searchSpaceIndex];
            fill_keypoint_mask(projectedMapPoint, keypointIndexContainer, keyPointMask);
        }
    }
    std::vector<std::vector<cv::DMatch>> knnMatches;
    _featuresMatcher->knnMatch(mapPointDescriptor, _descriptors, knnMatches, MAX_POINT_MATCH, keyPointMask, true);

    if (knnMatches.empty())
        return matchSet;

    // check the neighbors
    const std::vector<cv::DMatch>& firstMatch = knnMatches[0];
    for (const auto& match: firstMatch)
    {
        int id = match.trainIdx;
        matchSet.emplace(id);
    }
    return matchSet;
}

} // namespace rgbd_slam::features::keypoints
