#ifndef RGBDSLAM_POSEOPTIMIZATION_RANSAC_HPP
#define RGBDSLAM_POSEOPTIMIZATION_RANSAC_HPP

#include "utils/random.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace rgbd_slam::pose_optimization::ransac {

/**
 * \brief Return a random subset of unique elements, of size n. It is inefficient if n is very inferior to
 * inContainer.size
 * \param[in] inContainer A container of size > numberOfElementsToChoose, in which this function will pick elements
 * \param[in] numberOfElementsToChoose The number of elements we which to find in the final container
 * \return A container of size numberOfElementsToChoose, with no duplicated elements.
 */
template<template<typename> class Container, typename T>
[[nodiscard]] Container<T> get_random_subset(const Container<T>& inContainer, const uint numberOfElementsToChoose)
{
    if (numberOfElementsToChoose > inContainer.size())
    {
        throw std::invalid_argument("get_random_subset: numberOfElementsToChoose should be >= to the elements count");
    }

    // get a vector of references and shuffle it
    std::vector<std::reference_wrapper<const T>> copyVector(inContainer.cbegin(), inContainer.cend());
    std::ranges::shuffle(copyVector, utils::Random::get_random_engine());
    if (copyVector.size() != inContainer.size())
    {
        throw std::invalid_argument("get_random_subset: failed to create a reference vector");
    }

    // copy the first matches, they will be randoms
    Container<T> outContainer;
    for (uint i = 0; i < numberOfElementsToChoose; ++i)
        outContainer.insert(outContainer.begin(), copyVector[i]);
    return outContainer;
}

/**
 * \brief Return a random subset of elements, of size n
 * \param[in] inContainer A container of size > numberOfElementsToChoose, in which this function will pick elements
 * \param[in] numberOfElementsToChoose The number of elements we which to find in the final container
 * \return A container of size numberOfElementsToChoose
 */
template<template<typename> class Container, typename T>
[[nodiscard]] Container<T> get_random_subset_with_duplicates(const Container<T>& inContainer,
                                                             const uint numberOfElementsToChoose)
{
    const uint inContainerSize = inContainer.size();
    if (numberOfElementsToChoose > inContainer.size())
    {
        throw std::invalid_argument("get_random_subset: numberOfElementsToChoose should be >= to the elements count");
    }

    // get a vector of references
    std::vector<std::reference_wrapper<const T>> copyVector(inContainer.cbegin(), inContainer.cend());

    // copy the first matches, they will be randoms
    Container<T> outContainer;
    while (outContainer.size() < numberOfElementsToChoose)
    {
        outContainer.insert(outContainer.begin(), copyVector[utils::Random::get_random_uint(inContainerSize)]);
    }
    return outContainer;
}

/**
 * \brief Return a random subset of unique elements, of size n. It is inefficient if n is very inferior to
 * inContainer.size
 * \param[in] inContainer A container of unknown size, in which this function will pick elements
 * \param[in] targetScore The total score to pick
 * \return A container of unknown size, with no duplicated elements.
 */
template<template<typename> class Container, typename T>
[[nodiscard]] Container<T> get_random_subset_with_score(const Container<T>& inContainer, const double targetScore)
{
    // get a vector of references and shuffle it
    std::vector<std::reference_wrapper<const T>> copyVector(inContainer.cbegin(), inContainer.cend());
    std::ranges::shuffle(copyVector, utils::Random::get_random_engine());
    if (copyVector.size() != inContainer.size())
    {
        throw std::invalid_argument("get_random_subset: failed to create a reference vector");
    }

    // copy the first matches, they will be randoms
    double cumulatedScore = 0.0;
    Container<T> outContainer;
    for (const T& picked: copyVector)
    {
        cumulatedScore += picked->get_score();
        outContainer.insert(outContainer.begin(), picked);
        if (cumulatedScore >= targetScore)
        {
            return outContainer;
        }
    }

    throw std::invalid_argument("get_random_subset: failed to pick enough feature to respect targetScore");
    return outContainer;
}

/**
 * \brief Return a random subset of elements, of size n
 * \param[in] inContainer A container of unknown size, in which this function will pick elements
 * \param[in] targetScore The number of elements we which to find in the final container
 * \return A container of unknown size
 */
template<template<typename> class Container, typename T>
[[nodiscard]] Container<T> get_random_subset_with_score_with_duplicates(const Container<T>& inContainer,
                                                                        const double targetScore)
{
    // get a vector of references
    std::vector<std::reference_wrapper<const T>> copyVector(inContainer.cbegin(), inContainer.cend());

    // copy the first matches, they will be randoms
    Container<T> outContainer;
    double cumulatedScore = 0.0;
    while (cumulatedScore < targetScore)
    {
        const T& picked = copyVector[utils::Random::get_random_uint(inContainer.size())];

        cumulatedScore += picked.get_score();
        outContainer.insert(outContainer.begin(), picked);
    }
    return outContainer;
}

} // namespace rgbd_slam::pose_optimization::ransac

#endif
