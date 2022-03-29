#ifndef RGBDSLAM_POSEOPTIMIZATION_RANSAC_HPP
#define RGBDSLAM_POSEOPTIMIZATION_RANSAC_HPP


namespace rgbd_slam {
    namespace pose_optimization {

        /**
         * \brief Return a random subset of unique elements, of size n. It is inefficient if n is very inferior to inContainer.size
         * \param[in] inContainer A container of size > numberOfElementsToChoose, in which this function will pick elements
         * \param[in] numberOfElementsToChoose The number of elements we which to find in the final container
         * \return A container of size numberOfElementsToChoose, with no duplicated elements.
         */
        template< template<typename> class Container, typename T>
            Container<T> get_random_subset(const Container<T>& inContainer, const uint numberOfElementsToChoose)
            {
                const size_t maxIndex = inContainer.size();
                assert(numberOfElementsToChoose <= maxIndex);

                // get a vector of references and shuffle it
                std::vector<std::reference_wrapper<const T>> copyVector(inContainer.cbegin(), inContainer.cend());
                std::random_shuffle(copyVector.begin(), copyVector.end());

                assert(copyVector.size() == inContainer.size());

                // copy the first matches, they will be randoms
                Container<T> outContainer;
                for(uint i = 0; i < numberOfElementsToChoose; ++i)
                    outContainer.insert(outContainer.begin(), copyVector[i]);
                return outContainer;
            }

    }   /* pose_optimization */
}   /* rgbd_slam */


#endif
