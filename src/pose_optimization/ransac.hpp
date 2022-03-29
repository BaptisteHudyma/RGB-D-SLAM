#ifndef RGBDSLAM_POSEOPTIMIZATION_RANSAC_HPP
#define RGBDSLAM_POSEOPTIMIZATION_RANSAC_HPP


namespace rgbd_slam {
    namespace pose_optimization {

        /**
         * \brief Return a random subset of matches, of size n. It is inefficient for if n is very inferior to matchedPoints.size
         */
        template< template<typename> class Container, typename T>
            Container<T> get_n_random_matches(const Container<T>& inContainer, const uint n)
            {
                const size_t maxIndex = inContainer.size();
                assert(n <= maxIndex);

                // get a reference vector and shuffle it
                std::vector<std::reference_wrapper<const T>> copyVector(inContainer.cbegin(), inContainer.cend());
                std::random_shuffle(copyVector.begin(), copyVector.end());

                assert(copyVector.size() == inContainer.size());

                // copy the first matches, they will be randoms
                Container<T> selectedMatches;
                for(uint i = 0; i < n; ++i)
                    selectedMatches.insert(selectedMatches.begin(), copyVector[i]);
                return selectedMatches;
            }

    }   /* pose_optimization */
}   /* rgbd_slam */


#endif
