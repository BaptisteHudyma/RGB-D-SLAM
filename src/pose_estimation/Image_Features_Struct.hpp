#ifndef IMAGE_FEATURES_STRUCT_HPP
#define IMAGE_FEATURES_STRUCT_HPP

#include "Pose.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

namespace poseEstimation {

    /**
      * \brief Stores a list of keypoints (features) and provide methods to find a match of a point in those features
      */
    class Image_Features_Struct
    {
        public:
            Image_Features_Struct();

            /**
              * \param[in] image Image where features were found. Used only to get the number of rows/columns
              * \param[in] keypoints Keypoints in image
              * \param[in] descs Descriptors of map features
              * \param[in] trackingRadius Maximum distance between two features to consider a match
              * \param[in] hashingCellSize Size of the search match window in pixel.
              * \param[in] verticalSearchRadius Maximum row distance between two features to consider a match
              * \param[in] triangulationRatioTh
              * \param[in] trackingRatioTh 
              * \param[in] descDistTh Maximum distance of two descriptors to consider a match
              * \param[in] kpsDepth (optional) Vector storing the depth of each feature
              */
            void init(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descs,
                    int trackingRadius, int hashingCellSize, int verticalSearchRadius, float triangulationRatioTh,
                    float trackingRatioTh, float descDistTh, std::vector<float> *kpsDepth = nullptr);


            /**
             * \brief Searches or a match between the given point and our map features. It uses a k nearest neighbors method.
             *
             * \param[in] pt Point to match
             * \param[in] desc Descriptor of point pt
             * \param[out] d1 Distance of pt to the closest matching point
             * \param[out] d2 Distance of pt to the second closest matching point, or -1
             *
             * \return Index of the best feature or -1 if not matched
             */
            int find_match_index(const vector2 &pt, const cv::Mat &desc, float *d1, float *d2) const; 

            /**
             * \brief Searches for a match between a point and our map features. Reduces search space to rows. It uses a k nearest neighbors method.
             *
             * \param[in] pt The point to search a match for
             * \param[in] desc Descriptor of pt
             *
             * \return Index of the matched feature, or -1 if no match
             */
            int row_match(const cv::Point2f &pt, const cv::Mat &desc) const;

            /**
             * \brief Marks a feature id as been matched to a point
             */
            void mark_as_matched(unsigned int idx, bool val) { _matched_marks[idx] = val; }

            /**
             * \brief Checks if a feature as already been matched to a point
             */
            bool is_matched(unsigned int idx) const { return _matched_marks[idx]; }

            /**
             * \brief Reset the matched feature container
             */
            void reset_matched_marks() { _matched_marks = std::vector<bool>(_keypoints.size(), false); }

            /**
             * \brief Checks if the features are associated with depth values
             */
            bool is_depth_associated() const { return !_kps_depths.empty(); }


        public: //getters
            /**
             * \brief Return the knn search radius for matches
             */
            int get_tracking_radius() const { return _tracking_radius; }
            /**
             * \brief Return the feature count in the local map
             */
            unsigned int get_features_count() const { return static_cast<unsigned int>(_keypoints.size()); }
            /**
             * \brief Return the depth the feature with index idx
             */
            float get_keypoint_depth(unsigned int idx) const { return _kps_depths[idx]; }

            /**
             * \brief Return the descriptor of a feature
             */
            const cv::Mat get_descriptor(unsigned int index) const { return _descriptors.row(index); }

            const cv::KeyPoint& get_keypoint(unsigned int index) const { return _keypoints[index]; }

            //setter
            void set_tracking_radius(unsigned int tracking_radius) { _tracking_radius = tracking_radius; }


        private:
            typedef std::vector<cv::KeyPoint> keypoint_vector;

            keypoint_vector _keypoints;
            cv::Mat _descriptors;
            cv::Ptr<cv::DescriptorMatcher> _matcher;
            std::vector<bool> _matched_marks;
            unsigned int _cell_size;
            int _cell_count_x;
            int _cell_count_y;
            int _cell_search_radius;
            unsigned int _tracking_radius;
            int _img_rows;
            int _img_cols;
            int _vertical_search_radius;
            float _triangulation_ratio_th;
            float _tracking_ratio_th;
            float _desc_dist_th;
            std::vector<float> _kps_depths;

            typedef std::vector<int> index_list_t;
            std::vector<std::vector<index_list_t>> _index_hashmap;

            typedef std::pair<int, int> index_pair_t;
            /**
             * \brief Compute the index of the reduced map space to search
             *
             * \param[in] val Point that will be searched
             * \param[in] cell_size Size of the search window
             *
             * \return The index of the area to search in
             */
            inline index_pair_t compute_hashed_index(const cv::Point2f &val, float cellSize) const
            {
                return index_pair_t(floor(val.y / cellSize), floor(val.x / cellSize));
            }
    };

}

#endif
