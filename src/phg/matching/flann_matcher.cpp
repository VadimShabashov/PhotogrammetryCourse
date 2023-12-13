#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
    index_params = flannKdTreeIndexParams(2);
    search_params = flannKsTreeSearchParams(60);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_descs, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{

    int num_queries = query_descs.size[0];

    cv::Mat all_queries_indices(num_queries, k, CV_32SC1);;
    cv::Mat all_queries_dists(num_queries, k, CV_32FC1);;

    flann_index->knnSearch(
            query_descs,
            all_queries_indices,
            all_queries_dists,
            k,
            *search_params
    );

    matches.resize(num_queries);

    for (int query_ind = 0; query_ind < num_queries; ++query_ind) {
        cv::Mat indices = all_queries_indices.row(query_ind);
        cv::Mat dists = all_queries_dists.row(query_ind);

        for (int match_ind = 0; match_ind < k; ++match_ind) {
            matches[query_ind].emplace_back(
                    query_ind,
                    indices.at<int32_t>(match_ind),
                    std::sqrt(dists.at<float>(match_ind))
            );
        }
    }
}
