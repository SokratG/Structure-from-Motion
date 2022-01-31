#pragma once
#include "pch.hpp"


// Linear Least Square Model
class LLSModelRansac
{
public:
    LLSModelRansac(const int32_t in_columns_, const int32_t out_columns_) 
    {
        for (int32_t i = 0; i < in_columns_; ++i) {
		    in_columns.push_back(i);
	    }
	    for (int32_t i = in_columns_; i < (out_columns_+1); ++i) {
		    out_columns.push_back(i);
	    }
    }
    boost::tuple<Eigen::MatrixXf, vector<int32_t>> linefit(const Eigen::MatrixXf& data, 
                const int32_t max_iter, const int32_t min_num_data,
                const int32_t threshold, const int32_t d);
protected:
    Eigen::MatrixXf fit(const Eigen::MatrixXf& data); 
    Eigen::MatrixXf error(const Eigen::MatrixXf& data, const Eigen::MatrixXf& model);
private:
    vector<int32_t> filter_idxs(const vector<int32_t>& test_idxs, const Eigen::MatrixXf& data, const int32_t threshold);
private:
    vector<int32_t> in_columns;
    vector<int32_t> out_columns;
};