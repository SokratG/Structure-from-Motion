#include "linefit.hpp"
#include "utils.hpp"

#include <numeric>
#include <stdexcept>

#include <boost/range/algorithm_ext.hpp>

Eigen::MatrixXf LLSModelRansac::fit(const Eigen::MatrixXf& data)
{
    Eigen::MatrixXf A(data.rows(), data.cols()-1);
    Eigen::MatrixXf b(data.rows(), data.cols()-1);
    
    for (const int32_t i : in_columns) {
        A.col(i) = data.col(i);   
    }
    
    int32_t j = 0;
    for (const int32_t i : out_columns) {
        b.col(j) = data.col(i); 
        j += 1;  
    }

    // solve lls 
    Eigen::MatrixXf res = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    
    return res;
}

Eigen::MatrixXf LLSModelRansac::error(const Eigen::MatrixXf& data, const Eigen::MatrixXf& model)
{
    Eigen::MatrixXf A(data.rows(), data.cols()-1);
    Eigen::MatrixXf b(data.rows(), data.cols()-1);
    
    for (const int32_t i : in_columns) {
        A.col(i) = data.col(i);   
    }
    
    int32_t j = 0;
    for (const int32_t i : out_columns) {
        b.col(j) = data.col(i); 
        j += 1;  
    }
    Eigen::MatrixXf b_fit = A * model; 
    Eigen::MatrixXf err = (b - b_fit).array().pow(2).rowwise().sum();

    return err;
}


boost::tuple<Eigen::MatrixXf, vector<int32_t>> LLSModelRansac::linefit(
                const Eigen::MatrixXf& data, 
                const int32_t max_iter, const int32_t min_num_data,
                const int32_t threshold, const int32_t d)
{
    const int32_t size = data.rows();
    float test_error = std::numeric_limits<float>::infinity();
    Eigen::MatrixXf best_model_fit;
    vector<int32_t> best_idxs;

    //auto model = this->fit(data);
	//this->error(data, model);
    for (int32_t it = 0; it < max_iter; ++it) {
        boost::tuple<vector<int32_t>, vector<int32_t>> idxs = rand_partition(min_num_data, size);
       
        // hypothesis inliers
        Eigen::MatrixXf hyph_inliers = data(boost::get<0>(idxs), Eigen::all);
        Eigen::MatrixXf test_pts = data(boost::get<1>(idxs), Eigen::all);
        Eigen::MatrixXf model = fit(hyph_inliers);
        Eigen::MatrixXf hyph_error = error(test_pts, model);
        vector<int32_t> idxs_ = filter_idxs(boost::get<1>(idxs), hyph_error, threshold);
        Eigen::MatrixXf inliers_ = data(idxs_, Eigen::all);
        if (inliers_.size() > d) {
            Eigen::MatrixXf best_data(inliers_.rows() + hyph_inliers.rows(), hyph_inliers.cols());
            best_data << hyph_inliers, inliers_;
            Eigen::MatrixXf best_model = fit(best_data);
            Eigen::MatrixXf hyph_error = error(best_data, best_model);
            float tmp_err = hyph_error.mean();
            if (tmp_err < test_error) {
                test_error = tmp_err;
                best_model_fit = best_model;
                vector<int32_t> tmp_idx;
                boost::range::push_back(tmp_idx, boost::get<0>(idxs));
                boost::range::push_back(tmp_idx, idxs_);
                best_idxs = tmp_idx;
            }
        }
    }

    if (best_model_fit.size() == 0) {
        throw std::runtime_error("Error: algorithm not converge");
    }

    return boost::make_tuple(best_model_fit, best_idxs);
}

vector<int32_t> LLSModelRansac::filter_idxs(const vector<int32_t>& test_idxs, 
                                            const Eigen::MatrixXf& data, const int32_t threshold)
{
    vector<int32_t> res_idxs;
    for (int32_t i = 0; i < data.rows(); ++i) {
        if (data.row(i).value() < threshold) {
            res_idxs.push_back(test_idxs.at(i));
        }
    }
    
    return res_idxs;
}