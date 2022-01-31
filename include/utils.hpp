#pragma once
#include "pch.hpp"


template<class... Conts>
auto zip_range(Conts&... conts)
  -> decltype(boost::make_iterator_range(
  boost::make_zip_iterator(boost::make_tuple(conts.begin()...)),
  boost::make_zip_iterator(boost::make_tuple(conts.end()...))))
{
  return {boost::make_zip_iterator(boost::make_tuple(conts.begin()...)),
          boost::make_zip_iterator(boost::make_tuple(conts.end()...))};
}

Eigen::MatrixXf readMatrix(const std::string& filename);

boost::tuple<vector<int32_t>, vector<int32_t>> rand_partition(const int32_t n, const int32_t size);

cv::Mat eigen2CvRotate(const vec3& rvec);

cv::Mat eigen2CvTranlsate(const vec3& translate);

vector<vec6> prepareMesh(const vector<vec3>& pts, const vector<rgb>& colors);
vector<vec6> prepareMesh(const vector<vec3>& pts, const vec3& color);

void savePlyMesh(const vector<vec6>& mesh, const std::string& filename, const std::string& nameobj);