#pragma once
#include <sstream>
#include <iostream>
#include <fstream>
#include <utility>
#include <boost/unordered_map.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/container/vector.hpp>
#include <boost/container/string.hpp>
#include <boost/container/set.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/range/iterator_range_core.hpp>

using namespace boost::container;
using namespace boost::tuples;

#include <stdint.h>

#include <nlohmann/json.hpp>

#include <glog/logging.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "types.hpp"
#include "SfMException.hpp"


