#include "utils.hpp"

#include <vector>
#include <array>

#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm_ext.hpp>
#include <boost/range/irange.hpp>

#include "happly.h"

static int32_t BUFSIZE = 1e5;


Eigen::MatrixXf readMatrix(const std::string& filename)
{
    int32_t cols = 0, rows = 0;
    double buff[BUFSIZE];
 
    std::ifstream infile(filename);

    while (!infile.eof()) {
        std::string line;
        std::getline(infile, line);

        int32_t temp_cols = 0;
        std::stringstream stream(line);

        while(!stream.eof())
            stream >> buff[cols*rows+temp_cols++];

        if (temp_cols == 0)
            continue;

        if (cols == 0)
            cols = temp_cols;

        rows++;
    }

    infile.close();
    rows--;

    Eigen::MatrixXf result(rows, cols);
    for (int32_t i = 0; i < rows; i++)
        for (int32_t j = 0; j < cols; j++)
            result(i,j) = buff[ cols*i+j ];

    return result;
};


boost::tuple<vector<int32_t>, vector<int32_t>> rand_partition(const int32_t n, const int32_t size)
{
    vector<int32_t> idxs;
    boost::range::push_back(idxs, boost::irange(0, size));
    boost::range::random_shuffle(idxs);
    vector<int32_t> idxs1{idxs.begin(), idxs.begin() + n};
    vector<int32_t> idxs2{idxs.begin() + n, idxs.end()};

    return boost::make_tuple(idxs1, idxs2);
}


cv::Mat eigen2CvRotate(const vec3& rotate)
{
    cv::Vec3d rvec;
    cv::eigen2cv(rotate, rvec);
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    return R;
}

cv::Mat eigen2CvTranlsate(const vec3& translate)
{
    cv::Mat tvec = (cv::Mat_<double>(3, 1) << translate(0), translate(1), translate(2));
    return tvec;
}


vector<vec6> prepareMesh(const vector<vec3>& pts, const vector<rgb>& colors)
{
    assert(pts.size() == colors.size());

    vector<vec6> mesh;
    
	for (const auto& vert : zip_range(pts, colors)) {
        point3 pt = vert.get<0>();
        vec3 color = vert.get<1>();
		vec6 vertex(pt.x(), pt.y(), pt.z(), color.x(), color.y(), color.z());
		mesh.push_back(vertex);
	}
    return mesh;
}


vector<vec6> prepareMesh(const vector<vec3>& pts, const vec3& color)
{
	vector<vec3> colors(pts.size(), color);
    return prepareMesh(pts, colors);
}

void savePlyMesh(const vector<vec6>& mesh, const std::string& filename, const std::string& nameobj)
{
    // Create an empty object
    happly::PLYData plyOut;

    // header comment
    if (!nameobj.empty()) {
        std::string header_description = "object: " + nameobj;
        std::vector<std::string> comments {header_description};
        plyOut.comments = comments;
    }
    
    // prepare elements
    std::vector<std::array<double, 3>> meshVertexPosition;
    std::vector<std::array<unsigned char, 3>> meshVertexColors;

    std::array<double, 3> vertbuff;
    std::array<unsigned char, 3> buffcolor; 
    for(const auto& vert : mesh) {
        boost::copy_n(vert.head(3).array(), 3, vertbuff.begin());  
        meshVertexPosition.push_back(vertbuff);
        boost::copy_n(vert.tail(3).array(), 3, buffcolor.begin()); 
        meshVertexColors.push_back(buffcolor);
    }
    
    // add elements data
    plyOut.addVertexPositions(meshVertexPosition);
    plyOut.addVertexColors(meshVertexColors);
    
    // save data
    plyOut.write(filename.c_str(), happly::DataFormat::ASCII);
}
