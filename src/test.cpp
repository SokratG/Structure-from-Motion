#include "test.hpp"
#include "utils.hpp"
#include "linefit.hpp"


void test_sfm(const std::string& filepath, const std::string& outname, const std::string meshname)
{
	SfMParams param;
	param.readFromFile(filepath);
	
	SfM sfm(param);
	
	vector<point3> pts3D;
	vector<vector<point2>> pts2D;
    vector<vec3> colors;

	int32_t n_views = sfm.init(pts2D, pts3D, colors);

	auto mesh = sfm.run(n_views, pts2D, pts3D, colors);

    sfm.save2Ply(mesh, outname, meshname);
}
