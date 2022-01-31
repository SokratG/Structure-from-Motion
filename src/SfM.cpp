#include "SfM.hpp"
#include "bundleadjustment.hpp"
#include "utils.hpp"

#include <boost/range/algorithm_ext.hpp>

#include <vector> 
#include <filesystem>

using Graph = typename SparseStructureFromMotion::Graph;


// --------------- StructureFromMotionParams ---------------
StructureFromMotionParams::StructureFromMotionParams(const std::string& fileJSONname)
{
    bool res = readFromFile(fileJSONname);
    if (res == false)
        throw SfMException("Error: json file with Sfm parameters doesn't exist");
}


bool StructureFromMotionParams::readFromFile(const std::string& fileJSONname)
{
    std::ifstream ifs{fileJSONname};
    if (!ifs.is_open()) 
        return false;

    nlohmann::json jparam = nlohmann::json::parse(ifs)["SFM_Parameters"];
    
    focal = std::stod(jparam["focal_length"].get<std::string>());
	cx = std::stod(jparam["center_x"].get<std::string>()); 
    cy = std::stod(jparam["center_y"].get<std::string>());

	depth_init = std::stod(jparam["depth_inititially"].get<std::string>()); 
    depth_limit = std::stod(jparam["depth_limit"].get<std::string>());

	min_inlier = std::stoi(jparam["min_inlier_points"].get<std::string>()); 
    ba_max_iteration = std::stoi(jparam["BA_max_iteration"].get<std::string>());
	ba_loss_width = std::stod(jparam["BA_loss_width"].get<std::string>()); 

	brisk_threshold = std::stoi(jparam["brisk_threshold"].get<std::string>()); 
    n_octave = std::stoi(jparam["brisk_number_octave"].get<std::string>());
	patternScale = std::stod(jparam["brisk_pattern_scale"].get<std::string>());

	scale_img = std::stod(jparam["scale_images"].get<std::string>());
	ransac_threshold = std::stod(jparam["RANSAC_threshold"].get<std::string>());
	n_threads = std::stoi(jparam["number_threads"].get<std::string>());

    data_path = jparam["data_path"].get<std::string>();

    return true;
}


// --------------- VisibilityNode ---------------
size_t VisibilityNode::genKey(const uint32_t cam_idx, const uint32_t pt_idx)
{
    size_t seed = 0;
    boost::hash_combine(seed, cam_idx);
    boost::hash_combine(seed, pt_idx);
    return seed;
}


// --------------- SparseStructureFromMotion ---------------
bool SparseStructureFromMotion::setupImageDataset()
{
    boost::container::set<std::string> files;

    const std::filesystem::path dir{params.data_path};

    for (const auto& entry : std::filesystem::directory_iterator(dir, 
         std::filesystem::directory_options::skip_permission_denied)) {

        const std::string ext = entry.path().extension();
        
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".tiff") {
                continue;
        }

		const std::filesystem::file_status ft(status(entry));
		const auto type = ft.type();
        

		if (type == std::filesystem::file_type::directory ||
            type == std::filesystem::file_type::fifo || 
            type == std::filesystem::file_type::socket ||
            type == std::filesystem::file_type::unknown) {
			continue;
        }
		else {
			files.insert(canonical(entry.path()).string());
		}
    }

    if (files.size() < 2) 
        return false;

    for (const auto& filename : files) {
        cv::Mat img = cv::imread(filename, cv::IMREAD_UNCHANGED);

        if (img.empty()) 
			throw SfMException("Error: can't read image file");

        sequence_imgs.emplace_back(img);
    }  

	if (sequence_imgs.size() < 2) 
        return false;
    
    return true;
}


void SparseStructureFromMotion::featurePointsBuild(std::vector<std::vector<cv::KeyPoint>>& key_pts,
                                             std::vector<cv::Mat>& imgs, 
                                             std::vector<cv::Mat>& descriptors) const
{
    cv::Ptr<cv::FeatureDetector> feature_detector = cv::BRISK::create(params.brisk_threshold, params.n_octave, params.patternScale);    
    for (const auto& img : sequence_imgs) {
        cv::Mat resize_img, descriptor;
        std::vector<cv::KeyPoint> keypoints;
       
        cv::resize(img, resize_img, cv::Size(), params.scale_img, params.scale_img);
        
        feature_detector->detectAndCompute(resize_img, cv::noArray(), keypoints, descriptor);

        imgs.emplace_back(resize_img);
        descriptors.emplace_back(descriptor);
        key_pts.emplace_back(keypoints);   
    }
    
}


boost::tuple<std::vector<std::pair<uint32_t, uint32_t>>, std::vector<std::vector<cv::DMatch>>>
SparseStructureFromMotion::featurePointsMatching(std::vector<std::vector<cv::KeyPoint>>& key_pts, 
                                                const std::vector<cv::Mat>& resized_imgs,
                                                const std::vector<cv::Mat>& descriptors) const 
{    
    cv::Ptr<cv::DescriptorMatcher> dmatcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    std::vector<std::pair<uint32_t, uint32_t>> match_adjacent;
    std::vector<std::vector<cv::DMatch>> match_inlier_pts;
    const int32_t size = resized_imgs.size();

    for (int32_t i = 0; i < size; ++i) {
        for (int32_t j = i + 1; j < size; ++j) {
            // match pair for two adjacent images
            std::vector<cv::DMatch> matches, inliers;
            dmatcher->match(descriptors[i], descriptors[j], matches);
            
            std::vector<cv::Point2d> src, dst;
            for (auto m_it = matches.begin(); m_it != matches.end(); m_it++) {
                src.emplace_back(key_pts[i][m_it->queryIdx].pt);
                dst.emplace_back(key_pts[j][m_it->trainIdx].pt);
            }

            cv::Mat inlier_mask; // mask inlier points -> "status": 0 - outlier, 1 - inlier
            cv::findFundamentalMat(src, dst, inlier_mask, cv::FM_RANSAC, params.ransac_threshold);
            //std::cout << cv::countNonZero(inlier_mask) << "\n";
            for (int32_t k = 0; k < inlier_mask.rows; ++k) {
                if (inlier_mask.at<char>(k))
                    inliers.emplace_back(matches[k]);
            }
            
            if (inliers.size() < params.min_inlier)
                continue;

            match_adjacent.emplace_back(std::make_pair(i, j));
            
            match_inlier_pts.emplace_back(inliers);
        
        }
    }
    
    return boost::make_tuple(match_adjacent, match_inlier_pts);
}


boost::tuple<vector<vector<cv::DMatch>>, vector<std::pair<uint32_t, uint32_t>>, 
             vector<vector<cv::KeyPoint>>, vector<cv::Mat>> 
SparseStructureFromMotion::buildFeatures()
{
    if (sequence_imgs.size() < 2) 
        throw SfMException("Error: Not enough images in data set!");
    
    std::vector<std::vector<cv::KeyPoint>> key_pts;
    std::vector<cv::Mat> resized_imgs_, descriptors;

    featurePointsBuild(key_pts, resized_imgs_, descriptors);
    
    if (params.cx < 0 || params.cy < 0) {
        params.cx = resized_imgs_.at(0).cols / 2;
        params.cy = resized_imgs_.at(0).rows / 2;
    }

    auto matches = featurePointsMatching(key_pts, resized_imgs_, descriptors);
    std::vector<std::pair<uint32_t, uint32_t>> match_adj_ = matches.get<0>(); 
    
    std::vector<std::vector<cv::DMatch>> match_pts_ = matches.get<1>();
    
    if (match_adj_.size() < 1) 
        throw SfMException("Error: Not enough adjacent image feature points in data set!");

    vector<vector<cv::DMatch>> matches_inlier;
    for (const auto& m : match_pts_) {
        vector<cv::DMatch> vmatch{m.begin(), m.end()};
        matches_inlier.emplace_back(vmatch);
    }
    
    vector<vector<cv::KeyPoint>> keypoints;
    for (const auto& key : key_pts) {
        vector<cv::KeyPoint> vkey{key.begin(), key.end()};
        keypoints.emplace_back(vkey);
    } 

    vector<std::pair<uint32_t, uint32_t>> match_adj{match_adj_.begin(), match_adj_.end()};
    vector<cv::Mat> resized_imgs{resized_imgs_.begin(), resized_imgs_.end()};

    return boost::make_tuple(matches_inlier, match_adj, keypoints, resized_imgs);
}


Graph SparseStructureFromMotion::initPoints(const vector<std::pair<uint32_t, uint32_t>>& match_adj,
                                            const vector<vector<cv::DMatch>>& match_inlier,
                                            const vector<cv::Mat>& imgs,
                                            const vector<vector<cv::KeyPoint>>& key_pts,
                                            vector<point3>& pts3D, vector<rgb>& colors)
{
    const double z = params.depth_init;
    Graph localVisibilityGraph;
    
    for (int32_t i = 0; i < match_adj.size(); ++i) {
        for (int32_t j = 0; j < match_inlier[i].size(); ++j) {

            uint32_t cam1_idx = match_adj[i].first; 
            uint32_t cam2_idx = match_adj[i].second;

            uint32_t p1_idx = match_inlier[i][j].queryIdx; 
            uint32_t p2_idx = match_inlier[i][j].trainIdx;

            size_t key1 = VisibilityNode::genKey(cam1_idx, p1_idx); 
            size_t key2 = VisibilityNode::genKey(cam2_idx, p2_idx); 
            
            auto visitView1 = localVisibilityGraph.find(key1);
            auto visitView2 = localVisibilityGraph.find(key2);
            if (visitView1 != localVisibilityGraph.end() && visitView2 != localVisibilityGraph.end()) {
                
                // remove previous view points
                if (visitView1->second->pt3d_idx() != visitView2->second->pt3d_idx()) {
                    localVisibilityGraph.erase(visitView1);
                    localVisibilityGraph.erase(visitView2);
                }
                continue;
            } 
            
            uint32_t visibility_idx = 0;
            if (visitView1 != localVisibilityGraph.end())
                visibility_idx = visitView1->second->pt3d_idx();
            else if (visitView2 != localVisibilityGraph.end())
                visibility_idx = visitView2->second->pt3d_idx();
            else {
                // add new point for current view 
                visibility_idx = pts3D.size();
                pts3D.emplace_back(point3(0., 0., z));
                cv::Vec3b rgb = imgs[cam1_idx].at<cv::Vec3b>(key_pts[cam1_idx][p1_idx].pt);
                colors.emplace_back(vec3(rgb[0], rgb[1], rgb[2]));
            }

            if (visitView1 == localVisibilityGraph.end())
                localVisibilityGraph[key1] = boost::make_shared<VisibilityNode>(cam1_idx, p1_idx, visibility_idx);
            if (visitView2 == localVisibilityGraph.end())
                localVisibilityGraph[key2] = boost::make_shared<VisibilityNode>(cam2_idx, p2_idx, visibility_idx);   
        }
    }

    return localVisibilityGraph;
}


int32_t SparseStructureFromMotion::init(vector<vector<point2>>& pts2D, vector<point3>& pts3D, vector<rgb>& colors)
{
    bool res = setupImageDataset();
    if (res == false)
        throw SfMException("Error: not enough image in data set");

    auto packFeature = buildFeatures();

    vector<vector<cv::DMatch>> matches_inlier = packFeature.get<0>();
    vector<std::pair<uint32_t, uint32_t>> match_adj = packFeature.get<1>(); 
    vector<vector<cv::KeyPoint>> keypoints = packFeature.get<2>();
    vector<cv::Mat> resized_imgs = packFeature.get<3>();  

    // init array of 3D points with color and graph visibility points 
    visibilityGraph = initPoints(match_adj, matches_inlier, resized_imgs, keypoints, pts3D, colors);

    for (size_t i = 0; i < keypoints.size(); ++i) {
        vector<point2> tmp_pts;
        for (const auto& k_pt : keypoints[i]) {
            auto pt = k_pt.pt;
            tmp_pts.emplace_back(point2(pt.x, pt.y));
        }
        pts2D.emplace_back(tmp_pts);
    }

    const int n_views = resized_imgs.size();

    return n_views;
}


vector<bool> SparseStructureFromMotion::filterNoisePts(const vector<point3>& pts3D, 
                                                       const vector<vector<point2>>& pts2D, 
                                                       const vector<Camera>& cameras,
                                                       const double reproj_error)
{
    vector<bool> is_noisy(pts3D.size(), false);

    if (reproj_error < 0)
        return is_noisy;
    
    for (const auto& node : visibilityGraph)
    {
        const point3 P3D = pts3D[node.second->pt3d_idx()];

        if (P3D.z() < 0) 
            continue;

        int32_t cam_idx = node.second->cam_idx();
        int32_t pt_idx = node.second->pt2d_idx();
        const point2 x = pts2D[cam_idx][pt_idx];
        
        // Project the P
        Eigen::Matrix3d R, K;
        cv::cv2eigen(cameras[cam_idx].rotateMat(), R);
        cv::cv2eigen(cameras[cam_idx].K(), K);

        point3 P = R * P3D + cameras[cam_idx].translate();
        point3 p = K * P;
        point2 x_p = point2(p.x() / p.z(), p.y() / p.z());
       
        // Calculate reprojection error -> x and x_p
        point2 d = x - x_p;
        if ((d.x() * d.x() + d.y() * d.y()) > reproj_error) 
            is_noisy[node.second->pt3d_idx()] = true;
    }
    
    return is_noisy;
}


vector<vec6> SparseStructureFromMotion::run(const int n_views, const vector<vector<point2>>& pts2D,
                                            const vector<point3>& pts3D, const vector<rgb>& colors)
{   
    if (n_views < 2) 
        throw SfMException("Error: not enough views for run SfM pipeline");
    
    BundleAdjustment ba(n_views);
    ba.initProblem(params.focal, point2(params.cx, params.cy));

    vector<point3> localPts3D = pts3D;
    // compute camera poses and 3D point position
    for (const auto& node : visibilityGraph) {
        uint32_t cam_idx = node.second->cam_idx();
        uint32_t pt_idx = node.second->pt2d_idx(); 
        uint32_t pt3d_idx = node.second->pt3d_idx();
        point2 pt = pts2D[cam_idx][pt_idx];
        ba.formulateProblem(pt, localPts3D[pt3d_idx], cam_idx, params.ba_loss_width);
    }
    
    ba.solveProblem(params.ba_max_iteration, params.n_threads, false);
    vector<bool> noise_pts = filterNoisePts(localPts3D, pts2D, ba.get_cameras(), params.ba_loss_width);

    vector<point3> pts3D_;
    vector<rgb> color_;
    for (size_t i = 0; i < localPts3D.size(); ++i) {
        if (localPts3D[i].z() < -params.depth_limit || localPts3D[i].z() > params.depth_limit || noise_pts[i])
            continue;
        pts3D_.emplace_back(localPts3D[i]);
        color_.emplace_back(colors[i]);
    }
    
    vector<vec6> mesh = prepareMesh(pts3D_, color_);
    return mesh;
}


void SparseStructureFromMotion::save2Ply(const vector<vec6>& mesh, const std::string& filename, 
                                         const std::string& meshname)
{
    if (filename.empty()) 
        throw SfMException("Error: empty file name");
    savePlyMesh(mesh, filename, meshname);
}
