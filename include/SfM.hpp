#pragma once
#include "pch.hpp"
#include "camera.hpp"

// --------------- StructureFromMotionParams ---------------
struct StructureFromMotionParams
{
    int32_t brisk_threshold; // theshold for BRISK feature detector
    int32_t n_octave; // number of octave
    float patternScale;
    double scale_img;
    int32_t min_inlier; // BA inliers
    int32_t ba_max_iteration; // number of BA iterations
    double ba_loss_width; // BA loss function 
    double ransac_threshold;
    double depth_init;
    double depth_limit;
    double cx;
    double cy;
    double focal;
    int32_t n_threads; // number of threads for BA
    std::string data_path;
    StructureFromMotionParams() {}
    StructureFromMotionParams(const std::string& fileJSONname);
    bool readFromFile(const std::string& fileJSONname);
};


// --------------- SparseStructureFromMotion ---------------
class VisibilityNode
{
public:
    VisibilityNode() {}
    VisibilityNode(const uint32_t cam_idx, const uint32_t pt2d_idx, const uint32_t pt3d_idx) :
                    cam_idx_(cam_idx), pt2d_idx_(pt2d_idx), pt3d_idx_(pt3d_idx) { }
    static size_t genKey(const uint32_t cam_idx, const uint32_t pt_idx);
    bool operator==(const VisibilityNode& rhs) {
        return (cam_idx_ == rhs.cam_idx_) && (pt2d_idx_ == rhs.pt2d_idx_) && (pt3d_idx_ == rhs.pt3d_idx_);
    }
    bool operator!=(const VisibilityNode& rhs) {
        return !((*this) == rhs);
    }

    uint32_t cam_idx() const {return cam_idx_;}
    uint32_t cam_idx(const uint32_t cam_idx) {cam_idx_=cam_idx; return cam_idx_;}

    uint32_t pt2d_idx() const {return pt2d_idx_;}
    uint32_t pt2d_idx(const uint32_t pt2d_idx) {pt2d_idx_=pt2d_idx; return pt2d_idx_;}

    uint32_t pt3d_idx() const {return pt3d_idx_;}
    uint32_t pt3d_idx(const uint32_t pt3d_idx) {pt3d_idx_=pt3d_idx; return pt3d_idx_;}

private:
    uint32_t cam_idx_;
    uint32_t pt2d_idx_;
    uint32_t pt3d_idx_;
};


class SparseStructureFromMotion 
{
public:
    using Graph = boost::unordered_map<size_t, boost::shared_ptr<VisibilityNode>>;

public:
    SparseStructureFromMotion(const StructureFromMotionParams& sfmparam) : params(sfmparam) 
    { }
    int32_t init(vector<vector<point2>>& pts2D, vector<point3>& pts3D, vector<vec3>& colors);
    vector<vec6> run(const int n_views, const vector<vector<point2>>& pts2D,
                    const vector<point3>& pts3D, const vector<rgb>& colors);
    void save2Ply(const vector<vec6>& mesh, const std::string& filename, const std::string& meshname);
    void parameters(const StructureFromMotionParams& sfmparam) {params = sfmparam;}
    StructureFromMotionParams parameters() const {return params;}
private:
    bool setupImageDataset();
    void featurePointsBuild(std::vector<std::vector<cv::KeyPoint>>& key_pts, 
                            std::vector<cv::Mat>& resized_imgs, 
                            std::vector<cv::Mat>& descriptors) const;

    boost::tuple<std::vector<std::pair<uint32_t, uint32_t>>, std::vector<std::vector<cv::DMatch>>>
        featurePointsMatching(std::vector<std::vector<cv::KeyPoint>>& key_pts, 
                               const std::vector<cv::Mat>& resized_imgs, 
                               const std::vector<cv::Mat>& descriptors) const;

    boost::tuple<vector<vector<cv::DMatch>>, vector<std::pair<uint32_t, uint32_t>>, 
                 vector<vector<cv::KeyPoint>>, vector<cv::Mat>> buildFeatures();

    Graph initPoints(const vector<std::pair<uint32_t, uint32_t>>& match_adj,
                     const vector<vector<cv::DMatch>>& match_inlier,
                     const vector<cv::Mat>& imgs,
                     const vector<vector<cv::KeyPoint>>& key_pts,
                     vector<point3>& pts3D, vector<rgb>& colors);
    
    vector<bool> filterNoisePts(const vector<point3>& pts3D, const vector<vector<point2>>& pts2D, 
                                const vector<Camera>& cameras, const double reproj_error2);

private:
    StructureFromMotionParams params;
    vector<cv::Mat> sequence_imgs; 
    Graph visibilityGraph;
};


// --------------- type alias ---------------
using SfM = SparseStructureFromMotion;
using SfMParams = StructureFromMotionParams;
