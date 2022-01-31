#pragma once
#include "pch.hpp"
#include "camera.hpp"

enum class TypeReprojection {
    BA_REPROJECTION_RT = 0,
    BA_REPROJECTION_P = 1,
    BA_UNKNOWN
};

// --------------- CeresReprojectionErrorP ---------------
// Reprojection error camera matrix P = K[R t] -> 7 degree of freedom
struct CeresReprojectionErrorP 
{
public:
    using ReprojectionError = CeresReprojectionErrorP; 
public:
    CeresReprojectionErrorP(const point2& pt_, const point2& center_) :
                            pt(pt_), center(center_) {}
    template<typename T>
    bool operator()(const T* camera, const T* point, T* residuals) const {
        // project to camera coordinate system: P' = R*P + t
        T P[3]; 
        ceres::AngleAxisRotatePoint(camera, point, P); 
        P[0] += camera[3];
        P[1] += camera[4];
        P[2] += camera[5];  

        // project to camera image: p = K * P'
        const T& focal = camera[6];
        T px = focal * P[0] / P[2] + center.x();
        T py = focal * P[1] / P[2] + center.y();

        // residual: p - p'
        residuals[0] = pt.x() - px;
        residuals[1] = pt.y() - py;
        return true;
    }
    static ceres::CostFunction* create(const point2& pt_, const point2& center_) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 7, 3>(new ReprojectionError(pt_, center_)));
    }
private:
    point2 pt;
    point2 center; // image center
};


// --------------- CeresReprojectionErrorRt ---------------
// Reprojection error camera coordinate system matrix [R t] -> 6 degree of freedom
struct CeresReprojectionErrorRt
{
public:
    using ReprojectionError = CeresReprojectionErrorRt; 
public:
    CeresReprojectionErrorRt(const point2& pt_, const double focal_, const point2& center_) :
                            pt(pt_), focal(focal_), center(center_) {}
    template<typename T>
    bool operator()(const T* camera, const T* point, T* residuals) const {
        // project to camera coordinate system: P' = R*P + t
        T P[3]; 
        ceres::AngleAxisRotatePoint(camera, point, P); 
        P[0] += camera[3];
        P[1] += camera[4];
        P[2] += camera[5];  
        // project to camera image: p = K * P'
        T px = focal * P[0] / P[2] + center.x();
        T py = focal * P[1] / P[2] + center.y();

        // residual: p - p'
        residuals[0] = pt.x() - px;
        residuals[1] = pt.y() - py;
        return true;
    }
    static ceres::CostFunction* create(const point2& pt_, const double focal_, const point2& center_) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(new ReprojectionError(pt_, focal_, center_)));
    }
private:
    point2 pt;
    double focal; // focal length
    point2 center; // image center
};


// --------------- BundleAdjustment ---------------
class BundleAdjustment
{
public:
    BundleAdjustment(const int32_t n_cameras_) : n_cameras(n_cameras_)
    {
        Camera dummy(0., point2(0., 0.), vec6(0., 0., 0., 0., 0., 0.));
        cameras = std::move(vector<Camera>(number_views(), dummy));
    }

    int32_t number_views() const {return n_cameras;}

    void update_cameras(const vector<Camera>& cams) {cameras = cams;}
    void set_focals(const double focal);
    void set_centers(const point2& center);
    void update_intrinsic(const double focal, const point2& center);
    void update_extrinsic(const vector<vec3>& rvec, const vector<vec3>& tvec);

    vector<Camera> get_cameras() {return cameras;}
    vector<vec6> get_cameras_pose();

    void initProblem(const double focal, const point2& center); 
    void formulateProblem(const point2& pt2D, 
                          point3& pt3D,
                          const int32_t cam_idx,
                          const double loss_width,
                          const TypeReprojection typereproj=TypeReprojection::BA_REPROJECTION_RT);

    void solveProblem(const int32_t ba_iteration=0,
                      const int32_t num_threads=1, 
                      const bool fullreport=false);

protected:
    ceres::CostFunction* getCostFunction(const TypeReprojection typereproj, const point2& pt, const int32_t idx);
private:
    int32_t n_cameras; // number of views 
    boost::shared_ptr<ceres::Problem> optim_problem;
protected:
    vector<Camera> cameras;
};
