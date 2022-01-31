#include "bundleadjustment.hpp"


// --------------- BundleAdjustment ---------------
void BundleAdjustment::set_focals(const double focal) 
{
    for (auto& cam : cameras) {
        cam.focal(focal);
    }
}
void BundleAdjustment::set_centers(const point2& center) 
{
    for (auto& cam : cameras) {
        cam.center(center);
    }
}

void BundleAdjustment::update_intrinsic(const double focal, const point2& center)
{
    for (auto& cam : cameras) {
        cam.focal(focal);
        cam.center(center);
    }
}

void BundleAdjustment::update_extrinsic(const vector<vec3>& rvec, const vector<vec3>& tvec)
{
    assert(rvec.size() == tvec.size());
    assert(rvec.size() == cameras.size());

    for (size_t i = 0; i < cameras.size(); ++i) {
        cameras[i].rotate(rvec[i]);
        cameras[i].translate(tvec[i]);
    }   
}

vector<vec6> BundleAdjustment::get_cameras_pose()
{
    vector<vec6> cam_poses;
    for (const auto& cam : cameras) {
        cam_poses.push_back(cam.Rt());
    }
    return cam_poses;
}

ceres::CostFunction* BundleAdjustment::getCostFunction(const TypeReprojection typereproj, const point2& pt, const int32_t idx)
{
    switch (typereproj) 
    {
        case TypeReprojection::BA_REPROJECTION_RT:
            return CeresReprojectionErrorRt::create(pt, cameras[idx].focal(), cameras[idx].center());
        case TypeReprojection::BA_REPROJECTION_P:
            return CeresReprojectionErrorP::create(pt, cameras[idx].center());
        default:
            throw SfMException("Error: Unknown type of reprojection error!");
    }
}

void BundleAdjustment::initProblem(const double focal, const point2& center) 
{
    optim_problem = boost::make_shared<ceres::Problem>();
    update_intrinsic(focal, center);
}

void BundleAdjustment::formulateProblem(const point2& pt2D, 
                                              point3& pt3D,
                                              const int32_t cam_idx,
                                              const double loss_width,
                                              const TypeReprojection typereproj)
{
    ceres::CostFunction* cost_f = getCostFunction(typereproj, pt2D, cam_idx);
    ceres::LossFunction* loss_f = loss_width > 0 ? new ceres::CauchyLoss(loss_width) : nullptr;
    double* cam_param = cameras[cam_idx].rawP();
    double* P = pt3D.data();
    optim_problem->AddResidualBlock(cost_f, loss_f, cam_param, P);
}

void BundleAdjustment::solveProblem(const int32_t ba_iteration,
                                          const int32_t num_threads, 
                                          const bool fullreport)
{
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.num_threads = num_threads;
    if (ba_iteration > 0) 
        options.max_num_iterations = ba_iteration;
    options.minimizer_progress_to_stdout = true; // false

    ceres::Solver::Summary summary;
    ceres::Solve(options, optim_problem.get(), &summary);

    if (fullreport)
        std::cout << summary.FullReport() << "\n";

}
