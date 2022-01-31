#include "camera.hpp"
#include "utils.hpp"

// --------------- Camera ---------------
cv::Mat Camera::rotateMat() const
{
    return eigen2CvRotate(rotate());
}

cv::Mat Camera::K() const
{
    cv::Mat_<double> K = cv::Mat::eye(3, 3, CV_64F);
    K(0,0) = focal(); 
    K(1,1) = focal() * aspect; 
    point2 c = center();
    K(0,2) = c.x();
    K(1,2) = c.y();
    return K;
}

cv::Mat Camera::P() const
{
    cv::Mat K = this->K();
    cv::Mat t = eigen2CvTranlsate(translate());
    cv::Mat R = rotateMat();
    cv::Mat Rt;
    cv::hconcat(R, t, Rt);
    cv::Mat P = K * Rt;
    return P;
}

void Camera::updatePose(const vec3& rvec, const vec3& tvec)
{
    rotate(rvec);
    translate(tvec);
}

void Camera::updatePose(const cv::Mat& rvec, const cv::Mat& tvec)
{
    vec3 rvec_(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));
    vec3 tvec_(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    updatePose(rvec_, tvec_);
}