#pragma once
#include "pch.hpp"

// --------------- Camera ---------------
class Camera
{
public:
    Camera() {}
    Camera(const double focal, const point2& center, const vec6& Rt, const float aspect_ratio=1.0)
    {
        // Euler angle
        parameters[0] = Rt(0);
        parameters[1] = Rt(1);
        parameters[2] = Rt(2);
        // translation
        parameters[3] = Rt(3);
        parameters[4] = Rt(4);
        parameters[5] = Rt(5);
        // intrinsic
        parameters[6] = focal;
        parameters[7] = center.x();
        parameters[8] = center.y();
        
        // aspect ratio
        aspect = aspect_ratio;
    }
    double focal() const {return parameters[6];}
    double focal(const double focal) {return parameters[6]=focal;}

    float aspect_ratio() const {return aspect;}
    float aspect_ratio(const float aspect_ratio) {aspect=aspect_ratio; return aspect;}

    point2 center() const {return point2(parameters[7], parameters[8]);}
    point2 center(const point2 center) {parameters[7] = center.x(), parameters[8] = center.y(); return this->center();}

    vec3 rotate() const {return vec3(parameters[0], parameters[1], parameters[2]);}
    vec3 rotate(const vec3 rotate) {
        parameters[0] = rotate(0);
        parameters[1] = rotate(1);
        parameters[2] = rotate(2);
        return this->rotate();
    }

    vec3 translate() const {return vec3(parameters[3], parameters[4], parameters[5]);}
    vec3 translate(const vec3 translate) {
        parameters[3] = translate(0);
        parameters[4] = translate(1);
        parameters[5] = translate(2);
        return this->translate();
    }
    
    vec6 Rt() const {return vec6(parameters[0], parameters[1], parameters[2], 
                                 parameters[3], parameters[4], parameters[5]);}
    cv::Mat rotateMat() const;
    cv::Mat K() const;
    cv::Mat P() const; // camera projection matrix
    
    double* rawP() {
        return &(parameters[0]);
    }

    vec9 vecP() {
        return vec9(parameters[0], parameters[1], parameters[2],
                    parameters[3], parameters[4], parameters[5],
                    parameters[6], parameters[7], parameters[8]);
    }

    void updatePose(const vec3& rvec, const vec3& tvec);
    void updatePose(const cv::Mat& rvec, const cv::Mat& tvec);
private:
    // TODO add distortion coeff
    double parameters[9];
    float aspect;
};