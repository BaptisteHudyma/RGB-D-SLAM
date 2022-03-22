// Copyright (c) 2020, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#include <Eigen/Dense>
#include "p3p.hpp"
#include <iostream>
#include <random>


#define TEST(FUNC) if(!FUNC()) { std::cout << #FUNC"\033[1m\033[31m FAILED!\033[0m\n"; } else { std::cout << #FUNC"\033[1m\033[32m PASSED!\033[0m\n"; passed++;} num_tests++; 
#define REQUIRE(COND) if(!(COND)) { std::cout << "Failure: "#COND" was not satisfied.\n"; return false; }


using namespace lambdatwist;

void setup_instance(std::vector<Eigen::Vector3d> &x, std::vector<Eigen::Vector3d> &X, CameraPose &pose_gt) {
    pose_gt.R = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    pose_gt.t.setRandom();

    std::uniform_real_distribution<double> unif_depths(0.1, 10);
    std::default_random_engine re;


    for(int i = 0; i < 3; ++i) {
        Eigen::Vector3d z = Eigen::Vector3d::Random();
        z(2) = 1.0;
        z.normalize();

        double depth = unif_depths(re);

        Eigen::Vector3d Z = depth * z;

        Z = pose_gt.R.transpose() * (depth*z - pose_gt.t);

        x.push_back(z);
        X.push_back(Z);
    }
}

bool is_valid(const CameraPose &pose, const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X) {
    for(int k = 0; k < x.size(); ++k) {
        Eigen::Vector3d z = pose.R * X[k] + pose.t;
        double res = std::abs(1 - x[k].normalized().dot(z.normalized()));
        if(res > 1e-8) {        
            return false;
        }
    }
    return true;
}

bool test_simple_instance() {
    Eigen::Vector3d y1, y2, y3, X1, X2, X3;

    y1(0) = 0.0; y1(1) = 0.0; y1(2) = 1.0;
    y2(0) = 1.0; y2(1) = 0.0; y2(2) = 1.0;
    y3(0) = 2.0; y3(1) = 1.0; y3(2) = 1.0;

    y1.normalize();
    y2.normalize();
    y3.normalize();

    X1(0) = 0.0; X1(1) = 0.0; X1(2) = 2.0;
    X2(0) = 1.41421356237309; X2(1) = 0.0; X2(2) = 1.41421356237309;
    X3(0) = 1.63299316185545; X3(1) = 0.816496580927726; X3(2) =0.816496580927726;

    std::vector<Eigen::Vector3d> x{y1,y2,y3};
    std::vector<Eigen::Vector3d> X{X1,X2,X3};


    std::vector<CameraPose> poses;

    p3p(x,X, &poses);

    for(CameraPose &pose : poses) {
        if(!is_valid(pose,x,X))
            return false;
        double err_R = (pose.R - Eigen::Matrix3d::Identity()).norm();
        double err_t = (pose.t - Eigen::Vector3d::Zero()).norm();

        if( err_R < 1e-8 && err_t < 1e-8)
            return true;

    }

    return false;

}

bool test_random_instance() {
    std::vector<Eigen::Vector3d> x;
    std::vector<Eigen::Vector3d> X;
    CameraPose pose_gt;

    std::vector<CameraPose> poses;

    setup_instance(x, X, pose_gt);

    int n_sols = p3p(x, X, &poses);

    REQUIRE( n_sols > 0 );

    for(CameraPose &pose : poses) {
        if(!is_valid(pose,x,X))
            return false;
        double err_R = (pose.R - pose_gt.R).norm();
        double err_t = (pose.t - pose_gt.t).norm();

        if( err_R < 1e-8 && err_t < 1e-8)
            return true;
    }


    return false;

}


// Test case related to issue
// https://github.com/vlarsson/lambdatwist/issues/1
bool test_simple_instance2() {
    Eigen::Vector3d y1, y2, y3, X1, X2, X3;

    y1 << 0.0, 0.0, 1.0;
    y2 << 2.0, 0.0, 1.0;
    y3 << 0.0, 2.0, 1.0;

    y1.normalize();
    y2.normalize();
    y3.normalize();

    X1 << 0.0, 0.0, 0.0;
    X2 << 1.0, 0.0, 0.0;
    X3 << 0.0, 1.0, 0.0;

    std::vector<Eigen::Vector3d> x{y1,y2,y3};
    std::vector<Eigen::Vector3d> X{X1,X2,X3};


    std::vector<CameraPose> poses;

    p3p(x,X, &poses);

    Eigen::Matrix3d R_gt;
    R_gt.setIdentity();
    Eigen::Vector3d t_gt;
    t_gt << 0.0, 0.0, 0.5;

    for(CameraPose &pose : poses) {
        if(!is_valid(pose,x,X))
            return false;
        double err_R = (pose.R - R_gt).norm();
        double err_t = (pose.t - t_gt).norm();

        if( err_R < 1e-8 && err_t < 1e-8)
            return true;        
    }

    return false;
}

int main() {
    unsigned int seed = (unsigned int)time(0);	
    srand(seed);

    std::cout << "Running tests... (seed = " << seed << ")\n\n";

    int passed = 0;
    int num_tests = 0;

    for(int i = 0; i < 10; ++i) {
        TEST(test_random_instance);
    }

    TEST(test_simple_instance);

    TEST(test_simple_instance2);

    std::cout << "\nDone! Passed " << passed << "/" << num_tests << " tests.\n";
}
