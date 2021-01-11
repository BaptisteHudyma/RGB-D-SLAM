#include "CylinderSegment.hpp"

using namespace std;
using namespace Eigen;
using namespace planeDetection;

#include <iostream>

Cylinder_Segment::Cylinder_Segment(Cylinder_Segment& seg, int subRegionId) {
    //copy stored data
    this->radius.push_back(seg.radius[subRegionId]);
    this->centers.push_back(seg.centers[subRegionId]);
    this->axis[0] = seg.axis[0];
    this->axis[1] = seg.axis[1];
    this->axis[2] = seg.axis[2];
}

Cylinder_Segment::Cylinder_Segment(std::vector<std::unique_ptr<Plane_Segment>>& planeGrid, const bool* activatedMask, const int cellActivatedCount) {
    int samplesCount = planeGrid.size();

    this->segmentCount = 0;
    this->local2globalMap = new int[cellActivatedCount];

    Eigen::MatrixXd N(3, 2 * cellActivatedCount);
    Eigen::MatrixXd P(3, cellActivatedCount);

    // Init. P and N
    int j = 0;
    for(int i = 0; i < samplesCount; i++){
        if (activatedMask[i]){
            const Vector3d& planeNormal = planeGrid[i]->get_normal();
            const Vector3d& planeMean = planeGrid[i]->get_mean();
            N(0, j) = planeNormal[0];
            N(1, j) = planeNormal[1];
            N(2, j) = planeNormal[2];
            P(0, j) = planeMean[0];
            P(1, j) = planeMean[1];
            P(2, j) = planeMean[2];
            local2globalMap[j] = i;
            j++;
        }
    }
    // Concatenate [N -N]
    for(int i = 0; i < samplesCount; i++){
        if (activatedMask[i]){
            const Vector3d& planeNormal = planeGrid[i]->get_normal();
            N(0, j) = -planeNormal[0];
            N(1, j) = -planeNormal[1];
            N(2, j) = -planeNormal[2];
            j++;
        }
    }

    // Compute covariance
    Eigen::MatrixXd cov = (N * N.adjoint()) / double(N.cols() - 1);

    // PCA using QR decomposition for symmetric matrices
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    Eigen::Vector3d S = es.eigenvalues();
    double score = S(2)/S(0);

    // Checkpoint 1
    if(score < CYLINDER_SCORE_MIN){
        return;
    }

    Eigen::Vector3d vec = es.eigenvectors().col(0);
    axis[0] = vec(0); 
    axis[1] = vec(1); 
    axis[2] = vec(2);

    Eigen::MatrixXd NCpy = N.block(0, 0, 3, cellActivatedCount);
    N = NCpy; /* This avoids memory issues */
    Eigen::MatrixXd PProj(3, cellActivatedCount);

    // Projection to plane: P' = P-theta*<P.theta>
    Eigen::MatrixXd PDotTheta = vec.transpose() * P;
    PProj.row(0) = P.row(0) - PDotTheta * vec(0);
    PProj.row(1) = P.row(1) - PDotTheta * vec(1);
    PProj.row(2) = P.row(2) - PDotTheta * vec(2);
    Eigen::MatrixXd NDotTheta = vec.transpose() * N;
    N.row(0) -= NDotTheta * vec(0);
    N.row(1) -= NDotTheta * vec(1);
    N.row(2) -= NDotTheta * vec(2);

    // Normalize projected normals
    Eigen::MatrixXd normalsNorm = N.colwise().norm();
    N.row(0) = N.row(0).array() / normalsNorm.array();
    N.row(1) = N.row(1).array() / normalsNorm.array();
    N.row(2) = N.row(2).array() / normalsNorm.array();

    // Ransac params
    float pSuccess = 0.8;
    float w = 0.33;
    float K = log(1 - pSuccess) / log(1 - pow(w, 3));

    int mLeft = cellActivatedCount;
    MatrixXb idsLeftMask(1, cellActivatedCount);

    vector<int> idsLeft;
    for(int i = 0; i < cellActivatedCount; i++){
        idsLeft.push_back(i);
        idsLeftMask(i) = true;
    }
    // Sequential RANSAC main loop
    while(mLeft > 5 and mLeft > 0.1 * cellActivatedCount){
        Eigen::MatrixXd A, center, e1, e2;
        Eigen::MatrixXd D(1, cellActivatedCount);
        MatrixXb I(1, cellActivatedCount);
        MatrixXb IFinal(1, cellActivatedCount);
        double minHypothesisDist = CYLINDER_RANSAC_SQR_MAX_DIST * mLeft;
        int inliersAcceptedCount = 0.9 * mLeft;
        int maxInliersCount = 0;

        // RANSAC loop
        int k = 0;
        while(k < K){
            // Random triplet
            int id_1 = idsLeft[rand() % mLeft];
            int id_2 = idsLeft[rand() % mLeft];
            int id_3 = idsLeft[rand() % mLeft];

            e1 = (N.col(id_1) + N.col(id_2) + N.col(id_3));
            e2 = (PProj.col(id_1) + PProj.col(id_2) + PProj.col(id_3));

            // LLS solution for triplets
            A = e1.transpose() * e1;
            double a = 1.0-A(0)/9.0;
            double b = (
                    (N.col(id_1).array() * PProj.col(id_1).array()) + 
                    (N.col(id_2).array() * PProj.col(id_2).array()) + 
                    (N.col(id_3).array() * PProj.col(id_3).array())
                    ).sum() / 3 
                - ((e1.transpose() * e2)(0) / 9);
            double r = b / a;

            center = (e2 - r * e1) / 3;

            // Unnecessary calculations here
            // Normal dist
            D = ((PProj - r * N).colwise()- center.col(0)).colwise().squaredNorm() / (r * r);

            // Rectify radius if concave
            if (r < 0)
                r = -r;

            // Inliers
            I = D.array() < CYLINDER_RANSAC_SQR_MAX_DIST;

            //MSAC truncated distance
            double dist = 0.0;
            int inliersCount = 0;
            for(int i = 0;i < cellActivatedCount; i++){
                if(idsLeftMask(i)){
                    if(I(i)) {
                        inliersCount += 1;
                        dist += D(i);
                    }else{
                        dist += CYLINDER_RANSAC_SQR_MAX_DIST;
                    }
                }
            }

            if(dist < minHypothesisDist){
                minHypothesisDist = dist;
                maxInliersCount = inliersCount;
                for(int i = 0; i < cellActivatedCount; i++){
                    if(idsLeftMask(i)){
                        IFinal(i) = I(i);
                    }else{
                        IFinal(i) = false;
                    }
                }
                if(inliersCount > inliersAcceptedCount)
                    break;
            }
            k++;
        }

        // Checkpoint 2
        if(maxInliersCount < 6)
            break;

        // Increase prob. of finding inlier for next RANSAC runs
        K = log(1 - pSuccess) / log(1 - pow(0.5, 3));

        // Remove cells from list of remaining cells
        idsLeft.clear();
        for(int i = 0; i < cellActivatedCount; i++){
            if(IFinal(i)) {
                idsLeftMask(i) = false;
                mLeft--;
            }else if(idsLeftMask(i)) {
                idsLeft.push_back(i);
            }
        }

        // LLS solution using all inliers
        e1.setZero();
        e2.setZero();
        double b = 0;
        for(int i = 0; i < cellActivatedCount; i++){
            if(IFinal(i)) {
                e1 += N.col(i);
                e2 += PProj.col(i);
                b += (N.col(i).array() * PProj.col(i).array()).sum();
            }
        }

        A = e1.transpose() * e1;

        double a = 1 - A(0) / (maxInliersCount * maxInliersCount);
        b /= maxInliersCount;
        b -= (e1.transpose() * e2)(0) / (maxInliersCount * maxInliersCount);
        double r = b / a;
        center = (e2 - r * e1) / maxInliersCount;

        // Rectify radius if concave
        if (r < 0)
            r = -r;

        // Add cylinder
        this->segmentCount += 1;
        this->radius.push_back(r);
        this->centers.push_back(center);
        this->inliers.push_back(IFinal);

        // Save points on axis
        Eigen::Vector3d P1d = center;
        Eigen::Vector3d P2d = center + vec;
        double P1P2d = (P2d - P1d).norm();
        // Use point-to-line distances (same as cylinder distances)
        Eigen::Vector3d P3;
        for(int i = 0; i < cellActivatedCount; i++){
            if(IFinal(i)) {
                P3 = P.block<3,1>(0, i);
                //D(i) = (P3-center).norm()-r;
                D(i) = ((P2d - P1d).cross(P3 - P2d)).norm() / P1P2d - r;
            }
        }
        D = D.array().square();

        double mse = 0; 
        for(int i = 0; i < cellActivatedCount; i++){
            if(IFinal(i))
                mse += D(i);
        }
        mse /= maxInliersCount;
        MSE.push_back(mse);

        // Save points on axis, useful for computing distances afterwards
        this->pointsAxis1.push_back(P1d); 
        this->pointsAxis2.push_back(P2d);
        this->normalsAxis1Axis2.push_back(P1P2d);
        this->cylindricalMask.push_back(true);
    }
}



double Cylinder_Segment::distance(const Eigen::Vector3d& point, int id) {
    return ((pointsAxis2[id] - pointsAxis1[id]).cross(point - pointsAxis2[id])).norm() / normalsAxis1Axis2[id] - radius[id];
}




Cylinder_Segment::~Cylinder_Segment() {
    //delete []this->local2globalMap;    
}
