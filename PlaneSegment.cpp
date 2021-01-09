#include "PlaneSegment.hpp"

using namespace planeDetection;
using namespace Eigen;

Plane_Segment::Plane_Segment(Eigen::MatrixXf& depthCloudArray, int cellId, int ptsPerCellCount, int cellWidth) {
    clear_plane_parameters();
    this->isPlanar = true;
    this->hasEmpty = false;
    this->hasDepthDiscontinuity = false;

    this->minZeroPointCount = ptsPerCellCount/2;
    int offset = cellId * ptsPerCellCount;
    int cellHeight = ptsPerCellCount / cellWidth;
    double maxDiff = 100;

    //get z of depth points
    Eigen::MatrixXf Z_matrix = depthCloudArray.block(offset, 2, ptsPerCellCount, 1);

    // Check nbr of missing depth points
    this->pointCount =  (Z_matrix.array() > 0).count();
    if (this->pointCount < this->minZeroPointCount){
        this->hasEmpty = true;
        this->isPlanar = false;
        return;
    }

    //get points x and y coords
    Eigen::MatrixXf X_matrix = depthCloudArray.block(offset, 0, ptsPerCellCount, 1);
    Eigen::MatrixXf Y_matrix = depthCloudArray.block(offset, 1, ptsPerCellCount, 1);

    // Check for discontinuities using cross search
    int discontinuityCounter = 0;
    int i = cellWidth * (cellHeight / 2);
    int j = i + cellWidth;
    float z = 0;
    float zLast = std::max(Z_matrix(i), Z_matrix(i+1)); /* handles missing pixels on the borders*/
    i++;
    // Scan horizontally through the middle
    while(i<j){
        z = Z_matrix(i);
        if(z > 0 and abs(z - zLast) < maxDiff){
            zLast = z;
        }
        else if(z > 0) {
            discontinuityCounter++;
            if(discontinuityCounter > 1) {
                this->hasDepthDiscontinuity = true;
                this->isPlanar = false;
                return;
            }
        }
        i++;
    }

    // Scan vertically through the middle
    i = cellWidth/2;
    j = ptsPerCellCount - i;
    zLast = std::max(Z_matrix(i), Z_matrix(i + cellWidth));  /* handles missing pixels on the borders*/
    i += cellWidth;
    //discontinuityCounter = 0;
    while(i < j){
        z = Z_matrix(i);
        if(z > 0 and abs(z - zLast) < maxDiff){
            zLast = z;
        }
        else if(z > 0) {
            discontinuityCounter++;
            if(discontinuityCounter > 1) {
                this->hasDepthDiscontinuity = true;
                this->isPlanar = false;
                return;
            }
        }
        i += cellWidth;
    }

    this->Sx = X_matrix.sum();
    this->Sy = Y_matrix.sum();
    this->Sz = Z_matrix.sum();
    this->Sxs = (X_matrix.array() * X_matrix.array()).sum();
    this->Sys = (Y_matrix.array() * Y_matrix.array()).sum();
    this->Szs = (Z_matrix.array() * Z_matrix.array()).sum();
    this->Sxy = (X_matrix.array() * Y_matrix.array()).sum();
    this->Szx = (X_matrix.array() * Z_matrix.array()).sum();
    this->Syz = (Y_matrix.array() * Z_matrix.array()).sum();


    //check plane MSE
    if(this->isPlanar) {
        fit_plane();
        if(this->MSE > pow(DEPTH_SIGMA_COEFF * pow(mean[2], 2) + DEPTH_SIGMA_MARGIN, 2))
            this->isPlanar = false;
    }
}

void Plane_Segment::expand_segment(const Plane_Segment& ps) {
    //merge this nodes' PCA plane characteristics with those of another node
    //DO NOT MAKE PLANE CALCULATIONS
    this->Sx += ps.Sx;
    this->Sy += ps.Sy;
    this->Sz += ps.Sz;

    this->Sxs += ps.Sxs;
    this->Sys += ps.Sys;
    this->Szs += ps.Szs;

    this->Sxy += ps.Sxy;
    this->Syz += ps.Syz;
    this->Szx += ps.Szx;

    this->pointCount += ps.pointCount;
}

void Plane_Segment::expand_segment(const std::unique_ptr<Plane_Segment>& ps) {
    //merge this nodes' PCA plane characteristics with those of another node
    //DO NOT MAKE PLANE CALCULATIONS
    this->Sx += ps->Sx;
    this->Sy += ps->Sy;
    this->Sz += ps->Sz;

    this->Sxs += ps->Sxs;
    this->Sys += ps->Sys;
    this->Szs += ps->Szs;

    this->Sxy += ps->Sxy;
    this->Syz += ps->Syz;
    this->Szx += ps->Szx;

    this->pointCount += ps->pointCount;
}

void Plane_Segment::fit_plane() {
    //fit a plane to the stored points
    this->mean = Vector3d(Sx, Sy, Sz);
    this->mean /= this->pointCount;
    //this->mean[0] = Sx / this->pointCount;
    //this->mean[1] = Sy / this->pointCount;
    //this->mean[2] = Sz / this->pointCount;

    // Expressing covariance as E[PP^t] + E[P]*E[P^T]
    double cov[3][3] = {
        {Sxs - Sx * Sx / this->pointCount, Sxy - Sx * Sy / this->pointCount, Szx - Sx * Sz / this->pointCount},
        {0                               , Sys - Sy * Sy / this->pointCount, Syz - Sy * Sz / this->pointCount},
        {0                               , 0,                                Szs - Sz * Sz / this->pointCount }
    };
    cov[1][0] = cov[0][1]; 
    cov[2][0] = cov[0][2];
    cov[2][1] = cov[1][2];

    // This uses QR decomposition for symmetric matrices
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(Eigen::Map<Eigen::Matrix3d>(cov[0], 3, 3) );
    Eigen::Vector3d v = es.eigenvectors().col(0);

    d = - (v[0] * this->mean[0] + v[1] * this->mean[1] + v[2] * this->mean[2]);
    // Enforce normal orientation
    this->normal = Vector3d(v);
    if (this->d <= 0) {
        this->normal.inverse(); 
        this->d = -this->d;
    }
    /*
       if(this->d > 0) {
       this->normal[0] = v[0];
       this->normal[1] = v[1];
       this->normal[2] = v[2];
       } else {
       this->normal[0] = -v[0];
       this->normal[1] = -v[1];
       this->normal[2] = -v[2];
       this->d = -this->d;
       } 
     */

    const Eigen::VectorXd& eigenValues = es.eigenvalues();
    this->MSE   = eigenValues[0] / this->pointCount;
    this->score = eigenValues[1] / eigenValues[0];

}

void Plane_Segment::clear_plane_parameters() {
    //Clear saved plane parameters
    this->isPlanar = false; 
    this->Sx = 0; 
    this->Sy = 0;  
    this->Sz = 0;  
    this->Sxs = 0; 
    this->Sys = 0; 
    this->Szs = 0; 
    this->Sxy = 0; 
    this->Syz = 0; 
    this->Szx = 0; 

    this->MSE = 0;
    this->score = 0;

    this->normal[0] = 0;
    this->normal[1] = 0;
    this->normal[2] = 0;
    this->d = 0;

    this->mean[0] = 0;
    this->mean[1] = 0;
    this->mean[2] = 0;

    this->pointCount = 0;
}


Plane_Segment::~Plane_Segment() {
    //destructor
}
