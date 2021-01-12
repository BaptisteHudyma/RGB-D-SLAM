#include "PlaneSegment.hpp"

using namespace planeDetection;
using namespace Eigen;

#include <iostream>

/*
 * Initialize the plane segment with the points from the depth matrix
 * 
 */
Plane_Segment::Plane_Segment(int cellWidth, int ptsPerCellCount)
    : ptsPerCellCount(ptsPerCellCount), minZeroPointCount(ptsPerCellCount/2.0), cellWidth(cellWidth), cellHeight(ptsPerCellCount / cellWidth)
{
    clear_plane_parameters();
    this->isPlanar = true;
}

/*
 *  Copy constructor
 */
Plane_Segment::Plane_Segment(const Plane_Segment& seg) 
    : ptsPerCellCount(seg.ptsPerCellCount), minZeroPointCount(seg.minZeroPointCount), cellWidth(seg.cellWidth), cellHeight(seg.cellHeight)
{
    pointCount = seg.pointCount;
    score = seg.score;
    MSE = seg.MSE;
    isPlanar = seg.isPlanar;

    mean = seg.mean;
    normal = seg.normal;
    d = seg.d;

    Sx = seg.Sx;
    Sy = seg.Sy;
    Sz = seg.Sz;
    Sxs = seg.Sxs;
    Sys = seg.Sys;
    Szs = seg.Szs;
    Sxy = seg.Sxy;
    Syz = seg.Syz;
    Szx = seg.Szx;
}

void Plane_Segment::init_plane_segment(Eigen::MatrixXf& depthCloudArray, int cellId) {
    clear_plane_parameters();
    this->isPlanar = true;

    int offset = cellId * this->ptsPerCellCount;

    //get z of depth points
    Eigen::MatrixXf Z_matrix = depthCloudArray.block(offset, 2, this->ptsPerCellCount, 1);

    // Check nbr of missing depth points
    this->pointCount =  (Z_matrix.array() > 0).count();
    if (this->pointCount < this->minZeroPointCount){
        this->isPlanar = false;
        return;
    }

    //get points x and y coords
    Eigen::MatrixXf X_matrix = depthCloudArray.block(offset, 0, this->ptsPerCellCount, 1);
    Eigen::MatrixXf Y_matrix = depthCloudArray.block(offset, 1, this->ptsPerCellCount, 1);

    // Check for discontinuities using cross search
    //Search discontinuities only in a vertical line passing through the center, than an horizontal line passing through the center.
    int discontinuityCounter = 0;
    int i = this->cellWidth * (this->cellHeight / 2);
    int j = i + this->cellWidth;
    float zLast = std::max(Z_matrix(i), Z_matrix(i + 1)); /* handles missing pixels on the borders*/
    i++;
    // Scan horizontally through the middle
    while(i < j){
        float z = Z_matrix(i);
        if(z > 0) {
            if(abs(z - zLast) < DEPTH_ALPHA * (abs(z) + 0.5)) {
                zLast = z;
            }
            else { 
                discontinuityCounter += 1;
                if(discontinuityCounter > DEPTH_DISCONTINUITY_LIMIT) {
                    this->isPlanar = false;
                    return;
                }
            }
        }
        i++;
    }
    // Scan vertically through the middle
    i = this->cellWidth/2;
    j = this->ptsPerCellCount - i;
    zLast = std::max(Z_matrix(i), Z_matrix(i + this->cellWidth));  /* handles missing pixels on the borders*/
    i += this->cellWidth;
    discontinuityCounter = 0;
    while(i < j){
        float z = Z_matrix(i);
        if(z > 0) {
            if(abs(z - zLast) < DEPTH_ALPHA * (abs(z) + 0.5)) {
                zLast = z;
            }
            else {
                discontinuityCounter += 1;
                if(discontinuityCounter > DEPTH_DISCONTINUITY_LIMIT) {
                    this->isPlanar = false;
                    return;
                }
            }
        }
        i += this->cellWidth;
    }

    //set PCA components
    this->Sx = X_matrix.sum();
    this->Sy = Y_matrix.sum();
    this->Sz = Z_matrix.sum();
    this->Sxs = (X_matrix.array() * X_matrix.array()).sum();
    this->Sys = (Y_matrix.array() * Y_matrix.array()).sum();
    this->Szs = (Z_matrix.array() * Z_matrix.array()).sum();
    this->Sxy = (X_matrix.array() * Y_matrix.array()).sum();
    this->Szx = (X_matrix.array() * Z_matrix.array()).sum();
    this->Syz = (Y_matrix.array() * Z_matrix.array()).sum();

    //fit a plane to those points 
    if(this->isPlanar) {
        fit_plane();
        //MSE > T_MSE
        if(this->MSE > pow(DEPTH_SIGMA_COEFF * pow(this->mean[2], 2) + DEPTH_SIGMA_MARGIN, 2))
            this->isPlanar = false;
    }

}

/*
 * True if this plane segment presents a depth discontinuity with another one.
 * False if there is no depth discontinuity
 */
bool Plane_Segment::is_depth_discontinuous(const Plane_Segment& planeSegment) {
    return abs(mean[2] - planeSegment.mean[2]) < 2 * DEPTH_ALPHA * (abs(mean[2]) + 0.5);
}


/*
 * Merge the PCA saved values in prevision of a plane fitting
 * This function do not make any plane calculations
 */
void Plane_Segment::expand_segment(const Plane_Segment& ps) {
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

/*
 * Merge the PCA saved values in prevision of a plane fitting
 * This function do not make any plane calculations
 */
void Plane_Segment::expand_segment(const std::unique_ptr<Plane_Segment>& ps) {
    expand_segment(*ps);
}

/*
 * Fit a plane to the contained points using PCA
 */
void Plane_Segment::fit_plane() {
    const double oneOverCount = 1.0 / (double)this->pointCount;
    //fit a plane to the stored points
    this->mean = Vector3d(Sx, Sy, Sz);
    this->mean *= oneOverCount;

    // Expressing covariance as E[PP^t] + E[P]*E[P^T]
    double cov[3][3] = {
        {Sxs - Sx * Sx * oneOverCount, Sxy - Sx * Sy * oneOverCount, Szx - Sx * Sz * oneOverCount},
        {0                           , Sys - Sy * Sy * oneOverCount, Syz - Sy * Sz * oneOverCount},
        {0                           , 0,                            Szs - Sz * Sz * oneOverCount }
    };
    cov[1][0] = cov[0][1]; 
    cov[2][0] = cov[0][2];
    cov[2][1] = cov[1][2];

    // This uses QR decomposition for symmetric matrices
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(Eigen::Map<Eigen::Matrix3d>(cov[0], 3, 3) );
    Eigen::Vector3d v = es.eigenvectors().col(0);

    this->d = -v.dot(this->mean);
    // Enforce normal orientation
    if(this->d > 0) {   //point normal toward the camera
        this->normal[0] = v[0];
        this->normal[1] = v[1];
        this->normal[2] = v[2];
    } else {
        this->normal[0] = -v[0];
        this->normal[1] = -v[1];
        this->normal[2] = -v[2];
        this->d = -this->d;
    } 

    const Eigen::VectorXd& eigenValues = es.eigenvalues();
    this->MSE   = eigenValues[0] * oneOverCount; 
    //this->score = eigenValues[0] / (eigenValues[0] + eigenValues[1] + eigenValues[2]);
    this->score = eigenValues[1] / eigenValues[0];

}

/*
 * Sets all the plane parameters to zero 
 */
void Plane_Segment::clear_plane_parameters() {
    //Clear saved plane parameters
    this->Sx = 0; 
    this->Sy = 0;  
    this->Sz = 0;  
    this->Sxs = 0; 
    this->Sys = 0; 
    this->Szs = 0; 
    this->Sxy = 0; 
    this->Syz = 0; 
    this->Szx = 0; 

    this->isPlanar = false; 
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


/*
 *   Check similarity of two planes. 
 * Return value of 1 indicates similar normals.
 * Return Value of 0 indicates perpendicular normals
 */
double Plane_Segment::get_normal_similarity(const Plane_Segment& p) {
    return abs(normal.dot(p.normal));
}

/*
 *  Return the signed distance form a point to this plane
 */
double Plane_Segment::get_signed_distance(const double point[3]) {
    return 
        normal[0] * (point[0] - mean[0]) + 
        normal[1] * (point[1] - mean[1]) + 
        normal[2] * (point[2] - mean[2]); 
}
/*
 *  Return the signed distance form a point to this plane
 */
double Plane_Segment::get_signed_distance(const Eigen::Vector3d& point) {
    return normal.dot(point - mean);
}



Plane_Segment::~Plane_Segment() {
    //destructor

}
