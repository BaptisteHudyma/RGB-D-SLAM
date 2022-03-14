#include "plane_segment.hpp"
#include "eig33sym.hpp"

#include "parameters.hpp"
#include "logger.hpp"

namespace rgbd_slam {
namespace features {
namespace primitives {


    Plane_Segment::Plane_Segment(const uint cellWidth, const uint ptsPerCellCount) : 
        _ptsPerCellCount(ptsPerCellCount), 
        _minZeroPointCount(_ptsPerCellCount/2.0), 
        _cellWidth(cellWidth), 
        _cellHeight(_ptsPerCellCount / _cellWidth)
    {
        clear_plane_parameters();
        _isPlanar = true;
    }

    Plane_Segment::Plane_Segment(const Plane_Segment& seg) :
        _ptsPerCellCount(seg._ptsPerCellCount), 
        _minZeroPointCount(seg._minZeroPointCount), 
        _cellWidth(seg._cellWidth), 
        _cellHeight(seg._cellHeight),
        _pointCount(seg._pointCount),
        _score(seg._score),
        _MSE(seg._MSE),
        _isPlanar(seg._isPlanar),
        _mean(seg._mean),
        _normal(seg._normal),
        _d(seg._d),
        _Sx(seg._Sx),
        _Sy(seg._Sy),
        _Sz(seg._Sz),
        _Sxs(seg._Sxs),
        _Sys(seg._Sys),
        _Szs(seg._Szs),
        _Sxy(seg._Sxy),
        _Syz(seg._Syz),
        _Szx(seg._Szx)
    {
    }

    void Plane_Segment::init_plane_segment(const Eigen::MatrixXf& depthCloudArray, const uint cellId) 
    {
        clear_plane_parameters();
        _isPlanar = true;

        uint offset = cellId * _ptsPerCellCount;

        //get z of depth points
        const Eigen::MatrixXf& Z_matrix = depthCloudArray.block(offset, 2, _ptsPerCellCount, 1);

        // Check nbr of missing depth points
        _pointCount =  (Z_matrix.array() > 0).count();
        if (_pointCount < _minZeroPointCount){
            _isPlanar = false;
            return;
        }

        //get points x and y coords
        const Eigen::MatrixXf& X_matrix = depthCloudArray.block(offset, 0, _ptsPerCellCount, 1);
        const Eigen::MatrixXf& Y_matrix = depthCloudArray.block(offset, 1, _ptsPerCellCount, 1);

        // Check for discontinuities using cross search
        //Search discontinuities only in a vertical line passing through the center, than an horizontal line passing through the center.
        uint discontinuityCounter = 0;
        uint i = _cellWidth * (_cellHeight / 2);
        uint j = i + _cellWidth;
        float zLast = std::max(Z_matrix(i), Z_matrix(i + 1)); /* handles missing pixels on the borders*/
        
        const double depthAlphaValue = Parameters::get_depth_alpha();
        const uint depthDiscontinuityLimit = Parameters::get_depth_discontinuity_limit(); 

        i++;
        // Scan horizontally through the middle
        while(i < j){
            float z = Z_matrix(i);
            if(z > 0) {
                if(abs(z - zLast) < depthAlphaValue * (abs(z) + 0.5)) {
                    zLast = z;
                }
                else { 
                    discontinuityCounter += 1;
                    if(discontinuityCounter > depthDiscontinuityLimit) {
                        _isPlanar = false;
                        return;
                    }
                }
            }
            i++;
        }
        // Scan vertically through the middle
        i = _cellWidth/2;
        j = _ptsPerCellCount - i;
        zLast = std::max(Z_matrix(i), Z_matrix(i + _cellWidth));  /* handles missing pixels on the borders*/
        i += _cellWidth;
        discontinuityCounter = 0;
        while(i < j){
            float z = Z_matrix(i);
            if(z > 0) {
                if(abs(z - zLast) < depthAlphaValue * (abs(z) + 0.5)) {
                    zLast = z;
                }
                else {
                    discontinuityCounter += 1;
                    if(discontinuityCounter > depthDiscontinuityLimit) {
                        _isPlanar = false;
                        return;
                    }
                }
            }
            i += _cellWidth;
        }

        //set PCA components
        _Sx = X_matrix.sum();
        _Sy = Y_matrix.sum();
        _Sz = Z_matrix.sum();
        _Sxs = (X_matrix.array() * X_matrix.array()).sum();
        _Sys = (Y_matrix.array() * Y_matrix.array()).sum();
        _Szs = (Z_matrix.array() * Z_matrix.array()).sum();
        _Sxy = (X_matrix.array() * Y_matrix.array()).sum();
        _Szx = (X_matrix.array() * Z_matrix.array()).sum();
        _Syz = (Y_matrix.array() * Z_matrix.array()).sum();

        //fit a plane to those points 
        if(_isPlanar) {
            fit_plane();
            //MSE > T_MSE
            if(_MSE > pow(Parameters::get_depth_sigma_error() * pow(_mean[2], 2) + Parameters::get_depth_sigma_margin(), 2))
                _isPlanar = false;
        }

    }

    bool Plane_Segment::is_depth_discontinuous(const Plane_Segment& planeSegment) {
        return abs(_mean[2] - planeSegment._mean[2]) < 2 * Parameters::get_depth_alpha() * (abs(_mean[2]) + 0.5);
    }


    void Plane_Segment::expand_segment(const Plane_Segment& planeSegment) {
        _Sx += planeSegment._Sx;
        _Sy += planeSegment._Sy;
        _Sz += planeSegment._Sz;

        _Sxs += planeSegment._Sxs;
        _Sys += planeSegment._Sys;
        _Szs += planeSegment._Szs;

        _Sxy += planeSegment._Sxy;
        _Syz += planeSegment._Syz;
        _Szx += planeSegment._Szx;

        _pointCount += planeSegment._pointCount;
    }

    /*
     * Merge the PCA saved values in prevision of a plane fitting
     * This function do not make any plane calculations
     */
    void Plane_Segment::expand_segment(const std::unique_ptr<Plane_Segment>& planeSegment) {
        expand_segment(*planeSegment);
    }

    void Plane_Segment::fit_plane() {
        const double oneOverCount = 1.0 / (double)_pointCount;
        //fit a plane to the stored points
        _mean = vector3(_Sx, _Sy, _Sz);
        _mean *= oneOverCount;

        // Expressing covariance as E[PP^t] + E[P]*E[P^T]
        double cov[3][3] = {
            {_Sxs - _Sx * _Sx * oneOverCount, _Sxy - _Sx * _Sy * oneOverCount,  _Szx - _Sx * _Sz * oneOverCount},
            {0                              , _Sys - _Sy * _Sy * oneOverCount,  _Syz - _Sy * _Sz * oneOverCount},
            {0                              , 0,                                _Szs - _Sz * _Sz * oneOverCount }
        };
        cov[1][0] = cov[0][1]; 
        cov[2][0] = cov[0][2];
        cov[2][1] = cov[1][2];

        // This uses QR decomposition for symmetric matrices
        double sv[3] = {0, 0, 0};
        double v[3] = {0};
        if(not LA::eig33sym(cov, sv, v))
            utils::log("Too much error");

        //_d = -v.dot(_mean);
        _d = -(v[0] * _mean[0] + v[1] * _mean[1] + v[2] * _mean[2]);

        // Enforce normal orientation
        if(_d > 0) {   //point normal toward the camera
            _normal[0] = v[0]; 
            _normal[1] = v[1];
            _normal[2] = v[2];
        } else {
            _normal[0] = -v[0];
            _normal[1] = -v[1];
            _normal[2] = -v[2];
            _d = -_d;
        } 

        //_score = sv[0] / (sv[0] + sv[1] + sv[2]);
        _MSE = sv[0] * oneOverCount;
        _score = sv[1] / sv[0];
    }

    /*
     * Sets all the plane parameters to zero 
     */
    void Plane_Segment::clear_plane_parameters() {
        _pointCount = 0;
        _score = 0;
        _MSE = 0;
        _isPlanar = false; 

        _mean[0] = 0;
        _mean[1] = 0;
        _mean[2] = 0;
        _normal[0] = 0;
        _normal[1] = 0;
        _normal[2] = 0;
        _d = 0;

        //Clear saved plane parameters
        _Sx = 0; 
        _Sy = 0;  
        _Sz = 0;  
        _Sxs = 0; 
        _Sys = 0; 
        _Szs = 0; 
        _Sxy = 0; 
        _Syz = 0; 
        _Szx = 0; 
    }


    double Plane_Segment::get_normal_similarity(const Plane_Segment& p) const {
        return (_normal.dot(p._normal));
    }

    double Plane_Segment::get_signed_distance(const double point[3]) const {
        return 
            _normal[0] * (point[0] - _mean[0]) + 
            _normal[1] * (point[1] - _mean[1]) + 
            _normal[2] * (point[2] - _mean[2]); 
    }

    double Plane_Segment::get_signed_distance(const vector3& point) const {
        return _normal.dot(point - _mean);
    }


}
}
}
