#include "plane_segment.hpp"
#include "eig33sym.hpp"

#include "../../parameters.hpp"
#include "../../outputs/logger.hpp"
#include <iostream>

namespace rgbd_slam {
    namespace features {
        namespace primitives {



            Plane_Segment::Plane_Segment(const uint cellWidth, const uint ptsPerCellCount) : 
                _ptsPerCellCount(ptsPerCellCount), 
                _minZeroPointCount( static_cast<uint>(_ptsPerCellCount / 2.0)), 
                _cellWidth(cellWidth), 
                _cellHeight(_ptsPerCellCount / _cellWidth)
            {
                assert(ptsPerCellCount > 0);
                assert(cellWidth > 0);

                clear_plane_parameters();
                _isPlanar = false;
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

            /**
             * \brief Runs a check on a depth value, and increment a discontinuity counter if needed
             *
             * \param[in] depthAlphaValue
             * \param[in] depthDiscontinuityLimit Limit of maximum depth discontinuities
             * \param[in] z The depth value to check
             * \param[in,out] discontinuityCounter The count of discontinuities
             * \param[in,out] zLast Last depht value to pass the continuity test
             *
             * \return False if too much continuities are detected
             */
            bool check_discontinuities(const double depthAlphaValue, const uint depthDiscontinuityLimit, const float z, uint& discontinuityCounter, float& zLast)
            {
                if(z > 0 and abs(z - zLast) < depthAlphaValue * (abs(z) + 0.5)) 
                {
                    zLast = z;
                }
                else if(++discontinuityCounter > depthDiscontinuityLimit)
                {
                    return false;
                }
                return true;
            }

            bool Plane_Segment::is_cell_vertical_continuous(const matrixf& depthMatrix, const double depthAlphaValue, const uint depthDiscontinuityLimit) const
            {
                const uint startValue = _cellWidth / 2;
                const uint endValue = _ptsPerCellCount - startValue;

                uint discontinuityCounter = 0;
                float zLast = std::max(depthMatrix(startValue), depthMatrix(startValue + _cellWidth));  /* handles missing pixels on the borders*/

                // Scan vertically through the middle
                for(uint i = startValue + _cellWidth; i < endValue; i += _cellWidth)
                {
                    const float z = depthMatrix(i);
                    if (not check_discontinuities(depthAlphaValue, depthDiscontinuityLimit, z, discontinuityCounter, zLast))
                        return false;
                }
                // continuous
                return true;
            }

            bool Plane_Segment::is_cell_horizontal_continuous(const matrixf& depthMatrix, const double depthAlphaValue, const uint depthDiscontinuityLimit) const
            {
                const uint startValue = static_cast<uint>(_cellWidth * (_cellHeight / 2.0));
                const uint endValue = startValue + _cellWidth;

                uint discontinuityCounter = 0;
                float zLast = std::max(depthMatrix(startValue), depthMatrix(startValue + 1)); /* handles missing pixels on the borders*/

                // Scan horizontally through the middle
                for(uint i = startValue + 1; i < endValue; ++i) 
                {
                    const float z = depthMatrix(i);
                    if (not check_discontinuities(depthAlphaValue, depthDiscontinuityLimit, z, discontinuityCounter, zLast))
                        return false;
                }
                // continuous
                return true;
            }

            void Plane_Segment::init_plane_segment(const matrixf& depthCloudArray, const uint cellId) 
            {
                clear_plane_parameters();
                _isPlanar = true;

                const uint offset = cellId * _ptsPerCellCount;

                //get z of depth points
                const matrixf& depthMatrix = depthCloudArray.block(offset, 2, _ptsPerCellCount, 1);

                // Check number of missing depth points
                _pointCount =  (depthMatrix.array() > 0).count();
                if (_pointCount < _minZeroPointCount){
                    _isPlanar = false;
                    return;
                }

                //get points x and y coords
                const matrixf& xMatrix = depthCloudArray.block(offset, 0, _ptsPerCellCount, 1);
                const matrixf& yMatrix = depthCloudArray.block(offset, 1, _ptsPerCellCount, 1);

                // Check for discontinuities using cross search
                //Search discontinuities only in a vertical line passing through the center, than an horizontal line passing through the center.
                const static double depthAlphaValue = Parameters::get_depth_alpha();
                const static uint depthDiscontinuityLimit = Parameters::get_depth_discontinuity_limit(); 

                if (not is_cell_horizontal_continuous(depthMatrix, depthAlphaValue, depthDiscontinuityLimit) or 
                        not is_cell_vertical_continuous(depthMatrix, depthAlphaValue, depthDiscontinuityLimit))
                {
                    // this segment is not continuous...
                    _isPlanar = false;
                    return;
                }

                //set PCA components
                _Sx = xMatrix.sum();
                _Sy = yMatrix.sum();
                _Sz = depthMatrix.sum();
                _Sxs = (xMatrix.array() * xMatrix.array()).sum();
                _Sys = (yMatrix.array() * yMatrix.array()).sum();
                _Szs = (depthMatrix.array() * depthMatrix.array()).sum();
                _Sxy = (xMatrix.array() * yMatrix.array()).sum();
                _Szx = (xMatrix.array() * depthMatrix.array()).sum();
                _Syz = (yMatrix.array() * depthMatrix.array()).sum();

                //fit a plane to those points 
                fit_plane();
                //MSE > T_MSE
                const static double depthSigmaError = Parameters::get_depth_sigma_error();
                const static double depthSigmaMargin = Parameters::get_depth_sigma_margin();
                if(_MSE > pow(depthSigmaError * pow(_mean.z(), 2) + depthSigmaMargin, 2))
                    _isPlanar = false;

            }


            bool Plane_Segment::is_depth_discontinuous(const Plane_Segment& planeSegment) const
            {
                return is_depth_discontinuous(planeSegment._mean);
            }
            bool Plane_Segment::is_depth_discontinuous(const vector3& planeMean) const
            {
                const static double depthAlpha = Parameters::get_depth_alpha();
                return abs(_mean.z() - planeMean.z()) < 2.0 * depthAlpha * (abs(_mean.z()) + 0.5);
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

            void Plane_Segment::fit_plane() {
                assert(_pointCount > 0);

                const double oneOverCount = 1.0 / static_cast<double>(_pointCount);
                //fit a plane to the stored points
                _mean = vector3(_Sx, _Sy, _Sz) * oneOverCount;

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
                vector3 sv = vector3::Zero();
                vector3 v = vector3::Zero();
                // Pass those vectors as C style arrays of 3 elements
                if(not LA::eig33sym(cov, &sv(0), &v(0)))
                    outputs::log("Too much error");

                _d = -v.dot(_mean);

                // Enforce normal orientation
                if(_d > 0) {   //point normal toward the camera
                    _normal = v; 
                } else {
                    _normal = -v;
                    _d = -_d;
                } 
                // some values have floatting points errors, renormalize
                _normal.normalize();

                //_score = sv[0] / (sv[0] + sv[1] + sv[2]);
                _MSE = sv.x() * oneOverCount;
                _score = sv.y() / sv.x();
                _isPlanar = true;
            }

            /*
             * Sets all the plane parameters to zero 
             */
            void Plane_Segment::clear_plane_parameters() {
                _pointCount = 0;
                _score = 0;
                _MSE = 0;
                _isPlanar = false; 

                _mean = vector3::Zero();
                _normal = vector3::Zero();
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
                assert(_isPlanar);
                return (_normal.dot(p._normal));
            }

            double Plane_Segment::get_signed_distance(const vector3& point) const {
                assert(_isPlanar);
                return _normal.dot(point - _mean);
            }


        }
    }
}
