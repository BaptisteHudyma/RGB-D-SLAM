#include "cylinder_segment.hpp"
#include "parameters.hpp"
#include "logger.hpp"

namespace rgbd_slam {
    namespace features {
        namespace primitives {

            using namespace std;

            Cylinder_Segment::Cylinder_Segment(const Cylinder_Segment& seg, const uint subRegionId) :
                _cellActivatedCount(0),
                _segmentCount(0)
            {
                //copy stored data
                _radius.push_back(seg._radius[subRegionId]);
                _centers.push_back(seg._centers[subRegionId]);
                //_pointsAxis1 = seg.pointsAxis1; 
                //_pointsAxis2 = seg.pointsAxis2;
                //_normalsAxis1Axis2 = seg._normalsAxis1Axis2;
                _axis = seg._axis;
            }

            /*
             *  Copy constructor
             */
            Cylinder_Segment::Cylinder_Segment(const Cylinder_Segment& seg) :
                _centers(seg._centers),
                _radius(seg._radius),
                _cellActivatedCount(0),
                _segmentCount(0)
            {
                //_pointsAxis1 = seg.pointsAxis1; 
                //_pointsAxis2 = seg.pointsAxis2;
                //_normalsAxis1Axis2 = seg._normalsAxis1Axis2;
                _axis = seg._axis;
            }

            Cylinder_Segment::Cylinder_Segment(const std::vector<plane_segment_unique_ptr>& planeGrid, const std::vector<bool>& activatedMask, const uint cellActivatedCount) :
                _cellActivatedCount(cellActivatedCount),
                _segmentCount(0)
            {
                const uint samplesCount = activatedMask.size();

                const float minimumCyinderScore = Parameters::get_cylinder_ransac_minimm_score();
                const float maximumSqrtDistance = Parameters::get_cylinder_ransac_max_distance();

                _local2globalMap.assign(_cellActivatedCount, 0);

                Eigen::MatrixXd N(3, 2 * _cellActivatedCount);
                Eigen::MatrixXd P(3, _cellActivatedCount);

                // Init. P and N
                uint j = 0;
                for(uint i = 0; i < samplesCount; i++)
                {
                    if (activatedMask[i])
                    {
                        const vector3& planeNormal = planeGrid[i]->get_normal();
                        const vector3& planeMean = planeGrid[i]->get_mean();
                        N(0, j) = planeNormal.x();
                        N(1, j) = planeNormal.y();
                        N(2, j) = planeNormal.z();
                        P(0, j) = planeMean.x();
                        P(1, j) = planeMean.y();
                        P(2, j) = planeMean.z();

                        assert(j < _local2globalMap.size());
                        _local2globalMap[j] = i;
                        j++;
                    }
                }
                // Concatenate [N -N]
                for(uint i = 0; i < samplesCount; i++)
                {
                    if (activatedMask[i])
                    {
                        const vector3& planeNormal = planeGrid[i]->get_normal();
                        N(0, j) = -planeNormal.x();
                        N(1, j) = -planeNormal.y();
                        N(2, j) = -planeNormal.z();
                        j++;
                    }
                }

                // Compute covariance
                const Eigen::MatrixXd cov = (N * N.adjoint()) / static_cast<double>(N.cols() - 1);

                // PCA using QR decomposition for symmetric matrices
                Eigen::SelfAdjointEigenSolver<matrix33> es(cov);
                const vector3& S = es.eigenvalues();
                const double score = S(2)/S(0);

                // Checkpoint 1
                if(score < minimumCyinderScore)
                    return;

                const vector3& vec = es.eigenvectors().col(0);
                _axis = vec; 

                Eigen::MatrixXd NCpy = N.block(0, 0, 3, _cellActivatedCount);
                N = NCpy; /* This avoids memory issues */
                Eigen::MatrixXd PProj(3, _cellActivatedCount);

                // Projection to plane: P' = P-theta*<P.theta>
                const Eigen::MatrixXd& PDotTheta = vec.transpose() * P;
                PProj.row(0) = P.row(0) - PDotTheta * vec(0);
                PProj.row(1) = P.row(1) - PDotTheta * vec(1);
                PProj.row(2) = P.row(2) - PDotTheta * vec(2);
                const Eigen::MatrixXd& NDotTheta = vec.transpose() * N;
                N.row(0) -= NDotTheta * vec(0);
                N.row(1) -= NDotTheta * vec(1);
                N.row(2) -= NDotTheta * vec(2);

                // Normalize projected normals
                const Eigen::MatrixXd& normalsNorm = N.colwise().norm();
                N.row(0) = N.row(0).array() / normalsNorm.array();
                N.row(1) = N.row(1).array() / normalsNorm.array();
                N.row(2) = N.row(2).array() / normalsNorm.array();

                // Ransac params
                const float pSuccess = 0.8;
                const float w = 0.33;
                float K = log(1 - pSuccess) / log(1 - pow(w, 3));

                uint mLeft = _cellActivatedCount;
                Matrixb idsLeftMask(1, _cellActivatedCount);

                vector<uint> idsLeft;
                for(uint i = 0; i < _cellActivatedCount; i++)
                {
                    idsLeft.push_back(i);
                    idsLeftMask(i) = true;
                }
                // Sequential RANSAC main loop
                while(mLeft > 5 and mLeft > 0.1 * _cellActivatedCount)
                {
                    Eigen::MatrixXd A, center, e1, e2;
                    Eigen::MatrixXd D(1, _cellActivatedCount);
                    Matrixb I(1, _cellActivatedCount);
                    Matrixb IFinal(1, _cellActivatedCount);
                    double minHypothesisDist = maximumSqrtDistance * mLeft;
                    const uint inliersAcceptedCount = 0.9 * mLeft;
                    uint maxInliersCount = 0;

                    // RANSAC loop
                    int k = 0;
                    while(k < K)
                    {
                        // Random triplet
                        const uint id_1 = idsLeft[rand() % mLeft];
                        const uint id_2 = idsLeft[rand() % mLeft];
                        const uint id_3 = idsLeft[rand() % mLeft];

                        e1 = (N.col(id_1) + N.col(id_2) + N.col(id_3));
                        e2 = (PProj.col(id_1) + PProj.col(id_2) + PProj.col(id_3));

                        // LLS solution for triplets
                        A = e1.transpose() * e1;
                        const double a = 1.0 - A(0) / 9.0;
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
                        I = (D.array() < maximumSqrtDistance);

                        //MSAC truncated distance
                        double dist = 0.0;
                        uint inliersCount = 0;
                        for(uint i = 0;i < _cellActivatedCount; i++)
                        {
                            if(idsLeftMask(i))
                            {
                                if(I(i)) 
                                {
                                    inliersCount += 1;
                                    dist += D(i);
                                }
                                else
                                {
                                    dist += maximumSqrtDistance;
                                }
                            }
                        }

                        if(dist < minHypothesisDist)
                        {
                            minHypothesisDist = dist;
                            maxInliersCount = inliersCount;
                            for(uint i = 0; i < _cellActivatedCount; i++)
                            {
                                if(idsLeftMask(i))
                                {
                                    IFinal(i) = I(i);
                                }
                                else
                                {
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
                    for(uint i = 0; i < _cellActivatedCount; i++)
                    {
                        if(IFinal(i)) 
                        {
                            idsLeftMask(i) = false;
                            mLeft--;
                        }
                        else if(idsLeftMask(i)) 
                        {
                            idsLeft.push_back(i);
                        }
                    }

                    // LLS solution using all inliers
                    e1.setZero();
                    e2.setZero();
                    double b = 0;
                    for(uint i = 0; i < _cellActivatedCount; i++)
                    {
                        if(IFinal(i)) 
                        {
                            e1 += N.col(i);
                            e2 += PProj.col(i);
                            b += (N.col(i).array() * PProj.col(i).array()).sum();
                        }
                    }

                    A = e1.transpose() * e1;

                    const double a = 1 - A(0) / (maxInliersCount * maxInliersCount);
                    b /= maxInliersCount;
                    b -= (e1.transpose() * e2)(0) / (maxInliersCount * maxInliersCount);
                    double r = b / a;
                    center = (e2 - r * e1) / maxInliersCount;

                    // Rectify radius if concave
                    if (r < 0)
                        r = -r;

                    // Add cylinder
                    _segmentCount += 1;
                    _radius.push_back(r);
                    _centers.push_back(center);
                    _inliers.push_back(IFinal);

                    // Save points on axis
                    const vector3& P1d = center;
                    const vector3& P2d = center + vec;
                    const double P1P2d = (P2d - P1d).norm();
                    // Use point-to-line distances (same as cylinder distances)
                    vector3 P3;
                    for(uint i = 0; i < _cellActivatedCount; i++)
                    {
                        if(IFinal(i)) 
                        {
                            P3 = P.block<3,1>(0, i);
                            //D(i) = (P3-center).norm()-r;
                            D(i) = ((P2d - P1d).cross(P3 - P2d)).norm() / P1P2d - r;
                        }
                    }
                    D = D.array().square();

                    double mse = 0; 
                    for(uint i = 0; i < _cellActivatedCount; i++)
                    {
                        if(IFinal(i))
                            mse += D(i);
                    }
                    mse /= maxInliersCount;
                    _MSE.push_back(mse);

                    // Save points on axis, useful for computing distances afterwards
                    _pointsAxis1.push_back(P1d); 
                    _pointsAxis2.push_back(P2d);
                    _normalsAxis1Axis2.push_back(P1P2d);
                }
            }

            double Cylinder_Segment::get_distance(const vector3& point) const 
            {
                double minDist = this->get_distance(point, 0);
                for(vector3_vector::size_type i = 1; i < _pointsAxis1.size(); i++) 
                {
                    const double nd = this->get_distance(point, i);
                    if (minDist > nd)
                        minDist = nd;
                }
                return minDist;
            }

            double Cylinder_Segment::get_distance(const vector3& point, const uint segmentId) const 
            {
                const vector3 pointAxis2to1 = _pointsAxis2[segmentId] - _pointsAxis1[segmentId];
                const vector3 pointAxisTo2 = point - _pointsAxis2[segmentId]; 
                const double pointAxisNorm = (pointAxis2to1.cross(pointAxisTo2)).norm();
                return  pointAxisNorm / _normalsAxis1Axis2[segmentId] - _radius[segmentId];
            }

            /*
             *  Getters
             */
            uint Cylinder_Segment::get_segment_count() const 
            { 
                return _segmentCount; 
            }

            double Cylinder_Segment::get_MSE_at(const uint index) const 
            { 
                assert(index < _MSE.size());
                return _MSE[index]; 
            }

            bool Cylinder_Segment::is_inlier_at (const uint indexA, const uint indexB) const 
            { 
                assert(indexA < _inliers.size() and indexB < _inliers[indexA].size());
                return _inliers[indexA](indexB); 
            }

            uint Cylinder_Segment::get_local_to_global_mapping(const uint index) const 
            {
                assert(index < _local2globalMap.size());
                return _local2globalMap[index]; 
            }

            const vector3& Cylinder_Segment::get_axis1_point(const uint index) const 
            { 
                assert(index < _pointsAxis1.size());
                return _pointsAxis1[index];
            }

            const vector3& Cylinder_Segment::get_axis2_point(const uint index) const {
                assert(index < _pointsAxis2.size()); 
                return _pointsAxis2[index];
            }

            double Cylinder_Segment::get_axis_normal(const uint index) const 
            { 
                assert(index < _normalsAxis1Axis2.size());
                return _normalsAxis1Axis2[index]; 
            }

            double Cylinder_Segment::get_radius(const uint index) const
            {
                assert(index < _radius.size());
                return _radius[index]; 
            }

            double Cylinder_Segment::get_normal_similarity(const Cylinder_Segment& other) 
            {
                return std::abs(_axis.dot(other._axis));
            }

            const vector3 Cylinder_Segment::get_normal() const
            {
                return _axis;
            }



            Cylinder_Segment::~Cylinder_Segment() 
            {
                _cellActivatedCount = 0;
            }


        }
    }
}
