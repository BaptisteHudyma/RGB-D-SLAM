#include "cylinder_segment.hpp"
#include "parameters.hpp"
#include "logger.hpp"

namespace rgbd_slam {
namespace features {
namespace primitives {

    using namespace std;

    Cylinder_Segment::Cylinder_Segment(const Cylinder_Segment& seg, const uint subRegionId) :
        _cellActivatedCount(0),
        _segmentCount(0),
        _local2globalMap(nullptr)
    {
        //copy stored data
        _radius.push_back(seg._radius[subRegionId]);
        _centers.push_back(seg._centers[subRegionId]);
        //_pointsAxis1 = seg.pointsAxis1; 
        //_pointsAxis2 = seg.pointsAxis2;
        //_normalsAxis1Axis2 = seg._normalsAxis1Axis2;
        _axis[0] = seg._axis[0];
        _axis[1] = seg._axis[1];
        _axis[2] = seg._axis[2];
    }

    /*
     *  Copy constructor
     */
    Cylinder_Segment::Cylinder_Segment(const Cylinder_Segment& seg) :
        _centers(seg._centers),
        _radius(seg._radius),
        _cellActivatedCount(0),
        _segmentCount(0),
        _local2globalMap(nullptr)
    {
        //_pointsAxis1 = seg.pointsAxis1; 
        //_pointsAxis2 = seg.pointsAxis2;
        //_normalsAxis1Axis2 = seg._normalsAxis1Axis2;
        _axis[0] = seg._axis[0];
        _axis[1] = seg._axis[1];
        _axis[2] = seg._axis[2];
    }

    Cylinder_Segment::Cylinder_Segment(const std::unique_ptr<Plane_Segment>* planeGrid, const uint planeCount, const bool* activatedMask, const uint cellActivatedCount) :
        _cellActivatedCount(cellActivatedCount),
        _segmentCount(0),
        _local2globalMap(nullptr)
    {
        uint samplesCount = planeCount;

        const float minimumCyinderScore = Parameters::get_cylinder_ransac_minimm_score();
        const float maximumSqrtDistance = Parameters::get_cylinder_ransac_max_distance();

        _local2globalMap = new uint[_cellActivatedCount];

        Eigen::MatrixXd N(3, 2 * _cellActivatedCount);
        Eigen::MatrixXd P(3, _cellActivatedCount);

        // Init. P and N
        int j = 0;
        for(uint i = 0; i < samplesCount; i++){
            if (activatedMask[i]){
                const vector3& planeNormal = planeGrid[i]->get_normal();
                const vector3& planeMean = planeGrid[i]->get_mean();
                N(0, j) = planeNormal[0];
                N(1, j) = planeNormal[1];
                N(2, j) = planeNormal[2];
                P(0, j) = planeMean[0];
                P(1, j) = planeMean[1];
                P(2, j) = planeMean[2];
                _local2globalMap[j] = i;
                j++;
            }
        }
        // Concatenate [N -N]
        for(uint i = 0; i < samplesCount; i++){
            if (activatedMask[i]){
                const vector3& planeNormal = planeGrid[i]->get_normal();
                N(0, j) = -planeNormal[0];
                N(1, j) = -planeNormal[1];
                N(2, j) = -planeNormal[2];
                j++;
            }
        }

        // Compute covariance
        Eigen::MatrixXd cov = (N * N.adjoint()) / static_cast<double>(N.cols() - 1);

        // PCA using QR decomposition for symmetric matrices
        Eigen::SelfAdjointEigenSolver<matrix33> es(cov);
        const vector3& S = es.eigenvalues();
        double score = S(2)/S(0);

        // Checkpoint 1
        if(score < minimumCyinderScore){
            return;
        }

        const vector3& vec = es.eigenvectors().col(0);
        _axis[0] = vec(0); 
        _axis[1] = vec(1); 
        _axis[2] = vec(2);

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
        float pSuccess = 0.8;
        float w = 0.33;
        float K = log(1 - pSuccess) / log(1 - pow(w, 3));

        int mLeft = _cellActivatedCount;
        Matrixb idsLeftMask(1, _cellActivatedCount);

        vector<int> idsLeft;
        for(uint i = 0; i < _cellActivatedCount; i++){
            idsLeft.push_back(i);
            idsLeftMask(i) = true;
        }
        // Sequential RANSAC main loop
        while(mLeft > 5 and mLeft > 0.1 * _cellActivatedCount){
            Eigen::MatrixXd A, center, e1, e2;
            Eigen::MatrixXd D(1, _cellActivatedCount);
            Matrixb I(1, _cellActivatedCount);
            Matrixb IFinal(1, _cellActivatedCount);
            double minHypothesisDist = maximumSqrtDistance * mLeft;
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
                I = (D.array() < maximumSqrtDistance);

                //MSAC truncated distance
                double dist = 0.0;
                int inliersCount = 0;
                for(uint i = 0;i < _cellActivatedCount; i++){
                    if(idsLeftMask(i)){
                        if(I(i)) {
                            inliersCount += 1;
                            dist += D(i);
                        }else{
                            dist += maximumSqrtDistance;
                        }
                    }
                }

                if(dist < minHypothesisDist){
                    minHypothesisDist = dist;
                    maxInliersCount = inliersCount;
                    for(uint i = 0; i < _cellActivatedCount; i++){
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
            for(uint i = 0; i < _cellActivatedCount; i++){
                if(IFinal(i)) {
                    idsLeftMask(i) = false;
                    mLeft--;
                }
                else if(idsLeftMask(i)) {
                    idsLeft.push_back(i);
                }
            }

            // LLS solution using all inliers
            e1.setZero();
            e2.setZero();
            double b = 0;
            for(uint i = 0; i < _cellActivatedCount; i++){
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
            _segmentCount += 1;
            _radius.push_back(r);
            _centers.push_back(center);
            _inliers.push_back(IFinal);

            // Save points on axis
            const vector3& P1d = center;
            const vector3& P2d = center + vec;
            double P1P2d = (P2d - P1d).norm();
            // Use point-to-line distances (same as cylinder distances)
            vector3 P3;
            for(uint i = 0; i < _cellActivatedCount; i++){
                if(IFinal(i)) {
                    P3 = P.block<3,1>(0, i);
                    //D(i) = (P3-center).norm()-r;
                    D(i) = ((P2d - P1d).cross(P3 - P2d)).norm() / P1P2d - r;
                }
            }
            D = D.array().square();

            double mse = 0; 
            for(uint i = 0; i < _cellActivatedCount; i++){
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

    double Cylinder_Segment::get_distance(const vector3& point) const {
        double minDist = this->get_distance(point, 0);
        for(vector3_vector::size_type i = 1; i < _pointsAxis1.size(); i++) {
            double nd = this->get_distance(point, i);
            if (minDist > nd)
                minDist = nd;
        }
        return minDist;
    }

    double Cylinder_Segment::get_distance(const vector3& point, const uint segmentId) const {
        return (
                (_pointsAxis2[segmentId] - _pointsAxis1[segmentId]).cross(
                    point - _pointsAxis2[segmentId]
                    )
               ).norm() / _normalsAxis1Axis2[segmentId] - _radius[segmentId];
    }

    /*
     *  Getters
     */
    uint Cylinder_Segment::get_segment_count() const { 
        return _segmentCount; 
    }

    double Cylinder_Segment::get_MSE_at(const uint index) const { 
        if(index >= _MSE.size()) {
            utils::log_error("get_MSE required index over MSE vector size");
            exit(-1);
        }
        return _MSE[index]; 
    }

    bool Cylinder_Segment::is_inlier_at (const uint indexA, const uint indexB) const { 
        if(indexA >= _inliers.size() or indexB >= _inliers[indexA].size()) {
            utils::log_error("get_inlier required index over inlier vector size");
            exit(-1);
        }
        return _inliers[indexA](indexB); 
    }

    uint Cylinder_Segment::get_local_to_global_mapping(const uint index) const {
        if (index >= _cellActivatedCount) {
            utils::log_error("get_local_to_global required index over array size");
            exit(-1);
        }
        if(_local2globalMap == nullptr) {
            utils::log_error("_local2globalMap not initialized");
            exit(-1);
        }
        return _local2globalMap[index]; 
    }

    const vector3& Cylinder_Segment::get_axis1_point(const uint index) const { 
        if(index >= _pointsAxis1.size()) {
            utils::log_error("get_axis_1 required index over axis1 vector size");
            exit(-1);
        }
        return _pointsAxis1[index];
    }

    const vector3& Cylinder_Segment::get_axis2_point(const uint index) const {
        if(index >= _pointsAxis2.size()) {
            utils::log_error("get_axis_2 required index over axis2 size");
            exit(-1);
        }
        return _pointsAxis2[index];
    }

    double Cylinder_Segment::get_axis_normal(const uint index) const { 
        if(index >= _normalsAxis1Axis2.size()) {
            utils::log_error("get_axis_normal required index over normals vector size");
            exit(-1);
        }
        return _normalsAxis1Axis2[index]; 
    }

    double Cylinder_Segment::get_radius(const uint index) const {
        if(index >= _radius.size()) {
            utils::log_error("get_radius required index over radius vector size");
            exit(-1);
        }
        return _radius[index]; 
    }

    double Cylinder_Segment::get_normal_similarity(const Cylinder_Segment& other) {
        return std::abs( _axis[0] * other._axis[0] + _axis[1] * other._axis[1] + _axis[2] * other._axis[2]  );
    }

    const vector3 Cylinder_Segment::get_normal() const {
        return _axis;
    }



    Cylinder_Segment::~Cylinder_Segment() {
        if(_local2globalMap != nullptr) {
            delete[] _local2globalMap;    
            _local2globalMap = nullptr;
        }
        _cellActivatedCount = 0;
    }


}
}
}
