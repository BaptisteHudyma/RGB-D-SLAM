#include "PNPSolver.hpp"

#include "Constants.hpp"
#include "LocalMap.hpp"

#include <Eigen/StdVector>
#include <numeric>
#include <opencv2/opencv.hpp>

#include <g2o/config.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/types/icp/types_icp.h>
#include <g2o/solvers/pcg/linear_solver_pcg.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#define N_PASSES 2

namespace poseEstimation {

    PNP_Solver::PNP_Solver(double fx, double fy, double cx, double cy, double baseline) :
        _fx(fx), _fy(fy),
        _cx(cx), _cy(cy),
        _baseline(baseline)
    {
        _optimizer.setVerbose(false);

        auto linearSolver = std::make_unique<g2o::LinearSolverPCG<g2o::BlockSolver_6_3::PoseMatrixType>>();

        //will be freed by the optimizer destructor
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
        _optimizer.setAlgorithm(solver);
    }

    PNP_Solver::~PNP_Solver() {
    }

    Pose PNP_Solver::compute_pose(const Pose& camPose, Image_Features_Struct& features, const vector3_array& matchedPoints, const std::vector<int>& matchLeft) {
        quaternion orientation = camPose.get_orientation_quaternion();
        vector3 position = camPose.get_position();

        //add camera as a free point (it's position/orientation will be optimized)
        g2o::SBACam sbaCam(orientation, position);
        sbaCam.setKcam(_fx, _fy, _cx, _cy, _baseline);
        g2o::VertexCam *camVertex = new g2o::VertexCam();
        camVertex->setId(0);
        camVertex->setEstimate(sbaCam);
        camVertex->setFixed(false);
        _optimizer.addVertex(camVertex);

        // add mono measurments
        static const double monoChi = sqrt(LVT_REPROJECTION_TH2);
        int vertexID = 1;
        std::vector<g2o::EdgeProjectP2MC *> monoEdges(matchedPoints.size());
        for (size_t i = 0, count = matchedPoints.size(); i < count; i++)
        {
            //set the feature as a fixed point in graph
            g2o::VertexPointXYZ *pointVertex = new g2o::VertexPointXYZ();
            pointVertex->setId(vertexID++);
            pointVertex->setMarginalized(false);
            pointVertex->setEstimate(matchedPoints[i]);
            pointVertex->setFixed(true);
            _optimizer.addVertex(pointVertex);

            //set an edge between point and camera
            g2o::EdgeProjectP2MC *edge = new g2o::EdgeProjectP2MC();
            edge->setVertex(0, pointVertex);
            edge->setVertex(1, camVertex);

            //set an edge between this frame point and map point
            cv::Point2f mpCv = features.get_keypoint(matchLeft[i]).pt;
            vector2 imgPt;
            imgPt << mpCv.x, mpCv.y;
            edge->setMeasurement(imgPt);
            edge->information() = Eigen::Matrix2d::Identity();
            g2o::RobustKernel *rkh = new g2o::RobustKernelCauchy;
            rkh->setDelta(monoChi);
            edge->setRobustKernel(rkh);

            _optimizer.addEdge(edge);
            monoEdges[i] = edge;
        }

        // perform optimization
        std::vector<int> monoInlierMarks(monoEdges.size(), 1);
        for (int i = 0; i < N_PASSES; i++)
        {
            _optimizer.initializeOptimization(0);
            _optimizer.optimize(5);
            for (std::vector<g2o::EdgeProjectP2MC *>::size_type k = 0; k < monoEdges.size(); k++)
            {
                if (monoEdges[k]->chi2() > LVT_REPROJECTION_TH2)
                {
                    monoEdges[k]->setLevel(1);
                    monoInlierMarks[k] = 0;
                }
            }
        }

        // retrieve optimized pose
        const Eigen::Vector3d opPosition = camVertex->estimate().translation();
        const Eigen::Quaterniond opOrientation = camVertex->estimate().rotation();
        Pose opPose = Pose(opPosition, opOrientation);

        _optimizer.clear();
        return opPose;
    }

}
