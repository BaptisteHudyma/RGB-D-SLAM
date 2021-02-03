#include "PNPSolver.hpp"

#include "Constants.hpp"
#include "LocalMap.hpp"

#include <Eigen/StdVector>
#include <numeric>
#include <opencv2/opencv.hpp>

#include <g2o/config.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/types/icp/types_icp.h>
#include <g2o/solvers/pcg/linear_solver_pcg.h>
#include <g2o/types/sba/types_six_dof_expmap.h>


using namespace poseEstimation;

#define N_PASSES 2

PNP_Solver::PNP_Solver(double fx, double fy, double cx, double cy, double baseline) :
    fx(fx), fy(fy),
    cx(cx), cy(cy),
    baseline(baseline),
    optimizer(nullptr)
{
    this->optimizer = new g2o::SparseOptimizer();
    this->optimizer->setVerbose(false);

    auto linearSolver = std::make_unique<g2o::LinearSolverPCG<g2o::BlockSolver_6_3::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg solver{std::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))};

    //this->optimizer->setAlgorithm(std::move(solver));
    this->optimizer->setAlgorithm(&solver);
}

PNP_Solver::~PNP_Solver() {
    delete this->optimizer;
}

Pose PNP_Solver::compute_pose(const Pose& camPose, Image_Features_Struct& features, const vector3_array& matchedPoints, const std::vector<int>& matchOutliers) {
    quaternion orientation = camPose.get_orientation_quaternion();
    vector3 position = camPose.get_position();

    g2o::SBACam sbaCam(orientation, position);
    sbaCam.setKcam(this->fx, this->fy, this->cx, this->cy, this->baseline);
    g2o::VertexCam *camVertex = new g2o::VertexCam();
    camVertex->setId(0);
    camVertex->setEstimate(sbaCam);
    camVertex->setFixed(false);
    this->optimizer->addVertex(camVertex);

    // add mono measurments
    static const double monoChi = sqrt(LVT_REPROJECTION_TH2);
    int vertexID = 1;
    std::vector<g2o::EdgeProjectP2MC *> monoEdges(matchedPoints.size());
    for (size_t i = 0, count = matchedPoints.size(); i < count; i++)
    {
        g2o::VertexPointXYZ *pointVertex = new g2o::VertexPointXYZ();
        pointVertex->setId(vertexID++);
        pointVertex->setMarginalized(false);
        pointVertex->setEstimate(matchedPoints[i]);
        pointVertex->setFixed(true);
        this->optimizer->addVertex(pointVertex);

        g2o::EdgeProjectP2MC *edge = new g2o::EdgeProjectP2MC();
        edge->setVertex(0, pointVertex);
        edge->setVertex(1, camVertex);
        cv::Point2f mpCv = features.get_keypoint(matchOutliers[i]).pt;
        vector2 imgPt;
        imgPt << mpCv.x, mpCv.y;
        edge->setMeasurement(imgPt);
        edge->information() = Eigen::Matrix2d::Identity();
        g2o::RobustKernel *rkh = new g2o::RobustKernelCauchy;
        edge->setRobustKernel(rkh);
        rkh->setDelta(monoChi);
        this->optimizer->addEdge(edge);
        monoEdges[i] = edge;
    }

    // perform optimzation
    std::vector<int> monoInlierMarks(monoEdges.size(), 1);
    for (int i = 0; i < N_PASSES; i++)
    {
        this->optimizer->initializeOptimization(0);
        this->optimizer->optimize(5);
        for (int k = 0; k < monoEdges.size(); k++)
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

    this->optimizer->clear();
    return opPose;
}
