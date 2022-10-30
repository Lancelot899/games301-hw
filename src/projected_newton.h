//
// Created by 范帝楷 on 2022/10/29.
//

#ifndef PMP_TEMPLATE_PROJECTED_NEWTON_H
#define PMP_TEMPLATE_PROJECTED_NEWTON_H
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <pmp/SurfaceMesh.h>


namespace games301 {

class ProjectNewton {
public:
    ProjectNewton(uint16_t max_iter, double min_error);
    virtual ~ProjectNewton() = default;
    virtual bool Run(pmp::SurfaceMesh& mesh);

protected:
    void ComputeLocalCoord(pmp::SurfaceMesh& mesh);
    void ComputeAreaAndNormal(pmp::SurfaceMesh& mesh);
    void ComputePointsArea(pmp::SurfaceMesh& mesh);
    void ComputeFaceCenter(pmp::SurfaceMesh& mesh);
    void ComputeEdgeLength(pmp::SurfaceMesh& mesh);
    void CleanFaceCenter(pmp::SurfaceMesh& mesh);
    void CleanEdgeLength(pmp::SurfaceMesh& mesh);
    void CleanLocalCoord(pmp::SurfaceMesh& mesh);
    void CleanAreaAndNormal(pmp::SurfaceMesh& mesh);
    void CleanPointsArea(pmp::SurfaceMesh& mesh);

    void ComputeDVertexPerFace(pmp::SurfaceMesh& mesh);
    void CleanDVertexPerFace(pmp::SurfaceMesh& mesh);
    virtual Eigen::Matrix<double, 2, 3> ComputeDVertex(const Eigen::Vector2d local_vertex[3], double area);
    virtual Eigen::Matrix2d ComputeDeltaUV(const Eigen::Matrix<double, 2, 3> &dvertex, const Eigen::Vector2d uv[3]);
    virtual Eigen::Matrix2d ComputeDeltaUV(const Eigen::Vector2d local_vertex[3], double area, const Eigen::Vector2d uv[3]);
    virtual Eigen::Matrix2d ComputeDeltaUV(const Eigen::Vector3d vertex[3], const Eigen::Vector2d uv[3]);
    virtual Eigen::VectorXd ComputeEnergyGrad(const pmp::SurfaceMesh& mesh, const pmp::VertexProperty<Eigen::Vector2d> &tex) = 0;
    virtual Eigen::SparseMatrix<double> ProjectHessian(const pmp::SurfaceMesh& mesh, const pmp::VertexProperty<Eigen::Vector2d> &tex) = 0;
    virtual double LineSearch(const pmp::SurfaceMesh& mesh, const pmp::VertexProperty<Eigen::Vector2d> &tex, const Eigen::VectorXd &d) = 0;


protected:
    int    max_iter_ = 0;
    double min_error_ = 0;
};



}

#endif //PMP_TEMPLATE_PROJECTED_NEWTON_H
