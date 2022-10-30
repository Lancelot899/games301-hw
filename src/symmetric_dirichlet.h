//
// Created by 范帝楷 on 2022/10/29.
//

#ifndef PMP_TEMPLATE_SYMMETRIC_DIRICHLET_H
#define PMP_TEMPLATE_SYMMETRIC_DIRICHLET_H

#include "projected_newton.h"

namespace games301 {

class SymmetricDirichlet : public ProjectNewton {
public:
    SymmetricDirichlet(uint16_t max_iter, double min_error);
    ~SymmetricDirichlet();

protected:
    Eigen::VectorXd ComputeEnergyGrad(const pmp::SurfaceMesh& mesh, const pmp::VertexProperty<Eigen::Vector2d> &tex);
    Eigen::SparseMatrix<double> ProjectHessian(const pmp::SurfaceMesh& mesh, const pmp::VertexProperty<Eigen::Vector2d> &tex);
    double LineSearch(const pmp::SurfaceMesh& mesh, const pmp::VertexProperty<Eigen::Vector2d> &tex, const Eigen::VectorXd &d);
private:

};


}

#endif //PMP_TEMPLATE_SYMMETRIC_DIRICHLET_H
