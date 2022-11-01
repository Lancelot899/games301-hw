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
    void PreRun(pmp::SurfaceMesh& mesh);
    void PostRun(pmp::SurfaceMesh& mesh);
    Eigen::VectorXd ComputeEnergyGrad(const pmp::SurfaceMesh& mesh, const Eigen::VectorXd &tex);
    Eigen::SparseMatrix<double> ProjectHessian(const pmp::SurfaceMesh& mesh, const Eigen::VectorXd &tex);
    double LineSearch(const pmp::SurfaceMesh& mesh, const Eigen::VectorXd &tex, const Eigen::VectorXd &d);
private:
    double last_cost = 1e9;
    double last_alpha_ = 1.0;
};


}

#endif //PMP_TEMPLATE_SYMMETRIC_DIRICHLET_H
