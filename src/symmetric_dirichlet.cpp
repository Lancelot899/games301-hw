#include "symmetric_dirichlet.h"

namespace games301 {

SymmetricDirichlet::SymmetricDirichlet(uint16_t max_iter, double min_error)
    : ProjectNewton(max_iter, min_error)
{}

void SymmetricDirichlet::PreRun(pmp::SurfaceMesh& mesh) {
    mesh.add_face_property<Eigen::Matrix<double, 2, 3>>("f:affine_matrix_I2_I3");
}

void SymmetricDirichlet::PostRun(pmp::SurfaceMesh& mesh) {
    auto f_affines = mesh.get_face_property<Eigen::Matrix<double, 2, 3>>("f:affine_matrix_I2_I3");
    mesh.remove_face_property(f_affines);
}

SymmetricDirichlet::~SymmetricDirichlet() {}

Eigen::VectorXd SymmetricDirichlet::ComputeEnergyGrad(
    const pmp::SurfaceMesh &mesh, const pmp::VertexProperty<Eigen::Vector2d> &tex)
{
    auto dvertexs = mesh.get_face_property<Eigen::Matrix2d>("f:dvertex");
    auto areas = mesh.get_face_property<double>("f:area");
    auto f_affines = mesh.get_face_property<Eigen::Matrix<double, 2, 3>>("f:affine_matrix_I2_I3");
    Eigen::VectorXd ret = Eigen::VectorXd::Zero(mesh.vertices_size() * 2);
    double cost = 0.0f;
    for(auto fiter = mesh.faces().begin(); fiter != mesh.faces().end(); ++fiter) {
        const pmp::Face &f = *fiter;
        Eigen::Matrix2d dv = dvertexs[f];
        Eigen::Vector2d uv[3];
        pmp::Vertex vs[3];
        pmp::IndexType vids[3];
        auto fviter = mesh.vertices(f);
        for(int i = 0; i < 3; ++i) {
            vs[i] = *fviter;
            uv[i] = tex[*fviter];
            vids[i] = vs[i].idx();
            fviter++;
        }
        Eigen::Matrix2d F = ComputeDeltaUV(dv, uv);
        cost += (F.squaredNorm() + 1.0 / F.squaredNorm());
//        std::cout << "cost = " << (F.squaredNorm() + 1.0 / F.squaredNorm())  << std::endl;
        Eigen::Map<Eigen::Vector4d> Fvec(F.data());
        Eigen::Matrix2d G = T_core_ * F * T_core_.transpose();
        Eigen::Map<Eigen::Vector4d> Gvec(G.data());
        double I2 = F.squaredNorm();
        double I3 = F.determinant();
        Eigen::Matrix<double, 2, 3> affine;
        affine.block<2, 2>(0, 0) = F;
        affine(0, 2) = I2;
        affine(1, 2) = I3;
        f_affines[f] = affine;
        double I32 = I3 * I3;
        double I33 = I32 * I3;

        Eigen::Vector4d dphidf =  (1.0 + 1.0 / I32) * Fvec - I2 / I33 * Gvec; // dphi / df
        Eigen::Matrix<double, 4, 6> dfdx = Eigen::Matrix<double, 4, 6>::Zero();
//        dfdx(0, 0) = dv(0, 0); dfdx(0, 2) = dv(0, 1); dfdx(0, 4) = dv(0, 2);
//        dfdx(1, 0) = dv(1, 0); dfdx(1, 2) = dv(1, 1); dfdx(1, 4) = dv(1, 2);
//        dfdx(2, 1) = dv(0, 0); dfdx(2, 3) = dv(0, 1); dfdx(2, 5) = dv(0, 2);
//        dfdx(3, 1) = dv(1, 0); dfdx(3, 3) = dv(1, 1); dfdx(3, 5) = dv(1, 2);
        dfdx(0, 0) = -dv(0, 0) - dv(1, 0); dfdx(0, 2) = dv(0, 0); dfdx(0, 4) = dv(1, 0);
        dfdx(1, 1) = -dv(0, 0) - dv(1, 0); dfdx(1, 3) = dv(0, 0); dfdx(0, 5) = dv(1, 0);
        dfdx(2, 0) = -dv(0, 1) - dv(1, 1); dfdx(2, 2) = dv(0, 1); dfdx(2, 4) = dv(1, 1);
        dfdx(3, 1) = -dv(0, 1) - dv(1, 1); dfdx(3, 3) = dv(0, 1); dfdx(3, 5) = dv(1, 1);

        Eigen::Matrix<double, 6, 1> dphidx = dfdx.transpose() * dphidf;
//        ret.block<2, 1>(vids[0] * 2, 0) += areas[vs[0]] * dphidx.block<2, 1>(0, 0);
//        ret.block<2, 1>(vids[1] * 2, 0) += areas[vs[1]] * dphidx.block<2, 1>(2, 0);
//        ret.block<2, 1>(vids[2] * 2, 0) += areas[vs[2]] * dphidx.block<2, 1>(4, 0);
//        ret.block<2, 1>(vids[0] * 2, 0) += areas[f] * dphidx.block<2, 1>(0, 0);
//        ret.block<2, 1>(vids[1] * 2, 0) += areas[f] * dphidx.block<2, 1>(2, 0);
//        ret.block<2, 1>(vids[2] * 2, 0) += areas[f] * dphidx.block<2, 1>(4, 0);
        ret.block<2, 1>(vids[0] * 2, 0) += dphidx.block<2, 1>(0, 0);
        ret.block<2, 1>(vids[1] * 2, 0) += dphidx.block<2, 1>(2, 0);
        ret.block<2, 1>(vids[2] * 2, 0) += dphidx.block<2, 1>(4, 0);
    }
    std::cout << "cost = " << cost << std::endl;
    return ret;
}

Eigen::SparseMatrix<double> SymmetricDirichlet::ProjectHessian(
    const pmp::SurfaceMesh &mesh, const pmp::VertexProperty<Eigen::Vector2d> &tex)
{
    auto dvertexs = mesh.get_face_property<Eigen::Matrix2d>("f:dvertex");
    auto point_areas = mesh.get_vertex_property<double>("v:area");
    auto f_affines = mesh.get_face_property<Eigen::Matrix<double, 2, 3>>("f:affine_matrix_I2_I3");
    Eigen::VectorXd ret = Eigen::VectorXd::Zero(mesh.vertices_size() * 2);
    std::vector<Eigen::Triplet<double>> h_vec;
    h_vec.reserve(mesh.faces_size() * 6 * 4 * 4);
    for(auto fiter = mesh.faces().begin(); fiter != mesh.faces().end(); ++fiter) {
        const pmp::Face &f = *fiter;
        Eigen::Matrix<double, 2, 2> dv = dvertexs[f];
        Eigen::Vector2d uv[3];
        pmp::Vertex vs[3];
        pmp::IndexType vids[3];
        auto fviter = mesh.vertices(f);
        for(int i = 0; i < 3; ++i) {
            vs[i] = *fviter;
            uv[i] = tex[*fviter];
            vids[i] = vs[i].idx();
            fviter++;
        }
        Eigen::Matrix<double, 4, 6> dfdx = Eigen::Matrix<double, 4, 6>::Zero();
        //        dfdx(0, 0) = dv(0, 0); dfdx(0, 2) = dv(0, 1); dfdx(0, 4) = dv(0, 2);
        //        dfdx(1, 0) = dv(1, 0); dfdx(1, 2) = dv(1, 1); dfdx(1, 4) = dv(1, 2);
        //        dfdx(2, 1) = dv(0, 0); dfdx(2, 3) = dv(0, 1); dfdx(2, 5) = dv(0, 2);
        //        dfdx(3, 1) = dv(1, 0); dfdx(3, 3) = dv(1, 1); dfdx(3, 5) = dv(1, 2);
        dfdx(0, 0) = -dv(0, 0) - dv(1, 0); dfdx(0, 2) = dv(0, 0); dfdx(0, 4) = dv(1, 0);
        dfdx(1, 1) = -dv(0, 0) - dv(1, 0); dfdx(1, 3) = dv(0, 0); dfdx(0, 5) = dv(1, 0);
        dfdx(2, 0) = -dv(0, 1) - dv(1, 1); dfdx(2, 2) = dv(0, 1); dfdx(2, 4) = dv(1, 1);
        dfdx(3, 1) = -dv(0, 1) - dv(1, 1); dfdx(3, 3) = dv(0, 1); dfdx(3, 5) = dv(1, 1);
        Eigen::Matrix<double, 2, 3> affine = f_affines[f];
        Eigen::Matrix2d F = affine.block<2, 2>(0, 0);
        double I2 = affine(0, 2);
        double I3 = affine(1, 2);

        Eigen::JacobiSVD<Eigen::Matrix2d, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(F);
        Eigen::Vector2d sv = svd.singularValues();
        Eigen::Matrix2d U = svd.matrixU();
        Eigen::Matrix2d V = svd.matrixV();
        Eigen::Matrix2d D1 = U * D1_core_ * V.transpose();
        Eigen::Matrix2d D2 = U * D2_core_ * V.transpose();
        Eigen::Matrix2d L = U * L_core_ * V.transpose();
        Eigen::Matrix2d T = U * T_core_ * V.transpose();

        double lambda0 = 1 + 3.0 / sv[0] / sv[0] / sv[0] / sv[0];
        double lambda1 = 1 + 3.0 / sv[1] / sv[1] / sv[1] / sv[1];
        double lambda2 = 1 + 1.0 / I3 / I3;
        double I2I3 = I2 / I3 / I3 / I3;
        double lambda3 = lambda2 - I2I3;
        lambda2 += I2I3;

        Eigen::Map<Eigen::Vector4d> d1(D1.data());
        Eigen::Map<Eigen::Vector4d> d2(D2.data());
        Eigen::Map<Eigen::Vector4d> t(T.data());
        Eigen::Map<Eigen::Vector4d> l(L.data());
        Eigen::Matrix4d hessian = lambda0 * d1 * d1.transpose() + lambda1 * d2 * d2.transpose() + lambda2 * t * t.transpose() + lambda3 * l * l.transpose();

        Eigen::Matrix<double, 6, 6> dphidxdx = dfdx.transpose() * hessian * dfdx;

        h_vec.emplace_back(vids[0] * 2, vids[0] * 2, dphidxdx(0, 0));
        h_vec.emplace_back(vids[0] * 2, vids[0] * 2 + 1, dphidxdx(0, 1));
        h_vec.emplace_back(vids[0] * 2 + 1, vids[0] * 2 + 1, dphidxdx(1, 1));
        h_vec.emplace_back(vids[0] * 2 + 1, vids[0] * 2, dphidxdx(1, 0));

        h_vec.emplace_back(vids[1] * 2, vids[1] * 2, dphidxdx(2, 2));
        h_vec.emplace_back(vids[1] * 2, vids[1] * 2 + 1, dphidxdx(2, 3));
        h_vec.emplace_back(vids[1] * 2 + 1, vids[1] * 2 + 1, dphidxdx(3, 3));
        h_vec.emplace_back(vids[1] * 2 + 1, vids[1] * 2, dphidxdx(3, 2));

        h_vec.emplace_back(vids[2] * 2, vids[2] * 2, dphidxdx(4, 4));
        h_vec.emplace_back(vids[2] * 2, vids[2] * 2 + 1, dphidxdx(4, 5));
        h_vec.emplace_back(vids[2] * 2 + 1, vids[2] * 2 + 1, dphidxdx(5, 5));
        h_vec.emplace_back(vids[2] * 2 + 1, vids[2] * 2, dphidxdx(5, 4));

        h_vec.emplace_back(vids[0] * 2, vids[1] * 2, dphidxdx(0, 2));
        h_vec.emplace_back(vids[0] * 2, vids[1] * 2 + 1, dphidxdx(0, 3));
        h_vec.emplace_back(vids[0] * 2 + 1, vids[1] * 2 + 1, dphidxdx(1, 3));
        h_vec.emplace_back(vids[0] * 2 + 1, vids[1] * 2, dphidxdx(1, 2));

        h_vec.emplace_back(vids[0] * 2, vids[2] * 2, dphidxdx(0, 4));
        h_vec.emplace_back(vids[0] * 2, vids[2] * 2 + 1, dphidxdx(0, 5));
        h_vec.emplace_back(vids[0] * 2 + 1, vids[2] * 2 + 1, dphidxdx(1, 5));
        h_vec.emplace_back(vids[0] * 2 + 1, vids[2] * 2, dphidxdx(1, 4));

        h_vec.emplace_back(vids[1] * 2, vids[0] * 2, dphidxdx(2, 0));
        h_vec.emplace_back(vids[1] * 2, vids[0] * 2 + 1, dphidxdx(2, 1));
        h_vec.emplace_back(vids[1] * 2 + 1, vids[0] * 2 + 1, dphidxdx(3, 1));
        h_vec.emplace_back(vids[1] * 2 + 1, vids[0] * 2, dphidxdx(3, 0));

        h_vec.emplace_back(vids[2] * 2, vids[0] * 2, dphidxdx(4, 0));
        h_vec.emplace_back(vids[2] * 2, vids[0] * 2 + 1, dphidxdx(4, 1));
        h_vec.emplace_back(vids[2] * 2 + 1, vids[0] * 2 + 1, dphidxdx(5, 1));
        h_vec.emplace_back(vids[2] * 2 + 1, vids[0] * 2, dphidxdx(5, 0));

        h_vec.emplace_back(vids[1] * 2, vids[2] * 2, dphidxdx(2, 4));
        h_vec.emplace_back(vids[1] * 2, vids[2] * 2 + 1, dphidxdx(2, 5));
        h_vec.emplace_back(vids[1] * 2 + 1, vids[2] * 2 + 1, dphidxdx(3, 5));
        h_vec.emplace_back(vids[1] * 2 + 1, vids[2] * 2, dphidxdx(3, 4));

        h_vec.emplace_back(vids[2] * 2, vids[1] * 2, dphidxdx(4, 2));
        h_vec.emplace_back(vids[2] * 2, vids[1] * 2 + 1, dphidxdx(4, 3));
        h_vec.emplace_back(vids[2] * 2 + 1, vids[1] * 2 + 1, dphidxdx(5, 3));
        h_vec.emplace_back(vids[2] * 2 + 1, vids[1] * 2, dphidxdx(5, 2));
    }

    Eigen::SparseMatrix<double> H(mesh.vertices_size() * 2, mesh.vertices_size() * 2);
    H.setFromTriplets(h_vec.begin(), h_vec.end());
    return H;
}

double SymmetricDirichlet::LineSearch(const pmp::SurfaceMesh &mesh,
                                      const pmp::VertexProperty<Eigen::Vector2d> &tex,
                                      const Eigen::VectorXd &d) {
    return 1;
}

}