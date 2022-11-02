#include "symmetric_dirichlet.h"
#include "Scalar.hh"
#include "TinyAD/Utils/HessianProjection.hh"
#include <TinyAD/Operations/SVD.hh>

namespace games301 {
const bool number_check_ = true;
SymmetricDirichlet::SymmetricDirichlet(uint16_t max_iter, double min_error)
    : ProjectNewton(max_iter, min_error)
{}

void SymmetricDirichlet::PreRun(pmp::SurfaceMesh& mesh) {
    mesh.add_face_property<Eigen::Matrix<double, 2, 3>>("f:affine_matrix_I2_I3");
    mesh.add_vertex_property<Eigen::Vector2i>("v:vid");
}

void SymmetricDirichlet::PostRun(pmp::SurfaceMesh& mesh) {
    auto f_affines = mesh.get_face_property<Eigen::Matrix<double, 2, 3>>("f:affine_matrix_I2_I3");
    mesh.remove_face_property(f_affines);

    auto vids = mesh.get_vertex_property<Eigen::Vector2i>("v:vid");
    mesh.remove_vertex_property(vids);
}

SymmetricDirichlet::~SymmetricDirichlet() {}

Eigen::VectorXd SymmetricDirichlet::ComputeEnergyGrad(
    const pmp::SurfaceMesh &mesh, const Eigen::VectorXd &tex)
{
    auto dvertexs = mesh.get_face_property<Eigen::Matrix2d>("f:dvertex");
    auto areas = mesh.get_face_property<double>("f:area");
    auto f_affines = mesh.get_face_property<Eigen::Matrix<double, 2, 3>>("f:affine_matrix_I2_I3");
    Eigen::VectorXd ret = Eigen::VectorXd::Zero(mesh.vertices_size() * 2);
    last_cost = 0.0;
    for(auto fiter = mesh.faces().begin(); fiter != mesh.faces().end(); ++fiter) {
        const pmp::Face &f = *fiter;
        Eigen::Matrix2d dv = dvertexs[f];
        Eigen::Vector2d uv[3];
        pmp::Vertex vs[3];
        pmp::IndexType vids[3];
        auto fviter = mesh.vertices(f);
        for(int i = 0; i < 3; ++i) {
            vs[i] = *fviter;
            vids[i] = vs[i].idx();
            uv[i] = tex.block<2, 1>(vids[i] * 2, 0);
            fviter++;
        }


#ifdef CHECK_GRAD
        Eigen::Vector<double, 6>  uv3;
        uv3 << uv[0] , uv[1] , uv[2];
        typedef std::conditional<number_check_, TinyAD::Double<6>, double>::type T;
        Eigen::Vector<TinyAD::Double<6>, 6> uv3ad = TinyAD::Double<6>::make_active(uv3);
        Eigen::Vector2<T> a(uv3ad[0], uv3ad[1]);
        Eigen::Vector2<T> b(uv3ad[2], uv3ad[3]);
        Eigen::Vector2<T> c(uv3ad[4], uv3ad[5]);

        Eigen::Matrix2<T> target;
        target << b - a , c - a;
        Eigen::Matrix2<T> Fad = target * dv;

        T costad = 0.5 * areas[f] * (Fad.squaredNorm() + Fad.inverse().squaredNorm());
#endif //#ifdef CHECK_GRAD
        Eigen::Matrix2d F = ComputeDeltaUV(dv, uv);

#ifdef CHECK_GRAD
        typedef std::conditional<number_check_, TinyAD::Double<4>, double>::type T4;
        Eigen::Vector4<T4> Fvvad = T4::make_active({F(0, 0), F(1, 0), F(0, 1), F(1, 1)});
        Eigen::Matrix2<T4> FFad;
        FFad << Fvvad[0],  Fvvad[1], Fvvad[2], Fvvad[3];

        auto cost_F = 0.5 * (FFad.squaredNorm() + FFad.inverse().squaredNorm());
#endif //#ifdef CHECK_GRAD
        last_cost += (F.squaredNorm() +  F.inverse().squaredNorm());

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
#ifdef CHECK_GRAD
        std::cout << "df\n";
        std::cout << cost_F.grad.transpose() << std::endl;
        std::cout << dphidf.transpose() << std::endl;
#endif //#ifdef CHECK_GRAD
        Eigen::Matrix<double, 4, 6> dfdx = Eigen::Matrix<double, 4, 6>::Zero();
        dfdx(0, 0) = -dv(0, 0) - dv(1, 0); dfdx(0, 2) = dv(0, 0); dfdx(0, 4) = dv(1, 0);
        dfdx(1, 1) = -dv(0, 0) - dv(1, 0); dfdx(1, 3) = dv(0, 0); dfdx(1, 5) = dv(1, 0);
        dfdx(2, 0) = -dv(0, 1) - dv(1, 1); dfdx(2, 2) = dv(0, 1); dfdx(2, 4) = dv(1, 1);
        dfdx(3, 1) = -dv(0, 1) - dv(1, 1); dfdx(3, 3) = dv(0, 1); dfdx(3, 5) = dv(1, 1);
//        std::cout << dfdx << std::endl;
        Eigen::Matrix<double, 6, 1> dphidx = areas[f] * dfdx.transpose() * dphidf;
#ifdef CHECK_GRAD
        // std::cout << "grad:\n";
        // std::cout << costad.grad.transpose() << std::endl;
        // std::cout << dphidx.transpose() << std::endl;
//        dphidx = costad.grad;
#endif //#ifdef CHECK_GRAD
//        ret.block<2, 1>(vids[0] * 2, 0) += areas[vs[0]] * dphidx.block<2, 1>(0, 0);
//        ret.block<2, 1>(vids[1] * 2, 0) += areas[vs[1]] * dphidx.block<2, 1>(2, 0);
//        ret.block<2, 1>(vids[2] * 2, 0) += areas[vs[2]] * dphidx.block<2, 1>(4, 0);
//        ret.block<2, 1>(vids[0] * 2, 0) += areas[f] * dphidx.block<2, 1>(0, 0);
//        ret.block<2, 1>(vids[1] * 2, 0) += areas[f] * dphidx.block<2, 1>(2, 0);
//        ret.block<2, 1>(vids[2] * 2, 0) += areas[f] * dphidx.block<2, 1>(4, 0);
//        std::cout << dphidx.transpose() << std::endl;
//        exit(-1);
        ret.block<2, 1>(vids[0] * 2, 0) += dphidx.block<2, 1>(0, 0);
        ret.block<2, 1>(vids[1] * 2, 0) += dphidx.block<2, 1>(2, 0);
        ret.block<2, 1>(vids[2] * 2, 0) += dphidx.block<2, 1>(4, 0);
    }
    std::cout << "cost = " << last_cost << std::endl;
    return ret;
}

Eigen::SparseMatrix<double> SymmetricDirichlet::ProjectHessian(
    const pmp::SurfaceMesh &mesh, const Eigen::VectorXd &tex) {
    auto dvertexs = mesh.get_face_property<Eigen::Matrix2d>("f:dvertex");
    auto areas = mesh.get_face_property<double>("f:area");
    auto f_affines = mesh.get_face_property<Eigen::Matrix<double, 2, 3>>("f:affine_matrix_I2_I3");
    std::vector<Eigen::Triplet<double>> h_vec;
    h_vec.reserve(mesh.faces_size() * 6 * 4 * 4);
    for(auto fiter = mesh.faces().begin(); fiter != mesh.faces().end(); ++fiter) {
        const pmp::Face &f = *fiter;
        Eigen::Matrix2d dv = dvertexs[f];
        Eigen::Vector2d uv[3];
        pmp::Vertex vs[3];
        pmp::IndexType vids[3];
        auto fviter = mesh.vertices(f);
        for(int i = 0; i < 3; ++i) {
            vs[i] = *fviter;
            vids[i] = vs[i].idx();
            uv[i] = tex.block<2, 1>(vids[i] * 2, 0);
            fviter++;
        }

        Eigen::Matrix<double, 4, 6> dfdx = Eigen::Matrix<double, 4, 6>::Zero();
        dfdx(0, 0) = -dv(0, 0) - dv(1, 0); dfdx(0, 2) = dv(0, 0); dfdx(0, 4) = dv(1, 0);
        dfdx(1, 1) = -dv(0, 0) - dv(1, 0); dfdx(1, 3) = dv(0, 0); dfdx(1, 5) = dv(1, 0);
        dfdx(2, 0) = -dv(0, 1) - dv(1, 1); dfdx(2, 2) = dv(0, 1); dfdx(2, 4) = dv(1, 1);
        dfdx(3, 1) = -dv(0, 1) - dv(1, 1); dfdx(3, 3) = dv(0, 1); dfdx(3, 5) = dv(1, 1);

        Eigen::Matrix<double, 2, 3> affine = f_affines[f];
        Eigen::Matrix2d F = affine.block<2, 2>(0, 0);

        double I2 = affine(0, 2);
        double I3 = affine(1, 2);

        // Eigen::Matrix2d U, V;
        // Eigen::Vector2d sigma;
        // TinyAD::svd(F, U, sigma, V);
       Eigen::JacobiSVD<Eigen::Matrix2d> svd(F, Eigen::ComputeThinU | Eigen::ComputeThinV);
       Eigen::Vector2d sigma = svd.singularValues();
       const Eigen::Matrix2d& U = svd.matrixU();
       const Eigen::Matrix2d& V = svd.matrixV();

        Eigen::Matrix2d twist;
        twist << 0, -1, 1, 0;

        Eigen::Matrix2d flip;
        flip << 0, 1, 1, 0;

        Eigen::Matrix2d D1 = U * D1_core_ * V.transpose();
        Eigen::Matrix2d D2 = U * D2_core_ * V.transpose();
        Eigen::Matrix2d L = U * L_core_ * V.transpose() / std::sqrt(2.0);
        Eigen::Matrix2d T = U * T_core_ * V.transpose() / std::sqrt(2.0);

        double lam1 = 1.0 + 3.0 / (sigma[0] * sigma[0] * sigma[0] * sigma[0]);
        double lam2 = 1.0 + 3.0 / (sigma[1] * sigma[1] * sigma[1] * sigma[1]);
        double lam3 = 1.0 + 1.0 / (I3 * I3) - I2 / (I3 * I3 * I3);
        double lam4 = 1.0 + 1.0 / (I3 * I3) + I2 / (I3 * I3 * I3);

//        lam3 = std::max(lam3, 0.0);
//        lam4 = std::max(lam4, 0.0);

        Eigen::Map<Eigen::Vector4d> d1(D1.data());
        Eigen::Map<Eigen::Vector4d> d2(D2.data());
        Eigen::Map<Eigen::Vector4d> l(L.data());
        Eigen::Map<Eigen::Vector4d> t(T.data());

        Eigen::Matrix4d hessian = lam1 * d1 * d1.transpose() + lam2 * d2 * d2.transpose()
                                    + lam3 * l * l.transpose() + lam4 * t * t.transpose();
        Eigen::Matrix<double, 6, 6> dphidxdx = areas[f] * dfdx.transpose() * hessian * dfdx;

#define CHECK_GRAD
#ifdef CHECK_GRAD
        // typedef std::conditional<number_check_, TinyAD::Double<4>, double>::type T40;
        // Eigen::Vector<TinyAD::Double<4>, 4> Fvvad = T40::make_active({F(0, 0), F(1, 0), F(0, 1), F(1, 1)});
        // Eigen::Matrix2<T40> FFad;
        // FFad(0, 0) = Fvvad(0);
        // FFad(1, 0) = Fvvad(1);
        // FFad(0, 1) = Fvvad(2);
        // FFad(1, 1) = Fvvad(3);

        // auto cost_F = 0.5 * (FFad.squaredNorm() + FFad.inverse().squaredNorm());

        // std::cout << cost_F.grad.transpose() << std::endl;
        // std::cout << "\nhessian :\n";
        // std::cout << cost_F.Hess << std::endl;
        // std::cout << "\n";
        // std::cout << hessian << std::endl;

        Eigen::Vector<double, 6>  uv3;
        uv3 << uv[0] , uv[1] , uv[2];
        typedef std::conditional<number_check_, TinyAD::Double<6>, double>::type T6;
        Eigen::Vector<TinyAD::Double<6>, 6> uv3ad = TinyAD::Double<6>::make_active(uv3);
        Eigen::Vector2<T6> a(uv3ad[0], uv3ad[1]);
        Eigen::Vector2<T6> b(uv3ad[2], uv3ad[3]);
        Eigen::Vector2<T6> c(uv3ad[4], uv3ad[5]);

        Eigen::Matrix2<T6> target;
        target << b - a , c - a;
        Eigen::Matrix2<T6> Fad = target * dv;

        T6 costad = 0.5 * areas[f] * (Fad.squaredNorm() + Fad.inverse().squaredNorm());
        TinyAD::project_positive_definite(costad.Hess, TinyAD::default_hessian_projection_eps);
       // dphidxdx = costad.Hess;
    //    std::cout << "\nhessian :\n";
    //    std::cout << costad.Hess << std::endl;
    //    std::cout << "\n";
    //    std::cout << dphidxdx << std::endl;
        dphidxdx = costad.Hess;
#endif //#ifdef CHECK_GRAD

//        std::cout << "hessian = " << dphidxdx << std::endl;
//        exit(-1);
        pmp::IndexType toGlobal[6] = {2 * vids[0], 2 * vids[0] + 1, 2 * vids[1], 2 * vids[1] + 1, 2 * vids[2], 2 * vids[2] + 1};
        for(int n = 0; n < 6; ++n) {
            for(int m = 0; m < 6; ++m) {
                h_vec.emplace_back(toGlobal[n], toGlobal[m], dphidxdx(n, m));
            }
        }
//        h_vec.emplace_back(vids[0] * 2, vids[0] * 2, dphidxdx(0, 0));
//        h_vec.emplace_back(vids[0] * 2, vids[0] * 2 + 1, dphidxdx(0, 1));
//        h_vec.emplace_back(vids[0] * 2 + 1, vids[0] * 2 + 1, dphidxdx(1, 1));
//        h_vec.emplace_back(vids[0] * 2 + 1, vids[0] * 2, dphidxdx(1, 0));
//
//        h_vec.emplace_back(vids[1] * 2, vids[1] * 2, dphidxdx(2, 2));
//        h_vec.emplace_back(vids[1] * 2, vids[1] * 2 + 1, dphidxdx(2, 3));
//        h_vec.emplace_back(vids[1] * 2 + 1, vids[1] * 2 + 1, dphidxdx(3, 3));
//        h_vec.emplace_back(vids[1] * 2 + 1, vids[1] * 2, dphidxdx(3, 2));
//
//        h_vec.emplace_back(vids[2] * 2, vids[2] * 2, dphidxdx(4, 4));
//        h_vec.emplace_back(vids[2] * 2, vids[2] * 2 + 1, dphidxdx(4, 5));
//        h_vec.emplace_back(vids[2] * 2 + 1, vids[2] * 2 + 1, dphidxdx(5, 5));
//        h_vec.emplace_back(vids[2] * 2 + 1, vids[2] * 2, dphidxdx(5, 4));
//
//        h_vec.emplace_back(vids[0] * 2, vids[1] * 2, dphidxdx(0, 2));
//        h_vec.emplace_back(vids[0] * 2, vids[1] * 2 + 1, dphidxdx(0, 3));
//        h_vec.emplace_back(vids[0] * 2 + 1, vids[1] * 2 + 1, dphidxdx(1, 3));
//        h_vec.emplace_back(vids[0] * 2 + 1, vids[1] * 2, dphidxdx(1, 2));
//
//        h_vec.emplace_back(vids[0] * 2, vids[2] * 2, dphidxdx(0, 4));
//        h_vec.emplace_back(vids[0] * 2, vids[2] * 2 + 1, dphidxdx(0, 5));
//        h_vec.emplace_back(vids[0] * 2 + 1, vids[2] * 2 + 1, dphidxdx(1, 5));
//        h_vec.emplace_back(vids[0] * 2 + 1, vids[2] * 2, dphidxdx(1, 4));
//
//        h_vec.emplace_back(vids[1] * 2, vids[0] * 2, dphidxdx(2, 0));
//        h_vec.emplace_back(vids[1] * 2, vids[0] * 2 + 1, dphidxdx(2, 1));
//        h_vec.emplace_back(vids[1] * 2 + 1, vids[0] * 2 + 1, dphidxdx(3, 1));
//        h_vec.emplace_back(vids[1] * 2 + 1, vids[0] * 2, dphidxdx(3, 0));
//
//        h_vec.emplace_back(vids[2] * 2, vids[0] * 2, dphidxdx(4, 0));
//        h_vec.emplace_back(vids[2] * 2, vids[0] * 2 + 1, dphidxdx(4, 1));
//        h_vec.emplace_back(vids[2] * 2 + 1, vids[0] * 2 + 1, dphidxdx(5, 1));
//        h_vec.emplace_back(vids[2] * 2 + 1, vids[0] * 2, dphidxdx(5, 0));
//
//        h_vec.emplace_back(vids[1] * 2, vids[2] * 2, dphidxdx(2, 4));
//        h_vec.emplace_back(vids[1] * 2, vids[2] * 2 + 1, dphidxdx(2, 5));
//        h_vec.emplace_back(vids[1] * 2 + 1, vids[2] * 2 + 1, dphidxdx(3, 5));
//        h_vec.emplace_back(vids[1] * 2 + 1, vids[2] * 2, dphidxdx(3, 4));
//
//        h_vec.emplace_back(vids[2] * 2, vids[1] * 2, dphidxdx(4, 2));
//        h_vec.emplace_back(vids[2] * 2, vids[1] * 2 + 1, dphidxdx(4, 3));
//        h_vec.emplace_back(vids[2] * 2 + 1, vids[1] * 2 + 1, dphidxdx(5, 3));
//        h_vec.emplace_back(vids[2] * 2 + 1, vids[1] * 2, dphidxdx(5, 2));
    }

    Eigen::SparseMatrix<double> H(mesh.vertices_size() * 2, mesh.vertices_size() * 2);
    H.setFromTriplets(h_vec.begin(), h_vec.end());
    return H;
}

double SymmetricDirichlet::LineSearch(const pmp::SurfaceMesh &mesh,
                                      const Eigen::VectorXd &tex,
                                      const Eigen::VectorXd &d, const Eigen::VectorXd &grad) {
    auto dvertexs = mesh.get_face_property<Eigen::Matrix2d>("f:dvertex");
    auto areas = mesh.get_face_property<double>("f:area");
    auto f_affines = mesh.get_face_property<Eigen::Matrix<double, 2, 3>>("f:affine_matrix_I2_I3");
    double current_cost = 0.0f;
    Eigen::VectorXd newX;
    double s = 1.0;
    constexpr double shrink = 0.8;
    constexpr double armijoConst = 1e-4;
    for (int j = 0; j < 64; j++) {
        newX = tex + s * d;
        for(auto fiter = mesh.faces().begin(); fiter != mesh.faces().end(); ++fiter)
        {
            const pmp::Face &f = *fiter;
            Eigen::Matrix2d dv = dvertexs[f];
            Eigen::Vector2d uv[3];
            pmp::Vertex vs[3];
            auto fviter = mesh.vertices(f);
            for (int i = 0; i < 3; ++i)
            {
                vs[i] = *fviter;
                uv[i] = tex.block<2, 1>(vs[i].idx() * 2, 0) + last_alpha_ * d.block<2, 1>(vs[i].idx() * 2, 0);
                fviter++;
            }
            Eigen::Matrix2d F = ComputeDeltaUV(dv, uv);
            current_cost += (F.squaredNorm() +  F.inverse().squaredNorm());
        }
        if (current_cost <= last_cost + armijoConst * s * d.dot(grad)) {
            std::cout << "old cost = " << last_cost << ", new cost = " << current_cost << std::endl;
            last_cost = current_cost;
            break;
        }

        s *= shrink;
    }

    return s;
}

}