#include "projected_newton.h"


namespace games301 {

ProjectNewton::ProjectNewton(uint16_t max_iter, double min_error) {
    max_iter_  = max_iter;
    min_error_ = min_error;
}

Eigen::Matrix<double, 2, 3> ProjectNewton::ComputeDVertex(const Eigen::Vector2d v_local[3], double area) {
    Eigen::Matrix<double, 2, 3> dvertex;
    dvertex.block<2, 1>(0, 0) = v_local[2] - v_local[1];
    dvertex.block<2, 1>(1, 1) = v_local[0] - v_local[2];
    dvertex.block<2, 1>(2, 2) = v_local[1] - v_local[0];
    dvertex.block<1, 3>(0, 0) = -dvertex.block<1, 3>(0, 0);
    return dvertex / area;
}

Eigen::Matrix2d ProjectNewton::ComputeDeltaUV(const Eigen::Matrix<double, 2, 3> &dvertex, const Eigen::Vector2d uv[3]) {
    Eigen::Matrix<double, 3, 2> uv_stack;
    uv_stack.block<1, 2>(0, 0) = uv[0].transpose();
    uv_stack.block<1, 2>(1, 0) = uv[1].transpose();
    uv_stack.block<1, 2>(2, 0) = uv[2].transpose();
    return dvertex * uv_stack;
}

Eigen::Matrix2d ProjectNewton::ComputeDeltaUV(const Eigen::Vector2d v_local[3], double area, const Eigen::Vector2d uv[3]) {
    Eigen::Matrix<double, 2, 3> dvertex = ComputeDVertex(v_local, area);
    return ComputeDeltaUV(dvertex, uv);
}

Eigen::Matrix2d ProjectNewton::ComputeDeltaUV(const Eigen::Vector3d vertex[3], const Eigen::Vector2d uv[3]) {
    Eigen::Vector3d e1 = vertex[1] - vertex[0];
    Eigen::Vector3d e2 = vertex[2] - vertex[0];
    Eigen::Vector3d normal = e1.cross(e2);
    double area = normal.norm();
    normal /= area;

    double e1n = e1.norm();
    Eigen::Vector3d x_local = e1 / e1n;
    Eigen::Vector3d y_local = normal.cross(x_local);
    Eigen::Vector2d v_local[3];
    v_local[0] = Eigen::Vector2d::Zero();
    v_local[1] = Eigen::Vector2d(e1n, 0);
    v_local[2](0) = e2.dot(x_local);
    v_local[2](1) = e2.dot(y_local);

    return ComputeDeltaUV(v_local, area, uv);
}

void ProjectNewton::ComputeLocalCoord(pmp::SurfaceMesh& mesh) {
    auto local_coords = mesh.add_face_property<std::vector<Eigen::Vector2d>>("f:local_coord");
    auto f_normals = mesh.get_face_property<Eigen::Vector3d>("f:normal");
    auto points = mesh.vertex_property<pmp::Point>("v:point");
    auto edge_length = mesh.get_edge_property<double>("e:edge_length");
    for(auto fiter = mesh.faces().begin(); fiter != mesh.faces_end(); ++fiter) {
        const pmp::Face &f = *fiter;
        std::vector<Eigen::Vector2d>& coord = local_coords[f];
        auto viter = mesh.vertices(f);
        Eigen::Vector3d v_pos[3];
        pmp::Vertex vs[3];
        for(int i = 0; i < 3; ++i) {
            const pmp::Vertex &v = *viter;
            vs[i] = v;
            pmp::Point pt = points[v];
            v_pos[i](0) = pt[0];
            v_pos[i](1) = pt[1];
            v_pos[i](2) = pt[2];
            viter++;
        }

        Eigen::Vector3d e1 = v_pos[1] - v_pos[0];
        Eigen::Vector3d e2 = v_pos[2] - v_pos[0];
        Eigen::Vector3d normal = f_normals[f];
        auto e1_id = mesh.find_edge(vs[1], vs[0]);
        double e1n = edge_length[e1_id];
        Eigen::Vector3d x_local = e1 / e1n;
        Eigen::Vector3d y_local = normal.cross(x_local);
        coord.resize(3, Eigen::Vector2d::Zero());
        coord[1] = Eigen::Vector2d(e1n, 0);
        coord[2](0) = e2.dot(x_local);
        coord[2](1) = e2.dot(y_local);
    }
}

void ProjectNewton::ComputeFaceCenter(pmp::SurfaceMesh& mesh) {
    auto face_centers = mesh.add_face_property<Eigen::Vector3d>("f:face_center");
    auto edge_lengths = mesh.get_edge_property<double>("e:edge_length");
    auto face_normals = mesh.get_face_property<Eigen::Vector3d>("f:normal");
    auto points = mesh.get_vertex_property<pmp::Point>("v:point");
    auto facets = mesh.faces();
    for(auto fiter = facets.begin(); fiter != facets.end(); ++fiter) {
        auto f = *fiter;
        auto fvs = mesh.vertices(f);
        Eigen::Vector3d pts[3];
        pmp::Vertex vs[3];
        int vvid = 0;
        for(auto viter = fvs.begin(); viter != fvs.end(); viter++) {
            vs[vvid] = *viter;
            auto &pt = points[vs[vvid]];
            pts[vvid](0) = pt[0];
            pts[vvid](1) = pt[1];
            pts[vvid](2) = pt[2];
            vvid++;
        }
        Eigen::Vector3d e0v =  pts[1] - pts[0];
        Eigen::Vector3d e1v =  pts[2] - pts[0];
        pmp::Edge e0 = mesh.find_edge(vs[0], vs[1]);
        pmp::Edge e1 = mesh.find_edge(vs[0], vs[2]);
        e0v = e0v / edge_lengths[e0];
        e1v = e1v / edge_lengths[e1];
        const Eigen::Vector3d &fnormal = face_normals[f];
        Eigen::Vector3d d0 = fnormal.cross(e0v);
        Eigen::Vector3d d1 = fnormal.cross(e1v);
        Eigen::Vector3d x0 = (pts[1] + pts[0]) * 0.5;
        Eigen::Vector3d x1 = (pts[2] + pts[0]) * 0.5;
        Eigen::Matrix<double, 3, 2> A;
        A.block<3, 1>(0, 0) = d0;
        A.block<3, 1>(0, 1) = -d1;
        Eigen::Vector2d b = A.transpose() * (x1 - x0);
        Eigen::Vector2d lambda = (A.transpose() * A).ldlt().solve(b);
        face_centers[f] = x0 + lambda[0] * d0;
    }
}

void ProjectNewton::ComputeEdgeLength(pmp::SurfaceMesh& mesh) {
    auto edge_lengths = mesh.add_edge_property<double>("e:edge_length");
    auto edges = mesh.edges();
    auto points = mesh.get_vertex_property<pmp::Point>("v:point");
    for(auto eiter = edges.begin(); eiter != edges.end(); eiter++) {
        auto v0 = mesh.vertex(*eiter, 0);
        auto v1 = mesh.vertex(*eiter, 1);
        Eigen::Vector3d p0, p1;
        p0[0] = points[v0][0]; p0[1] = points[v0][1]; p0[2] = points[v0][2];
        p1[0] = points[v1][0]; p1[1] = points[v1][1]; p1[2] = points[v1][2];
        edge_lengths[*eiter] = (p0 - p1).norm();
    }
}
void ProjectNewton::CleanFaceCenter(pmp::SurfaceMesh& mesh) {
    auto face_centers = mesh.get_face_property<Eigen::Vector3d>("f:face_center");
    mesh.remove_face_property(face_centers);
}
void ProjectNewton::CleanEdgeLength(pmp::SurfaceMesh& mesh) {
    auto edge_lengths = mesh.get_edge_property<double>("e:edge_length");
    mesh.remove_edge_property(edge_lengths);
}

void ProjectNewton::ComputePointsArea(pmp::SurfaceMesh& mesh) {
    auto areas = mesh.add_vertex_property<double>("v:area");
    auto face_center = mesh.get_face_property<Eigen::Vector3d>("f:face_center");
    auto edge_length = mesh.get_edge_property<double>("e:edge_length");
    auto points = mesh.get_vertex_property<pmp::Point>("v:point");
    auto vs = mesh.vertices();
    for(auto viter = vs.begin(); viter != vs.end(); ++viter) {
        auto v = *viter;
        auto vfs = mesh.faces(v);
        Eigen::Vector3d vp;
        vp[0] = points[v][0]; vp[1] = points[v][1]; vp[2] = points[v][2];
        double v_area = 0.0;
        for(auto fiter = vfs.begin(); fiter != vfs.end(); fiter++) {
            auto f = *fiter;
            const Eigen::Vector3d &fc = face_center[f];
            Eigen::Vector3d vjp[2];
            pmp::Vertex vj[2];
            bool is_zero = true;
            auto vvs = mesh.vertices(f);
            for(auto vviter = vvs.begin(); vviter != vvs.end(); ++vviter) {
                if(*vviter == v)
                    continue ;
                if(is_zero) {
                    vjp[0] = points[*vviter];
                    vj[0] = *vviter;
                    is_zero = false;
                } else {
                    vjp[1] = points[*vviter];
                    vj[1] = *vviter;
                }
            }
            Eigen::Vector3d e0 = vjp[0] - vp;
            Eigen::Vector3d e1 = vjp[1] - vp;
            Eigen::Vector3d c = fc - vp;
            v_area += (e0.cross(c)).norm() * 0.25 + (e1.cross(c)).norm() * 0.25;
        }
        areas[v] = v_area;
    }
}

void ProjectNewton::CleanPointsArea(pmp::SurfaceMesh& mesh) {
    auto areas = mesh.get_vertex_property<double>("v:area");
    mesh.remove_vertex_property(areas);
}

void ProjectNewton::ComputeDVertexPerFace(pmp::SurfaceMesh& mesh) {
    auto dvertexs = mesh.add_face_property<Eigen::Matrix<double, 2, 3>>("f:dvertex");
    auto local_coords = mesh.get_face_property<std::vector<Eigen::Vector2d>>("f:local_coord");
    auto face_areas = mesh.get_face_property<double>("f:area");
    for(auto fiter = mesh.faces().begin(); fiter != mesh.faces_end(); ++fiter) {
        const pmp::Face &f = *fiter;
        std::vector<Eigen::Vector2d>& coord = local_coords[f];
        dvertexs[f] = ComputeDVertex(coord.data(), face_areas[f]);
    }
}

void ProjectNewton::CleanDVertexPerFace(pmp::SurfaceMesh& mesh) {
    auto dvertexs = mesh.get_face_property<Eigen::Matrix<double, 2, 3>>("f:dvertex");
    mesh.remove_face_property(dvertexs);

}

void ProjectNewton::ComputeAreaAndNormal(pmp::SurfaceMesh& mesh) {
    auto areas = mesh.add_face_property<double>("f:area");
    auto normals = mesh.add_face_property<Eigen::Vector3d>("f:normal");
    auto points = mesh.vertex_property<pmp::Point>("v:point");
    for(auto fiter = mesh.faces().begin(); fiter != mesh.faces_end(); ++fiter) {
        const pmp::Face &f = *fiter;
        auto viter = mesh.vertices(f);
        Eigen::Vector3d v_pos[3];
        for(int i = 0; i < 3; ++i) {
            const pmp::Vertex &v = *viter;
            pmp::Point pt = points[v];
            v_pos[i](0) = pt[0];
            v_pos[i](1) = pt[1];
            v_pos[i](2) = pt[2];
            viter++;
        }

        Eigen::Vector3d e1 = v_pos[1] - v_pos[0];
        Eigen::Vector3d e2 = v_pos[2] - v_pos[0];
        Eigen::Vector3d normal = e1.cross(e2);
        areas[f] = normal.norm();
        normals[f] = normal / areas[f];
    }
}

void ProjectNewton::CleanLocalCoord(pmp::SurfaceMesh& mesh) {
    auto local_coords = mesh.get_face_property<std::vector<Eigen::Vector2d>>("f:local_coord");
    mesh.remove_face_property(local_coords);
}

void ProjectNewton::CleanAreaAndNormal(pmp::SurfaceMesh& mesh) {
    auto areas = mesh.get_face_property<double>("f:area");
    auto normals = mesh.get_face_property<Eigen::Vector3d>("f:normal");
    mesh.remove_face_property(areas);
    mesh.remove_face_property(normals);
}

bool ProjectNewton::Run(pmp::SurfaceMesh& mesh) {
//    auto points = mesh.vertex_property<pmp::Point>("v:point");
    ComputeAreaAndNormal(mesh);
    ComputeEdgeLength(mesh);
    ComputeFaceCenter(mesh);
    ComputePointsArea(mesh);
    ComputeLocalCoord(mesh);
    ComputeDVertexPerFace(mesh);


    auto v_tex = mesh.get_vertex_property<pmp::TexCoord>("v:tex");
    auto tex = mesh.add_vertex_property<Eigen::Vector2d>("v:tex_new");

    for(size_t i = 0; i < v_tex.vector().size(); ++i) {
        auto &t = v_tex.vector()[i];
        tex.vector()[i][0] = t[0];
        tex.vector()[i][1] = t[1];
    }

    for(int iter = 0; iter < max_iter_; ++iter) {
        Eigen::VectorXd b = ComputeEnergyGrad(mesh, tex);
        double max_b = 0;
        for(int i = 0; i < b.size(); ++i) {
            if(max_b < std::abs(b[i])) {
                max_b = std::abs(b[i]);
            }
        }
        std::cout << "iter = " << iter << ": max distortion = " << max_b << std::endl;
        if(max_b < min_error_) {
            break;
        }
        Eigen::SparseMatrix<double> H = ProjectHessian(mesh, tex);
        Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Upper | Eigen::Lower> cg;
        cg.setTolerance(0.00005);
        cg.compute(H);
        Eigen::VectorXd direction = cg.solve(-b);

        double alpha = LineSearch(mesh, tex, direction);
//        tex += alpha * direction;
        for(size_t i = 0; i < tex.vector().size(); ++i) {
            auto &t = tex.vector()[i];
            t += alpha * direction.block<2, 1>(i * 2, 0);
        }
    }

    for(size_t i = 0; i < v_tex.vector().size(); ++i) {
        auto &t = v_tex.vector()[i];
        auto &tn = tex.vector()[i];
        t = tn;
    }
    mesh.remove_vertex_property(tex);
    CleanDVertexPerFace(mesh);
    CleanFaceCenter(mesh);
    CleanEdgeLength(mesh);
    CleanAreaAndNormal(mesh);
    CleanLocalCoord(mesh);
    CleanPointsArea(mesh);

    return true;
}

}