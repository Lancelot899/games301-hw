//
// Created by pupa on 2022/10/23.
//

#include "TutteEmbedding.h"

#include <pmp/algorithms/DifferentialGeometry.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>


games301::SurfaceTutteEmbedding::SurfaceTutteEmbedding(pmp::SurfaceMesh& mesh)
    :mesh_(mesh){
    bool has_boundary = false;
    for (auto v : mesh_.vertices())
        if (mesh_.is_boundary(v)){
            has_boundary = true;
            break;
        }

    if (!has_boundary)
    {
        auto what = "SurfaceParameterization: Mesh has no boundary.";
        throw std::logic_error(what);
    }
}

void games301::SurfaceTutteEmbedding::setup_boundary_constraints() {
    // get properties
    auto points = mesh_.vertex_property<pmp::Point>("v:point");
    auto tex = mesh_.vertex_property<pmp::TexCoord>("v:tex");

    pmp::SurfaceMesh::VertexIterator vit, vend = mesh_.vertices_end();
    pmp::Vertex vh;
    pmp::Halfedge hh;
    std::vector<pmp::Vertex> loop;

    // Initialize all texture coordinates to the origin.
    for (auto v : mesh_.vertices())
        tex[v] = pmp::TexCoord(0.5, 0.5);

    // find 1st boundary vertex
    for (vit = mesh_.vertices_begin(); vit != vend; ++vit)
        if (mesh_.is_boundary(*vit))
            break;

    // collect boundary loop
    vh = *vit;
    hh = mesh_.halfedge(vh);
    do
    {
        loop.push_back(mesh_.to_vertex(hh));
        hh = mesh_.next_halfedge(hh);
    } while (hh != mesh_.halfedge(vh));

    // map boundary loop to unit circle in texture domain
    unsigned int i, n = loop.size();
    pmp::Scalar angle, l, length;
    pmp::TexCoord t;

    // compute length of boundary loop
    for (i = 0, length = 0.0; i < n; ++i)
        length += distance(points[loop[i]], points[loop[(i + 1) % n]]);

    // map length intervalls to unit circle intervals
    for (i = 0, l = 0.0; i < n;)
    {
        // go from 2pi to 0 to preserve orientation
        angle = 2.0 * M_PI * (1.0 - l / length);

        t[0] = 0.5 + 0.5 * cosf(angle);
        t[1] = 0.5 + 0.5 * sinf(angle);

        tex[loop[i]] = t;

        ++i;
        if (i < n)
        {
            l += distance(points[loop[i]], points[loop[(i + 1) % n]]);
        }
    }
}

/**
 * @param hedge
 * @return cotangent<from(h), to(h), to(next(h))>
 */
float games301::SurfaceTutteEmbedding::corner_cotan(pmp::Halfedge hedge) {
    if(mesh_.is_boundary(hedge))
        throw std::logic_error("corner cotangent is not supported for boundary hedge");
    pmp::dvec3 pf = (pmp::dvec3)mesh_.position(mesh_.from_vertex(hedge));
    pmp::dvec3 pt = (pmp::dvec3)mesh_.position(mesh_.to_vertex(hedge));
    pmp::dvec3 pd = (pmp::dvec3)mesh_.position(mesh_.to_vertex(mesh_.next_halfedge(hedge)));
    auto d0 = pt - pf;
    auto d1 = pd - pf;
    const double area = norm(cross(d0, d1));
    if (area > std::numeric_limits<double>::min()){
        const double cot = dot(d0, d1) / area;
        return pmp::clamp_cot(cot);
    }else
        return 0.001;
}

float games301::SurfaceTutteEmbedding::corner_halftan(pmp::Halfedge hedge) {
    if(mesh_.is_boundary(hedge))
        throw std::logic_error("corner half tangent is not supported for boundary hedge");
    pmp::Point pf = mesh_.position(mesh_.from_vertex(hedge));
    pmp::Point pt = mesh_.position(mesh_.to_vertex(hedge));
    pmp::Point pd = mesh_.position(mesh_.to_vertex(mesh_.next_halfedge(hedge)));
    pmp::Point d0 = pt - pf;
    pmp::Point d1 = pd - pf;
    const double area = norm(cross(d0, d1));
    if (area > std::numeric_limits<double>::min()){
        double angle = pmp::angle(d0, d1);
        return std::tan(angle > M_PI? (angle < 0? 0.02*M_PI:angle/2):0.499*M_PI);
    }else
        return 0.01;
}

void games301::SurfaceTutteEmbedding::uniform_edge_wighting() {
    auto eweight = mesh_.edge_property<pmp::Scalar>("e:param");
    for (auto e : mesh_.edges()) eweight[e] = 1.0;
}

void games301::SurfaceTutteEmbedding::floater_edge_wighting() {
    auto eweight = mesh_.edge_property<pmp::Scalar>("e:param");
    for (auto e : mesh_.edges()) {
        eweight[e] = 0.01;
        auto h0 = mesh_.halfedge(e, 0);
        auto h1 = mesh_.halfedge(e, 1);
        if(!mesh_.is_boundary(h0))
            eweight[e] += corner_cotan(h0);
        if(!mesh_.is_boundary(h1))
            eweight[e] += corner_cotan(mesh_.prev_halfedge(h1));
        eweight[e] /= std::pow(mesh_.edge_length(e),2);
    }
}

void games301::SurfaceTutteEmbedding::mean_value_edge_wighting() {
    auto eweight = mesh_.edge_property<pmp::Scalar>("e:param");
    for (auto e : mesh_.edges()) {
        eweight[e] = 0.001;
        auto h0 = mesh_.halfedge(e, 0);
        auto h1 = mesh_.halfedge(e, 1);
        if(!mesh_.is_boundary(h0))
            eweight[e] += corner_halftan(mesh_.prev_halfedge(h0) );
        if(!mesh_.is_boundary(h1))
            eweight[e] += corner_halftan(h1);

        eweight[e] /= mesh_.edge_length(e);
    }
}

void games301::SurfaceTutteEmbedding::embedding() {
    // map boundary to circle
    setup_boundary_constraints();

    // get properties
    auto tex = mesh_.vertex_property<pmp::TexCoord>("v:tex");
    auto eweight = mesh_.get_edge_property<pmp::Scalar>("e:param");
    auto idx = mesh_.add_vertex_property<int>("v:idx", -1);

    // collect free (non-boundary) vertices in array free_vertices[]
    // assign indices such that idx[ free_vertices[i] ] == i
    unsigned i = 0;
    std::vector<pmp::Vertex> free_vertices;
    free_vertices.reserve(mesh_.n_vertices());
    for (auto v : mesh_.vertices())
    {
        if (!mesh_.is_boundary(v))
        {
            idx[v] = i++;
            free_vertices.push_back(v);
        }
    }

    // setup matrix A and rhs B
    const unsigned int n = free_vertices.size();
    Eigen::SparseMatrix<double> A(n, n);
    Eigen::MatrixXd B(n, 2);
    std::vector<Eigen::Triplet<double>> triplets;
    pmp::dvec2 b;
    double w, ww;
    pmp::Vertex v, vv;
    pmp::Edge e;
    for (i = 0; i < n; ++i)
    {
        v = free_vertices[i];

        // rhs row
        b = pmp::dvec2(0.0);

        // lhs row
        ww = 0.0;
        for (auto h : mesh_.halfedges(v))
        {
            vv = mesh_.to_vertex(h);
            e = mesh_.edge(h);
            w = eweight[e];
             ww += w;

            if (mesh_.is_boundary(vv))
            {
                b -= -w * static_cast<pmp::dvec2>(tex[vv]);
            }
            else
            {
                triplets.emplace_back(i, idx[vv], -w);
            }
        }
        triplets.emplace_back(i, i, ww);
        B.row(i) = (Eigen::Vector2d)b;
    }

    // build sparse matrix from triplets
    A.setFromTriplets(triplets.begin(), triplets.end());

    // solve A*X = B
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver(A);
    Eigen::MatrixXd X = solver.solve(B);
    if (solver.info() != Eigen::Success){
        // clean-up
        mesh_.remove_vertex_property(idx);
        mesh_.remove_edge_property(eweight);
        auto what = "SurfaceParameterization: Failed to solve linear system.";
        throw std::logic_error(what);
    } else {
        // copy solution
        for (i = 0; i < n; ++i) {
            tex[free_vertices[i]] = X.row(i);
        }
    }

    // clean-up
    mesh_.remove_vertex_property(idx);
    mesh_.remove_edge_property(eweight);
}
