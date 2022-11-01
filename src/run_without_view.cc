// Copyright 2011-2021 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#include "TutteEmbedding.h"
#include "symmetric_dirichlet.h"
#include <memory>

#include <pmp/visualization/MeshViewer.h>
#include <pmp/algorithms/SurfaceParameterization.h>

int main(int argc, char **argv)
{
    if(argc != 2) {
        std::cout << "input a mesh file" << std::endl;
        return -1;
    }
     pmp::SurfaceMesh mesh;
     mesh.read(argv[1]);
     std::shared_ptr<games301::ProjectNewton> project_newton
        = std::make_shared<games301::SymmetricDirichlet>(100, 1e-5);
    games301::SurfaceTutteEmbedding param(mesh);
    param.uniform_edge_wighting();
    param.embedding();
    project_newton->Run(mesh);
}