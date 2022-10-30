//
// Created by pupa on 2022/10/23.
//
#pragma once

#include <pmp/SurfaceMesh.h>

namespace games301{

//! \brief A class for surface parameterization.
//! \details See \cite levy_2002_least and \cite desbrun_2002_intrinsic
//! for more details.
//! \ingroup algorithms
class SurfaceTutteEmbedding
{
public:
    //! \brief Construct with mesh to be parameterized.
    //! \pre The mesh has a boundary.
    //! \throw InvalidInputException if the input precondition is violated.
    SurfaceTutteEmbedding(pmp::SurfaceMesh& mesh);


    void uniform_edge_wighting();

    void floater_edge_wighting();

    void mean_value_edge_wighting();

    void embedding();

private:
    // setup boundary constraints: map surface boundary to unit circle
    void setup_boundary_constraints();

    float corner_cotan(pmp::Halfedge hedge);

    float corner_halftan(pmp::Halfedge hedge);

    pmp::SurfaceMesh& mesh_;

};



}