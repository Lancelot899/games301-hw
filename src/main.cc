// Copyright 2011-2021 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#include "TutteEmbedding.h"
#include "symmetric_dirichlet.h"
#include <memory>

#include <pmp/visualization/MeshViewer.h>
#include <pmp/algorithms/SurfaceParameterization.h>
#include <imgui.h>

namespace games301{

class MeshViewer : public pmp::MeshViewer
{
public:
    //! constructor
    MeshViewer(const char* title, int width, int height)
        : pmp::MeshViewer(title, width, height)
    {
        set_draw_mode("Smooth Shading");
    }

protected:
    void process_imgui() override;
    void draw(const std::string& _draw_mode) override;
};


}



void games301::MeshViewer::process_imgui()
{
    pmp::MeshViewer::process_imgui();
    std::shared_ptr<games301::ProjectNewton> project_newton
        = std::make_shared<SymmetricDirichlet>(100, 1e-5);
    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Tutte Embedding",
                                ImGuiTreeNodeFlags_DefaultOpen))
    {
        bool is_recompute = false;
        bool is_project_nt = false;

        ImGui::Spacing();
        if (ImGui::Button("Uniform")) {
            try{
                SurfaceTutteEmbedding param(mesh_);
                param.uniform_edge_wighting();
                param.embedding();
            } catch (const std::exception& e) {
                std::cerr << e.what() << std::endl;
                return;
            }
            is_recompute = true;
        }
        if (ImGui::Button("Floater's shape-preserving")) {
            try{
                SurfaceTutteEmbedding param(mesh_);
                param.floater_edge_wighting();
                param.embedding();
            } catch (const std::exception& e) {
                std::cerr << e.what() << std::endl;
                return;
            }
            is_recompute = true;
        }

        if (ImGui::Button("Mean Value Coordinates")) {
            try{
                SurfaceTutteEmbedding param(mesh_);
                param.mean_value_edge_wighting();
                param.embedding();
            } catch (const std::exception& e) {
                std::cerr << e.what() << std::endl;
                return;
            }
            is_recompute = true;
        }

        if (ImGui::Button("Harmonic Param")){
            try{
                pmp::SurfaceParameterization param(mesh_);
                param.harmonic();
            } catch (const std::exception& e) {
                std::cerr << e.what() << std::endl;
                return;
            }
            is_recompute = true;
        }
        if(is_recompute) {
            project_newton->Run(mesh_);
            mesh_.use_checkerboard_texture();
            set_draw_mode("Texture");
            update_mesh();
        }
    }
}

void games301::MeshViewer::draw(const std::string& draw_mode)
{
    // normal mesh draw
    glViewport(0, 0, width(), height());
    mesh_.draw(projection_matrix_, modelview_matrix_, draw_mode);

    // draw uv layout
    {
        // clear depth buffer
        glClear(GL_DEPTH_BUFFER_BIT);

        // setup viewport
        GLint size = std::min(width(), height()) * 0.5;
        glViewport(width() - size - 1, height() - size - 1, size, size);

        // setup matrices
        pmp::mat4 P = pmp::ortho_matrix(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        pmp::mat4 M = pmp::mat4::identity();

        // draw mesh once more
        mesh_.draw(P, M, "Texture Layout");
    }

    // reset viewport
    glViewport(0, 0, width(), height());
}



int main(int argc, char **argv)
{
    games301::MeshViewer window("Games301-hw1", 800, 600);

    if (argc == 2)
        window.load_mesh(argv[1]);
#ifdef __EMSCRIPTEN__
    else
        window.load_mesh("input.off");
#endif

    return window.run();
}