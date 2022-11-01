file(GLOB SOURCES ./*.cpp)
file(GLOB HEADERS ./*.h)

add_executable(myviewer ${SOURCES} main.cc ${HEADERS})
target_link_libraries(myviewer pmp_vis)

add_executable(run_without_view ${SOURCES} run_without_view.cc ${HEADERS})
target_link_libraries(run_without_view pmp_vis)

if (EMSCRIPTEN)
    set_target_properties(myviewer PROPERTIES LINK_FLAGS "--shell-file ${PROJECT_SOURCE_DIR}/external/pmp-library/src/apps/data/shell.html --preload-file ${PROJECT_SOURCE_DIR}/external/pmp-library/external/pmp-data/off/bunny.off@input.off")
endif()
