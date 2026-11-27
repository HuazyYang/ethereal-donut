#
# Copyright (c) 2014-2026, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


file(GLOB donut_engine_tests CONFIGURE_DEPENDS src/engine/test_*.cpp)

foreach(test_src ${donut_engine_tests})

    get_filename_component(test_name "${test_src}" NAME_WE)
    #message(STATUS "Added test ${test_name}")

    add_executable("${test_name}" "${test_src}")
    # The Vulkan-Hpp dispatcher is defined by nvrhi_vk (nvrhi ADR 0014).
    target_link_libraries("${test_name}" donut_app donut_engine donut_core donut_tests_utils)

    add_dependencies(donut_all_tests "${test_name}")

    add_test(NAME "${test_name}" COMMAND "${test_name}")

    set_property(TARGET "${test_name}" PROPERTY FOLDER "Donut/donut_tests/donut_engine_tests")

endforeach()

# The scene-graph trace compares its records with golden files in the source tree; ECS_UPDATE_GOLDEN=1 rewrites them.
# They live under src/ because the VFS test requires tests/ to hold only include/ and src/.
target_compile_definitions(test_scene_graph_trace PRIVATE
    DONUT_TEST_GOLDEN_DIR="${CMAKE_CURRENT_LIST_DIR}/src/engine/golden")
# The ECS world-hash corpus (ecs_world_hashes.txt), the same way.
target_compile_definitions(test_ecs_snapshot PRIVATE
    DONUT_TEST_GOLDEN_DIR="${CMAKE_CURRENT_LIST_DIR}/src/engine/golden")
# The per-frame determinism hashes of the scheduler (ecs_scheduler_frames.txt) and of simulation domains
# (ecs_simulation_frames.txt), the same way.
target_compile_definitions(test_ecs_determinism PRIVATE
    DONUT_TEST_GOLDEN_DIR="${CMAKE_CURRENT_LIST_DIR}/src/engine/golden")
# The per-frame hashes of the replayed simulation script (simulation_frames.txt), the same way.
target_compile_definitions(test_scene_graph_simulation PRIVATE
    DONUT_TEST_GOLDEN_DIR="${CMAKE_CURRENT_LIST_DIR}/src/engine/golden")
# The per-frame hashes of the runtime-only mesh instance script (runtime_instances_frames.txt; plan step E7, item
# E7.1), the same way.
target_compile_definitions(test_runtime_instances PRIVATE
    DONUT_TEST_GOLDEN_DIR="${CMAKE_CURRENT_LIST_DIR}/src/engine/golden")

# The ECS tests reach the world's internals (allocation counters, chunk layout, entity table) through the private
# headers in src/engine (EcsWorldImpl.h and the ones beside it); so do the scene-graph storage test (step E3) and the
# GPU scene test (step E4), for the allocation counter, the leaf columns test (E6.3), for the records' counter and the
# failure seams, the simulation test (E6.2a), for the store's counters, the runtime-only mesh instances (E7.1), for the
# store and the entity table, and the exit test of the shared job system (SharedJobSystem.h).
foreach(test_src ${donut_engine_tests})
    get_filename_component(test_name "${test_src}" NAME_WE)
    if(test_name MATCHES "^test_ecs_" OR test_name STREQUAL "test_scene_graph_store"
       OR test_name STREQUAL "test_gpu_scene" OR test_name STREQUAL "test_scene_graph_leaf_columns"
       OR test_name STREQUAL "test_scene_graph_simulation" OR test_name STREQUAL "test_runtime_instances"
       OR test_name STREQUAL "test_job_system_exit")
        target_include_directories("${test_name}" PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../src/engine")
    endif()
endforeach()
# The inspector's test (plan step E7, item E7.7) reaches its formatter and test seam through the private header in
# src/app (WorldInspectorDetail.h), and its stage T3 cases the world's allocation counter (EcsWorldImpl.h, src/engine).
target_include_directories(test_world_inspector PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../src/app"
                           "${CMAKE_CURRENT_LIST_DIR}/../src/engine")
# Its cases that fail on an open backlog item (aggregate docs/plans/2026-11-26-donut-ecs-e7-rev2.md) run apart, so
# that test_world_inspector itself stays the pass rule's.
add_test(NAME test_world_inspector_backlog COMMAND test_world_inspector --backlog)
# The exit test runs ~130 child processes; a child that hangs is killed after 30 s (Windows), this bounds the rest.
set_tests_properties(test_job_system_exit PROPERTIES TIMEOUT 300)

# Baseline timings of the scene graph (aggregate docs/plans/2026-11-21-donut-ecs.md, step E0.4). The glob above skips
# it; ctest runs it only in its short smoke mode, the Release numbers come from `bench_scene_graph`.
add_executable(bench_scene_graph src/engine/bench_scene_graph.cpp)
target_link_libraries(bench_scene_graph donut_app donut_engine donut_core)
add_dependencies(donut_all_tests bench_scene_graph)
add_test(NAME bench_scene_graph_smoke COMMAND bench_scene_graph --smoke)
set_property(TARGET bench_scene_graph PROPERTY FOLDER "Donut/donut_tests/donut_engine_tests")

# The CPU cost of the instance upload (plan step E4): GpuScene against the old RefreshBuffers instance path, which the
# same source measures when it is compiled against the E0 tree with BENCH_GPU_SCENE_E0=1. ctest runs the smoke mode.
add_executable(bench_gpu_scene src/engine/bench_gpu_scene.cpp)
target_link_libraries(bench_gpu_scene donut_app donut_engine donut_core)
add_dependencies(donut_all_tests bench_gpu_scene)
add_test(NAME bench_gpu_scene_smoke COMMAND bench_gpu_scene --smoke)
set_property(TARGET bench_gpu_scene PROPERTY FOLDER "Donut/donut_tests/donut_engine_tests")

# What simulation-driven nodes cost (plan step E6, sub-plan item E6.2a). ctest runs the smoke mode.
add_executable(bench_simulation src/engine/bench_simulation.cpp)
target_link_libraries(bench_simulation donut_app donut_engine donut_core)
add_dependencies(donut_all_tests bench_simulation)
add_test(NAME bench_simulation_smoke COMMAND bench_simulation --smoke)
set_property(TARGET bench_simulation PROPERTY FOLDER "Donut/donut_tests/donut_engine_tests")

# The A/B dump of plan step E4 (a GPU program, not a ctest test): the instance, geometry and material buffers read back
# after Scene::Refresh, in GpuSceneDump.h's canonical form. The same source builds against the E0 tree from outside it.
add_executable(gpu_scene_dump src/engine/gpu_scene_dump.cpp src/engine/GpuSceneDump.h)
target_link_libraries(gpu_scene_dump donut_app donut_render donut_engine donut_core)
# `--path graph` (ECS step E5, sub-plan item E5.4): the render-graph path, which the E0 tree does not have.
target_compile_definitions(gpu_scene_dump PRIVATE GPU_SCENE_DUMP_GRAPH_PATH=1)
add_dependencies(donut_all_tests gpu_scene_dump)
set_property(TARGET gpu_scene_dump PROPERTY FOLDER "Donut/donut_tests/donut_engine_tests")
