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
# The scheduler's per-frame determinism hashes (ecs_scheduler_frames.txt), the same way.
target_compile_definitions(test_ecs_determinism PRIVATE
    DONUT_TEST_GOLDEN_DIR="${CMAKE_CURRENT_LIST_DIR}/src/engine/golden")

# The ECS tests reach the world's internals (allocation counters, chunk layout, entity table) through the private
# headers in src/engine (EcsWorldImpl.h and the ones beside it).
foreach(test_src ${donut_engine_tests})
    get_filename_component(test_name "${test_src}" NAME_WE)
    if(test_name MATCHES "^test_ecs_")
        target_include_directories("${test_name}" PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../src/engine")
    endif()
endforeach()

# Baseline timings of the scene graph (aggregate docs/plans/2026-11-21-donut-ecs.md, step E0.4). The glob above skips
# it; ctest runs it only in its short smoke mode, the Release numbers come from `bench_scene_graph`.
add_executable(bench_scene_graph src/engine/bench_scene_graph.cpp)
target_link_libraries(bench_scene_graph donut_app donut_engine donut_core)
add_dependencies(donut_all_tests bench_scene_graph)
add_test(NAME bench_scene_graph_smoke COMMAND bench_scene_graph --smoke)
set_property(TARGET bench_scene_graph PROPERTY FOLDER "Donut/donut_tests/donut_engine_tests")

