#
# Copyright (c) 2026, ethereal contributors. All rights reserved.
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


# The render graph's device-free tests (aggregate docs/rdg-design.md; plan step P1.3).

add_library(donut_rdg_corpus STATIC src/render/rdg_corpus.cpp src/render/rdg_corpus.h
    src/render/rdg_donut_frame.cpp src/render/rdg_donut_frame.h)
target_link_libraries(donut_rdg_corpus PUBLIC donut_render donut_engine donut_core)
set_property(TARGET donut_rdg_corpus PROPERTY FOLDER "Donut/donut_tests/donut_render_tests")

# Recording, validation, culling, the plan cache, golden barrier lists and the verifier corpus.
add_executable(test_render_graph src/render/test_render_graph.cpp)
target_link_libraries(test_render_graph donut_rdg_corpus donut_render donut_engine donut_core donut_tests_utils)
add_dependencies(donut_all_tests test_render_graph)
add_test(NAME test_render_graph COMMAND test_render_graph)
set_property(TARGET test_render_graph PROPERTY FOLDER "Donut/donut_tests/donut_render_tests")

# donut's passes in a graph, device-free (plan step P1.5): golden access lists of every AddToGraph, per-view folding.
add_executable(test_render_graph_passes src/render/test_render_graph_passes.cpp)
target_link_libraries(test_render_graph_passes donut_rdg_corpus donut_render donut_engine donut_core)
add_dependencies(donut_all_tests test_render_graph_passes)
add_test(NAME test_render_graph_passes COMMAND test_render_graph_passes)
set_property(TARGET test_render_graph_passes PROPERTY FOLDER "Donut/donut_tests/donut_render_tests")

# The render service's extraction into the frame snapshot, device-free (ECS step E5, sub-plan item E5.1). It reaches
# the ECS allocation counter through the private headers in src/engine, as the engine's GPU scene test does, and the
# limits of the serial extraction through src/render (RenderVisibility.h).
add_executable(test_render_extract src/render/test_render_extract.cpp)
target_link_libraries(test_render_extract donut_render donut_engine donut_core donut_tests_utils)
target_include_directories(test_render_extract PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../src/engine"
    "${CMAKE_CURRENT_LIST_DIR}/../src/render")
add_dependencies(donut_all_tests test_render_extract)
add_test(NAME test_render_extract COMMAND test_render_extract)
set_property(TARGET test_render_extract PROPERTY FOLDER "Donut/donut_tests/donut_render_tests")

# Pipelined recording against serial recording, device-free (aggregate docs/plans/2026-11-26-donut-ecs-e7-rev2.md,
# stage T6; revision 1's E7.8 tests).
add_executable(test_render_pipelining src/render/test_render_pipelining.cpp)
target_link_libraries(test_render_pipelining donut_render donut_engine donut_core donut_tests_utils)
add_dependencies(donut_all_tests test_render_pipelining)
add_test(NAME test_render_pipelining COMMAND test_render_pipelining)
set_property(TARGET test_render_pipelining PROPERTY FOLDER "Donut/donut_tests/donut_render_tests")

# The regression tests of the E7 fix track, render side (aggregate docs/plans/2026-11-26-donut-ecs-e7-rev2.md, stage
# F), device-free.
add_executable(test_e7_render_fixes src/render/test_e7_render_fixes.cpp)
target_link_libraries(test_e7_render_fixes donut_render donut_engine donut_core donut_tests_utils)
add_dependencies(donut_all_tests test_e7_render_fixes)
add_test(NAME test_e7_render_fixes COMMAND test_e7_render_fixes)
set_property(TARGET test_e7_render_fixes PROPERTY FOLDER "Donut/donut_tests/donut_render_tests")

# Visibility and draw packets against the legacy strategies, device-free (sub-plan item E5.2). The ECS allocation
# counter through src/engine, the radix sort through src/render (RenderVisibility.h); the per-frame packet hashes are
# compared with tests/src/render/golden/render_extract_packets.txt (ECS_UPDATE_GOLDEN=1 rewrites it).
add_executable(test_draw_packets src/render/test_draw_packets.cpp)
target_link_libraries(test_draw_packets donut_render donut_engine donut_core donut_tests_utils)
target_include_directories(test_draw_packets PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../src/engine"
    "${CMAKE_CURRENT_LIST_DIR}/../src/render")
target_compile_definitions(test_draw_packets PRIVATE
    DONUT_TEST_GOLDEN_DIR="${CMAKE_CURRENT_LIST_DIR}/src/render/golden")
add_dependencies(donut_all_tests test_draw_packets)
add_test(NAME test_draw_packets COMMAND test_draw_packets)
set_property(TARGET test_draw_packets PROPERTY FOLDER "Donut/donut_tests/donut_render_tests")

# The draw-strategy adapters over packets against the legacy graph walk, device-free (sub-plan item E5.3).
add_executable(test_draw_strategy_adapters src/render/test_draw_strategy_adapters.cpp)
target_link_libraries(test_draw_strategy_adapters donut_render donut_engine donut_core donut_tests_utils)
add_dependencies(donut_all_tests test_draw_strategy_adapters)
add_test(NAME test_draw_strategy_adapters COMMAND test_draw_strategy_adapters)
set_property(TARGET test_draw_strategy_adapters PROPERTY FOLDER "Donut/donut_tests/donut_render_tests")

# Shadow maps owned by the render service, keyed by light entity, device-free (sub-plan item E5.6): the shadow table's
# lifecycle, the lighting constants of both input forms (src/render/LightingConstants.h), the cascades against the
# application's set-up, and the depth passes' uses in a compiled graph, over stand-in textures.
add_executable(test_light_shadows src/render/test_light_shadows.cpp)
target_link_libraries(test_light_shadows donut_render donut_engine donut_core donut_tests_utils)
target_include_directories(test_light_shadows PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../src/render")
add_dependencies(donut_all_tests test_light_shadows)
add_test(NAME test_light_shadows COMMAND test_light_shadows)
set_property(TARGET test_light_shadows PROPERTY FOLDER "Donut/donut_tests/donut_render_tests")

# The CPU cost of visibility and packets against the legacy strategy walk (sub-plan item E5.2; design 10.1). The same
# source measures the walk alone when compiled against the E0 tree with BENCH_RENDER_EXTRACT_E0=1. ctest runs the smoke
# mode; the Release numbers come from `bench_render_extract --workers N`.
add_executable(bench_render_extract src/render/bench_render_extract.cpp)
target_link_libraries(bench_render_extract donut_render donut_engine donut_core)
add_dependencies(donut_all_tests bench_render_extract)
add_test(NAME bench_render_extract_smoke COMMAND bench_render_extract --smoke)
set_property(TARGET bench_render_extract PROPERTY FOLDER "Donut/donut_tests/donut_render_tests")

# The benchmark of design 10.3. The device-free budgets: a smoke run in ctest, measured in Release with
# `rdg_bench --gate`. The parallel-recording budget needs a device (plan step P1.7): `rdg_bench --parallel d3d12|vulkan
# --gate`, with GPU smoke runs in ctest.
add_executable(rdg_bench src/render/rdg_bench.cpp src/render/rdg_bench_parallel.cpp)
target_link_libraries(rdg_bench donut_rdg_corpus donut_render donut_engine donut_core)
if(TARGET nvrhi_factory)
    target_link_libraries(rdg_bench nvrhi_factory)
endif()
target_link_libraries(rdg_bench nvrhi)
add_dependencies(donut_all_tests rdg_bench)
add_test(NAME rdg_bench_smoke COMMAND rdg_bench --quick)
if(DONUT_WITH_DX12)
    if(TARGET Microsoft::DirectX-Headers)
        target_link_libraries(rdg_bench Microsoft::DirectX-Headers)
    endif()
    target_link_libraries(rdg_bench dxguid)
    target_compile_definitions(rdg_bench PRIVATE RDG_BENCH_D3D12=1)
    add_test(NAME rdg_bench_parallel_d3d12 COMMAND rdg_bench --parallel d3d12 --quick)
    set_tests_properties(rdg_bench_parallel_d3d12 PROPERTIES RUN_SERIAL TRUE LABELS gpu)
endif()
if(DONUT_WITH_VULKAN)
    if(TARGET Vulkan-Headers)
        target_link_libraries(rdg_bench Vulkan-Headers)
    elseif(TARGET Vulkan::Headers)
        target_link_libraries(rdg_bench Vulkan::Headers)
    endif()
    target_compile_definitions(rdg_bench PRIVATE RDG_BENCH_VULKAN=1)
    add_test(NAME rdg_bench_parallel_vulkan COMMAND rdg_bench --parallel vulkan --quick)
    set_tests_properties(rdg_bench_parallel_vulkan PROPERTIES RUN_SERIAL TRUE LABELS gpu)
endif()
set_property(TARGET rdg_bench PROPERTY FOLDER "Donut/donut_tests/donut_render_tests")

# Headless graphs on a device, per backend, under the debug layers (plan step P1.4). The nvrhi backends are static
# libraries linked through nvrhi_factory; a shared nvrhi build exports them from its DLL.
add_executable(test_render_graph_gpu src/render/test_render_graph_gpu.cpp)
target_link_libraries(test_render_graph_gpu donut_rdg_corpus donut_render donut_engine donut_core)
if(TARGET nvrhi_factory)
    target_link_libraries(test_render_graph_gpu nvrhi_factory)
endif()
target_link_libraries(test_render_graph_gpu nvrhi)
if(WIN32)
    target_link_libraries(test_render_graph_gpu dbghelp)   # RDG_ALLOC_TRACE
endif()
if(DONUT_WITH_DX11)
    target_compile_definitions(test_render_graph_gpu PRIVATE TEST_D3D11=1)
    add_test(NAME test_render_graph_gpu_d3d11 COMMAND test_render_graph_gpu d3d11)
endif()
if(DONUT_WITH_DX12)
    if(TARGET Microsoft::DirectX-Headers)
        target_link_libraries(test_render_graph_gpu Microsoft::DirectX-Headers)
    endif()
    target_link_libraries(test_render_graph_gpu dxguid)
    target_compile_definitions(test_render_graph_gpu PRIVATE TEST_D3D12=1)
    add_test(NAME test_render_graph_gpu_d3d12 COMMAND test_render_graph_gpu d3d12)
endif()
if(DONUT_WITH_VULKAN)
    if(TARGET Vulkan-Headers)
        target_link_libraries(test_render_graph_gpu Vulkan-Headers)
    elseif(TARGET Vulkan::Headers)
        target_link_libraries(test_render_graph_gpu Vulkan::Headers)
    endif()
    target_compile_definitions(test_render_graph_gpu PRIVATE TEST_VULKAN=1)
    add_test(NAME test_render_graph_gpu_vulkan COMMAND test_render_graph_gpu vulkan)
endif()
if(NVRHI_WITH_CUDA)
    target_compile_definitions(test_render_graph_gpu PRIVATE TEST_CUDA=1)
    add_test(NAME test_render_graph_gpu_cuda COMMAND test_render_graph_gpu cuda)
    add_test(NAME test_render_graph_gpu_cuda_overlapped COMMAND test_render_graph_gpu cuda-overlapped)
endif()
# One GPU process at a time.
set(_rdg_gpu_tests test_render_graph_gpu_d3d11 test_render_graph_gpu_d3d12 test_render_graph_gpu_vulkan test_render_graph_gpu_cuda
    test_render_graph_gpu_cuda_overlapped)
foreach(_test ${_rdg_gpu_tests})
    if(TEST ${_test})
        set_tests_properties(${_test} PROPERTIES RUN_SERIAL TRUE LABELS gpu)
    endif()
endforeach()
add_dependencies(donut_all_tests test_render_graph_gpu)
set_property(TARGET test_render_graph_gpu PROPERTY FOLDER "Donut/donut_tests/donut_render_tests")

# app::RenderGraphLayer on a headless DeviceManager (plan step P1.5).
add_executable(test_render_graph_layer src/render/test_render_graph_layer.cpp)
target_link_libraries(test_render_graph_layer donut_app donut_render donut_engine donut_core)
foreach(_backend d3d11 d3d12 vulkan)
    string(TOUPPER "${_backend}" _upper)
    if((_backend STREQUAL "d3d11" AND DONUT_WITH_DX11) OR (_backend STREQUAL "d3d12" AND DONUT_WITH_DX12) OR
       (_backend STREQUAL "vulkan" AND DONUT_WITH_VULKAN))
        add_test(NAME test_render_graph_layer_${_backend} COMMAND test_render_graph_layer ${_backend})
        set_tests_properties(test_render_graph_layer_${_backend} PROPERTIES RUN_SERIAL TRUE LABELS gpu)
    endif()
endforeach()
add_dependencies(donut_all_tests test_render_graph_layer)
set_property(TARGET test_render_graph_layer PROPERTY FOLDER "Donut/donut_tests/donut_render_tests")

# The GpuScene upload pass on a headless device (ECS step E5, sub-plan item E5.4): after every frame of a scripted scene
# the buffers read back equal a CPU-mode GpuScene's images, under the debug runtime and the nvrhi validation layer.
add_executable(test_render_extract_gpu src/render/test_render_extract_gpu.cpp)
target_link_libraries(test_render_extract_gpu donut_app donut_render donut_engine donut_core)
if(MSVC)
    # The steady-state case attributes its counted allocations with the linker's map: it has every function, those of the
    # static libraries and their internal ones too, whether or not the configuration makes debug information. No folding:
    # a folded function has the name of another one (a link item, so that it comes after ASTEROIDS_MATCH_CODEGEN's
    # /OPT:ICF). dbghelp: UnDecorateSymbolName.
    target_link_options(test_render_extract_gpu PRIVATE /MAP /INCREMENTAL:NO)
    target_link_libraries(test_render_extract_gpu dbghelp -OPT:NOICF)
endif()
foreach(_backend d3d11 d3d12 vulkan)
    if((_backend STREQUAL "d3d11" AND DONUT_WITH_DX11) OR (_backend STREQUAL "d3d12" AND DONUT_WITH_DX12) OR
       (_backend STREQUAL "vulkan" AND DONUT_WITH_VULKAN))
        add_test(NAME test_render_extract_gpu_${_backend} COMMAND test_render_extract_gpu ${_backend})
        set_tests_properties(test_render_extract_gpu_${_backend} PROPERTIES RUN_SERIAL TRUE LABELS gpu)
    endif()
endforeach()
add_dependencies(donut_all_tests test_render_extract_gpu)
set_property(TARGET test_render_extract_gpu PROPERTY FOLDER "Donut/donut_tests/donut_render_tests")

# Header hygiene (design 3.2): the RenderGraph* files include only nvrhi/, donut/core/, other RenderGraph* headers
# and the standard library.
add_test(NAME donut_render_graph_header_hygiene
    COMMAND ${CMAKE_COMMAND} -DDONUT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/..
        -P ${CMAKE_CURRENT_SOURCE_DIR}/render_graph_hygiene.cmake)

# Every public RenderGraph header is self-contained: one compile-only translation unit each, which includes it twice.
set(_rdg_header_tus "")
foreach(_header RenderGraph RenderGraphLambdaPass RenderGraphFrameData RenderGraphWorkerPool)
    set(_tu "${CMAKE_CURRENT_BINARY_DIR}/render_graph_header_tus/${_header}.cpp")
    file(WRITE "${_tu}.in" "#include <donut/render/${_header}.h>\n#include <donut/render/${_header}.h>\n")
    configure_file("${_tu}.in" "${_tu}" COPYONLY)
    list(APPEND _rdg_header_tus "${_tu}")
endforeach()
add_library(donut_render_graph_header_tus OBJECT ${_rdg_header_tus})
target_link_libraries(donut_render_graph_header_tus PRIVATE donut_render)
if(MSVC)
    target_compile_options(donut_render_graph_header_tus PRIVATE /W4 /permissive-)
endif()
add_dependencies(donut_all_tests donut_render_graph_header_tus)
set_property(TARGET donut_render_graph_header_tus PROPERTY FOLDER "Donut/donut_tests/donut_render_tests")

# The render-extraction headers of ECS step E5 (aggregate docs/plans/2026-11-21-donut-ecs-e5.md, item E5.0) and the
# headers it extends are self-contained too, the same way. Module-private C++ (they name scene-graph classes), so
# they are not under the ABI lint; the scatter kernel's constants header is compiled as C++ the way donut's sources
# include the shader headers (after `using namespace donut::math`). The new headers compile at /W4; the extended ones at
# the libraries' level (their older inline code has unused parameters).
set(_render_extract_header_tus "")
set(_render_extract_w4_tus "")
foreach(_header render/RenderExtract render/LightShadows render/GpuSceneUploadPass render/SkinningPass engine/GpuScene
        render/DrawStrategy render/GeometryPasses render/DeferredLightingPass render/ForwardShadingPass
        engine/Scene engine/SceneGraph)
    string(REPLACE "/" "_" _name "${_header}")
    set(_tu "${CMAKE_CURRENT_BINARY_DIR}/render_extract_header_tus/${_name}.cpp")
    file(WRITE "${_tu}.in" "#include <donut/${_header}.h>\n#include <donut/${_header}.h>\n")
    configure_file("${_tu}.in" "${_tu}" COPYONLY)
    list(APPEND _render_extract_header_tus "${_tu}")
    if(_header MATCHES "^render/(RenderExtract|LightShadows|GpuSceneUploadPass|SkinningPass)$|^engine/GpuScene$")
        list(APPEND _render_extract_w4_tus "${_tu}")
    endif()
endforeach()
set(_tu "${CMAKE_CURRENT_BINARY_DIR}/render_extract_header_tus/shaders_gpu_scene_scatter_cb.cpp")
file(WRITE "${_tu}.in" "#include <donut/core/math/math.h>\nusing namespace donut::math;\n#include <donut/shaders/gpu_scene_scatter_cb.h>\n#include <donut/shaders/gpu_scene_scatter_cb.h>\nstatic_assert(sizeof(GpuSceneTransformRecord) == 64);\nstatic_assert(sizeof(GpuSceneRowRecord) == 128);\nstatic_assert(sizeof(GpuSceneScatterConstants) == 32);\n")
configure_file("${_tu}.in" "${_tu}" COPYONLY)
list(APPEND _render_extract_header_tus "${_tu}")
list(APPEND _render_extract_w4_tus "${_tu}")
add_library(donut_render_extract_header_tus OBJECT ${_render_extract_header_tus})
target_link_libraries(donut_render_extract_header_tus PRIVATE donut_render)
if(MSVC)
    target_compile_options(donut_render_extract_header_tus PRIVATE /permissive-)
    set_source_files_properties(${_render_extract_w4_tus} PROPERTIES COMPILE_OPTIONS "/W4")
endif()
add_dependencies(donut_all_tests donut_render_extract_header_tus)
set_property(TARGET donut_render_extract_header_tus PROPERTY FOLDER "Donut/donut_tests/donut_render_tests")

# RenderGraph.h follows nvrhi's ABI rules (design 3.2): nvrhi's abi_lint with the nvrhi headers' types known.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND AND TARGET nvrhi)
    get_target_property(_nvrhi_source_dir nvrhi SOURCE_DIR)
    if(EXISTS "${_nvrhi_source_dir}/tests/abi_lint.py")
        add_test(NAME donut_render_graph_abi_lint
            COMMAND ${Python3_EXECUTABLE} -I ${_nvrhi_source_dir}/tests/abi_lint.py ${_nvrhi_source_dir}/include/nvrhi
                --api-header ${CMAKE_CURRENT_SOURCE_DIR}/../include/donut/render/RenderGraph.h)
    endif()
endif()
