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

add_library(donut_rdg_corpus STATIC src/render/rdg_corpus.cpp src/render/rdg_corpus.h)
target_link_libraries(donut_rdg_corpus PUBLIC donut_render)
set_property(TARGET donut_rdg_corpus PROPERTY FOLDER "Donut/donut_tests/donut_render_tests")

# Recording, validation, culling, the plan cache, golden barrier lists and the verifier corpus.
add_executable(test_render_graph src/render/test_render_graph.cpp)
target_link_libraries(test_render_graph donut_rdg_corpus donut_render donut_engine donut_core donut_tests_utils)
add_dependencies(donut_all_tests test_render_graph)
add_test(NAME test_render_graph COMMAND test_render_graph)
set_property(TARGET test_render_graph PROPERTY FOLDER "Donut/donut_tests/donut_render_tests")

# The device-free benchmark of design 10.3. The ctest entry is a smoke run; budgets are measured in Release with
# `rdg_bench --gate`.
add_executable(rdg_bench src/render/rdg_bench.cpp)
target_link_libraries(rdg_bench donut_rdg_corpus donut_render donut_engine donut_core)
add_dependencies(donut_all_tests rdg_bench)
add_test(NAME rdg_bench_smoke COMMAND rdg_bench --quick)
set_property(TARGET rdg_bench PROPERTY FOLDER "Donut/donut_tests/donut_render_tests")

# Header hygiene (design 3.2): the RenderGraph* files include only nvrhi/, donut/core/, other RenderGraph* headers
# and the standard library.
add_test(NAME donut_render_graph_header_hygiene
    COMMAND ${CMAKE_COMMAND} -DDONUT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/..
        -P ${CMAKE_CURRENT_SOURCE_DIR}/render_graph_hygiene.cmake)

# Every public RenderGraph header is self-contained: one compile-only translation unit each, which includes it twice.
set(_rdg_header_tus "")
foreach(_header RenderGraph RenderGraphLambdaPass RenderGraphFrameData)
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
