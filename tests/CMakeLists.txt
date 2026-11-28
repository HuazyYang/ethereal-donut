#
# Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
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


# build small library of common untilities for tests

add_library(donut_tests_utils STATIC src/utils.cpp)
target_include_directories(donut_tests_utils PUBLIC "include")
set_property(TARGET donut_tests_utils PROPERTY FOLDER "Donut/donut_tests")

# XXXX mk : CTest does not create (yet?) a default build target for all tests
add_custom_target(donut_all_tests)
set_property(TARGET donut_all_tests PROPERTY FOLDER "Donut/donut_tests")

#
# Add tests for each donut module
#

add_definitions(-DDONUT_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DDONUT_TEST_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}")

include(test-core.cmake)

if (DONUT_WITH_NVRHI)
    include(test-engine.cmake)
    include(test-render.cmake)

    # The switches of plan step E7 (sub-plan revision 2, parts I4 and I5) from the environment (src/e7_switches.cpp:
    # DONUT_E7_SWITCHES=all or a list; nothing without it). Linked into the device-free tests and smoke benches that build
    # scene graphs, run queries or extract; each also runs as `<test>_switches_on` with every switch on (stage T1).
    add_library(donut_tests_e7_switches OBJECT src/e7_switches.cpp)
    target_link_libraries(donut_tests_e7_switches PRIVATE donut_render donut_engine donut_core donut_tests_utils)
    set_property(TARGET donut_tests_e7_switches PROPERTY FOLDER "Donut/donut_tests")
    set(_e7_switch_tests test_asset_tables test_gpu_scene test_runtime_instances test_world_inspector test_e7_fixes
        test_render_extract test_draw_packets test_draw_strategy_adapters test_light_shadows test_render_pipelining
        test_e7_render_fixes)
    foreach(test_src ${donut_engine_tests})
        get_filename_component(test_name "${test_src}" NAME_WE)
        if(test_name MATCHES "^test_(ecs|scene_graph)_")
            list(APPEND _e7_switch_tests "${test_name}")
        endif()
    endforeach()
    foreach(_test ${_e7_switch_tests})
        if(TARGET ${_test})
            target_link_libraries(${_test} donut_tests_e7_switches)
            add_test(NAME ${_test}_switches_on COMMAND ${_test})
            set_tests_properties(${_test}_switches_on PROPERTIES ENVIRONMENT "DONUT_E7_SWITCHES=all")
        endif()
    endforeach()
    foreach(_bench bench_scene_graph bench_gpu_scene bench_simulation bench_render_extract)
        target_link_libraries(${_bench} donut_tests_e7_switches)
        add_test(NAME ${_bench}_smoke_switches_on COMMAND ${_bench} --smoke)
        set_tests_properties(${_bench}_smoke_switches_on PROPERTIES ENVIRONMENT "DONUT_E7_SWITCHES=all")
    endforeach()
    # The benches' runtime-only population (revision 2, stage M), smoke size; the default population's tests above.
    foreach(_bench bench_gpu_scene bench_render_extract)
        add_test(NAME ${_bench}_smoke_runtime COMMAND ${_bench} --smoke --population runtime)
    endforeach()
    # The GPU case of the extraction (stage T2), per backend with every switch on: named so that `_switches_on$` (the
    # device-free set) does not match them; GPU tests, run serially.
    target_link_libraries(test_render_extract_gpu donut_tests_e7_switches)
    foreach(_backend d3d11 d3d12 vulkan)
        if(TEST test_render_extract_gpu_${_backend})
            add_test(NAME test_render_extract_gpu_switches_on_${_backend} COMMAND test_render_extract_gpu ${_backend})
            set_tests_properties(test_render_extract_gpu_switches_on_${_backend} PROPERTIES
                ENVIRONMENT "DONUT_E7_SWITCHES=all" RUN_SERIAL TRUE LABELS gpu)
        endif()
    endforeach()
endif()
