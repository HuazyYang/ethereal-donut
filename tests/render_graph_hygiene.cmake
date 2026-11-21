# Header hygiene of the render graph (aggregate docs/rdg-design.md, 3.2), run by ctest:
#
#     cmake -DDONUT_SOURCE_DIR=<donut> -P render_graph_hygiene.cmake
#
# The graph code stays generic: include/donut/render/RenderGraph*.h and src/render/RenderGraph*.{h,cpp} may include
# only nvrhi/ headers, donut/core/ headers, other RenderGraph* headers and the standard library. It fails when one
# of them includes any other donut/render/ or donut/engine/ header, or any other project header. One exception: the
# host's worker pool, RenderGraphWorkerPool.h, is a client of the engine's job system (aggregate docs/ecs-design.md,
# D10) and includes donut/engine/JobSystem.h; the graph code itself still reaches threads only through
# ExecuteDesc::parallelFor.

if(NOT DONUT_SOURCE_DIR)
    message(FATAL_ERROR "render_graph_hygiene.cmake: set DONUT_SOURCE_DIR")
endif()

file(GLOB files
    "${DONUT_SOURCE_DIR}/include/donut/render/RenderGraph*.h"
    "${DONUT_SOURCE_DIR}/src/render/RenderGraph*.h"
    "${DONUT_SOURCE_DIR}/src/render/RenderGraph*.cpp")
list(SORT files)
list(LENGTH files count)
if(count LESS 3)
    message(FATAL_ERROR "render_graph_hygiene.cmake: found ${count} RenderGraph files under ${DONUT_SOURCE_DIR}")
endif()

set(violations "")
foreach(file IN LISTS files)
    file(STRINGS "${file}" lines REGEX "^[ \t]*#[ \t]*include")
    foreach(line IN LISTS lines)
        if(line MATCHES "#[ \t]*include[ \t]*[<\"]([^>\"]+)[>\"]")
            set(header "${CMAKE_MATCH_1}")
            if(header MATCHES "^nvrhi/" OR header MATCHES "^donut/core/" OR header MATCHES "^donut/render/RenderGraph[A-Za-z]*\\.h$"
               OR header MATCHES "^RenderGraph[A-Za-z]*\\.h$")
                continue()
            endif()
            if(header STREQUAL "donut/engine/JobSystem.h"
               AND file MATCHES "/include/donut/render/RenderGraphWorkerPool\\.h$")
                continue()
            endif()
            # The standard library: a bare name without a path or an extension.
            if(NOT header MATCHES "[/.]")
                continue()
            endif()
            file(RELATIVE_PATH relative "${DONUT_SOURCE_DIR}" "${file}")
            string(APPEND violations "  ${relative}: #include ${header}\n")
        endif()
    endforeach()
endforeach()

if(violations)
    message(FATAL_ERROR "render graph header hygiene (design 3.2) failed:\n${violations}")
endif()
message(STATUS "render graph header hygiene: ${count} files OK")
