#
# Copyright (c) 2014-2025, NVIDIA CORPORATION. All rights reserved.
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


# Third-party code is resolved through EPM (ethereal-nvrhi/cmake/EPM.cmake). glfw, imgui, stb
# and cgltf use the installed copy (vcpkg) when there is a usable one, and are otherwise fetched
# at a fixed revision into EPM's source directory (EPM_SOURCE_CACHE, default <build>/_epm/src).
# shader-tool is a git submodule. Each package is skipped when a parent project already
# provides its target.
set(third_party_folder "Third-Party Libraries")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# vcpkg's imgui has the GLFW backend (imgui_impl_glfw.h) only with its glfw-binding feature,
# and Donut needs at least the version of the revision it is built against otherwise.
function(donut_accept_imgui name result)
    set(${result} FALSE PARENT_SCOPE)
    if (NOT TARGET imgui::imgui)
        return()
    endif()
    get_target_property(_dirs imgui::imgui INTERFACE_INCLUDE_DIRECTORIES)
    foreach (_dir IN LISTS _dirs)
        if (EXISTS "${_dir}/imgui_impl_glfw.h" AND EXISTS "${_dir}/imgui.h")
            file(STRINGS "${_dir}/imgui.h" _line REGEX "^#define IMGUI_VERSION_NUM +[0-9]+")
            string(REGEX MATCH "[0-9]+$" _version "${_line}")
            if (_version GREATER_EQUAL 19220)
                set(${result} TRUE PARENT_SCOPE)
            endif()
        endif()
    endforeach()
endfunction()

# An INTERFACE target for a header-only library, from the installed copy or the fetched sources.
function(donut_header_only_library name)
    if (TARGET ${name})
        return()
    endif()
    set(_dir "${${name}_SOURCE_DIR}")
    if (${name}_INCLUDE_DIR)
        set(_dir "${${name}_INCLUDE_DIR}")
    endif()
    add_library(${name} INTERFACE)
    target_include_directories(${name} INTERFACE "${_dir}")
endfunction()

if (DONUT_WITH_NVRHI)
    epm_add_package(
        NAME imgui
        PACKAGE imgui
        FIND_PACKAGE_ARGUMENTS CONFIG
        FIND_ACCEPT donut_accept_imgui
        GITHUB_REPOSITORY ocornut/imgui
        GIT_TAG 45acd5e0e82f4c954432533ae9985ff0e1aad6d5
        TARGETS imgui
        DOWNLOAD_ONLY)
    if (imgui_SOURCE STREQUAL installed)
        add_library(imgui INTERFACE)
        target_link_libraries(imgui INTERFACE imgui::imgui)
    elseif (NOT imgui_SOURCE STREQUAL target)
        # The fetched sources: the core, the demo window and the GLFW backend.
        add_library(imgui STATIC
            ${imgui_SOURCE_DIR}/imconfig.h
            ${imgui_SOURCE_DIR}/imgui_draw.cpp
            ${imgui_SOURCE_DIR}/imgui_tables.cpp
            ${imgui_SOURCE_DIR}/imgui_widgets.cpp
            ${imgui_SOURCE_DIR}/imgui_internal.h
            ${imgui_SOURCE_DIR}/imgui.cpp
            ${imgui_SOURCE_DIR}/imgui.h
            ${imgui_SOURCE_DIR}/imstb_rectpack.h
            ${imgui_SOURCE_DIR}/imstb_textedit.h
            ${imgui_SOURCE_DIR}/imstb_truetype.h
            ${imgui_SOURCE_DIR}/imgui_demo.cpp
            ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.h
            ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp)
        target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR})
        target_link_libraries(imgui PRIVATE glfw)
        set_target_properties(imgui PROPERTIES FOLDER ${third_party_folder})
    endif()
endif()

if (NOT TARGET jsoncpp_static)
    add_library(jsoncpp_static)
    set(jsoncpp_path "${CMAKE_CURRENT_SOURCE_DIR}/jsoncpp-amalgam")
    target_include_directories(jsoncpp_static PUBLIC "${jsoncpp_path}")
    target_sources(jsoncpp_static PRIVATE "${jsoncpp_path}/jsoncpp.cpp")
    set_target_properties(jsoncpp_static PROPERTIES FOLDER ${third_party_folder})
endif()

# KTX2 texture loading (BCn + Zstandard supercompression) needs Zstandard.
if (DONUT_WITH_KTX)
    # Leave ZSTD_USE_STATIC_RUNTIME off: it rewrites /MD to /MT in CMAKE_*_FLAGS,
    # while the CRT actually comes from CMAKE_MSVC_RUNTIME_LIBRARY, which zstd inherits.
    epm_add_package(
        NAME zstd
        GITHUB_REPOSITORY facebook/zstd
        GIT_TAG v1.5.6
        SOURCE_SUBDIR build/cmake
        TARGETS libzstd_static
        OPTIONS
            ZSTD_BUILD_PROGRAMS=OFF
            ZSTD_BUILD_SHARED=OFF
            ZSTD_BUILD_STATIC=ON
            ZSTD_BUILD_TESTS=OFF
            ZSTD_LEGACY_SUPPORT=OFF)
    if (TARGET libzstd_static AND NOT zstd_SOURCE STREQUAL target)
        set_target_properties(libzstd_static PROPERTIES FOLDER ${third_party_folder})
    endif()
endif()

epm_add_package(
    NAME stb
    FIND_HEADER stb_image.h
    GITHUB_REPOSITORY nothings/stb
    GIT_TAG 2e2bef463a5b53ddf8bb788e25da6b8506314c08
    TARGETS stb
    DOWNLOAD_ONLY)
donut_header_only_library(stb)

epm_add_package(
    NAME cgltf
    FIND_HEADER cgltf.h
    GITHUB_REPOSITORY jkuhlmann/cgltf
    GIT_TAG fa3b80fa762790192c9532b63c441627416ff300
    TARGETS cgltf
    DOWNLOAD_ONLY)
donut_header_only_library(cgltf)

# Optional dependencies

if (DONUT_WITH_TINYEXR AND NOT TARGET tinyexr)
    add_library(tinyexr INTERFACE)
    target_include_directories(tinyexr INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/tinyexr)
endif()

if (DONUT_WITH_NVRHI)
    epm_add_package(
        NAME glfw
        PACKAGE glfw3
        VERSION 3.4
        FIND_PACKAGE_ARGUMENTS CONFIG
        GITHUB_REPOSITORY glfw/glfw
        GIT_TAG 3.4
        TARGETS glfw
        OPTIONS
            GLFW_BUILD_EXAMPLES=OFF
            GLFW_BUILD_TESTS=OFF
            GLFW_BUILD_DOCS=OFF
            GLFW_INSTALL=OFF)
    if (glfw_SOURCE MATCHES "^(local|existing|cached|downloaded)$")
        set_target_properties(glfw PROPERTIES FOLDER ${third_party_folder})
        set_target_properties(update_mappings PROPERTIES FOLDER ${third_party_folder})
    endif()

    epm_add_package(NAME ShaderTool SOURCE_DIR shader-tool TARGETS ShaderTool)
    if (ShaderTool_SOURCE STREQUAL submodule)
        set_target_properties(ShaderTool PROPERTIES FOLDER ${third_party_folder})
        set_target_properties(ShaderToolBlob PROPERTIES FOLDER ${third_party_folder})
    endif()
endif()

# if(NOT TARGET assimp)
# set(BUILD_SHARED_LIBS OFF)
# add_subdirectory(assimp-6.0.4)
# endif()
