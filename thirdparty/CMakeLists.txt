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


# Third-party code is resolved through EPM (ethereal-nvrhi/cmake/EPM.cmake). glfw, imgui, stb,
# cgltf and shader-tool are fetched into this directory (git-ignored) at fixed revisions; a
# checkout that is already here is used as it is. zstd goes to the EPM source cache. Each
# package is skipped when a parent project already provides its target.
set(third_party_folder "Third-Party Libraries")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

if (DONUT_WITH_NVRHI)
    epm_add_package(
        NAME imgui
        GITHUB_REPOSITORY ocornut/imgui
        GIT_TAG 45acd5e0e82f4c954432533ae9985ff0e1aad6d5
        FETCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/imgui"
        TARGETS imgui
        DOWNLOAD_ONLY)
    if (NOT imgui_SOURCE STREQUAL target)
        include(imgui.cmake)
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
    GITHUB_REPOSITORY nothings/stb
    GIT_TAG 2e2bef463a5b53ddf8bb788e25da6b8506314c08
    FETCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/stb"
    TARGETS stb
    DOWNLOAD_ONLY)
if (NOT stb_SOURCE STREQUAL target)
    add_library(stb INTERFACE)
    target_include_directories(stb INTERFACE ${stb_SOURCE_DIR})
endif()

epm_add_package(
    NAME cgltf
    GITHUB_REPOSITORY jkuhlmann/cgltf
    GIT_TAG fa3b80fa762790192c9532b63c441627416ff300
    FETCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cgltf"
    TARGETS cgltf
    DOWNLOAD_ONLY)
if (NOT cgltf_SOURCE STREQUAL target)
    add_library(cgltf INTERFACE)
    target_include_directories(cgltf INTERFACE ${cgltf_SOURCE_DIR})
endif()

# Optional dependencies

if (DONUT_WITH_TINYEXR AND NOT TARGET tinyexr)
    add_library(tinyexr INTERFACE)
    target_include_directories(tinyexr INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/tinyexr)
endif()

if (DONUT_WITH_NVRHI)
    epm_add_package(
        NAME glfw
        GITHUB_REPOSITORY glfw/glfw
        GIT_TAG 3.4
        FETCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/glfw"
        TARGETS glfw
        OPTIONS
            GLFW_BUILD_EXAMPLES=OFF
            GLFW_BUILD_TESTS=OFF
            GLFW_BUILD_DOCS=OFF
            GLFW_INSTALL=OFF)
    if (glfw_ADDED AND NOT glfw_SOURCE STREQUAL target)
        set_target_properties(glfw PROPERTIES FOLDER ${third_party_folder})
        set_target_properties(update_mappings PROPERTIES FOLDER ${third_party_folder})
    endif()

    epm_add_package(
        NAME ShaderTool
        GITHUB_REPOSITORY HuazyYang/shader-tool
        GIT_TAG afbf2734e3d8dcad3553467d1e4427f20378feca
        FETCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shader-tool"
        TARGETS ShaderTool)
    if (ShaderTool_ADDED AND NOT ShaderTool_SOURCE STREQUAL target)
        set_target_properties(ShaderTool PROPERTIES FOLDER ${third_party_folder})
        set_target_properties(ShaderToolBlob PROPERTIES FOLDER ${third_party_folder})
    endif()
endif()

# if(NOT TARGET assimp)
# set(BUILD_SHARED_LIBS OFF)
# add_subdirectory(assimp-6.0.4)
# endif()
