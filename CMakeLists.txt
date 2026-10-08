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


project(donut)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# EPM, the package manager of the Ethereal projects, lives in nvrhi (a submodule of this repository).
include("${CMAKE_CURRENT_SOURCE_DIR}/ethereal-nvrhi/cmake/EPM.cmake")

set_property( GLOBAL PROPERTY USE_FOLDERS ON)

if (MSVC)
    string(APPEND CMAKE_C_FLAGS " /utf-8")
    string(APPEND CMAKE_CXX_FLAGS " /W3 /utf-8")
    if (NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
        string(APPEND CMAKE_CXX_FLAGS " /MP")
    endif()
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        string(APPEND CMAKE_CXX_FLAGS " -mf16c")
    endif()
endif()

string(APPEND CMAKE_CXX_FLAGS_DEBUG " -D_DEBUG")

include(CMakeDependentOption)

option(DONUT_WITH_NVRHI "Enable NVRHI and related projects" ON)

cmake_dependent_option(DONUT_WITH_DX11 "Enable the DX11 version of Donut" ON "WIN32" OFF)
cmake_dependent_option(DONUT_WITH_DX12 "Enable the DX12 version of Donut" ON "WIN32" OFF)
option(DONUT_WITH_VULKAN "Enable the Vulkan version of Donut" ON)
option(DONUT_WITH_AFTERMATH "Enable Aftermath crash dump generation with Donut" OFF)
option(DONUT_EMBED_SHADER_PDBS "Embed shader PDBs with shader binary files" OFF)

option(DONUT_WITH_STATIC_SHADERS "Build Donut with statically linked shaders" OFF)

option(DONUT_WITH_AUDIO "Include Audio features (XAudio2)" OFF)
option(DONUT_WITH_TINYEXR "Include TinyEXR" ON)
option(DONUT_WITH_UNIT_TESTS "Donut unit-tests (see CMake/CTest documentation)" OFF)

option(DONUT_WITH_STREAMLINE "Enable Streamline, separate package required" OFF)
set(DONUT_STREAMLINE_FETCH_URL "" CACHE STRING "URL to Streamline package to fetch from https://github.com/NVIDIA-RTX/Streamline/releases")
set(DONUT_STREAMLINE_FETCH_SHA256 "" CACHE STRING "SHA256 hash of Streamline package from https://github.com/NVIDIA-RTX/Streamline/releases")
set(DONUT_STREAMLINE_FETCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/streamline" CACHE STRING "Directory to fetch Streamline to, empty uses build directory default")
set(DONUT_STREAMLINE_SEARCH_PATHS "" CACHE STRING "Search paths for Streamline package")

option(DONUT_WITH_DLSS "Enable the NGX DLSS integration, separate package required" OFF)
set(DONUT_DLSS_FETCH_URL "https://github.com/NVIDIA/DLSS.git" CACHE STRING "URL of the DLSS Git repository to fetch from")
set(DONUT_DLSS_FETCH_TAG "a8ed84e2d1cc1efa7bef63fb9394be6cb06c0d74" CACHE STRING "Git commit hash or tag for DLSS")
set(DONUT_DLSS_FETCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/DLSS" CACHE STRING "Directory to fetch DLSS to, empty uses build directory default")

option(DONUT_WITH_KTX "Build KTX2 (BCn + Zstandard) texture loading support" ON)

if (WIN32)
    option(DONUT_FORCE_DISCRETE_GPU "Declare symbols to make the OS run the app on discrete GPU on laptops" OFF)
endif()

add_subdirectory(thirdparty)

if (DONUT_WITH_NVRHI)
    set(NVRHI_WITH_DX11 "${DONUT_WITH_DX11}" CACHE BOOL "" FORCE)
    set(NVRHI_WITH_DX12 "${DONUT_WITH_DX12}" CACHE BOOL "" FORCE)
    set(NVRHI_WITH_VULKAN "${DONUT_WITH_VULKAN}" CACHE BOOL "" FORCE)
    set(NVRHI_WITH_AFTERMATH "${DONUT_WITH_AFTERMATH}" CACHE BOOL "" FORCE)

    epm_add_package(NAME nvrhi SOURCE_DIR ethereal-nvrhi)

    set(DONUT_SHADER_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include CACHE PATH "" FORCE)
    
    add_subdirectory(shaders)
endif()

if (DONUT_WITH_DLSS)
    epm_add_asset(
        NAME DLSS
        GIT_REPOSITORY ${DONUT_DLSS_FETCH_URL}
        GIT_TAG ${DONUT_DLSS_FETCH_TAG}
        DESTINATION "${DONUT_DLSS_FETCH_DIR}")

    include(cmake/DLSS.cmake)
endif()

include(donut-core.cmake)
if (DONUT_WITH_NVRHI)
    include(donut-engine.cmake)
    include(donut-render.cmake)
    include(donut-app.cmake)
endif()

if (DONUT_WITH_UNIT_TESTS)
    include(CTest)
    add_subdirectory(tests)
endif()

if (DONUT_WITH_STREAMLINE)
    # Validate that CMAKE_RUNTIME_OUTPUT_DIRECTORY is set.
    # The Streamline CMake script uses it to copy DLLs, and it will fail at compile time with obscure messages
    # if the variable is empty.
    if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
        message(SEND_ERROR "Streamline integration requires that CMAKE_RUNTIME_OUTPUT_DIRECTORY is non-empty.")
    endif()
    
    if (DONUT_STREAMLINE_FETCH_TAG)
        message(WARNING "DONUT_STREAMLINE_FETCH_TAG is deprecated. Use DONUT_STREAMLINE_FETCH_SHA256 instead. Tag fetching from the public repo no longer has the binaries.")
    endif()

    # XOR logic to make sure both options are set 
    if ((DONUT_STREAMLINE_FETCH_URL AND NOT DONUT_STREAMLINE_FETCH_SHA256) OR 
        (NOT DONUT_STREAMLINE_FETCH_URL AND DONUT_STREAMLINE_FETCH_SHA256))
        message(SEND_ERROR "Both DONUT_STREAMLINE_FETCH_URL and DONUT_STREAMLINE_FETCH_SHA256 must be defined together")
    endif()

    if (DONUT_STREAMLINE_FETCH_URL AND DONUT_STREAMLINE_FETCH_SHA256)
        epm_add_package(
            NAME streamline
            URL ${DONUT_STREAMLINE_FETCH_URL}
            URL_HASH SHA256=${DONUT_STREAMLINE_FETCH_SHA256}
            FETCH_DIR "${DONUT_STREAMLINE_FETCH_DIR}")
    else ()
        if (DONUT_STREAMLINE_SEARCH_PATHS)
            set(STREAMLINE_SEARCH_PATHS ${DONUT_STREAMLINE_SEARCH_PATHS})
        endif()
        find_package(STREAMLINE REQUIRED)

        if (STREAMLINE_FOUND)
            epm_add_package(NAME streamline SOURCE_DIR "${STREAMLINE_SOURCE_DIR}" BINARY_DIR "${CMAKE_BINARY_DIR}/streamline")
        endif()
    endif()
endif()
