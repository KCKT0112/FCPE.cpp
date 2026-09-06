# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.

# ggml v0.19.0's Vulkan backend rounds F32 matrices/activations to F16 in
# cooperative-matrix paths, even with GGML_PREC_F32. Select its existing
# full-F32 shaders for this precision-sensitive model. Generate a build-local
# source copy so the user's ggml checkout and process environment stay intact.
function(fcpe_vulkan_precision)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/VulkanMatmul.cmake")
    get_target_property(_sources ggml-vulkan SOURCES)
    get_target_property(_source_dir ggml-vulkan SOURCE_DIR)
    set(_patched_sources "")
    set(_found FALSE)
    foreach(_source IN LISTS _sources)
        get_filename_component(_name "${_source}" NAME)
        if(_name STREQUAL "ggml-vulkan.cpp")
            if(IS_ABSOLUTE "${_source}")
                set(_original "${_source}")
            else()
                set(_original "${_source_dir}/${_source}")
            endif()
            file(READ "${_original}" _text)
            foreach(_flag GGML_VK_DISABLE_F16 GGML_VK_DISABLE_COOPMAT GGML_VK_DISABLE_COOPMAT2)
                set(_needle "getenv(\"${_flag}\")")
                string(FIND "${_text}" "${_needle}" _position)
                if(_position EQUAL -1)
                    message(FATAL_ERROR "FCPE Vulkan precision patch cannot find ${_flag}; use pinned ggml v0.19.0")
                endif()
                string(REPLACE "${_needle}" "\"1\" /* FCPE: preserve F32 input precision */" _text "${_text}")
            endforeach()
            # Avoid the slow host-visible/ReBAR path observed on discrete RTX
            # hardware. Upstream handles UMA before consulting this flag.
            fcpe_patch_unique(_text
                "device->disable_host_visible_vidmem = GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM != nullptr;"
                "device->disable_host_visible_vidmem = GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM != nullptr || !getenv(\"FCPE_VK_HOST_VISIBLE\");"
                )
            fcpe_patch_unique(_text
                "buf->memory_property_flags = mem_props.memoryTypes[*mtype_it].propertyFlags;"
                "buf->memory_property_flags = mem_props.memoryTypes[*mtype_it].propertyFlags;\n                    if (getenv(\"FCPE_VK_LOG_MEMORY_TYPE\")) GGML_LOG_INFO(\"FCPE Vulkan allocation: bytes=%zu type=%u heap=%u flags=%u\\n\", (size_t)mem_req.size, *mtype_it, mem_props.memoryTypes[*mtype_it].heapIndex, (unsigned)VkMemoryPropertyFlags(buf->memory_property_flags));"
                )
            if(FCPE_VULKAN_TILED_F32)
                fcpe_vulkan_matmul(_text)
            endif()
            if(FCPE_VULKAN_FUSION)
                include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/VulkanFusion.cmake")
                fcpe_vulkan_fusion(_text)
            endif()
            set(_output "${CMAKE_CURRENT_BINARY_DIR}/fcpe-vulkan/ggml-vulkan.cpp")
            file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fcpe-vulkan")
            # Avoid recompiling this large source when configuration did not change.
            set(_previous "")
            if(EXISTS "${_output}")
                file(READ "${_output}" _previous)
            endif()
            if(NOT _previous STREQUAL _text)
                file(WRITE "${_output}" "${_text}")
            endif()
            list(APPEND _patched_sources "${_output}")
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_original}")
            set(_found TRUE)
        else()
            list(APPEND _patched_sources "${_source}")
        endif()
    endforeach()
    if(NOT _found)
        message(FATAL_ERROR "Cannot locate ggml-vulkan.cpp for FCPE's F32 precision configuration")
    endif()
    set_property(TARGET ggml-vulkan PROPERTY SOURCES "${_patched_sources}")
    if(FCPE_VULKAN_TILED_F32 OR FCPE_VULKAN_FUSION)
        find_program(FCPE_GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin" REQUIRED)
        target_include_directories(ggml-vulkan PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/fcpe-vulkan")
    endif()
    if(FCPE_VULKAN_TILED_F32)
        fcpe_vulkan_matmul_shader("${CMAKE_CURRENT_BINARY_DIR}/fcpe-vulkan")
    endif()
    if(FCPE_VULKAN_FUSION)
        fcpe_vulkan_fusion_shader("${CMAKE_CURRENT_BINARY_DIR}/fcpe-vulkan")
    endif()
    target_include_directories(ggml-vulkan PRIVATE "${_source_dir}")
    message(STATUS "FCPE Vulkan: preserve F32 arithmetic; cooperative/F16 arithmetic paths disabled")
endfunction()
