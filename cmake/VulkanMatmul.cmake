# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.

# Called on the build-local ggml-vulkan.cpp source in VulkanPrecision.cmake.
# All replacements require unique anchors, so a changed ggml version fails at
# configure time instead of silently losing precision/performance guarantees.
function(fcpe_patch_unique variable needle replacement)
    string(FIND "${${variable}}" "${needle}" _first)
    if(_first EQUAL -1)
        message(FATAL_ERROR "FCPE Vulkan patch anchor missing: ${needle}")
    endif()
    string(REPLACE "${needle}" "" _without "${${variable}}")
    string(LENGTH "${${variable}}" _before_length)
    string(LENGTH "${_without}" _after_length)
    string(LENGTH "${needle}" _needle_length)
    math(EXPR _removed "${_before_length} - ${_after_length}")
    if(NOT _removed EQUAL _needle_length)
        message(FATAL_ERROR "FCPE Vulkan patch anchor is not unique: ${needle}")
    endif()
    string(REPLACE "${needle}" "${replacement}" _result "${${variable}}")
    set(${variable} "${_result}" PARENT_SCOPE)
endfunction()

function(fcpe_vulkan_matmul variable)
    set(_text "${${variable}}")
    fcpe_patch_unique(_text "#include \"ggml-vulkan.h\""
        "#include \"ggml-vulkan.h\"\n#include \"fcpe-matmul-f32-spv.h\"")
    fcpe_patch_unique(_text "    vk_matmul_pipeline pipeline_matmul_f32 {};"
        "    vk_pipeline pipeline_fcpe_matmul_f32;\n    std::array<vk_pipeline,6> pipeline_fcpe_mm_tuned;\n    vk_matmul_pipeline pipeline_matmul_f32 {};")
    set(_anchor "    // FA scalar has two SPIR-V modules (MMQ vs non-MMQ); FA cm1 has one. K/V")
    set(_insert [=[
    if (device->properties.limits.maxComputeWorkGroupInvocations >= 256 &&
        device->properties.limits.maxComputeWorkGroupSize[0] >= 16 &&
        device->properties.limits.maxComputeWorkGroupSize[1] >= 16 &&
        device->properties.limits.maxComputeSharedMemorySize >= 8320) {
        ggml_vk_create_pipeline(device, device->pipeline_fcpe_matmul_f32,
            "fcpe_matmul_f32_64x64", sizeof(fcpe_matmul_f32_spv), fcpe_matmul_f32_spv,
            "main", 3, sizeof(vk_mat_mat_push_constants), {64, 64, 1}, {}, 1);
    }
]=])
    fcpe_patch_unique(_text "${_anchor}" "${_insert}\n${_anchor}")
    set(_anchor "#undef CREATE_MM\n\n    // mul mat vec")
    set(_insert [=[
    // F32-only register/warp tile candidates. BK is fixed to 32 in upstream's
    // dense F32 shader. Keep the exact upstream code and vary specialization.
    const std::vector<std::vector<uint32_t>> fcpe_tiles = {
        {128,128,64,32,64,32,2,4,4,1,32},
        {128,64,128,32,32,64,2,4,4,1,32},
        {256,128,128,32,64,32,2,4,4,1,32},
        {128,64,64,32,32,32,1,4,4,1,32},
        {64,64,64,32,32,64,1,4,4,1,32},
        {256,128,64,32,32,32,1,4,4,1,32}
    };
    if (device->subgroup_size == 32 && device->properties.limits.maxComputeSharedMemorySize >= 34816 &&
        device->properties.limits.maxComputeWorkGroupInvocations >= 256 &&
        device->properties.limits.maxComputeWorkGroupSize[0] >= 256) {
        for (uint32_t v=0;v<fcpe_tiles.size();++v) {
            ggml_vk_create_pipeline2(device,device->pipeline_fcpe_mm_tuned[v],"fcpe_mm_tuned_"+std::to_string(v),
                matmul_f32_f32_fp32_len,matmul_f32_f32_fp32_data,"main",3,sizeof(vk_mat_mat_push_constants),
                {fcpe_tiles[v][1],fcpe_tiles[v][2],1},ggml_vk_mul_mm_spec(fcpe_tiles[v],true),4);
        }
    }
]=])
    fcpe_patch_unique(_text "${_anchor}" "${_insert}\n${_anchor}")
    set(_anchor "    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline(ctx, mmp, ne01, ne11, aligned, qx_needs_dequant ? f16_type : src0->type, effective_src1_type);")
    set(_insert [=[
    // Conservative specialization; all other operations retain upstream paths.
    // The environment switch supports reproducible before/after measurements.
    const bool fcpe_fast_matmul = !getenv("FCPE_VK_DISABLE_TILED_F32") &&
        (getenv("FCPE_VK_TILED_F32") || !ctx->device->disable_host_visible_vidmem) &&
        src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        !qx_needs_dequant && !qy_needs_dequant && !quantize_y &&
        ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
        ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 &&
        ctx->device->properties.limits.maxComputeWorkGroupInvocations >= 256 &&
        ctx->device->properties.limits.maxComputeWorkGroupSize[0] >= 16 &&
        ctx->device->properties.limits.maxComputeWorkGroupSize[1] >= 16 &&
        ctx->device->properties.limits.maxComputeSharedMemorySize >= 8320 &&
        ggml_nbytes(src0) <= ctx->device->properties.limits.maxStorageBufferRange &&
        ggml_nbytes(src1) <= ctx->device->properties.limits.maxStorageBufferRange &&
        ggml_nbytes(dst) <= ctx->device->properties.limits.maxStorageBufferRange;
    if (fcpe_fast_matmul) pipeline = ctx->device->pipeline_fcpe_matmul_f32;
    const char * fcpe_variant = getenv("FCPE_VK_MM_VARIANT");
    if (!fcpe_variant && !getenv("FCPE_VK_DISABLE_TILED_F32") && !getenv("FCPE_VK_TILED_F32") &&
        ctx->device->disable_host_visible_vidmem && ctx->device->vendor_id==VK_VENDOR_ID_NVIDIA && ne11>32) {
        fcpe_variant = ne11<=128 ? "3" : "1";
    }
    const bool fcpe_tuned = fcpe_variant && fcpe_variant[0]>='0' && fcpe_variant[0]<='5' && fcpe_variant[1]=='\0' &&
        src0->type==GGML_TYPE_F32 && src1->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_F32 &&
        ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
        !qx_needs_dequant && !qy_needs_dequant && !quantize_y &&
        ne02==1 && ne03==1 && ne12==1 && ne13==1 && ne10%4==0 &&
        ctx->device->subgroup_size==32 && ctx->device->properties.limits.maxComputeSharedMemorySize>=34816 &&
        ctx->device->properties.limits.maxComputeWorkGroupInvocations>=256 &&
        ctx->device->properties.limits.maxComputeWorkGroupSize[0]>=256 &&
        ggml_nbytes(src0)<=ctx->device->properties.limits.maxStorageBufferRange &&
        ggml_nbytes(src1)<=ctx->device->properties.limits.maxStorageBufferRange &&
        ggml_nbytes(dst)<=ctx->device->properties.limits.maxStorageBufferRange;
    if (fcpe_tuned) pipeline=ctx->device->pipeline_fcpe_mm_tuned[fcpe_variant[0]-'0'];
]=])
    fcpe_patch_unique(_text "${_anchor}" "${_anchor}\n${_insert}")
    fcpe_patch_unique(_text
        "    const uint32_t split_k = ggml_vk_guess_split_k(ctx, ne01, ne11, ne10, disable_split_k, pipeline);"
        "    const uint32_t split_k = (fcpe_fast_matmul || fcpe_tuned) ? 1 : ggml_vk_guess_split_k(ctx, ne01, ne11, ne10, disable_split_k, pipeline);")
    set(${variable} "${_text}" PARENT_SCOPE)
endfunction()

function(fcpe_vulkan_matmul_shader output_dir)
    find_program(FCPE_GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin" REQUIRED)
    set(_shader "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/fcpe-matmul-f32.comp")
    set(_spv "${output_dir}/fcpe-matmul-f32.spv")
    set(_header "${output_dir}/fcpe-matmul-f32-spv.h")
    add_custom_command(OUTPUT "${_header}" "${_spv}"
        COMMAND "${FCPE_GLSLC}" -O --target-env=vulkan1.2 "${_shader}" -o "${_spv}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${_spv}" "-DOUTPUT=${_header}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedSpirv.cmake"
        DEPENDS "${_shader}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedSpirv.cmake"
        VERBATIM)
    add_custom_target(fcpe-vulkan-shader DEPENDS "${_header}" "${_spv}")
    add_dependencies(ggml-vulkan fcpe-vulkan-shader)
    target_include_directories(ggml-vulkan PRIVATE "${output_dir}")
endfunction()
