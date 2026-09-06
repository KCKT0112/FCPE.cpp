# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.

function(fcpe_vulkan_fusion variable)
    set(_text "${${variable}}")
    fcpe_patch_unique(_text "#include \"ggml-vulkan.h\""
        "#include \"ggml-vulkan.h\"\n#include \"fcpe-fused-spv.h\"")
    fcpe_patch_unique(_text "    vk_matmul_pipeline pipeline_matmul_f32 {};"
        "    std::array<vk_pipeline, 6> pipeline_fcpe_fused;\n    vk_matmul_pipeline pipeline_matmul_f32 {};")
    set(_anchor "    // FA scalar has two SPIR-V modules (MMQ vs non-MMQ); FA cm1 has one. K/V")
    set(_insert [=[
    if (device->properties.limits.maxComputeWorkGroupInvocations >= 512 &&
        device->properties.limits.maxComputeWorkGroupSize[0] >= 512) {
        for (uint32_t mode=0; mode<6; ++mode) {
            ggml_vk_create_pipeline2(device, device->pipeline_fcpe_fused[mode],
                "fcpe_fused_"+std::to_string(mode), sizeof(fcpe_fused_spv), fcpe_fused_spv,
                "main",4,40,{1,1,1},{mode},1);
        }
    }
]=])
    fcpe_patch_unique(_text "${_anchor}" "${_insert}\n${_anchor}")
    set(_anchor "// Returns true if node has enqueued work into the queue, false otherwise")
    fcpe_patch_unique(_text "${_anchor}" "#include \"fcpe-fusion.inc\"\n\n${_anchor}")
    set(_anchor "    switch (node->op) {\n    case GGML_OP_REPEAT:")
    fcpe_patch_unique(_text "${_anchor}"
        "    if (!fcpe_vk_dispatch_fusion(ctx, compute_ctx, cgraph, node_idx))\n${_anchor}")
    set(_anchor "            if (num_adds) {")
    set(_insert [=[
            const int fcpe_fusion = ctx->device->properties.limits.maxComputeWorkGroupInvocations >= 512 &&
                ctx->device->properties.limits.maxComputeWorkGroupSize[0] >= 512 ? fcpe_vk_fusion(cgraph,i) : -1;
            if (fcpe_fusion >= 0) {
                ctx->num_additional_fused_ops=fcpe_fusion == 0 ? 2 : 1;
                fusion_string="FCPE_FUSED";
                // Conservatively reject all source/destination overlap.
                std::fill_n(op_srcs_fused_elementwise,ctx->num_additional_fused_ops+1,false);
            } else
]=])
    fcpe_patch_unique(_text "${_anchor}" "${_insert}\n${_anchor}")
    set(${variable} "${_text}" PARENT_SCOPE)
endfunction()

function(fcpe_vulkan_fusion_shader output_dir)
    set(_shader "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/fcpe-fused.comp")
    set(_spv "${output_dir}/fcpe-fused.spv")
    set(_header "${output_dir}/fcpe-fused-spv.h")
    add_custom_command(OUTPUT "${_header}" "${_spv}"
        COMMAND "${FCPE_GLSLC}" -O --target-env=vulkan1.2 "${_shader}" -o "${_spv}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${_spv}" "-DOUTPUT=${_header}" -DSYMBOL=fcpe_fused_spv
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedSpirv.cmake"
        DEPENDS "${_shader}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedSpirv.cmake" VERBATIM)
    add_custom_target(fcpe-vulkan-fused-shader DEPENDS "${_header}" "${_spv}")
    add_dependencies(ggml-vulkan fcpe-vulkan-fused-shader)
    target_include_directories(ggml-vulkan PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan")
endfunction()
