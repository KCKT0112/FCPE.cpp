# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.

function(fcpe_metal_replace variable before after)
    string(FIND "${${variable}}" "${before}" _pos)
    if(_pos EQUAL -1)
        message(FATAL_ERROR "FCPE Metal precision anchor missing; use pinned ggml v0.19.0: ${before}")
    endif()
    string(LENGTH "${before}" _length)
    math(EXPR _end "${_pos} + ${_length}")
    string(SUBSTRING "${${variable}}" ${_end} -1 _tail)
    string(FIND "${_tail}" "${before}" _duplicate)
    if(NOT _duplicate EQUAL -1)
        message(FATAL_ERROR "FCPE Metal precision anchor is not unique: ${before}")
    endif()
    string(REPLACE "${before}" "${after}" _text "${${variable}}")
    set(${variable} "${_text}" PARENT_SCOPE)
endfunction()

function(fcpe_metal_write path text)
    set(_previous "")
    if(EXISTS "${path}")
        file(READ "${path}" _previous)
    endif()
    if(NOT _previous STREQUAL text)
        file(WRITE "${path}" "${text}")
    endif()
endfunction()

function(fcpe_metal_precision)
    if(NOT TARGET ggml-metal OR NOT GGML_METAL_EMBED_LIBRARY)
        message(FATAL_ERROR "FCPE Metal strict F32 requires GGML_METAL_EMBED_LIBRARY=ON")
    endif()
    get_target_property(_dir ggml-metal SOURCE_DIR)
    set(_output "${CMAKE_CURRENT_BINARY_DIR}/fcpe-metal")
    file(MAKE_DIRECTORY "${_output}")
    file(READ "${_dir}/ggml-metal.metal" _shader)
    file(READ "${_dir}/ggml-metal-device.cpp" _device)
    file(READ "${_dir}/ggml-metal-context.m" _context)
    file(READ "${_dir}/ggml-metal-ops.cpp" _ops)
    file(READ "${_dir}/ggml-metal.cpp" _backend)

    # Reuse the upstream 64x32 SIMD-group algorithm with float operands. Keep
    # this specialization outside HAS_TENSOR: newer GPUs use the same verified
    # arithmetic instead of implicitly selecting a different tensor kernel.
    set(_start "template<\n    typename S0, typename S0_4x4, typename S0_8x8,")
    string(FIND "${_shader}" "${_start}" _begin)
    string(FIND "${_shader}" "#endif // GGML_METAL_HAS_TENSOR" _end)
    if(_begin EQUAL -1 OR _end LESS_EQUAL _begin)
        message(FATAL_ERROR "Cannot locate ggml v0.19.0 Metal SIMD-group matmul")
    endif()
    math(EXPR _length "${_end} - ${_begin}")
    string(SUBSTRING "${_shader}" ${_begin} ${_length} _kernel)
    fcpe_metal_replace(_kernel "kernel void kernel_mul_mm(" "kernel void kernel_fcpe_mul_mm(")
    fcpe_metal_replace(_kernel "shmem + 4096" "shmem + 64 * 32 * sizeof(S0)")
    string(APPEND _shader "\n// FCPE strict F32 specialization of the upstream MIT kernel above.\n${_kernel}\n")
    string(APPEND _shader "template [[host_name(\"kernel_fcpe_mul_mm_f32_f32\")]] kernel mul_mm_t kernel_fcpe_mul_mm<float, float4x4, simdgroup_float8x8, float, float2x4, simdgroup_float8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>;\n")

    fcpe_metal_replace(_device
        "const bool has_tensor = ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->has_tensor;"
        "const bool strict_f32 = tsrc0 == GGML_TYPE_F32 && tsrc1 == GGML_TYPE_F32;\n    const bool has_tensor = !strict_f32 && ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->has_tensor;")
    fcpe_metal_replace(_device
        "snprintf(base, 256, \"kernel_mul_mm_%s_%s\", ggml_type_name(tsrc0), ggml_type_name(tsrc1));"
        "snprintf(base, 256, strict_f32 ? \"kernel_fcpe_mul_mm_%s_%s\" : \"kernel_mul_mm_%s_%s\", ggml_type_name(tsrc0), ggml_type_name(tsrc1));")
    fcpe_metal_replace(_device "res.smem = bc_out ? 8192 : (4096 + 2048);"
        "res.smem = strict_f32 ? (64 + 32) * 32 * sizeof(float) : (bc_out ? 8192 : (4096 + 2048));")

    # Diagnostic only: isolated command buffers provide GPU timestamps for each
    # original node. These samples deliberately exclude fusion and never enter
    # the normal benchmark. Default execution remains asynchronous upstream.
    file(READ "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/metal/profile.inc" _profile)
    fcpe_metal_replace(_context
        "    // number of nodes encoded by the main thread (empirically determined)"
        "${_profile}\n    // number of nodes encoded by the main thread (empirically determined)")
    fcpe_metal_replace(_context "const int n_main = MAX(64, 0.1*gf->n_nodes);"
        "const int n_main = getenv(\"FCPE_METAL_SINGLE_SUBMIT\") ? gf->n_nodes : MAX(64, 0.1*gf->n_nodes);")
    fcpe_metal_replace(_ops "    ggml_metal_device_t  dev;"
        "    bool can_fuse_span(int i, int count) const {\n        if (i + count > n_nodes()) return false;\n        const int begin = idxs[i], end = idxs[i + count - 1];\n        std::vector<int> nodes; std::vector<ggml_op> ops;\n        for (int n = begin; n <= end; ++n) { nodes.push_back(n); ops.push_back(gf->nodes[n]->op); }\n        return ggml_can_fuse_subgraph_ext(gf, nodes.data(), nodes.size(), ops.data(), &end, 1);\n    }\n\n    ggml_metal_device_t  dev;")
    file(READ "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/metal/kernels.metal" _extra_shader)
    string(APPEND _shader "\n${_extra_shader}")
    fcpe_metal_replace(_ops "static int ggml_metal_op_encode_impl(ggml_metal_op_t ctx, int idx) {"
        "#include \"dispatch.inc\"\n\nstatic int ggml_metal_op_encode_impl(ggml_metal_op_t ctx, int idx) {")
    fcpe_metal_replace(_ops "int ggml_metal_op_cpy(ggml_metal_op_t ctx, int idx) {"
        "int ggml_metal_op_cpy(ggml_metal_op_t ctx, int idx) {\n    if (fcpe_metal_copy(ctx, idx)) return 1;")
    fcpe_metal_replace(_ops "int ggml_metal_op_conv_2d_dw(ggml_metal_op_t ctx, int idx) {"
        "int ggml_metal_op_conv_2d_dw(ggml_metal_op_t ctx, int idx) {\n    if (fcpe_metal_depthwise(ctx, idx)) return 1;")
    fcpe_metal_replace(_ops "int ggml_metal_op_group_norm(ggml_metal_op_t ctx, int idx) {"
        "int ggml_metal_op_group_norm(ggml_metal_op_t ctx, int idx) {\n    if (fcpe_metal_group_norm(ctx, idx)) return 1;")
    fcpe_metal_replace(_ops "int ggml_metal_op_im2col(ggml_metal_op_t ctx, int idx) {"
        "int ggml_metal_op_im2col(ggml_metal_op_t ctx, int idx) {\n    if (fcpe_metal_im2col(ctx, idx)) return 1;")
    fcpe_metal_replace(_backend "    // some operations require additional memory for fleeting data:"
        "    if (tensor->op == GGML_OP_GROUP_NORM) {\n        const int groups = ((const int32_t *) tensor->op_params)[0];\n        const size_t chunk_count = (ggml_nelements(tensor) / groups + 1023) / 1024;\n        res = GGML_PAD(res, 16) + 2 * groups * chunk_count * sizeof(float);\n    }\n\n    // some operations require additional memory for fleeting data:")
    fcpe_metal_replace(_ops "    switch (node->op) {\n        case GGML_OP_CONCAT:"
        "    if (const int fused = fcpe_metal_fuse(ctx, idx)) {\n        for (int i = 0; i < fused; ++i) {\n            if (!ggml_metal_op_concurrency_add(ctx, ctx->node(idx + i))) ggml_metal_op_concurrency_reset(ctx);\n        }\n        return fused;\n    }\n\n    switch (node->op) {\n        case GGML_OP_CONCAT:")

    # Embed a merged shader under the existing library symbols. Replacing the
    # assembly source disconnects upstream's generation rule, avoiding races.
    file(READ "${_dir}/../ggml-common.h" _common)
    file(READ "${_dir}/ggml-metal-impl.h" _impl)
    fcpe_metal_replace(_shader "__embed_ggml-common.h__" "${_common}")
    fcpe_metal_replace(_shader "#include \"ggml-metal-impl.h\"" "${_impl}")
    fcpe_metal_write("${_output}/ggml-metal.metal" "${_shader}")
    fcpe_metal_write("${_output}/ggml-metal-device.cpp" "${_device}")
    fcpe_metal_write("${_output}/ggml-metal-context.m" "${_context}")
    fcpe_metal_write("${_output}/ggml-metal-ops.cpp" "${_ops}")
    fcpe_metal_write("${_output}/ggml-metal.cpp" "${_backend}")
    fcpe_metal_write("${_output}/ggml-metal-embed.s"
        ".section __DATA,__ggml_metallib\n.globl _ggml_metallib_start\n_ggml_metallib_start:\n.incbin \"${_output}/ggml-metal.metal\"\n.globl _ggml_metallib_end\n_ggml_metallib_end:\n")
    set_source_files_properties("${_output}/ggml-metal-embed.s" DIRECTORY "${_dir}" PROPERTIES
        OBJECT_DEPENDS "${_output}/ggml-metal.metal")
    get_target_property(_sources ggml-metal SOURCES)
    set(_patched "")
    set(_replaced 0)
    foreach(_source IN LISTS _sources)
        get_filename_component(_name "${_source}" NAME)
        if(_name STREQUAL "ggml-metal-device.cpp" OR _name STREQUAL "ggml-metal-embed.s" OR _name STREQUAL "ggml-metal-context.m" OR _name STREQUAL "ggml-metal-ops.cpp" OR _name STREQUAL "ggml-metal.cpp")
            list(APPEND _patched "${_output}/${_name}")
            math(EXPR _replaced "${_replaced} + 1")
        else()
            list(APPEND _patched "${_source}")
        endif()
    endforeach()
    if(NOT _replaced EQUAL 5)
        message(FATAL_ERROR "Cannot replace ggml Metal device and embedded assembly sources")
    endif()
    set_property(TARGET ggml-metal PROPERTY SOURCES "${_patched}")
    target_include_directories(ggml-metal PRIVATE "${_dir}")
    target_include_directories(ggml-metal PRIVATE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/metal")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${_dir}/ggml-metal.metal" "${_dir}/ggml-metal-device.cpp"
        "${_dir}/ggml-metal-impl.h" "${_dir}/../ggml-common.h" "${_dir}/ggml-metal-context.m"
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/metal/profile.inc"
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/metal/kernels.metal" "${_dir}/ggml-metal-ops.cpp" "${_dir}/ggml-metal.cpp")
    message(STATUS "FCPE Metal: F32 SIMD-group operands and accumulation; build-local embedded shader")
endfunction()
