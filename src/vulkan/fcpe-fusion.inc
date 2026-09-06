// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.

// Included into the build-local ggml Vulkan backend. Fusions use upstream graph
// consumer checks, alias checks, dependency tracking and command submission.
static bool fcpe_vk_matrix(const ggml_tensor * t) {
    return t && t->type == GGML_TYPE_F32 && t->ne[2] == 1 && t->ne[3] == 1 && t->nb[0] == 4;
}
static bool fcpe_vk_bias(const ggml_tensor * t, int64_t cols) {
    return fcpe_vk_matrix(t) && ggml_is_contiguous(t) && t->ne[0] == cols && t->ne[1] == 1;
}
static int fcpe_vk_fusion(const ggml_cgraph * g, int i) {
    if (getenv("FCPE_VK_DISABLE_FCPE_FUSION") || i+1 >= g->n_nodes) return -1;
    auto * a=g->nodes[i]; auto * b=g->nodes[i+1];
    if (!fcpe_vk_matrix(a) || !fcpe_vk_matrix(b) || !ggml_is_contiguous(a) || !ggml_is_contiguous(b)) return -1;
    const auto cols=a->ne[0];
    if (a->op == GGML_OP_NORM && i+2 < g->n_nodes &&
        ggml_can_fuse(g, i, {GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD})) {
        auto * c=g->nodes[i+2];
        if (b->src[0] == a && c->src[0] == b && fcpe_vk_matrix(a->src[0]) &&
            fcpe_vk_bias(b->src[1], cols) && fcpe_vk_bias(c->src[1], cols) && ggml_is_contiguous(c)) return 0;
    }
    if (a->op == GGML_OP_MUL && ggml_can_fuse(g,i,{GGML_OP_MUL,GGML_OP_ADD}) &&
        b->src[0] == a && fcpe_vk_matrix(a->src[0]) &&
        fcpe_vk_bias(a->src[1],cols) && fcpe_vk_bias(b->src[1],cols)) return 1;
    if (a->op == GGML_OP_UNARY && ggml_get_unary_op(a) == GGML_UNARY_OP_SIGMOID &&
        ggml_can_fuse(g,i,{GGML_OP_UNARY,GGML_OP_MUL}) && b->src[1] == a &&
        fcpe_vk_matrix(a->src[0]) && fcpe_vk_matrix(b->src[0]) &&
        ggml_are_same_shape(a,b->src[0])) return 2;
    if (a->op == GGML_OP_ADD && b->src[0] == a &&
        fcpe_vk_matrix(a->src[0]) && fcpe_vk_bias(a->src[1],cols)) {
        if (b->op == GGML_OP_UNARY && ggml_can_fuse(g,i,{GGML_OP_ADD,GGML_OP_UNARY})) {
            if (ggml_get_unary_op(b) == GGML_UNARY_OP_SIGMOID) return 3;
            if (ggml_get_unary_op(b) == GGML_UNARY_OP_SILU) return 4;
        }
        if (b->op == GGML_OP_LEAKY_RELU && ggml_can_fuse(g,i,{GGML_OP_ADD,GGML_OP_LEAKY_RELU})) return 5;
    }
    return -1;
}
static bool fcpe_vk_dispatch_fusion(ggml_backend_vk_context * ctx, vk_context & subctx, ggml_cgraph * g, int i) {
    if (!ctx->num_additional_fused_ops) return false;
    int mode=fcpe_vk_fusion(g,i);
    if (mode < 0 || ctx->num_additional_fused_ops != (mode == 0 ? 2 : 1)) return false;
    auto * first=g->nodes[i]; auto * next=g->nodes[i+1];
    auto * out=g->nodes[i+ctx->num_additional_fused_ops];
    auto * a=first->src[0];
    auto * b=mode == 0 ? next->src[1] : mode == 2 ? next->src[0] : first->src[1];
    auto * c=mode <= 1 ? out->src[1] : b;
    const std::array<uint32_t,8> shape={uint32_t(ggml_nelements(out)),uint32_t(out->ne[0]),
        uint32_t(a->nb[1]/4),uint32_t(mode == 2 ? b->nb[1]/4 : 0),
        uint32_t(get_misalign_bytes(ctx,a)/4),uint32_t(get_misalign_bytes(ctx,b)/4),
        uint32_t(get_misalign_bytes(ctx,c)/4),uint32_t(get_misalign_bytes(ctx,out)/4)};
    struct { std::array<uint32_t,8> shape; float eps; float alpha; } pc {shape,0.0f,0.0f};
    if (mode == 0) memcpy(&pc.eps,first->op_params,4);
    if (mode == 5) memcpy(&pc.alpha,out->op_params,4);
    auto & pipeline=ctx->device->pipeline_fcpe_fused[mode];
    ggml_pipeline_request_descriptor_sets(ctx,pipeline,1);
    uint32_t groups=mode == 0 ? out->ne[1] : CEIL_DIV(ggml_nelements(out),512);
    ggml_vk_dispatch_pipeline(ctx,subctx,pipeline,
        {ggml_vk_tensor_subbuffer(ctx,a,true),ggml_vk_tensor_subbuffer(ctx,b,true),
         ggml_vk_tensor_subbuffer(ctx,c,true),ggml_vk_tensor_subbuffer(ctx,out,true)},pc,
        {std::min(groups,65535u),CEIL_DIV(groups,65535),1});
    return true;
}
