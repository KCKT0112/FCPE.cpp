// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.

    if (getenv("FCPE_METAL_PROFILE")) {
        ggml_metal_synchronize(ctx);
        ggml_metal_device_rsets_keep_alive(ctx->dev);
        if (ctx->has_error) return GGML_STATUS_FAILED;
        GGML_LOG_INFO("FCPE_METAL_PROFILE_BEGIN nodes=%d\n", gf->n_nodes);
        for (int i = 0; i < gf->n_nodes; ++i) {
            const struct ggml_tensor * node = gf->nodes[i];
            if (ggml_op_is_empty(node->op) || ggml_is_empty(node) ||
                !(node->flags & GGML_TENSOR_FLAG_COMPUTE)) continue;
            @autoreleasepool {
                id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
                id<MTLCommandBuffer> command = [queue commandBuffer];
                ggml_metal_op_t op = ggml_metal_op_init(ctx->dev, command, gf,
                    i, i + 1, false, false, false, 0, 0);
                const int encoded = ggml_metal_op_encode(op, 0);
                ggml_metal_op_free(op);
                [command commit];
                [command waitUntilCompleted];
                if (!encoded || [command status] != MTLCommandBufferStatusCompleted) {
                    ctx->has_error = true;
                    GGML_LOG_ERROR("FCPE Metal profile compute failed\n");
                    return GGML_STATUS_FAILED;
                }
                GGML_LOG_INFO("FCPE_METAL_PROFILE index=%d op=%s shape=%lld,%lld,%lld,%lld gpu_ms=%.6f name=%s\n",
                    i, ggml_op_desc(node), (long long)node->ne[0], (long long)node->ne[1],
                    (long long)node->ne[2], (long long)node->ne[3],
                    1000.0 * (command.GPUEndTime - command.GPUStartTime), node->name);
            }
        }
        GGML_LOG_INFO("FCPE_METAL_PROFILE_END\n");
        return GGML_STATUS_SUCCESS;
    }
