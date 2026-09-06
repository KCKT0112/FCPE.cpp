// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Independent scalar oracle for the accelerator F32 specializations. Odd dimensions
// exercise all tile tails; batches exercise fallback, and an explicit CONT
// checks that a transposed input can feed the specialization without F16 loss.
int main(int argc, char ** argv) {
    try {
        ggml_backend_load_all();
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            ggml_backend_init_by_name(argc > 1 ? argv[1] : "CPU", nullptr), ggml_backend_free);
        if (!backend) throw std::runtime_error("Requested backend unavailable");
        std::mt19937 rng(20260906);
        std::uniform_real_distribution<float> random(-0.5f, 0.5f);
        // M, N, K, batches, transpose A
        const std::vector<std::array<int, 5>> shapes = {
            {1,1,31,1,0}, {17,11,15,1,0}, {63,63,31,1,0}, {64,64,64,1,0},
            {65,65,65,1,0}, {360,101,512,1,0}, {512,11,384,1,0},
            {512,65,1536,1,0}, {2048,65,512,1,0}, {512,65,1024,1,0},
            {17,65,31,3,0}, {17,65,31,1,1},
            {64,1,64,1,0}, {64,2,128,1,0}, {64,8,128,1,0}, {64,9,128,1,0},
            {65,33,65,3,0}, {65,33,65,1,1}, {512,129,512,1,0}};
        float worst = 0;
        for (const auto & s : shapes) {
            const int m=s[0], n=s[1], k=s[2], batches=s[3];
            std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
                ggml_init({ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true}), ggml_free);
            if (!ctx) throw std::runtime_error("Context allocation failed");
            auto * a_storage = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, s[4] ? m : k, s[4] ? k : m, batches);
            auto * a = s[4] ? ggml_cont(ctx.get(), ggml_transpose(ctx.get(), a_storage)) : a_storage;
            // Explicit CONT preserves full F32 for the noncontiguous input case:
            // upstream Vulkan's direct noncontiguous matmul path converts to F16.
            auto * b = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, n, batches);
            auto * d = ggml_mul_mat(ctx.get(), a, b);
            ggml_mul_mat_set_prec(d, GGML_PREC_F32);
            auto * graph = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(graph, d);
            std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
                ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
            if (!buffer) throw std::runtime_error("Tensor allocation failed");
            std::vector<float> av(m*k*batches), bv(n*k*batches), output(m*n*batches);
            for (auto & v : av) v = random(rng);
            for (auto & v : bv) v = random(rng);
            ggml_backend_tensor_set(a_storage, av.data(), 0, av.size()*sizeof(float));
            ggml_backend_tensor_set(b, bv.data(), 0, bv.size()*sizeof(float));
            if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("Backend compute failed");
            ggml_backend_tensor_get(d, output.data(), 0, output.size()*sizeof(float));
            double squared_error = 0, half_squared_error = 0;
            float shape_worst = 0;
            bool mismatch = false;
            std::vector<float> ah(av.size()), bh(bv.size());
            std::transform(av.begin(), av.end(), ah.begin(), [](float v) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(v)); });
            std::transform(bv.begin(), bv.end(), bh.begin(), [](float v) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(v)); });
            for (int z=0; z<batches; ++z) for (int j=0; j<n; ++j) for (int i=0; i<m; ++i) {
                double reference = 0;
                double half_reference = 0;
                for (int q=0; q<k; ++q) {
                    const auto ai = z*m*k + (s[4] ? q*m+i : i*k+q), bi = z*n*k+j*k+q;
                    reference += double(av[ai]) * bv[bi];
                    half_reference += double(ah[ai]) * bh[bi];
                }
                const float got = output[z*m*n+j*m+i];
                const float error = static_cast<float>(std::abs(got-reference));
                mismatch |= !std::isfinite(got) || error > 1e-4 + 5e-6*std::abs(reference);
                squared_error += (got-reference)*(got-reference);
                half_squared_error += (got-half_reference)*(got-half_reference);
                shape_worst = std::max(shape_worst, error);
                worst = std::max(worst, error);
            }
            if (mismatch) {
                std::ostringstream message;
                message << "Matmul mismatch at M=" << m << ", N=" << n << ", K=" << k
                        << ": max_abs=" << shape_worst << " rmse=" << std::sqrt(squared_error/output.size())
                        << " rmse_vs_half_operands=" << std::sqrt(half_squared_error/output.size());
                throw std::runtime_error(message.str());
            }
        }
        std::cout << "PASS: " << shapes.size() << " matmul shapes, " << ggml_backend_name(backend.get())
                  << ", max error vs double oracle=" << worst << '\n';
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
