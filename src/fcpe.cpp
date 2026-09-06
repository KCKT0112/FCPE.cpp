// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.
// Portions derived from FCPE: Copyright (c) 2023 CN_ChiTu.
// Upstream MIT notice: licenses/FCPE-MIT.txt; see NOTICE.md.

#include "fcpe/fcpe.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace fcpe {
namespace {
void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}
void init_backends() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_backend_load_all(); });
}
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Gguf = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using Scheduler = std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)>;

int64_t key(gguf_context * ctx, const std::string & name, gguf_type type) {
    auto i = gguf_find_key(ctx, name.c_str());
    require(i >= 0, "Missing GGUF metadata: " + name);
    require(gguf_get_kv_type(ctx, i) == type, "Invalid GGUF metadata type: " + name);
    return i;
}
int integer(gguf_context * ctx, const std::string & name, int lo, int hi) {
    auto v = gguf_get_val_u32(ctx, key(ctx, "fcpe." + name, GGUF_TYPE_UINT32));
    require(v >= static_cast<uint32_t>(lo) && v <= static_cast<uint32_t>(hi), "Invalid fcpe." + name);
    return static_cast<int>(v);
}
float real(gguf_context * ctx, const std::string & name) {
    float v = gguf_get_val_f32(ctx, key(ctx, "fcpe." + name, GGUF_TYPE_FLOAT32));
    require(std::isfinite(v) && v > 0, "Invalid fcpe." + name);
    return v;
}
void finite(const std::vector<float> & v, const char * name) {
    require(std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); }),
            std::string(name) + " contains NaN or infinity");
}
} // namespace

std::vector<std::string> devices() {
    init_backends();
    std::vector<std::string> out;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
        out.emplace_back(ggml_backend_dev_name(ggml_backend_dev_get(i)));
    return out;
}

struct Model::Impl {
    Config cfg;
    Backend primary{nullptr, ggml_backend_free};
    Backend cpu{nullptr, ggml_backend_free};
    Context weights{nullptr, ggml_free};
    Buffer buffer{nullptr, ggml_backend_buffer_free};
    Context graph_ctx{nullptr, ggml_free};
    Scheduler scheduler{nullptr, ggml_backend_sched_free};
    std::vector<float> cents, basis, window;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    size_t cached_frames = 0;
    bool optimize_graph = true;
    int frontend_threads = 1;

    ggml_tensor * tensor(const std::string & name) const {
        auto * t = ggml_get_tensor(weights.get(), name.c_str());
        require(t != nullptr, "Missing tensor: " + name);
        return t;
    }
    void shape(const std::string & name, std::array<int64_t, 4> expected, bool f32 = false) const {
        auto * t = tensor(name);
        require(std::equal(expected.begin(), expected.end(), t->ne), "Invalid tensor shape: " + name);
        require(t->type == GGML_TYPE_F32 || (!f32 && t->type == GGML_TYPE_F16),
                "Unsupported tensor type: " + name);
    }
    std::vector<float> read_floats(const std::string & name) const {
        auto * t = tensor(name);
        require(t->type == GGML_TYPE_F32, "Expected F32: " + name);
        std::vector<float> result(static_cast<size_t>(ggml_nelements(t)));
        ggml_backend_tensor_get(t, result.data(), 0, ggml_nbytes(t));
        finite(result, name.c_str());
        return result;
    }

    Impl(const std::string & path, const Options & options) {
        optimize_graph = options.optimize_graph;
        require(options.threads >= 0 && options.threads <= 1024, "threads must be between 0 and 1024");
        ggml_context * raw = nullptr;
        Gguf meta(gguf_init_from_file(path.c_str(), {true, &raw}), gguf_free);
        weights.reset(raw);
        require(meta && weights, "Cannot read GGUF model: " + path);
        require(std::string(gguf_get_val_str(meta.get(), key(meta.get(), "general.architecture", GGUF_TYPE_STRING))) == "fcpe",
                "Expected an FCPE GGUF model");
        integer(meta.get(), "version", 1, 1);
        cfg.sample_rate = integer(meta.get(), "sample_rate", 1000, 384000);
        cfg.hop_size = integer(meta.get(), "hop_size", 1, 16384);
        cfg.n_fft = integer(meta.get(), "n_fft", 2, 16384);
        cfg.win_size = integer(meta.get(), "win_size", 2, 16384);
        cfg.mel_bins = integer(meta.get(), "mel_bins", 1, 2048);
        cfg.hidden_dims = integer(meta.get(), "hidden_dims", 4, 4096);
        cfg.layers = integer(meta.get(), "layers", 1, 64);
        cfg.bins = integer(meta.get(), "bins", 2, 4096);
        cfg.f0_min = real(meta.get(), "f0_min");
        cfg.f0_max = real(meta.get(), "f0_max");
        require(cfg.f0_max > cfg.f0_min && cfg.hidden_dims % 4 == 0 &&
                cfg.n_fft == cfg.win_size && cfg.hop_size <= cfg.win_size, "Unsupported model configuration");
        const int h = cfg.hidden_dims;
        shape("input_stack.0.weight", {3, cfg.mel_bins, h, 1});
        shape("input_stack.3.weight", {3, h, h, 1});
        for (const auto & s : {"input_stack.0.bias", "input_stack.1.weight", "input_stack.1.bias",
                              "input_stack.3.bias", "norm.weight", "norm.bias"}) shape(s, {h, 1, 1, 1}, true);
        for (int i = 0; i < cfg.layers; ++i) {
            std::string p = "block." + std::to_string(i) + ".";
            shape(p + "norm.weight", {h, 1, 1, 1}, true);
            shape(p + "norm.bias", {h, 1, 1, 1}, true);
            shape(p + "in.weight", {h, 4 * h, 1, 1});
            shape(p + "in.bias", {4 * h, 1, 1, 1}, true);
            shape(p + "dw.weight", {31, 1, 2 * h, 1}, true);
            shape(p + "dw.bias", {2 * h, 1, 1, 1}, true);
            shape(p + "out.weight", {2 * h, h, 1, 1});
            shape(p + "out.bias", {h, 1, 1, 1}, true);
        }
        shape("output_proj.weight", {h, cfg.bins, 1, 1});
        shape("output_proj.bias", {cfg.bins, 1, 1, 1}, true);
        shape("cent_table", {cfg.bins, 1, 1, 1}, true);
        shape("mel.basis", {cfg.n_fft / 2 + 1, cfg.mel_bins, 1, 1}, true);
        shape("mel.window", {cfg.win_size, 1, 1, 1}, true);
        require(gguf_get_n_tensors(meta.get()) == 13 + 8 * cfg.layers, "Unexpected tensor count");

        // Check all file ranges before allocating or reading backend buffers.
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        require(bool(file), "Cannot open " + path);
        auto file_size = static_cast<uint64_t>(file.tellg());
        auto data_offset = gguf_get_data_offset(meta.get());
        require(data_offset <= file_size, "Truncated GGUF header");
        for (int64_t i = 0; i < gguf_get_n_tensors(meta.get()); ++i) {
            auto offset = gguf_get_tensor_offset(meta.get(), i);
            auto size = gguf_get_tensor_size(meta.get(), i);
            require(offset <= file_size - data_offset && size <= file_size - data_offset - offset,
                    "Truncated GGUF tensor data");
        }
        init_backends();
        auto * dev = options.backend == "cpu" ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) :
                     options.backend == "auto" ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) :
                     ggml_backend_dev_by_name(options.backend.c_str());
        if (!dev && options.backend == "auto") dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        require(dev != nullptr, "Backend not available: " + options.backend + " (use --list-backends)");
        primary.reset(ggml_backend_dev_init(dev, nullptr));
        require(bool(primary), "Cannot initialize backend: " + options.backend);
        std::vector<ggml_backend_t> backends{primary.get()};
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
            require(bool(cpu), "CPU fallback backend unavailable");
            backends.push_back(cpu.get());
        }
        const int threads = options.threads ? options.threads :
            static_cast<int>(std::max(1u, std::min(8u, std::thread::hardware_concurrency())));
        frontend_threads = threads;
        for (auto * backend : backends) {
            auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
            auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
            if (set_threads) set_threads(backend, threads);
        }
        buffer.reset(ggml_backend_alloc_ctx_tensors(weights.get(), primary.get()));
        require(bool(buffer), "Cannot allocate model weights");
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int64_t i = 0; i < gguf_get_n_tensors(meta.get()); ++i) {
            auto * t = tensor(gguf_get_tensor_name(meta.get(), i));
            std::vector<char> bytes(ggml_nbytes(t));
            file.seekg(static_cast<std::streamoff>(data_offset + gguf_get_tensor_offset(meta.get(), i)));
            file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
            require(bool(file), "Cannot read tensor: " + std::string(t->name));
            ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
        }
        cents = read_floats("cent_table");
        require(std::adjacent_find(cents.begin(), cents.end(), std::greater_equal<float>()) == cents.end(),
                "cent_table must be strictly increasing");
        basis = read_floats("mel.basis");
        window = read_floats("mel.window");
        scheduler.reset(ggml_backend_sched_new(backends.data(), nullptr, static_cast<int>(backends.size()),
                                               8192, false, true));
        require(bool(scheduler), "Cannot create backend scheduler");
    }

    ggml_tensor * affine(ggml_context * ctx, ggml_tensor * x, const std::string & prefix) {
        return ggml_add(ctx, ggml_mul(ctx, x, tensor(prefix + ".weight")), tensor(prefix + ".bias"));
    }
    ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, const std::string & prefix) {
        auto * w = tensor(prefix + ".weight");
        // CPU F16 dot kernels also round activations to F16. Expand storage
        // weights in the graph so F16 GGUF changes only the stored weights.
        if (w->type == GGML_TYPE_F16) w = ggml_cast(ctx, w, GGML_TYPE_F32);
        auto * y = ggml_mul_mat(ctx, w, x);
        ggml_mul_mat_set_prec(y, GGML_PREC_F32);
        return ggml_add(ctx, y, tensor(prefix + ".bias"));
    }
    ggml_tensor * conv(ggml_context * ctx, ggml_tensor * x, const std::string & prefix) {
        auto * w = tensor(prefix + ".weight");
        if (w->type == GGML_TYPE_F16) w = ggml_cast(ctx, w, GGML_TYPE_F32);
        auto * xt = ggml_cont(ctx, ggml_transpose(ctx, x)); // [T, C]
        // Explicit F32 im2col: ggml_conv_1d uses F16 activations even with F32 weights.
        auto * col = ggml_im2col(ctx, w, xt, 1, 0, static_cast<int>(w->ne[0] / 2), 0, 1, 0, false, GGML_TYPE_F32);
        col = ggml_reshape_2d(ctx, col, w->ne[0] * w->ne[1], x->ne[1]);
        auto * y = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]), col);
        ggml_mul_mat_set_prec(y, GGML_PREC_F32);
        return ggml_add(ctx, y, tensor(prefix + ".bias"));
    }
    ggml_tensor * depthwise(ggml_context * ctx, ggml_tensor * x, const std::string & prefix) {
        auto * w = tensor(prefix + ".weight");
        if (optimize_graph) {
#ifdef FCPE_METAL_OPTIMIZED
            if (std::string(ggml_backend_name(primary.get())).find("MTL") == 0 && !std::getenv("FCPE_METAL_LEGACY_DW_LAYOUT")) {
                auto * kernel = ggml_reshape_4d(ctx, w, 31, 1, 1, x->ne[0]);
                auto * view = ggml_permute(ctx, ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1], 1, 1), 2, 0, 1, 3);
                auto * y = ggml_conv_2d_dw_direct(ctx, kernel, view, 1, 1, 15, 0, 1, 1);
                // Preserve the logical WHCN shape with channels contiguous in memory.
                y->nb[0] = sizeof(float) * x->ne[0];
                y->nb[1] = y->nb[0] * x->ne[1];
                y->nb[2] = sizeof(float);
                y->nb[3] = y->nb[1];
                y = ggml_reshape_2d(ctx, ggml_permute(ctx, y, 1, 2, 0, 3), x->ne[0], x->ne[1]);
                return ggml_add(ctx, y, tensor(prefix + ".bias"));
            }
#endif
            // Direct convolution avoids a 31x im2col expansion and 1024 tiny
            // per-channel GEMMs. WHCN also supports singleton time dimensions.
            auto * kernel = ggml_reshape_4d(ctx, w, 31, 1, 1, x->ne[0]);
            auto * xt = ggml_cont(ctx, ggml_transpose(ctx, x));
            auto * input_whc = ggml_reshape_3d(ctx, xt, x->ne[1], 1, x->ne[0]);
            auto * y = ggml_conv_2d_dw_direct(ctx, kernel, input_whc, 1, 1, 15, 0, 1, 1);
            if (ggml_backend_supports_op(primary.get(), y)) {
                y = ggml_reshape_2d(ctx, y, x->ne[1], x->ne[0]);
                y = ggml_cont(ctx, ggml_transpose(ctx, y));
                return ggml_add(ctx, y, tensor(prefix + ".bias"));
            }
        }
        auto * xt = ggml_cont(ctx, ggml_transpose(ctx, x));
        xt = ggml_reshape_3d(ctx, xt, x->ne[1], 1, x->ne[0]); // channels as batch
        auto * col = ggml_im2col(ctx, w, xt, 1, 0, 15, 0, 1, 0, false, GGML_TYPE_F32);
        auto * y = ggml_mul_mat(ctx, w, col); // [1, T, C]
        ggml_mul_mat_set_prec(y, GGML_PREC_F32);
        y = ggml_reshape_2d(ctx, y, x->ne[1], x->ne[0]);
        y = ggml_cont(ctx, ggml_transpose(ctx, y));
        return ggml_add(ctx, y, tensor(prefix + ".bias"));
    }
    void build(size_t frames) {
        if (frames == cached_frames) return;
        cached_frames = 0;
        ggml_backend_sched_reset(scheduler.get());
        graph_ctx.reset(ggml_init({8192 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8192, false), nullptr, true}));
        require(bool(graph_ctx), "Cannot allocate graph metadata");
        auto * ctx = graph_ctx.get();
        const auto t = static_cast<int64_t>(frames);
        const auto h = cfg.hidden_dims;
        graph = ggml_new_graph_custom(ctx, 8192, false);
        input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.mel_bins, t);
        ggml_set_name(input, "mel_input");
        ggml_set_input(input);
        auto * x = conv(ctx, input, "input_stack.0");
        x = ggml_cont(ctx, ggml_transpose(ctx, x));
        x = ggml_reshape_3d(ctx, x, t, 1, h);
        x = ggml_group_norm(ctx, x, 4, 1e-5f); // normalize across time AND channels per group
        x = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, x, t, h)));
        x = affine(ctx, x, "input_stack.1");
        x = ggml_leaky_relu(ctx, x, 0.01f, false);
        x = conv(ctx, x, "input_stack.3");
        for (int i = 0; i < cfg.layers; ++i) {
            std::string p = "block." + std::to_string(i);
            auto * y = affine(ctx, ggml_norm(ctx, x, 1e-5f), p + ".norm");
            y = linear(ctx, y, p + ".in");
            // PyTorch GLU: first half * sigmoid(second half), split on channels.
            auto * a = ggml_view_2d(ctx, y, 2 * h, t, y->nb[1], 0);
            auto * b = ggml_view_2d(ctx, y, 2 * h, t, y->nb[1], 2 * h * sizeof(float));
            const std::string backend = ggml_backend_name(primary.get());
            bool strided_glu = backend.find("Vulkan") == 0 && !std::getenv("FCPE_VK_LEGACY_GRAPH");
#ifdef FCPE_METAL_OPTIMIZED
            strided_glu |= backend.find("MTL") == 0 && !std::getenv("FCPE_METAL_LEGACY_GRAPH");
#endif
            if (!optimize_graph || !strided_glu) {
                a = ggml_cont(ctx, a);
                b = ggml_cont(ctx, b);
            }
            y = ggml_mul(ctx, a, ggml_sigmoid(ctx, b));
            y = ggml_silu(ctx, depthwise(ctx, y, p + ".dw"));
            x = ggml_add(ctx, x, linear(ctx, y, p + ".out"));
        }
        x = affine(ctx, ggml_norm(ctx, x, 1e-5f), "norm");
        output = ggml_sigmoid(ctx, linear(ctx, x, "output_proj"));
        ggml_set_name(output, "probabilities");
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        require(ggml_backend_sched_alloc_graph(scheduler.get(), graph), "Cannot allocate inference graph; try a shorter recording");
        cached_frames = frames;
    }
};

Model::Model(const std::string & path, const Options & options) : impl_(new Impl(path, options)) {}
Model::~Model() = default;
Model::Model(Model &&) noexcept = default;
Model & Model::operator=(Model &&) noexcept = default;
const Config & Model::config() const { return impl_->cfg; }
std::string Model::backend_name() const { return ggml_backend_name(impl_->primary.get()); }
ExecutionInfo Model::execution_info() const {
    ExecutionInfo info;
    if (!impl_->graph) return info;
    info.graph_nodes = ggml_graph_n_nodes(impl_->graph);
    info.graph_splits = ggml_backend_sched_get_n_splits(impl_->scheduler.get());
    for (int i = 0; i < info.graph_nodes; ++i) {
        auto * node = ggml_graph_node(impl_->graph, i);
        if (node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
            node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) continue;
        auto * backend = ggml_backend_sched_get_tensor_backend(impl_->scheduler.get(), node);
        if (!backend) continue;
        if (ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU)
            ++info.cpu_compute_nodes;
        else ++info.accelerator_compute_nodes;
    }
    for (int i = 0; i < ggml_backend_sched_get_n_backends(impl_->scheduler.get()); ++i)
        info.compute_buffer_bytes += ggml_backend_sched_get_buffer_size(impl_->scheduler.get(),
            ggml_backend_sched_get_backend(impl_->scheduler.get(), i));
    return info;
}
std::vector<float> Model::probabilities(const std::vector<float> & mel) {
    require(!mel.empty() && mel.size() % config().mel_bins == 0, "Expected nonempty [frames, mel_bins] input");
    finite(mel, "Mel input");
    auto frames = mel.size() / config().mel_bins;
    require(frames <= 1000000, "Input exceeds one million frames");
    impl_->build(frames);
    ggml_backend_tensor_set(impl_->input, mel.data(), 0, mel.size() * sizeof(float));
    require(ggml_backend_sched_graph_compute(impl_->scheduler.get(), impl_->graph) == GGML_STATUS_SUCCESS,
            "ggml inference failed");
    std::vector<float> result(frames * config().bins);
    ggml_backend_tensor_get(impl_->output, result.data(), 0, result.size() * sizeof(float));
    finite(result, "Model output");
    return result;
}
Result Model::decode_probabilities(const std::vector<float> & p, const DecodeOptions & options) const {
    return decode(p, impl_->cents, options, config().f0_min);
}
Result Model::infer_mel(const std::vector<float> & mel, const DecodeOptions & options) {
    return decode_probabilities(probabilities(mel), options);
}
Result Model::infer(const std::vector<float> & audio, int sample_rate, const DecodeOptions & options) {
    return infer_mel(mel(audio, sample_rate), options);
}

// Defined in audio.cpp; keep FFT implementation private to the library.
std::vector<float> compute_mel(const std::vector<float> &, const Config &,
                               const std::vector<float> &, const std::vector<float> &, int);
std::vector<float> Model::mel(const std::vector<float> & audio, int sample_rate) const {
    require(!audio.empty(), "Audio is empty");
    finite(audio, "Audio");
    return compute_mel(resample(audio, sample_rate, config().sample_rate), config(), impl_->basis, impl_->window, impl_->frontend_threads);
}

Result decode(const std::vector<float> & p, const std::vector<float> & cents,
              const DecodeOptions & options, float model_f0_min) {
    require(cents.size() >= 2 && !p.empty() && p.size() % cents.size() == 0, "Invalid decoder input shape");
    require(std::isfinite(options.threshold) && options.threshold >= 0 && options.threshold <= 1,
            "threshold must be between 0 and 1");
    require(std::isfinite(options.f0_min) && options.f0_min >= 0 &&
            std::isfinite(options.f0_max) && options.f0_max >= 0 &&
            std::isfinite(model_f0_min) && model_f0_min > 0, "Invalid F0 range");
    require(options.decoder == Decoder::local_argmax || options.decoder == Decoder::argmax, "Unknown decoder");
    finite(cents, "Cent table");
    require(std::all_of(p.begin(), p.end(), [](float x) { return std::isfinite(x) && x >= 0 && x <= 1; }),
            "Probabilities must be finite and in [0, 1]");
    const size_t bins = cents.size(), frames = p.size() / bins;
    Result out;
    out.f0.resize(frames);
    out.confidence.resize(frames);
    out.unvoiced.resize(frames);
    for (size_t t = 0; t < frames; ++t) {
        const auto * row = p.data() + t * bins;
        auto peak = std::max_element(row, row + bins) - row;
        out.confidence[t] = row[peak];
        float sum = 0, weighted = 0;
        if (options.decoder == Decoder::local_argmax) {
            for (int d = -4; d <= 4; ++d) {
                // The original gathers nine CLAMPED indices, including duplicate edge bins.
                auto j = static_cast<size_t>(std::clamp<int64_t>(peak + d, 0, static_cast<int64_t>(bins) - 1));
                sum += row[j]; weighted += row[j] * cents[j];
            }
        } else {
            for (size_t j = 0; j < bins; ++j) { sum += row[j]; weighted += row[j] * cents[j]; }
        }
        float f0 = row[peak] > options.threshold && sum > 0 ? 10.0f * std::exp2((weighted / sum) / 1200.0f) : 0.0f;
        out.f0[t] = f0;
        out.unvoiced[t] = f0 < (options.f0_min > 0 ? options.f0_min : model_f0_min) ? 1.0f : 0.0f;
    }
    if (options.interpolate_unvoiced) {
        size_t first = 0;
        while (first < frames && out.unvoiced[first] != 0) ++first;
        if (first < frames) {
            std::fill(out.f0.begin(), out.f0.begin() + first, out.f0[first]);
            size_t left = first;
            for (size_t right = first + 1; right < frames; ++right) {
                if (out.unvoiced[right] != 0) continue;
                for (size_t j = left + 1; j < right; ++j) {
                    float frac = static_cast<float>(j - left) / static_cast<float>(right - left);
                    out.f0[j] = out.f0[left] + frac * (out.f0[right] - out.f0[left]);
                }
                left = right;
            }
            std::fill(out.f0.begin() + left + 1, out.f0.end(), out.f0[left]);
        }
    }
    if (options.f0_max > 0) for (auto & v : out.f0) v = std::min(v, options.f0_max);
    if (options.output_frames && options.output_frames != frames) {
        require(options.output_frames <= 100000000, "Too many requested output frames");
        for (auto * values : {&out.f0, &out.confidence, &out.unvoiced}) {
            std::vector<float> resized(options.output_frames);
            for (size_t i = 0; i < resized.size(); ++i)
                resized[i] = (*values)[static_cast<size_t>((static_cast<uint64_t>(i) * frames) / resized.size())];
            *values = std::move(resized);
        }
    }
    return out;
}
} // namespace fcpe
