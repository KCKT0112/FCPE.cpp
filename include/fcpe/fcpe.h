// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace fcpe {

struct Config {
    int sample_rate = 16000;
    int hop_size = 160;
    int n_fft = 1024;
    int win_size = 1024;
    int mel_bins = 128;
    int hidden_dims = 512;
    int layers = 6;
    int bins = 360;
    float f0_min = 32.7f;
    float f0_max = 1975.5f;
};

enum class Decoder { local_argmax, argmax };
struct Options {
    std::string backend = "cpu"; // "auto", "cpu", or a name from devices()
    int threads = 0; // 0: min(8, hardware concurrency)
    bool optimize_graph = true; // false: retain the original im2col graph for A/B measurements
};
struct ExecutionInfo {
    int graph_nodes = 0;
    int graph_splits = 0;
    int cpu_compute_nodes = 0;
    int accelerator_compute_nodes = 0;
    std::size_t compute_buffer_bytes = 0;
};
struct DecodeOptions {
    Decoder decoder = Decoder::local_argmax;
    float threshold = 0.006f;
    bool interpolate_unvoiced = false;
    float f0_min = 0.0f; // 0: model minimum (used for UV classification)
    float f0_max = 0.0f; // 0: no post-processing clamp
    std::size_t output_frames = 0; // 0: native length; otherwise nearest interpolation
};
struct Result {
    std::vector<float> f0;
    std::vector<float> confidence;
    std::vector<float> unvoiced; // 1 = unvoiced, before optional interpolation
};
struct Audio {
    std::vector<float> samples; // mono, normalized floating point
    int sample_rate = 0;
};

std::vector<std::string> devices();
Audio read_wav(const std::string & path); // PCM 8/16/24/32 and IEEE float 32/64
std::vector<float> resample(const std::vector<float> & audio, int from, int to);
Result decode(const std::vector<float> & probabilities, const std::vector<float> & cents,
              const DecodeOptions & options = {}, float model_f0_min = 32.7f);

// One model per concurrent caller. Repeated calls may have different lengths.
// All matrices are contiguous row-major [frames, channels]. Throws std::exception on errors.
class Model {
public:
    explicit Model(const std::string & gguf_path, const Options & options = {});
    ~Model();
    Model(Model &&) noexcept;
    Model & operator=(Model &&) noexcept;
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;

    const Config & config() const;
    std::string backend_name() const;
    ExecutionInfo execution_info() const; // valid after the first probabilities()/infer() call
    std::vector<float> mel(const std::vector<float> & audio, int sample_rate) const;
    std::vector<float> probabilities(const std::vector<float> & mel);
    Result infer_mel(const std::vector<float> & mel, const DecodeOptions & options = {});
    Result infer(const std::vector<float> & audio, int sample_rate,
                 const DecodeOptions & options = {});
    Result decode_probabilities(const std::vector<float> & probabilities,
                                const DecodeOptions & options = {}) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace fcpe
