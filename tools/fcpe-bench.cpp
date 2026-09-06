// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.

#include "fcpe/fcpe.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

using Clock = std::chrono::steady_clock;
double ms(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
std::vector<float> read_mel(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() <= 0 || f.tellg() % 4) throw std::runtime_error("Invalid mel file");
    std::vector<float> v(static_cast<size_t>(f.tellg()) / 4);
    f.seekg(0); f.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(v.size() * 4));
    if (!f) throw std::runtime_error("Cannot read mel file");
    return v;
}
int main(int argc, char ** argv) {
    try {
        std::string model_path, mel_path, output_path, dump_path, wav_path;
        int repeats = 20, warmup = 3;
        fcpe::Options opts;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto value = [&]() -> std::string { if (++i >= argc) throw std::runtime_error("Missing argument"); return argv[i]; };
            if (arg == "--model") model_path = value();
            else if (arg == "--mel") mel_path = value();
            else if (arg == "--wav") wav_path = value();
            else if (arg == "--output") output_path = value();
            else if (arg == "--dump") dump_path = value();
            else if (arg == "--backend") opts.backend = value();
            else if (arg == "--threads") opts.threads = std::stoi(value());
            else if (arg == "--repeats") repeats = std::stoi(value());
            else if (arg == "--warmup") warmup = std::stoi(value());
            else if (arg == "--reference-graph") opts.optimize_graph = false;
            else throw std::runtime_error("Unknown argument: " + arg);
        }
        if (model_path.empty() || mel_path.empty() || output_path.empty() || repeats < 1 || repeats > 10000 || warmup < 0)
            throw std::runtime_error("Usage: fcpe-bench --model MODEL --mel INPUT --output JSON [--backend cpu] [--repeats 20] [--warmup 3] [--reference-graph] [--dump probabilities.f32] [--wav audio.wav]");
        auto mel = read_mel(mel_path);
        auto start = Clock::now();
        fcpe::Model model(model_path, opts);
        double load_ms = ms(start);
        start = Clock::now(); auto output = model.probabilities(mel); double first_ms = ms(start);
        for (int i = 0; i < warmup; ++i) output = model.probabilities(mel);
        std::vector<double> samples;
        for (int i = 0; i < repeats; ++i) {
            start = Clock::now(); output = model.probabilities(mel); samples.push_back(ms(start));
        }
        double frontend_ms = 0, decode_ms = 0;
        std::vector<double> end_to_end_samples;
        if (!wav_path.empty()) {
            const auto audio = fcpe::read_wav(wav_path);
            for (int i = 0; i < repeats; ++i) {
                start = Clock::now(); auto m = model.mel(audio.samples, audio.sample_rate); frontend_ms += ms(start);
                if (m.empty()) throw std::runtime_error("Empty mel");
            }
            frontend_ms /= repeats;
            for (int i = 0; i < warmup; ++i) model.infer(audio.samples, audio.sample_rate);
            for (int i = 0; i < repeats; ++i) {
                start = Clock::now(); auto result = model.infer(audio.samples, audio.sample_rate);
                end_to_end_samples.push_back(ms(start));
                if (result.f0.empty()) throw std::runtime_error("Empty end-to-end result");
            }
        }
        for (int i = 0; i < repeats; ++i) {
            start = Clock::now(); auto f0 = model.decode_probabilities(output); decode_ms += ms(start);
            if (f0.f0.empty()) throw std::runtime_error("Empty output");
        }
        decode_ms /= repeats;
        if (!dump_path.empty()) {
            std::ofstream f(dump_path, std::ios::binary);
            f.write(reinterpret_cast<const char *>(output.data()), static_cast<std::streamsize>(output.size() * 4));
            if (!f) throw std::runtime_error("Cannot write output tensor");
        }
        auto sorted = samples; std::sort(sorted.begin(), sorted.end());
        auto percentile = [&](double fraction) {
            const double pos = fraction * (sorted.size() - 1);
            const size_t lo = static_cast<size_t>(pos), hi = static_cast<size_t>(std::ceil(pos));
            return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
        };
        double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / repeats;
        const auto info = model.execution_info();
        std::ofstream f(output_path);
        f << std::setprecision(10) << "{\n  \"backend\": \"" << model.backend_name() << "\",\n"
          << "  \"optimized\": " << (opts.optimize_graph ? "true" : "false") << ",\n"
          << "  \"frames\": " << mel.size() / model.config().mel_bins << ",\n"
          << "  \"threads\": " << opts.threads << ",\n  \"warmup\": " << warmup << ",\n"
          << "  \"load_ms\": " << load_ms << ",\n  \"first_ms\": " << first_ms << ",\n"
          << "  \"mean_ms\": " << mean << ",\n  \"median_ms\": " << percentile(0.5) << ",\n"
          << "  \"p95_ms\": " << percentile(0.95) << ",\n"
          << "  \"frontend_mean_ms\": " << frontend_ms << ",\n  \"decode_mean_ms\": " << decode_ms << ",\n"
          << "  \"graph_nodes\": " << info.graph_nodes << ",\n  \"graph_splits\": " << info.graph_splits << ",\n"
          << "  \"cpu_compute_nodes\": " << info.cpu_compute_nodes << ",\n"
          << "  \"accelerator_compute_nodes\": " << info.accelerator_compute_nodes << ",\n"
          << "  \"compute_buffer_bytes\": " << info.compute_buffer_bytes << ",\n  \"samples_ms\": [";
        for (size_t i = 0; i < samples.size(); ++i) f << (i ? ", " : "") << samples[i];
        f << "],\n  \"end_to_end_samples_ms\": [";
        for (size_t i = 0; i < end_to_end_samples.size(); ++i) f << (i ? ", " : "") << end_to_end_samples[i];
        f << "]\n}\n"; f.close();
        if (!f) throw std::runtime_error("Cannot write benchmark JSON");
        std::cout << model.backend_name() << " mean=" << mean << " ms first=" << first_ms
                  << " ms nodes=" << info.graph_nodes << " splits=" << info.graph_splits
                  << " CPU=" << info.cpu_compute_nodes << " GPU=" << info.accelerator_compute_nodes
                  << " compute_MiB=" << info.compute_buffer_bytes / 1048576.0 << '\n';
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
