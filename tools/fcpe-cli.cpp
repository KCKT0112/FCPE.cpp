// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.

#include "fcpe/fcpe.h"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void usage() {
    std::cout << "FCPE.cpp - native ggml pitch estimation\n"
        "Usage: fcpe-cli -m model.gguf -i input.wav -o pitch.csv [options]\n"
        "       fcpe-cli -m model.gguf --mel input.f32 -o pitch.csv\n"
        "  -m, --model PATH       Converted FCPE GGUF\n"
        "  -i, --input PATH       PCM / float WAV (channels averaged to mono)\n"
        "  -o, --output PATH      CSV: time,f0,confidence,unvoiced (default: stdout)\n"
        "  --mel PATH             Raw little-endian float32 [frames, mel_bins]\n"
        "  -t, --threads N        CPU threads (0 = automatic, default)\n"
        "  --backend NAME         cpu (default), auto, or an enumerated device\n"
        "  --list-backends        List available ggml devices\n"
        "  --threshold VALUE      Confidence threshold (default: 0.006)\n"
        "  --decoder NAME         local_argmax (default) or argmax\n"
        "  --interp-uv            Interpolate unvoiced F0; preserve UV flags\n"
        "  --f0-min HZ            Minimum voiced pitch, defaults to model minimum\n"
        "  --f0-max HZ            Clamp output F0 (0 = disabled)\n"
        "  --output-frames N      Nearest-neighbor output interpolation\n"
        "  --dump-prefix PATH     Write .mel.f32, .probabilities.f32, .f0.f32\n"
        "  --reference-graph      Use im2col depthwise graph for comparison\n"
        "  -h, --help             Show help\n";
}
std::vector<float> read_raw(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() <= 0 || f.tellg() % 4) throw std::runtime_error("Invalid raw float32 file: " + path);
    std::vector<float> result(static_cast<size_t>(f.tellg()) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(result.size() * 4));
    if (!f) throw std::runtime_error("Cannot read: " + path);
    return result;
}
void dump(const std::string & path, const std::vector<float> & v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(v.data()), static_cast<std::streamsize>(v.size() * 4));
    if (!f) throw std::runtime_error("Cannot write: " + path);
}
int parse_int(const std::string & s) {
    size_t end; int v = std::stoi(s, &end);
    if (end != s.size() || v < 0) throw std::runtime_error("Invalid nonnegative integer: " + s);
    return v;
}
float parse_float(const std::string & s) {
    size_t end; float v = std::stof(s, &end);
    if (end != s.size() || !std::isfinite(v)) throw std::runtime_error("Invalid number: " + s);
    return v;
}
}
int main(int argc, char ** argv) {
    try {
        std::string model_path, input_path, output_path, mel_path, dump_prefix;
        fcpe::Options options;
        fcpe::DecodeOptions decode;
        bool list = false;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto value = [&]() -> std::string {
                if (++i >= argc) throw std::runtime_error("Missing value for " + arg);
                return argv[i];
            };
            if (arg == "--help" || arg == "-h") { usage(); return 0; }
            else if (arg == "--model" || arg == "-m") model_path = value();
            else if (arg == "--input" || arg == "-i") input_path = value();
            else if (arg == "--output" || arg == "-o") output_path = value();
            else if (arg == "--threads" || arg == "-t") options.threads = parse_int(value());
            else if (arg == "--backend") options.backend = value();
            else if (arg == "--reference-graph") options.optimize_graph = false;
            else if (arg == "--list-backends") list = true;
            else if (arg == "--mel") mel_path = value();
            else if (arg == "--threshold") decode.threshold = parse_float(value());
            else if (arg == "--interp-uv") decode.interpolate_unvoiced = true;
            else if (arg == "--f0-min") decode.f0_min = parse_float(value());
            else if (arg == "--f0-max") decode.f0_max = parse_float(value());
            else if (arg == "--output-frames") decode.output_frames = static_cast<size_t>(parse_int(value()));
            else if (arg == "--dump-prefix") dump_prefix = value();
            else if (arg == "--decoder") {
                auto name = value();
                if (name == "local_argmax") decode.decoder = fcpe::Decoder::local_argmax;
                else if (name == "argmax") decode.decoder = fcpe::Decoder::argmax;
                else throw std::runtime_error("Unknown decoder: " + name);
            } else throw std::runtime_error("Unknown option: " + arg);
        }
        if (list) { for (const auto & device : fcpe::devices()) std::cout << device << '\n'; return 0; }
        if (model_path.empty() || input_path.empty() == mel_path.empty()) {
            usage(); throw std::runtime_error("Specify a model and exactly one of --input / --mel");
        }
        auto protect_input = [&](const std::string & output) {
            if (output.empty()) return;
            auto normalized = [](const std::string & path) {
                std::error_code error;
                auto canonical = std::filesystem::weakly_canonical(path, error);
                return error ? std::filesystem::absolute(path).lexically_normal() : canonical;
            };
            for (const auto & source : {model_path, input_path, mel_path}) {
                if (source.empty()) continue;
                std::error_code error;
                const bool same_file = std::filesystem::equivalent(output, source, error);
                if (same_file || normalized(output) == normalized(source))
                    throw std::runtime_error("Output path must differ from the input and model");
            }
        };
        protect_input(output_path);
        if (!dump_prefix.empty()) {
            for (const auto & suffix : {".mel.f32", ".probabilities.f32", ".f0.f32"})
                protect_input(dump_prefix + suffix);
        }
        auto start = std::chrono::steady_clock::now();
        fcpe::Model model(model_path, options);
        auto loaded = std::chrono::steady_clock::now();
        std::vector<float> mel;
        if (!mel_path.empty()) mel = read_raw(mel_path);
        else {
            auto audio = fcpe::read_wav(input_path);
            mel = model.mel(audio.samples, audio.sample_rate);
        }
        auto preprocessed = std::chrono::steady_clock::now();
        auto probabilities = model.probabilities(mel);
        auto result = model.decode_probabilities(probabilities, decode);
        auto end = std::chrono::steady_clock::now();
        if (!dump_prefix.empty()) {
            dump(dump_prefix + ".mel.f32", mel);
            dump(dump_prefix + ".probabilities.f32", probabilities);
            dump(dump_prefix + ".f0.f32", result.f0);
        }
        std::ofstream file;
        std::ostream * out = &std::cout;
        if (!output_path.empty()) {
            file.open(output_path);
            if (!file) throw std::runtime_error("Cannot open output CSV: " + output_path);
            out = &file;
        }
        const double hop = double(model.config().hop_size) / model.config().sample_rate;
        const auto native_frames = mel.size() / model.config().mel_bins;
        *out << "time,f0,confidence,unvoiced\n" << std::setprecision(9);
        for (size_t i = 0; i < result.f0.size(); ++i) {
            const double time = i * hop * double(native_frames) / result.f0.size();
            *out << time << ',' << result.f0[i] << ',' << result.confidence[i] << ',' << result.unvoiced[i] << '\n';
        }
        out->flush();
        if (!*out) throw std::runtime_error("Failed to write CSV output");
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        std::cerr << "backend=" << model.backend_name() << " frames=" << native_frames
                  << " load_ms=" << ms(start, loaded) << " mel_ms=" << ms(loaded, preprocessed)
                  << " inference_ms=" << ms(preprocessed, end) << '\n';
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "fcpe: " << e.what() << '\n';
        return 1;
    }
}
