// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.
// Portions derived from FCPE: Copyright (c) 2023 CN_ChiTu.
// Upstream MIT notice: licenses/FCPE-MIT.txt; see NOTICE.md.

#include "fcpe/fcpe.h"
#define POCKETFFT_NO_MULTITHREADING
#define POCKETFFT_CACHE_SIZE 16
#include "pocketfft_hdronly.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <future>
#include <numeric>
#include <stdexcept>

namespace fcpe {
namespace {
constexpr double pi = 3.14159265358979323846;
uint16_t u16(const unsigned char * p) { return uint16_t(p[0]) | (uint16_t(p[1]) << 8); }
uint32_t u32(const unsigned char * p) { return uint32_t(u16(p)) | (uint32_t(u16(p + 2)) << 16); }
void check(bool valid, const char * message) { if (!valid) throw std::runtime_error(message); }
}

Audio read_wav(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    check(bool(f), "Cannot open WAV file");
    const auto file_size = f.tellg();
    check(file_size >= 12, "Truncated WAV header");
    f.seekg(0);
    unsigned char header[12];
    f.read(reinterpret_cast<char *>(header), 12);
    check(std::memcmp(header, "RIFF", 4) == 0 && std::memcmp(header + 8, "WAVE", 4) == 0,
          "Expected a little-endian RIFF WAVE file");
    uint16_t format = 0, channels = 0, bits = 0, align = 0;
    uint32_t sr = 0;
    std::vector<unsigned char> data;
    bool have_fmt = false, have_data = false;
    const auto riff_end = static_cast<uint64_t>(u32(header + 4)) + 8;
    check(riff_end >= 12 && riff_end <= static_cast<uint64_t>(file_size), "Truncated RIFF container");
    while (static_cast<uint64_t>(f.tellg()) + 8 <= riff_end) {
        unsigned char chunk[8];
        f.read(reinterpret_cast<char *>(chunk), 8);
        auto size = u32(chunk + 4);
        auto position = static_cast<uint64_t>(f.tellg());
        check(position + size <= riff_end, "Truncated WAV chunk");
        if (std::memcmp(chunk, "fmt ", 4) == 0) {
            check(size >= 16 && size <= 65536, "Invalid WAV format chunk");
            std::vector<unsigned char> fmt(size);
            f.read(reinterpret_cast<char *>(fmt.data()), size);
            format = u16(fmt.data()); channels = u16(fmt.data() + 2); sr = u32(fmt.data() + 4);
            align = u16(fmt.data() + 12); bits = u16(fmt.data() + 14);
            if (format == 0xfffe) {
                check(size >= 40 && u16(fmt.data() + 16) >= 22, "Invalid extensible WAV format");
                static const unsigned char tail[14] = {0,0,0,0,16,0,128,0,0,170,0,56,155,113};
                check(std::memcmp(fmt.data() + 26, tail, 14) == 0, "Unsupported WAV subformat GUID");
                format = u16(fmt.data() + 24);
            }
            have_fmt = true;
        } else if (std::memcmp(chunk, "data", 4) == 0 && !have_data) {
            data.resize(size);
            f.read(reinterpret_cast<char *>(data.data()), size);
            have_data = true;
        }
        f.seekg(static_cast<std::streamoff>(position + size + (size & 1)));
    }
    check(have_fmt && have_data && !data.empty(), "WAV has no format or audio data");
    check(channels > 0 && channels <= 256 && sr >= 1000 && sr <= 384000, "Invalid WAV channel count or sample rate");
    check((format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) ||
          (format == 3 && (bits == 32 || bits == 64)), "Unsupported WAV encoding; use PCM or IEEE float");
    check(align == channels * (bits / 8) && data.size() % align == 0, "Invalid WAV block alignment");
    Audio result;
    result.sample_rate = static_cast<int>(sr);
    result.samples.resize(data.size() / align);
    for (size_t i = 0; i < result.samples.size(); ++i) {
        double sum = 0;
        for (int c = 0; c < channels; ++c) {
            const auto * p = data.data() + i * align + c * (bits / 8);
            double value = 0;
            if (format == 3) {
                if (bits == 32) { float x; std::memcpy(&x, p, 4); value = x; }
                else { double x; std::memcpy(&x, p, 8); value = x; }
            } else if (bits == 8) value = (int(p[0]) - 128) / 128.0;
            else if (bits == 16) value = static_cast<int16_t>(u16(p)) / 32768.0;
            else if (bits == 24) {
                int32_t x = int32_t(p[0]) | (int32_t(p[1]) << 8) | (int32_t(p[2]) << 16);
                if (x & 0x800000) x -= 0x1000000;
                value = x / 8388608.0;
            } else value = static_cast<int32_t>(u32(p)) / 2147483648.0;
            check(std::isfinite(value), "WAV contains non-finite samples");
            sum += value;
        }
        result.samples[i] = static_cast<float>(sum / channels);
        check(std::isfinite(result.samples[i]), "WAV sample exceeds float32 range");
    }
    return result;
}

std::vector<float> resample(const std::vector<float> & audio, int from, int to) {
    check(from >= 1000 && from <= 384000 && to >= 1000 && to <= 384000, "Sample rate must be 1000..384000 Hz");
    check(std::all_of(audio.begin(), audio.end(), [](float v) { return std::isfinite(v); }), "Non-finite audio input");
    if (from == to || audio.empty()) return audio;
    int divisor = std::gcd(from, to), orig = from / divisor, next = to / divisor;
    const double base = std::min(orig, next) * 0.99;
    const int width = static_cast<int>(std::ceil(128 * orig / base));
    // torchaudio Resample(lowpass_filter_width=128): sinc + Hann, zero-padded.
    struct Filter { int start; std::vector<float> weights; };
    std::vector<Filter> filters(next);
    for (int phase = 0; phase < next; ++phase) {
        const double phase_offset = static_cast<double>(-static_cast<float>(phase) / next);
        const double center = -phase_offset * orig;
        auto & filter = filters[phase];
        filter.start = std::max(-width, static_cast<int>(std::floor(center - width)));
        int end = std::min(width + orig - 1, static_cast<int>(std::ceil(center + width)));
        filter.weights.resize(end - filter.start + 1);
        for (int i = filter.start; i <= end; ++i) {
            double t = std::clamp((phase_offset + static_cast<double>(i) / orig) * base, -128.0, 128.0);
            double window = std::cos(t * pi / 256);
            double angle = t * pi;
            filter.weights[i - filter.start] = static_cast<float>((angle == 0 ? 1.0 : std::sin(angle) / angle) * window * window * base / orig);
        }
    }
    std::vector<float> out((static_cast<uint64_t>(audio.size()) * next + orig - 1) / orig);
    for (size_t j = 0; j < out.size(); ++j) {
        const auto & filter = filters[j % next];
        int64_t start = static_cast<int64_t>(j / next) * orig + filter.start;
        double sum = 0;
        size_t lo = static_cast<size_t>(std::max<int64_t>(0, -start));
        size_t hi = static_cast<size_t>(std::max<int64_t>(0, std::min<int64_t>(static_cast<int64_t>(filter.weights.size()), static_cast<int64_t>(audio.size()) - start)));
        for (size_t i = lo; i < hi; ++i) sum += static_cast<double>(filter.weights[i]) * audio[static_cast<size_t>(start + i)];
        out[j] = static_cast<float>(sum);
    }
    return out;
}

std::vector<float> compute_mel(const std::vector<float> & audio, const Config & cfg,
                             const std::vector<float> & basis, const std::vector<float> & window, int threads) {
    const auto n = static_cast<int64_t>(audio.size());
    const int64_t left = (cfg.win_size - cfg.hop_size) / 2;
    const int64_t right = std::max<int64_t>((cfg.win_size - cfg.hop_size + 1) / 2, cfg.win_size - n - left);
    const bool reflect = right < n;
    const auto stft_frames = static_cast<size_t>((n + left + right - cfg.n_fft) / cfg.hop_size + 1);
    const auto frames = audio.size() / cfg.hop_size + 1;
    const auto fft_bins = static_cast<size_t>(cfg.n_fft / 2 + 1);
    std::vector<float> out(frames * cfg.mel_bins);
    // Triangular Mel filters have short, contiguous nonzero support. Preserve
    // the original F32 accumulation order while skipping exact zero weights.
    std::vector<std::pair<size_t, size_t>> support(cfg.mel_bins);
    for (int m = 0; m < cfg.mel_bins; ++m) {
        size_t begin = 0, end = fft_bins;
        while (begin < end && basis[m * fft_bins + begin] == 0.0f) ++begin;
        while (end > begin && basis[m * fft_bins + end - 1] == 0.0f) --end;
        support[m] = {begin, end};
    }
    const auto work = [&](size_t begin, size_t end) {
        // Batched FFTs let pocketfft vectorize across frames, with bounded scratch
        // space. Each worker owns its arrays; no change to whole-recording GN.
        constexpr size_t batch_limit = 64;
        std::vector<float> time(batch_limit * cfg.n_fft), magnitude(fft_bins);
        std::vector<std::complex<float>> spectrum(batch_limit * fft_bins);
        for (size_t block = begin; block < end; block += batch_limit) {
            const size_t batch = std::min(batch_limit, end - block);
            for (size_t frame = 0; frame < batch; ++frame) {
                const size_t t = block + frame;
                for (int j = 0; j < cfg.n_fft; ++j) {
                    int64_t k = static_cast<int64_t>(t) * cfg.hop_size + j - left;
                    if (reflect) { if (k < 0) k = -k; if (k >= n) k = 2 * n - 2 - k; }
                    time[frame * cfg.n_fft + j] =
                        (k >= 0 && k < n ? audio[static_cast<size_t>(k)] : 0.0f) * window[j];
                }
            }
            pocketfft::r2c<float>({batch, static_cast<size_t>(cfg.n_fft)},
                {ptrdiff_t(cfg.n_fft * sizeof(float)), sizeof(float)},
                {ptrdiff_t(fft_bins * sizeof(std::complex<float>)), sizeof(std::complex<float>)},
                1, true, time.data(), spectrum.data(), 1.0f);
            for (size_t frame = 0; frame < batch; ++frame) {
                const size_t t = block + frame;
                for (size_t j = 0; j < fft_bins; ++j) {
                    float re = spectrum[frame * fft_bins + j].real(), im = spectrum[frame * fft_bins + j].imag();
                    magnitude[j] = std::sqrt(re * re + im * im + 1e-9f);
                }
                for (int m = 0; m < cfg.mel_bins; ++m) {
                    float sum = 0;
                    for (size_t j = support[m].first; j < support[m].second; ++j)
                        sum += basis[m * fft_bins + j] * magnitude[j];
                    out[t * cfg.mel_bins + m] = std::log(std::max(1e-5f, sum));
                }
            }
        }
    };
    const size_t count = std::min(stft_frames, frames);
    const size_t workers = std::min<size_t>(std::max(1, threads), std::max<size_t>(1, count / 64));
    std::vector<std::future<void>> jobs;
    for (size_t i = 1; i < workers; ++i)
        jobs.emplace_back(std::async(std::launch::async, work, count * i / workers, count * (i + 1) / workers));
    work(0, count / workers);
    for (auto & job : jobs) job.get();
    if (frames > stft_frames)
        std::copy_n(out.data() + (stft_frames - 1) * cfg.mel_bins, cfg.mel_bins, out.data() + stft_frames * cfg.mel_bins);
    return out;
}
} // namespace fcpe
