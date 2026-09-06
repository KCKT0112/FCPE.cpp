// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.

#include "fcpe/fcpe.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void check(bool ok, const char * msg) { if (!ok) throw std::runtime_error(msg); }
template<class F> void rejects(F fn) {
    bool caught = false;
    try { fn(); } catch (const std::exception &) { caught = true; }
    check(caught, "Expected invalid input to be rejected");
}
void unit() {
    std::vector<float> cents{1200, 2400, 3600, 4800, 6000, 7200};
    auto zeros = fcpe::decode(std::vector<float>(12, 0), cents);
    check(zeros.f0 == std::vector<float>({0, 0}), "Silence must decode to zero");
    auto edge = fcpe::decode({1, 0.5f, 0, 0, 0, 0}, cents);
    float expected = 10 * std::exp2(((5 * 1200.0f + 0.5f * 2400) / 5.5f) / 1200);
    check(std::abs(edge.f0[0] - expected) < 1e-4f, "Local decoder must duplicate clamped edge bins");
    fcpe::DecodeOptions opts;
    opts.decoder = fcpe::Decoder::argmax;
    auto global = fcpe::decode({1, 0.5f, 0, 0, 0, 0}, cents, opts);
    check(std::abs(global.f0[0] - 10 * std::exp2(1600.0f / 1200)) < 1e-4f, "Global decoder must weight all bins");
    opts.decoder = fcpe::Decoder::local_argmax;
    opts.threshold = 0.5;
    auto threshold = fcpe::decode({0, 0.5f, 0, 0, 0, 0}, cents, opts);
    check(threshold.f0[0] == 0, "Threshold comparison must be strict");
    opts.interpolate_unvoiced = true;
    opts.output_frames = 10;
    auto interpolated = fcpe::decode({0,0,0,0,0,0, 0,1,0,0,0,0, 0,0,0,0,0,0,
                                      0,0,1,0,0,0, 0,0,0,0,0,0}, cents, opts);
    check(interpolated.f0 == std::vector<float>({40,40,40,40,60,60,80,80,80,80}), "UV interpolation mismatch");
    check(interpolated.unvoiced == std::vector<float>({1,1,0,0,1,1,0,0,1,1}), "UV flags must be retained and resized");
    auto silent = fcpe::decode(std::vector<float>(12, 0), cents, opts);
    check(std::all_of(silent.f0.begin(), silent.f0.end(), [](float x) { return x == 0; }), "All-UV interpolation must stay zero");
    rejects([&] { fcpe::decode({}, cents); });
    rejects([&] { fcpe::decode({1}, cents); });
    rejects([&] { fcpe::decode({NAN,0,0,0,0,0}, cents); });
    rejects([&] { fcpe::resample({0}, 0, 16000); });
    for (int sr : {8000, 16000, 22050, 44100, 48000}) {
        auto z = fcpe::resample(std::vector<float>(sr / 100, 0), sr, 16000);
        check(z.size() == 160 && std::all_of(z.begin(), z.end(), [](float x) { return x == 0; }), "Resample silence/length mismatch");
    }
}
void model_test(const char * path, const char * backend) {
    fcpe::Model model(path, {backend, 2});
    // Variable lengths, reused graph, short-input zero padding, exact frame boundaries.
    for (int n : {1, 159, 160, 431, 432, 433, 1024, 1600, 3200, 160, 1}) {
        auto m = model.mel(std::vector<float>(n, 0), 16000);
        check(m.size() == size_t(n / 160 + 1) * model.config().mel_bins, "Mel frame count mismatch");
        check(std::all_of(m.begin(), m.end(), [](float v) { return std::abs(v - std::log(1e-5f)) < 1e-5f; }), "Silence mel mismatch");
        auto p = model.probabilities(m);
        auto p2 = model.probabilities(m);
        check(p == p2, "Repeated inference is not deterministic");
        check(p.size() == size_t(n / 160 + 1) * model.config().bins, "Probability shape mismatch");
    }
    rejects([&] { model.probabilities({1, 2, 3}); });
    rejects([&] { model.mel({}, 16000); });
    rejects([&] { model.mel({std::numeric_limits<float>::infinity()}, 16000); });
}
}
int main(int argc, char ** argv) {
    try { unit(); if (argc > 1) model_test(argv[1], argc > 2 ? argv[2] : "cpu"); std::cout << "All tests passed\n"; return 0; }
    catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
