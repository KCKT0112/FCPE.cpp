// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.

#include <fcpe/fcpe.h>
#include <iostream>

int main(int argc, char ** argv) {
    if (argc != 3) { std::cerr << "Usage: fcpe-example model.gguf input.wav\n"; return 1; }
    try {
        fcpe::Model model(argv[1]);
        const auto audio = fcpe::read_wav(argv[2]);
        const auto result = model.infer(audio.samples, audio.sample_rate);
        for (size_t i = 0; i < result.f0.size(); ++i)
            std::cout << double(i * model.config().hop_size) / model.config().sample_rate
                      << ',' << result.f0[i] << '\n';
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
