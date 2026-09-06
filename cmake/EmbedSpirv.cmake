# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.

file(READ "${INPUT}" _hex HEX)
string(REGEX REPLACE "(..)" "0x\\1," _bytes "${_hex}")
if(NOT SYMBOL)
    set(SYMBOL fcpe_matmul_f32_spv)
endif()
file(WRITE "${OUTPUT}" "// Generated SPIR-V; do not edit.\nalignas(4) static const unsigned char ${SYMBOL}[] = {${_bytes}};\n")
