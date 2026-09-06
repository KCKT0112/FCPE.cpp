# SPDX-License-Identifier: MPL-2.0
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.18)
project(fcpe_consumer LANGUAGES C CXX)
set(FCPE_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(FCPE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(.. fcpe)
add_executable(fcpe-example basic.cpp)
set_target_properties(fcpe-example PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(fcpe-example PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}/bin")
endforeach()
target_link_libraries(fcpe-example PRIVATE fcpe::fcpe)
target_compile_features(fcpe-example PRIVATE cxx_std_17)
