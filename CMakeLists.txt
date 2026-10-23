cmake_minimum_required(VERSION 3.16)
project(grok_from_scratch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to an optimized build; grokking runs tens of thousands of steps.
# Use -DCMAKE_BUILD_TYPE=Debug when chasing backprop bugs.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---- Reusable libraries -----------------------------------------------------
# Everything includes from the project root, e.g. #include "engine/tensor.hpp"

add_library(engine STATIC
    engine/tensor.cpp
    engine/autograd.cpp
    engine/ops.cpp
)
target_include_directories(engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(nn STATIC
    nn/layers.cpp
    nn/losses.cpp
    nn/optim.cpp
)
target_link_libraries(nn PUBLIC engine)

add_library(data STATIC
    data/modular_arithmetic.cpp
)
target_link_libraries(data PUBLIC engine)

# ---- Tests (run with: ctest --test-dir build) -------------------------------

enable_testing()

foreach(t test_tensor test_gradients test_xor)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE nn)
    add_test(NAME ${t} COMMAND ${t})
endforeach()

# ---- Experiments ------------------------------------------------------------

foreach(e train_linreg train_xor train_grokking)
    add_executable(${e} experiments/${e}.cpp)
    target_link_libraries(${e} PRIVATE nn data)
endforeach()
