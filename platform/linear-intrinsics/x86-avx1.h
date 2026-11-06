#include <immintrin.h>

// Every function in here (including the SSMF ones and `SimdFill`) gets compiled for AVX, so it must only be called after checking the CPU (and OS) support it.
#if defined(__clang__)
#	pragma clang attribute push(__attribute__((target("avx"))), apply_to=function)
#elif defined(__GNUC__)
#	pragma GCC push_options
#	pragma GCC target("avx")
#endif

namespace signalsmith { namespace linear { namespace simd_avx1 {

#include "./x86-avx.h"

}}}; // namespace

#if defined(__clang__)
#	pragma clang attribute pop
#elif defined(__GNUC__)
#	pragma GCC pop_options
#endif
