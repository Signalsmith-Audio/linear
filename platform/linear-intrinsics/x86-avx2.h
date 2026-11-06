#include <immintrin.h>

// Every function in here (including the SSMF ones and `SimdFill`) gets compiled for AVX2 and FMA, so it must only be called after checking the CPU (and OS) support it.
#if defined(__clang__)
#	pragma clang attribute push(__attribute__((target("avx2,fma"))), apply_to=function)
#elif defined(__GNUC__)
#	pragma GCC push_options
#	pragma GCC target("avx2,fma")
#endif

namespace signalsmith { namespace linear { namespace simd_avx2 {

#define SIGNALSMITH_LINEAR_AVX2
#include "./x86-avx.h"
#undef SIGNALSMITH_LINEAR_AVX2

}}}; // namespace

#if defined(__clang__)
#	pragma clang attribute pop
#elif defined(__GNUC__)
#	pragma GCC pop_options
#endif
