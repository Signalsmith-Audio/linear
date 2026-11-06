#include <immintrin.h>

// Every function in here (including the SSMF ones and `SimdFill`) gets compiled for SSE4.1, so it must only be called after checking the CPU supports it.
#if defined(__clang__)
#	pragma clang attribute push(__attribute__((target("sse4.1"))), apply_to=function)
#elif defined(__GNUC__)
#	pragma GCC push_options
#	pragma GCC target("sse4.1")
#endif

namespace signalsmith { namespace linear { namespace simd_sse41 {

#define SIGNALSMITH_LINEAR_SSE41
#include "./x86-sse.h"
#undef SIGNALSMITH_LINEAR_SSE41

}}}; // namespace

#if defined(__clang__)
#	pragma clang attribute pop
#elif defined(__GNUC__)
#	pragma GCC pop_options
#endif
