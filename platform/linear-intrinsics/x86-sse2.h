#include <immintrin.h>

#if defined(__GNUC__) && !defined(__clang__) // clang doesn't support `#pragma GCC target`
#	pragma GCC push_options
// Here's where we would put `#pragma GCC target("foo,bar")`, but SSE2 is always supported
#endif

namespace signalsmith { namespace linear { namespace simd_sse2 {

#include "./x86-sse.h"

}}}; // namespace

#if defined(__GNUC__) && !defined(__clang__) // clang doesn't support `#pragma GCC target`
#	pragma GCC pop_options
#endif
