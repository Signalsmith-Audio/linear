#include "./linear-intrinsics/scalar.h"
#include "./linear-intrinsics/arm-neon.h"
#include "./intrinsics.h"

namespace signalsmith { namespace linear {

// NEON is always there (and only one version), so there's no runtime dispatch
using ArchFillNeon = simd_neon::SimdFill;

template<>
struct LinearImpl<true> : public LinearImplIntrinsics<ArchFillNeon> {};

}}; // namespace
