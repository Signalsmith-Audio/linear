#include <cmath>
#include <cstdint>
#include <cstring>

// `ArchFloat`/`ArchDouble` and SSMF functions for plain float/double.  Used by SIMD fills for the final elements of odd-sized arrays.
namespace signalsmith { namespace linear { namespace simd_scalar {

// Bitwise ops go through same-sized integers
inline std::uint32_t bitsOf(float v) {
	std::uint32_t i;
	std::memcpy(&i, &v, sizeof(v));
	return i;
}
inline std::uint64_t bitsOf(double v) {
	std::uint64_t i;
	std::memcpy(&i, &v, sizeof(v));
	return i;
}
inline float fromBits(std::uint32_t i) {
	float v;
	std::memcpy(&v, &i, sizeof(v));
	return v;
}
inline double fromBits(std::uint64_t i) {
	double v;
	std::memcpy(&v, &i, sizeof(v));
	return v;
}
template<class V>
V vmax(V a, V b) {return a > b ? a : b;}
template<class V>
V vmin(V a, V b) {return a < b ? a : b;}

// Supplying BARRIER, ROUND/FLOOR or FMA aren't actually faster than SSMF's internal fallback, so we skip them



// The same for float and double, apart from the SSMF functions
template<class S>
struct ArchScalar {
	using V = S;
	using Scalar = S;

	static V splat(double k) {return V(k);}
	static V load(const S *p) {return *p;}
	static V loadu(const S *p) {return *p;}
	static void store(S *p, V v) {*p = v;}
	static void storeu(S *p, V v) {*p = v;}

	// One complex value is a real vector followed by an imaginary one
	static void deinterleave(V a, V b, V &real, V &imag) {
		real = a;
		imag = b;
	}
	static void interleave(V real, V imag, V &a, V &b) {
		a = real;
		b = imag;
	}

	// Sign bit, to match the SSE4.1/AVX blends
	static V ifNegative(V v, V a, V b) {return std::signbit(v) ? a : b;}

	static V add(V a, V b) {return a + b;}
	static V sub(V a, V b) {return a - b;}
	static V mul(V a, V b) {return a*b;}
	static V div(V a, V b) {return a/b;}
	static V max(V a, V b) {return vmax(a, b);}
	static V min(V a, V b) {return vmin(a, b);}
	// Not fused (std::fma() is slow without hardware support)
	static V fma(V a, V b, V c) {return a*b + c;}
	static V fms(V a, V b, V c) {return a*b - c;}
	static V fnma(V a, V b, V c) {return c - a*b;}
	static V fnms(V a, V b, V c) {return -(a*b + c);}
};

struct ArchFloat : public ArchScalar<float> {
#define SSMF_BITS 32
#if defined(SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY32)
#	define SSMF_ACCURACY SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY32
#elif defined(SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY)
#	define SSMF_ACCURACY SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY
#endif
#define SSMF_VF float
#define SSMF_VF_0 0.0f
#define SSMF_VF_F(v) float(v)
#define SSMF_VF_I(v) fromBits(std::uint32_t(v))
#define SSMF_VF_AND(a, b) fromBits(bitsOf(a) & bitsOf(b))
#define SSMF_VF_ANDNOT(a, b) fromBits(~bitsOf(a) & bitsOf(b))
#define SSMF_VF_OR(a, b) fromBits(bitsOf(a) | bitsOf(b))
#define SSMF_VF_XOR(a, b) fromBits(bitsOf(a) ^ bitsOf(b))
#define SSMF_VF_NOT(a) fromBits(~bitsOf(a))
#define SSMF_VF_RSHIFT_K(a, bits) fromBits(bitsOf(a) >> (bits))
#define SSMF_VF_LSHIFT_K(a, bits) fromBits(bitsOf(a) << (bits))
#define SSMF_VF_ADD(a, b) ((a) + (b))
#define SSMF_VF_SUB(a, b) ((a) - (b))
#define SSMF_VF_MUL(a, b) ((a)*(b))
#define SSMF_VF_DIV(a, b) ((a)/(b))
#define SSMF_VF_MAX(a, b) vmax(a, b)
#define SSMF_VF_MIN(a, b) vmin(a, b)
#define SSMF_VF_SQRT(v) std::sqrt(v)
#define SSMF_NAME(fn) fn // static member functions
#define SSMF_INLINE static inline
#include "../signalsmith-mathfun.h"
};
struct ArchDouble : public ArchScalar<double> {
#define SSMF_BITS 64
#if defined(SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY64)
#	define SSMF_ACCURACY SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY64
#elif defined(SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY)
#	define SSMF_ACCURACY SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY
#endif
#define SSMF_VF double
#define SSMF_VF_0 0.0
#define SSMF_VF_F(v) double(v)
#define SSMF_VF_I(v) fromBits(std::uint64_t(v))
#define SSMF_VF_AND(a, b) fromBits(bitsOf(a) & bitsOf(b))
#define SSMF_VF_ANDNOT(a, b) fromBits(~bitsOf(a) & bitsOf(b))
#define SSMF_VF_OR(a, b) fromBits(bitsOf(a) | bitsOf(b))
#define SSMF_VF_XOR(a, b) fromBits(bitsOf(a) ^ bitsOf(b))
#define SSMF_VF_NOT(a) fromBits(~bitsOf(a))
#define SSMF_VF_RSHIFT_K(a, bits) fromBits(bitsOf(a) >> (bits))
#define SSMF_VF_LSHIFT_K(a, bits) fromBits(bitsOf(a) << (bits))
#define SSMF_VF_ADD(a, b) ((a) + (b))
#define SSMF_VF_SUB(a, b) ((a) - (b))
#define SSMF_VF_MUL(a, b) ((a)*(b))
#define SSMF_VF_DIV(a, b) ((a)/(b))
#define SSMF_VF_MAX(a, b) vmax(a, b)
#define SSMF_VF_MIN(a, b) vmin(a, b)
#define SSMF_VF_SQRT(v) std::sqrt(v)
#define SSMF_NAME(fn) fn // static member functions
#define SSMF_INLINE static inline
#include "../signalsmith-mathfun.h"
};

#include "./fill.h"

}}}; // namespace
