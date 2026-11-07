#include <cmath>
#include <cstdint>
#include <cstring>

// `Arch` and SSMF functions for plain float/double.  Used by SIMD fills to for the final elements for odd-sized arrays.
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
#define SSMF_NAME(fn) ssmf_##fn##f
#include "../signalsmith-mathfun.h"

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
#define SSMF_NAME(fn) ssmf_##fn##d
#include "../signalsmith-mathfun.h"

struct Arch {
	using Float = float;
	using Double = double;

	static void splat(float &v, double k) {v = float(k);}
	static void splat(double &v, double k) {v = k;}

	template<class V>
	static V load(const V *p) {return *p;}
	template<class V>
	static V loadu(const V *p) {return *p;}
	template<class V>
	static void store(V *p, V v) {*p = v;}
	template<class V>
	static void storeu(V *p, V v) {*p = v;}

	// One complex value is a real vector followed by an imaginary one
	template<class V>
	static void deinterleave(V a, V b, V &real, V &imag) {
		real = a;
		imag = b;
	}
	template<class V>
	static void interleave(V real, V imag, V &a, V &b) {
		a = real;
		b = imag;
	}

	// Sign bit, to match the SSE4.1/AVX blends
	template<class V>
	static V ifNegative(V v, V a, V b) {return std::signbit(v) ? a : b;}

	template<class V>
	static V add(V a, V b) {return a + b;}
	template<class V>
	static V sub(V a, V b) {return a - b;}
	template<class V>
	static V mul(V a, V b) {return a*b;}
	template<class V>
	static V div(V a, V b) {return a/b;}
	template<class V>
	static V max(V a, V b) {return vmax(a, b);}
	template<class V>
	static V min(V a, V b) {return vmin(a, b);}
	template<class V>
	static V sqrt(V a) {return std::sqrt(a);}
	// Not fused (std::fma() is slow without hardware support)
	template<class V>
	static V fma(V a, V b, V c) {return a*b + c;}
	template<class V>
	static V fms(V a, V b, V c) {return a*b - c;}
	template<class V>
	static V fnma(V a, V b, V c) {return c - a*b;}
	template<class V>
	static V fnms(V a, V b, V c) {return -(a*b + c);}

#define SIGNALSMITH_SSMF1(name) \
	static float name(float a) {return ssmf_##name##f(a);} \
	static double name(double a) {return ssmf_##name##d(a);}
#define SIGNALSMITH_SSMF2(name) \
	static float name(float a, float b) {return ssmf_##name##f(a, b);} \
	static double name(double a, double b) {return ssmf_##name##d(a, b);}
	SIGNALSMITH_SSMF1(abs)
	SIGNALSMITH_SSMF1(neg)
	SIGNALSMITH_SSMF1(pow2)
	SIGNALSMITH_SSMF1(ceil)
	SIGNALSMITH_SSMF1(floor)
	SIGNALSMITH_SSMF1(trunc)
	SIGNALSMITH_SSMF1(round)
	SIGNALSMITH_SSMF1(cbrt)
	SIGNALSMITH_SSMF1(log)
	SIGNALSMITH_SSMF1(log2)
	SIGNALSMITH_SSMF1(log10)
	SIGNALSMITH_SSMF1(exp)
	SIGNALSMITH_SSMF1(exp2)
	SIGNALSMITH_SSMF1(sin)
	SIGNALSMITH_SSMF1(cos)
	SIGNALSMITH_SSMF1(tan)
	SIGNALSMITH_SSMF1(asin)
	SIGNALSMITH_SSMF1(acos)
	SIGNALSMITH_SSMF1(atan)
	SIGNALSMITH_SSMF1(sinh)
	SIGNALSMITH_SSMF1(cosh)
	SIGNALSMITH_SSMF1(tanh)
	SIGNALSMITH_SSMF1(erf)
	SIGNALSMITH_SSMF1(erfc)
	SIGNALSMITH_SSMF2(dim)
	SIGNALSMITH_SSMF2(copysign)
	SIGNALSMITH_SSMF2(pow)
	SIGNALSMITH_SSMF2(atan2)
	SIGNALSMITH_SSMF2(hypot)
#undef SIGNALSMITH_SSMF1
#undef SIGNALSMITH_SSMF2
	static void sincos(float v, float &sinOut, float &cosOut) {ssmf_sincosf(v, &sinOut, &cosOut);}
	static void sincos(double v, double &sinOut, double &cosOut) {ssmf_sincosd(v, &sinOut, &cosOut);}
};

#include "./fill.h"

}}}; // namespace
