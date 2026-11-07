/* SSMF functions and basic operations (`Arch`) for 256-bit AVX, followed by `SimdFill`.

Like fill.h, this is unguarded and namespace-agnostic: it's included from x86-avx1.h and x86-avx2.h, inside their own namespace and target #pragmas.  Defining SIGNALSMITH_LINEAR_AVX2 before including it switches on AVX2 (256-bit integer shifts) and FMA.
*/

#ifndef SIGNALSMITH_LINEAR_AVX2
// AVX has no 256-bit integer ops, so shifts are done on each 128-bit half
#	define SIGNALSMITH_AVX_SPLIT(op, a, bits) _mm256_insertf128_si256(_mm256_castsi128_si256(op(_mm256_castsi256_si128(a), bits)), op(_mm256_extractf128_si256(a, 1), bits), 1)
#endif

#define SSMF_BITS 32
#if defined(SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY32)
#	define SSMF_ACCURACY SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY32
#elif defined(SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY)
#	define SSMF_ACCURACY SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY
#endif
#define SSMF_VF __m256
#define SSMF_VF_0 _mm256_setzero_ps()
#define SSMF_VF_F(v) _mm256_set1_ps(float(v))
#define SSMF_VF_I(v) _mm256_castsi256_ps(_mm256_set1_epi32(v))
#define SSMF_VF_AND(a, b) _mm256_and_ps(a, b)
#define SSMF_VF_ANDNOT(a, b) _mm256_andnot_ps(a, b)
#define SSMF_VF_OR(a, b) _mm256_or_ps(a, b)
#define SSMF_VF_XOR(a, b) _mm256_xor_ps(a, b)
#define SSMF_VF_NOT(a) _mm256_xor_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(-1)))
#define SSMF_VF_ADD(a, b) _mm256_add_ps(a, b)
#define SSMF_VF_SUB(a, b) _mm256_sub_ps(a, b)
#define SSMF_VF_MUL(a, b) _mm256_mul_ps(a, b)
#define SSMF_VF_DIV(a, b) _mm256_div_ps(a, b)
#define SSMF_VF_MAX(a, b) _mm256_max_ps(a, b)
#define SSMF_VF_MIN(a, b) _mm256_min_ps(a, b)
#define SSMF_VF_SQRT(v) _mm256_sqrt_ps(v)
#define SSMF_VF_RSQRT_ESTIMATE(v) _mm256_rsqrt_ps(v)
#define SSMF_RSQRT_ESTIMATE_BITS 11 // relative error <= 1.5*2^-12
#ifdef SIGNALSMITH_LINEAR_AVX2
#	define SSMF_VF_RSHIFT_K(a, bits) _mm256_castsi256_ps(_mm256_srli_epi32(_mm256_castps_si256(a), bits))
#	define SSMF_VF_LSHIFT_K(a, bits) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(a), bits))
#	define SSMF_VF_FMA(a, b, c) _mm256_fmadd_ps(a, b, c)
#	define SSMF_VF_FMAK(a, b, k) _mm256_fmadd_ps(a, b, _mm256_set1_ps(float(k)))
#else
#	define SSMF_VF_RSHIFT_K(a, bits) _mm256_castsi256_ps(SIGNALSMITH_AVX_SPLIT(_mm_srli_epi32, _mm256_castps_si256(a), bits))
#	define SSMF_VF_LSHIFT_K(a, bits) _mm256_castsi256_ps(SIGNALSMITH_AVX_SPLIT(_mm_slli_epi32, _mm256_castps_si256(a), bits))
#endif
#define SSMF_VF_ROUND(v) _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC)
#define SSMF_VF_FLOOR(v) _mm256_floor_ps(v)
#if defined(__clang__) || defined(__GNUC__)
inline __m256 asmBarrier(__m256 v) {
	__asm__("" : "+x"(v)); // empty ASM block which the compiler doesn't re-associate through
	return v;
}
#	define SSMF_VF_BARRIER(v) asmBarrier(v)
#endif
#define SSMF_NAME(fn) ssmf_##fn##f
#include "../signalsmith-mathfun.h"

#define SSMF_BITS 64
#if defined(SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY64)
#	define SSMF_ACCURACY SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY64
#elif defined(SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY)
#	define SSMF_ACCURACY SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY
#endif
#define SSMF_VF __m256d
#define SSMF_VF_0 _mm256_setzero_pd()
#define SSMF_VF_F(v) _mm256_set1_pd(double(v))
#define SSMF_VF_I(v) _mm256_castsi256_pd(_mm256_set1_epi64x(v))
#define SSMF_VF_AND(a, b) _mm256_and_pd(a, b)
#define SSMF_VF_ANDNOT(a, b) _mm256_andnot_pd(a, b)
#define SSMF_VF_OR(a, b) _mm256_or_pd(a, b)
#define SSMF_VF_XOR(a, b) _mm256_xor_pd(a, b)
#define SSMF_VF_NOT(a) _mm256_xor_pd(a, _mm256_castsi256_pd(_mm256_set1_epi32(-1)))
#define SSMF_VF_ADD(a, b) _mm256_add_pd(a, b)
#define SSMF_VF_SUB(a, b) _mm256_sub_pd(a, b)
#define SSMF_VF_MUL(a, b) _mm256_mul_pd(a, b)
#define SSMF_VF_DIV(a, b) _mm256_div_pd(a, b)
#define SSMF_VF_MAX(a, b) _mm256_max_pd(a, b)
#define SSMF_VF_MIN(a, b) _mm256_min_pd(a, b)
#define SSMF_VF_SQRT(v) _mm256_sqrt_pd(v)
// Skipping RSQRT_EST here because it's not actually faster
#ifdef SIGNALSMITH_LINEAR_AVX2
#	define SSMF_VF_RSHIFT_K(a, bits) _mm256_castsi256_pd(_mm256_srli_epi64(_mm256_castpd_si256(a), bits))
#	define SSMF_VF_LSHIFT_K(a, bits) _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_castpd_si256(a), bits))
#	define SSMF_VF_FMA(a, b, c) _mm256_fmadd_pd(a, b, c)
#	define SSMF_VF_FMAK(a, b, k) _mm256_fmadd_pd(a, b, _mm256_set1_pd(double(k)))
#else
#	define SSMF_VF_RSHIFT_K(a, bits) _mm256_castsi256_pd(SIGNALSMITH_AVX_SPLIT(_mm_srli_epi64, _mm256_castpd_si256(a), bits))
#	define SSMF_VF_LSHIFT_K(a, bits) _mm256_castsi256_pd(SIGNALSMITH_AVX_SPLIT(_mm_slli_epi64, _mm256_castpd_si256(a), bits))
#endif
#define SSMF_VF_ROUND(v) _mm256_round_pd(v, _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC)
#define SSMF_VF_FLOOR(v) _mm256_floor_pd(v)
#if defined(__clang__) || defined(__GNUC__)
inline __m256d asmBarrier(__m256d v) {
	__asm__("" : "+x"(v));
	return v;
}
#	define SSMF_VF_BARRIER(v) asmBarrier(v)
#endif
#define SSMF_NAME(fn) ssmf_##fn##d
#include "../signalsmith-mathfun.h"

// Everything specific to AVX: vector types, memory access, complex (de-)interleaving and arithmetic, plus the SSMF functions above.
// The generic code (`SimdFill` in ./fill.h) works out lane counts and alignment from `sizeof(Float)`.
struct Arch {
	using Float = __m256;
	using Double = __m256d;

	static void splat(__m256 &v, double k) {v = _mm256_set1_ps(float(k));}
	static void splat(__m256d &v, double k) {v = _mm256_set1_pd(k);}

	static __m256 load(const float *p) {return _mm256_load_ps(p);}
	static __m256d load(const double *p) {return _mm256_load_pd(p);}
	static __m256 loadu(const float *p) {return _mm256_loadu_ps(p);}
	static __m256d loadu(const double *p) {return _mm256_loadu_pd(p);}
	static void store(float *p, __m256 v) {_mm256_store_ps(p, v);}
	static void store(double *p, __m256d v) {_mm256_store_pd(p, v);}
	static void storeu(float *p, __m256 v) {_mm256_storeu_ps(p, v);}
	static void storeu(double *p, __m256d v) {_mm256_storeu_pd(p, v);}

	// Interleaved complex values: (a, b) are two consecutive vectors of [r0, i0, r1, i1, ...]
	// The in-lane shuffles work on 128-bit halves, so we first swap the middle halves: [a.lo, b.lo], [a.hi, b.hi]
	static void deinterleave(__m256 a, __m256 b, __m256 &real, __m256 &imag) {
		__m256 lo = _mm256_permute2f128_ps(a, b, 0x20), hi = _mm256_permute2f128_ps(a, b, 0x31);
		real = _mm256_shuffle_ps(lo, hi, _MM_SHUFFLE(2, 0, 2, 0));
		imag = _mm256_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 1, 3, 1));
	}
	static void deinterleave(__m256d a, __m256d b, __m256d &real, __m256d &imag) {
		__m256d lo = _mm256_permute2f128_pd(a, b, 0x20), hi = _mm256_permute2f128_pd(a, b, 0x31);
		real = _mm256_unpacklo_pd(lo, hi);
		imag = _mm256_unpackhi_pd(lo, hi);
	}
	static void interleave(__m256 real, __m256 imag, __m256 &a, __m256 &b) {
		__m256 lo = _mm256_unpacklo_ps(real, imag), hi = _mm256_unpackhi_ps(real, imag);
		a = _mm256_permute2f128_ps(lo, hi, 0x20);
		b = _mm256_permute2f128_ps(lo, hi, 0x31);
	}
	static void interleave(__m256d real, __m256d imag, __m256d &a, __m256d &b) {
		__m256d lo = _mm256_unpacklo_pd(real, imag), hi = _mm256_unpackhi_pd(real, imag);
		a = _mm256_permute2f128_pd(lo, hi, 0x20);
		b = _mm256_permute2f128_pd(lo, hi, 0x31);
	}

	// v < 0 ? a : b, blending on the sign bit (so -0 counts as negative)
	static __m256 ifNegative(__m256 v, __m256 a, __m256 b) {return _mm256_blendv_ps(b, a, v);}
	static __m256d ifNegative(__m256d v, __m256d a, __m256d b) {return _mm256_blendv_pd(b, a, v);}

#define SIGNALSMITH_AVX_OP1(name, mmFloat, mmDouble) \
	static __m256 name(__m256 a) {return mmFloat(a);} \
	static __m256d name(__m256d a) {return mmDouble(a);}
#define SIGNALSMITH_AVX_OP2(name, mmFloat, mmDouble) \
	static __m256 name(__m256 a, __m256 b) {return mmFloat(a, b);} \
	static __m256d name(__m256d a, __m256d b) {return mmDouble(a, b);}
	SIGNALSMITH_AVX_OP2(add, _mm256_add_ps, _mm256_add_pd)
	SIGNALSMITH_AVX_OP2(sub, _mm256_sub_ps, _mm256_sub_pd)
	SIGNALSMITH_AVX_OP2(mul, _mm256_mul_ps, _mm256_mul_pd)
	SIGNALSMITH_AVX_OP2(div, _mm256_div_ps, _mm256_div_pd)
	SIGNALSMITH_AVX_OP2(max, _mm256_max_ps, _mm256_max_pd)
	SIGNALSMITH_AVX_OP2(min, _mm256_min_ps, _mm256_min_pd)
	SIGNALSMITH_AVX_OP1(sqrt, _mm256_sqrt_ps, _mm256_sqrt_pd)
	// a*b + c
#ifdef SIGNALSMITH_LINEAR_AVX2
	static __m256 fma(__m256 a, __m256 b, __m256 c) {return _mm256_fmadd_ps(a, b, c);}
	static __m256d fma(__m256d a, __m256d b, __m256d c) {return _mm256_fmadd_pd(a, b, c);}
#else
	static __m256 fma(__m256 a, __m256 b, __m256 c) {return _mm256_add_ps(_mm256_mul_ps(a, b), c);}
	static __m256d fma(__m256d a, __m256d b, __m256d c) {return _mm256_add_pd(_mm256_mul_pd(a, b), c);}
#endif
#define SIGNALSMITH_SSMF1(name) SIGNALSMITH_AVX_OP1(name, ssmf_##name##f, ssmf_##name##d)
#define SIGNALSMITH_SSMF2(name) SIGNALSMITH_AVX_OP2(name, ssmf_##name##f, ssmf_##name##d)
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
#undef SIGNALSMITH_AVX_OP1
#undef SIGNALSMITH_AVX_OP2
	static void sincos(__m256 v, __m256 &sinOut, __m256 &cosOut) {ssmf_sincosf(v, &sinOut, &cosOut);}
	static void sincos(__m256d v, __m256d &sinOut, __m256d &cosOut) {ssmf_sincosd(v, &sinOut, &cosOut);}
};

#include "./fill.h"

#undef SIGNALSMITH_AVX_SPLIT
