/* SSMF functions and basic operations (`ArchFloat`/`ArchDouble`) for 256-bit AVX, followed by `SimdFill`.

Like fill.h, this is unguarded and namespace-agnostic: it's included from x86-avx1.h and x86-avx2.h, inside their own namespace and target #pragmas.  Defining SIGNALSMITH_LINEAR_AVX2 before including it switches on AVX2 (256-bit integer shifts) and FMA.
*/

#ifndef SIGNALSMITH_LINEAR_AVX2
// AVX has no 256-bit integer ops, so shifts are done on each 128-bit half
#	define SIGNALSMITH_AVX_SPLIT(op, a, bits) _mm256_insertf128_si256(_mm256_castsi128_si256(op(_mm256_castsi256_si128(a), bits)), op(_mm256_extractf128_si256(a, 1), bits), 1)
#endif

#if defined(__clang__) || defined(__GNUC__)
inline __m256 asmBarrier(__m256 v) {
	__asm__("" : "+x"(v)); // empty ASM block which the compiler doesn't re-associate through
	return v;
}
#endif

#if defined(__clang__) || defined(__GNUC__)
inline __m256d asmBarrier(__m256d v) {
	__asm__("" : "+x"(v));
	return v;
}
#endif

// Everything specific to AVX: vector types, memory access, complex (de-)interleaving and arithmetic, plus the SSMF functions (included as static members).
// The generic code (`SimdFill` in ./fill.h) works out lane counts and alignment from `sizeof(V)`.
struct ArchFloat {
	using V = __m256;
	using Scalar = float;

	static V splat(double k) {return _mm256_set1_ps(float(k));}
	static V load(const float *p) {return _mm256_load_ps(p);}
	static V loadu(const float *p) {return _mm256_loadu_ps(p);}
	static void store(float *p, V v) {_mm256_store_ps(p, v);}
	static void storeu(float *p, V v) {_mm256_storeu_ps(p, v);}

	// Interleaved complex values: (a, b) are two consecutive vectors of [r0, i0, r1, i1, ...]
	// The in-lane shuffles work on 128-bit halves, so we first swap the middle halves: [a.lo, b.lo], [a.hi, b.hi]
	static void deinterleave(V a, V b, V &real, V &imag) {
		V lo = _mm256_permute2f128_ps(a, b, 0x20), hi = _mm256_permute2f128_ps(a, b, 0x31);
		real = _mm256_shuffle_ps(lo, hi, _MM_SHUFFLE(2, 0, 2, 0));
		imag = _mm256_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 1, 3, 1));
	}
	static void interleave(V real, V imag, V &a, V &b) {
		V lo = _mm256_unpacklo_ps(real, imag), hi = _mm256_unpackhi_ps(real, imag);
		a = _mm256_permute2f128_ps(lo, hi, 0x20);
		b = _mm256_permute2f128_ps(lo, hi, 0x31);
	}

	// v < 0 ? a : b, blending on the sign bit (so -0 counts as negative)
	static V ifNegative(V v, V a, V b) {return _mm256_blendv_ps(b, a, v);}

	static V add(V a, V b) {return _mm256_add_ps(a, b);}
	static V sub(V a, V b) {return _mm256_sub_ps(a, b);}
	static V mul(V a, V b) {return _mm256_mul_ps(a, b);}
	static V div(V a, V b) {return _mm256_div_ps(a, b);}
	static V max(V a, V b) {return _mm256_max_ps(a, b);}
	static V min(V a, V b) {return _mm256_min_ps(a, b);}
	// a*b + c, a*b - c, c - a*b and -(a*b + c)
#ifdef SIGNALSMITH_LINEAR_AVX2
	static V fma(V a, V b, V c) {return _mm256_fmadd_ps(a, b, c);}
	static V fms(V a, V b, V c) {return _mm256_fmsub_ps(a, b, c);}
	static V fnma(V a, V b, V c) {return _mm256_fnmadd_ps(a, b, c);}
	static V fnms(V a, V b, V c) {return _mm256_fnmsub_ps(a, b, c);}
#else
	static V fma(V a, V b, V c) {return _mm256_add_ps(_mm256_mul_ps(a, b), c);}
	static V fms(V a, V b, V c) {return _mm256_sub_ps(_mm256_mul_ps(a, b), c);}
	static V fnma(V a, V b, V c) {return _mm256_sub_ps(c, _mm256_mul_ps(a, b));}
	static V fnms(V a, V b, V c) {return _mm256_xor_ps(_mm256_add_ps(_mm256_mul_ps(a, b), c), _mm256_set1_ps(-0.0f));}
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
#define SSMF_VF_RSQRT_EST(v) _mm256_rsqrt_ps(v)
#define SSMF_RSQRT_EST_BITS 11.4 // relative error <= 1.5*2^-12
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
#	define SSMF_VF_BARRIER(v) asmBarrier(v)
#endif
#define SSMF_NAME(fn) fn // static member functions
#define SSMF_INLINE static inline
#include "../signalsmith-mathfun.h"
};

struct ArchDouble {
	using V = __m256d;
	using Scalar = double;

	static V splat(double k) {return _mm256_set1_pd(k);}
	static V load(const double *p) {return _mm256_load_pd(p);}
	static V loadu(const double *p) {return _mm256_loadu_pd(p);}
	static void store(double *p, V v) {_mm256_store_pd(p, v);}
	static void storeu(double *p, V v) {_mm256_storeu_pd(p, v);}

	static void deinterleave(V a, V b, V &real, V &imag) {
		V lo = _mm256_permute2f128_pd(a, b, 0x20), hi = _mm256_permute2f128_pd(a, b, 0x31);
		real = _mm256_unpacklo_pd(lo, hi);
		imag = _mm256_unpackhi_pd(lo, hi);
	}
	static void interleave(V real, V imag, V &a, V &b) {
		V lo = _mm256_unpacklo_pd(real, imag), hi = _mm256_unpackhi_pd(real, imag);
		a = _mm256_permute2f128_pd(lo, hi, 0x20);
		b = _mm256_permute2f128_pd(lo, hi, 0x31);
	}

	static V ifNegative(V v, V a, V b) {return _mm256_blendv_pd(b, a, v);}

	static V add(V a, V b) {return _mm256_add_pd(a, b);}
	static V sub(V a, V b) {return _mm256_sub_pd(a, b);}
	static V mul(V a, V b) {return _mm256_mul_pd(a, b);}
	static V div(V a, V b) {return _mm256_div_pd(a, b);}
	static V max(V a, V b) {return _mm256_max_pd(a, b);}
	static V min(V a, V b) {return _mm256_min_pd(a, b);}
#ifdef SIGNALSMITH_LINEAR_AVX2
	static V fma(V a, V b, V c) {return _mm256_fmadd_pd(a, b, c);}
	static V fms(V a, V b, V c) {return _mm256_fmsub_pd(a, b, c);}
	static V fnma(V a, V b, V c) {return _mm256_fnmadd_pd(a, b, c);}
	static V fnms(V a, V b, V c) {return _mm256_fnmsub_pd(a, b, c);}
#else
	static V fma(V a, V b, V c) {return _mm256_add_pd(_mm256_mul_pd(a, b), c);}
	static V fms(V a, V b, V c) {return _mm256_sub_pd(_mm256_mul_pd(a, b), c);}
	static V fnma(V a, V b, V c) {return _mm256_sub_pd(c, _mm256_mul_pd(a, b));}
	static V fnms(V a, V b, V c) {return _mm256_xor_pd(_mm256_add_pd(_mm256_mul_pd(a, b), c), _mm256_set1_pd(-0.0));}
#endif

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
#	define SSMF_VF_BARRIER(v) asmBarrier(v)
#endif
#define SSMF_NAME(fn) fn // static member functions
#define SSMF_INLINE static inline
#include "../signalsmith-mathfun.h"
};

#include "./fill.h"

#undef SIGNALSMITH_AVX_SPLIT
