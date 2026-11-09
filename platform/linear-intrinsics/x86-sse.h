/* SSMF functions and basic operations (`ArchFloat`/`ArchDouble`) for 128-bit SSE, followed by `SimdFill`.

Like fill.h, this is unguarded and namespace-agnostic: it's included from x86-sse2.h and x86-sse41.h, inside their own namespace and target #pragmas.  Defining SIGNALSMITH_LINEAR_SSE41 before including it switches on the SSE4.1 instructions.
*/

#if defined(__clang__) || defined(__GNUC__)
inline __m128 asmBarrier(__m128 v) {
	__asm__("" : "+x"(v)); // empty ASM block which the compiler doesn't re-associate through
	return v;
}
inline __m128d asmBarrier(__m128d v) {
	__asm__("" : "+x"(v));
	return v;
}
#endif

// Everything specific to SSE: vector types, memory access, complex (de-)interleaving and arithmetic, plus the SSMF functions (included as static members).
// The generic code (`SimdFill` in ./fill.h) works out lane counts and alignment from `sizeof(V)`.
struct ArchFloat {
	using V = __m128;
	using Scalar = float;

	static V splat(double k) {return _mm_set1_ps(float(k));}
	static V load(const float *p) {return _mm_load_ps(p);}
	static V loadu(const float *p) {return _mm_loadu_ps(p);}
	static void store(float *p, V v) {_mm_store_ps(p, v);}
	static void storeu(float *p, V v) {_mm_storeu_ps(p, v);}

	// Interleaved complex values: (a, b) are two consecutive vectors of [r0, i0, r1, i1, ...]
	static void deinterleave(V a, V b, V &real, V &imag) {
		real = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0));
		imag = _mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1));
	}
	static void interleave(V real, V imag, V &a, V &b) {
		a = _mm_unpacklo_ps(real, imag);
		b = _mm_unpackhi_ps(real, imag);
	}

	// v < 0 ? a : b
#ifdef SIGNALSMITH_LINEAR_SSE41
	// blends on the sign bit, so -0 counts as negative
	static V ifNegative(V v, V a, V b) {return _mm_blendv_ps(b, a, v);}
#else
	static V ifNegative(V v, V a, V b) {
		V mask = _mm_cmplt_ps(v, _mm_setzero_ps());
		return _mm_or_ps(_mm_and_ps(mask, a), _mm_andnot_ps(mask, b));
	}
#endif

	static V add(V a, V b) {return _mm_add_ps(a, b);}
	static V sub(V a, V b) {return _mm_sub_ps(a, b);}
	static V mul(V a, V b) {return _mm_mul_ps(a, b);}
	static V div(V a, V b) {return _mm_div_ps(a, b);}
	static V max(V a, V b) {return _mm_max_ps(a, b);}
	static V min(V a, V b) {return _mm_min_ps(a, b);}
	// a*b + c, a*b - c, c - a*b and -(a*b + c): there are no fused 128-bit instructions without AVX2+FMA
	static V fma(V a, V b, V c) {return _mm_add_ps(_mm_mul_ps(a, b), c);}
	static V fms(V a, V b, V c) {return _mm_sub_ps(_mm_mul_ps(a, b), c);}
	static V fnma(V a, V b, V c) {return _mm_sub_ps(c, _mm_mul_ps(a, b));}
	static V fnms(V a, V b, V c) {return _mm_xor_ps(_mm_add_ps(_mm_mul_ps(a, b), c), _mm_set1_ps(-0.0f));}

#define SSMF_BITS 32
#if defined(SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY32)
#	define SSMF_ACCURACY SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY32
#elif defined(SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY)
#	define SSMF_ACCURACY SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY
#endif
#define SSMF_VF __m128
#define SSMF_VF_0 _mm_setzero_ps()
#define SSMF_VF_F(v) _mm_set1_ps(float(v))
#define SSMF_VF_I(v) _mm_castsi128_ps(_mm_set1_epi32(v))
#define SSMF_VF_AND(a, b) _mm_and_ps(a, b)
#define SSMF_VF_ANDNOT(a, b) _mm_andnot_ps(a, b)
#define SSMF_VF_OR(a, b) _mm_or_ps(a, b)
#define SSMF_VF_XOR(a, b) _mm_xor_ps(a, b)
#define SSMF_VF_NOT(a) _mm_xor_ps(a, _mm_castsi128_ps(_mm_set1_epi32(-1)))
#define SSMF_VF_ADD(a, b) _mm_add_ps(a, b)
#define SSMF_VF_SUB(a, b) _mm_sub_ps(a, b)
#define SSMF_VF_MUL(a, b) _mm_mul_ps(a, b)
#define SSMF_VF_DIV(a, b) _mm_div_ps(a, b)
#define SSMF_VF_MAX(a, b) _mm_max_ps(a, b)
#define SSMF_VF_MIN(a, b) _mm_min_ps(a, b)
#define SSMF_VF_SQRT(v) _mm_sqrt_ps(v)
#define SSMF_VF_RSQRT_EST(v) _mm_rsqrt_ps(v)
#define SSMF_RSQRT_EST_BITS 11.4 // relative error <= 1.5*2^-12
#define SSMF_VF_RSHIFT_K(a, bits) _mm_castsi128_ps(_mm_srli_epi32(_mm_castps_si128(a), bits))
#define SSMF_VF_LSHIFT_K(a, bits) _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(a), bits))
#ifdef SIGNALSMITH_LINEAR_SSE41
#	define SSMF_VF_ROUND(v) _mm_round_ps(v, _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC)
#	define SSMF_VF_FLOOR(v) _mm_floor_ps(v)
#else
// Round-trip through int32 (round-to-nearest with the default MXCSR), valid for |v| < 2^31.
#	define SSMF_VF_ROUND(v) _mm_cvtepi32_ps(_mm_cvtps_epi32(v))
#endif
#if defined(__clang__) || defined(__GNUC__)
#	define SSMF_VF_BARRIER(v) asmBarrier(v)
#endif
#define SSMF_NAME(fn) fn // static member functions
#define SSMF_INLINE static inline
#include "../signalsmith-mathfun.h"
};

struct ArchDouble {
	using V = __m128d;
	using Scalar = double;

	static V splat(double k) {return _mm_set1_pd(k);}
	static V load(const double *p) {return _mm_load_pd(p);}
	static V loadu(const double *p) {return _mm_loadu_pd(p);}
	static void store(double *p, V v) {_mm_store_pd(p, v);}
	static void storeu(double *p, V v) {_mm_storeu_pd(p, v);}

	static void deinterleave(V a, V b, V &real, V &imag) {
		real = _mm_unpacklo_pd(a, b);
		imag = _mm_unpackhi_pd(a, b);
	}
	static void interleave(V real, V imag, V &a, V &b) {
		a = _mm_unpacklo_pd(real, imag);
		b = _mm_unpackhi_pd(real, imag);
	}

#ifdef SIGNALSMITH_LINEAR_SSE41
	static V ifNegative(V v, V a, V b) {return _mm_blendv_pd(b, a, v);}
#else
	static V ifNegative(V v, V a, V b) {
		V mask = _mm_cmplt_pd(v, _mm_setzero_pd());
		return _mm_or_pd(_mm_and_pd(mask, a), _mm_andnot_pd(mask, b));
	}
#endif

	static V add(V a, V b) {return _mm_add_pd(a, b);}
	static V sub(V a, V b) {return _mm_sub_pd(a, b);}
	static V mul(V a, V b) {return _mm_mul_pd(a, b);}
	static V div(V a, V b) {return _mm_div_pd(a, b);}
	static V max(V a, V b) {return _mm_max_pd(a, b);}
	static V min(V a, V b) {return _mm_min_pd(a, b);}
	static V fma(V a, V b, V c) {return _mm_add_pd(_mm_mul_pd(a, b), c);}
	static V fms(V a, V b, V c) {return _mm_sub_pd(_mm_mul_pd(a, b), c);}
	static V fnma(V a, V b, V c) {return _mm_sub_pd(c, _mm_mul_pd(a, b));}
	static V fnms(V a, V b, V c) {return _mm_xor_pd(_mm_add_pd(_mm_mul_pd(a, b), c), _mm_set1_pd(-0.0));}

#define SSMF_BITS 64
#if defined(SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY64)
#	define SSMF_ACCURACY SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY64
#elif defined(SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY)
#	define SSMF_ACCURACY SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY
#endif
#define SSMF_VF __m128d
#define SSMF_VF_0 _mm_setzero_pd()
#define SSMF_VF_F(v) _mm_set1_pd(double(v))
#define SSMF_VF_I(v) _mm_castsi128_pd(_mm_set1_epi64x(v))
#define SSMF_VF_AND(a, b) _mm_and_pd(a, b)
#define SSMF_VF_ANDNOT(a, b) _mm_andnot_pd(a, b)
#define SSMF_VF_OR(a, b) _mm_or_pd(a, b)
#define SSMF_VF_XOR(a, b) _mm_xor_pd(a, b)
#define SSMF_VF_NOT(a) _mm_xor_pd(a, _mm_castsi128_pd(_mm_set1_epi32(-1)))
#define SSMF_VF_ADD(a, b) _mm_add_pd(a, b)
#define SSMF_VF_SUB(a, b) _mm_sub_pd(a, b)
#define SSMF_VF_MUL(a, b) _mm_mul_pd(a, b)
#define SSMF_VF_DIV(a, b) _mm_div_pd(a, b)
#define SSMF_VF_MAX(a, b) _mm_max_pd(a, b)
#define SSMF_VF_MIN(a, b) _mm_min_pd(a, b)
#define SSMF_VF_SQRT(v) _mm_sqrt_pd(v)
// Skipping RSQRT_EST here because it's not actually faster
#define SSMF_VF_RSHIFT_K(a, b) _mm_castsi128_pd(_mm_srli_epi64(_mm_castpd_si128(a), b))
#define SSMF_VF_LSHIFT_K(a, b) _mm_castsi128_pd(_mm_slli_epi64(_mm_castpd_si128(a), b))
#ifdef SIGNALSMITH_LINEAR_SSE41
#	define SSMF_VF_ROUND(v) _mm_round_pd(v, _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC)
#	define SSMF_VF_FLOOR(v) _mm_floor_pd(v)
#endif
#if defined(__clang__) || defined(__GNUC__)
#	define SSMF_VF_BARRIER(v) asmBarrier(v)
#endif
#define SSMF_NAME(fn) fn // static member functions
#define SSMF_INLINE static inline
#include "../signalsmith-mathfun.h"
};

#include "./fill.h"
