/* SSMF functions and basic operations (`Arch`) for 128-bit SSE, followed by `SimdFill`.

Like fill.h, this is unguarded and namespace-agnostic: it's included from x86-sse2.h and x86-sse41.h, inside their own namespace and target #pragmas.  Defining SIGNALSMITH_LINEAR_SSE41 before including it switches on the SSE4.1 instructions.
*/

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
inline __m128 asmBarrier(__m128 v) {
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
#define SSMF_VF_RSHIFT_K(a, b) _mm_castsi128_pd(_mm_srli_epi64(_mm_castpd_si128(a), b))
#define SSMF_VF_LSHIFT_K(a, b) _mm_castsi128_pd(_mm_slli_epi64(_mm_castpd_si128(a), b))
#ifdef SIGNALSMITH_LINEAR_SSE41
#	define SSMF_VF_ROUND(v) _mm_round_pd(v, _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC)
#	define SSMF_VF_FLOOR(v) _mm_floor_pd(v)
#endif
#if defined(__clang__) || defined(__GNUC__)
inline __m128d asmBarrier(__m128d v) {
	__asm__("" : "+x"(v));
	return v;
}
#	define SSMF_VF_BARRIER(v) asmBarrier(v)
#endif
#define SSMF_NAME(fn) ssmf_##fn##d
#include "../signalsmith-mathfun.h"

// Everything specific to SSE: vector types, memory access, complex (de-)interleaving and arithmetic, plus the SSMF functions above.
// The generic code (`SimdFill` in ./fill.h) works out lane counts and alignment from `sizeof(Float)`.
struct Arch {
	using Float = __m128;
	using Double = __m128d;

	static void splat(__m128 &v, double k) {v = _mm_set1_ps(float(k));}
	static void splat(__m128d &v, double k) {v = _mm_set1_pd(k);}

	static __m128 load(const float *p) {return _mm_load_ps(p);}
	static __m128d load(const double *p) {return _mm_load_pd(p);}
	static __m128 loadu(const float *p) {return _mm_loadu_ps(p);}
	static __m128d loadu(const double *p) {return _mm_loadu_pd(p);}
	static void store(float *p, __m128 v) {_mm_store_ps(p, v);}
	static void store(double *p, __m128d v) {_mm_store_pd(p, v);}
	static void storeu(float *p, __m128 v) {_mm_storeu_ps(p, v);}
	static void storeu(double *p, __m128d v) {_mm_storeu_pd(p, v);}

	// Interleaved complex values: (a, b) are two consecutive vectors of [r0, i0, r1, i1, ...]
	static void deinterleave(__m128 a, __m128 b, __m128 &real, __m128 &imag) {
		real = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0));
		imag = _mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1));
	}
	static void deinterleave(__m128d a, __m128d b, __m128d &real, __m128d &imag) {
		real = _mm_unpacklo_pd(a, b);
		imag = _mm_unpackhi_pd(a, b);
	}
	static void interleave(__m128 real, __m128 imag, __m128 &a, __m128 &b) {
		a = _mm_unpacklo_ps(real, imag);
		b = _mm_unpackhi_ps(real, imag);
	}
	static void interleave(__m128d real, __m128d imag, __m128d &a, __m128d &b) {
		a = _mm_unpacklo_pd(real, imag);
		b = _mm_unpackhi_pd(real, imag);
	}

	// v < 0 ? a : b
#ifdef SIGNALSMITH_LINEAR_SSE41
	// blends on the sign bit, so -0 counts as negative
	static __m128 ifNegative(__m128 v, __m128 a, __m128 b) {return _mm_blendv_ps(b, a, v);}
	static __m128d ifNegative(__m128d v, __m128d a, __m128d b) {return _mm_blendv_pd(b, a, v);}
#else
	static __m128 ifNegative(__m128 v, __m128 a, __m128 b) {
		__m128 mask = _mm_cmplt_ps(v, _mm_setzero_ps());
		return _mm_or_ps(_mm_and_ps(mask, a), _mm_andnot_ps(mask, b));
	}
	static __m128d ifNegative(__m128d v, __m128d a, __m128d b) {
		__m128d mask = _mm_cmplt_pd(v, _mm_setzero_pd());
		return _mm_or_pd(_mm_and_pd(mask, a), _mm_andnot_pd(mask, b));
	}
#endif

#define SIGNALSMITH_SSE_OP1(name, mmFloat, mmDouble) \
	static __m128 name(__m128 a) {return mmFloat(a);} \
	static __m128d name(__m128d a) {return mmDouble(a);}
#define SIGNALSMITH_SSE_OP2(name, mmFloat, mmDouble) \
	static __m128 name(__m128 a, __m128 b) {return mmFloat(a, b);} \
	static __m128d name(__m128d a, __m128d b) {return mmDouble(a, b);}
	SIGNALSMITH_SSE_OP2(add, _mm_add_ps, _mm_add_pd)
	SIGNALSMITH_SSE_OP2(sub, _mm_sub_ps, _mm_sub_pd)
	SIGNALSMITH_SSE_OP2(mul, _mm_mul_ps, _mm_mul_pd)
	SIGNALSMITH_SSE_OP2(div, _mm_div_ps, _mm_div_pd)
	SIGNALSMITH_SSE_OP2(max, _mm_max_ps, _mm_max_pd)
	SIGNALSMITH_SSE_OP2(min, _mm_min_ps, _mm_min_pd)
	SIGNALSMITH_SSE_OP1(sqrt, _mm_sqrt_ps, _mm_sqrt_pd)
#define SIGNALSMITH_SSMF1(name) SIGNALSMITH_SSE_OP1(name, ssmf_##name##f, ssmf_##name##d)
#define SIGNALSMITH_SSMF2(name) SIGNALSMITH_SSE_OP2(name, ssmf_##name##f, ssmf_##name##d)
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
#undef SIGNALSMITH_SSE_OP1
#undef SIGNALSMITH_SSE_OP2
	static void sincos(__m128 v, __m128 &sinOut, __m128 &cosOut) {ssmf_sincosf(v, &sinOut, &cosOut);}
	static void sincos(__m128d v, __m128d &sinOut, __m128d &cosOut) {ssmf_sincosd(v, &sinOut, &cosOut);}
};

#include "./fill.h"
