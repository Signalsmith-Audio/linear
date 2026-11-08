#include <arm_neon.h>

/* SSMF functions and basic operations (`Arch`) for 128-bit ARM NEON, followed by `SimdFill`.

There's no runtime dispatch: NEON is always present on AArch64, and on 32-bit ARM we only get here if the compiler is targeting NEON (`__ARM_NEON`).

AArch64 has everything we want (division, sqrt, FMA, rounding, `float64x2_t`).  32-bit ARM NEON is float-only, without division or sqrt, and FMA/rounding depend on the target:
-	division is done one lane at a time (VFP division is exact, and an estimate+refine wouldn't be)
-	sqrt comes from SSMF's rsqrt()
-	`double` is scalar, using the functions from ./scalar.h

MSVC isn't supported: its NEON vector types are all the same `__n128`, so the float/double overloads in `Arch` would clash.
*/

#ifdef __aarch64__
#	define SIGNALSMITH_LINEAR_NEON_A64
#endif

namespace signalsmith { namespace linear { namespace simd_neon {

#define SIGNALSMITH_NEON_BITS32(op, a, b) vreinterpretq_f32_u32(op(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)))
#define SIGNALSMITH_NEON_BITS64(op, a, b) vreinterpretq_f64_u64(op(vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b)))

#ifndef SIGNALSMITH_LINEAR_NEON_A64
inline float32x4_t divLanes(float32x4_t a, float32x4_t b) {
	float fa[4], fb[4];
	vst1q_f32(fa, a);
	vst1q_f32(fb, b);
	for (int i = 0; i < 4; ++i) fa[i] /= fb[i];
	return vld1q_f32(fa);
}
#endif

#define SSMF_BITS 32
#if defined(SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY32)
#	define SSMF_ACCURACY SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY32
#elif defined(SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY)
#	define SSMF_ACCURACY SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY
#endif
#define SSMF_VF float32x4_t
#define SSMF_VF_0 vdupq_n_f32(0.0f)
#define SSMF_VF_F(v) vdupq_n_f32(float(v))
#define SSMF_VF_I(v) vreinterpretq_f32_s32(vdupq_n_s32(v))
#define SSMF_VF_AND(a, b) SIGNALSMITH_NEON_BITS32(vandq_u32, a, b)
#define SSMF_VF_ANDNOT(a, b) SIGNALSMITH_NEON_BITS32(vbicq_u32, b, a) // BIC is (first & ~second)
#define SSMF_VF_OR(a, b) SIGNALSMITH_NEON_BITS32(vorrq_u32, a, b)
#define SSMF_VF_XOR(a, b) SIGNALSMITH_NEON_BITS32(veorq_u32, a, b)
#define SSMF_VF_NOT(a) vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(a)))
#define SSMF_VF_RSHIFT_K(a, bits) vreinterpretq_f32_u32(vshrq_n_u32(vreinterpretq_u32_f32(a), bits))
#define SSMF_VF_LSHIFT_K(a, bits) vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_f32(a), bits))
#define SSMF_VF_ADD(a, b) vaddq_f32(a, b)
#define SSMF_VF_SUB(a, b) vsubq_f32(a, b)
#define SSMF_VF_MUL(a, b) vmulq_f32(a, b)
#define SSMF_VF_MAX(a, b) vmaxq_f32(a, b)
#define SSMF_VF_MIN(a, b) vminq_f32(a, b)
#define SSMF_VF_ABS(v) vabsq_f32(v)
#define SSMF_VF_NEG(v) vnegq_f32(v)
#define SSMF_VF_COPYSIGN(a, b) vbslq_f32(vdupq_n_u32(0x80000000u), b, a)
#define SSMF_VF_RSQRT_EST(v) vrsqrteq_f32(v)
#define SSMF_RSQRT_EST_BITS 8 // relative error <= 2^-8.25 (FRSQRTE, measured)
#ifdef SIGNALSMITH_LINEAR_NEON_A64
#	define SSMF_VF_DIV(a, b) vdivq_f32(a, b)
#	define SSMF_VF_SQRT(v) vsqrtq_f32(v)
#else
#	define SSMF_VF_DIV(a, b) divLanes(a, b)
#endif
#if defined(SIGNALSMITH_LINEAR_NEON_A64) || defined(__ARM_FEATURE_FMA)
#	define SSMF_VF_FMA(a, b, c) vfmaq_f32(c, a, b)
#	define SSMF_VF_FMAK(a, b, k) vfmaq_f32(vdupq_n_f32(float(k)), a, b)
#endif
#if defined(SIGNALSMITH_LINEAR_NEON_A64) || defined(__ARM_FEATURE_DIRECTED_ROUNDING)
#	define SSMF_VF_ROUND(v) vrndnq_f32(v)
#	define SSMF_VF_FLOOR(v) vrndmq_f32(v)
#endif
#if defined(__clang__) || defined(__GNUC__)
inline float32x4_t asmBarrier(float32x4_t v) {
	__asm__("" : "+w"(v)); // empty ASM block which the compiler doesn't re-associate through
	return v;
}
#	define SSMF_VF_BARRIER(v) asmBarrier(v)
#endif
#define SSMF_NAME(fn) ssmf_##fn##f
#include "../signalsmith-mathfun.h"

#ifdef SIGNALSMITH_LINEAR_NEON_A64
#define SSMF_BITS 64
#if defined(SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY64)
#	define SSMF_ACCURACY SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY64
#elif defined(SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY)
#	define SSMF_ACCURACY SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY
#endif
#define SSMF_VF float64x2_t
#define SSMF_VF_0 vdupq_n_f64(0.0)
#define SSMF_VF_F(v) vdupq_n_f64(double(v))
#define SSMF_VF_I(v) vreinterpretq_f64_s64(vdupq_n_s64(v))
#define SSMF_VF_AND(a, b) SIGNALSMITH_NEON_BITS64(vandq_u64, a, b)
#define SSMF_VF_ANDNOT(a, b) SIGNALSMITH_NEON_BITS64(vbicq_u64, b, a)
#define SSMF_VF_OR(a, b) SIGNALSMITH_NEON_BITS64(vorrq_u64, a, b)
#define SSMF_VF_XOR(a, b) SIGNALSMITH_NEON_BITS64(veorq_u64, a, b)
#define SSMF_VF_NOT(a) vreinterpretq_f64_u32(vmvnq_u32(vreinterpretq_u32_f64(a)))
#define SSMF_VF_RSHIFT_K(a, bits) vreinterpretq_f64_u64(vshrq_n_u64(vreinterpretq_u64_f64(a), bits))
#define SSMF_VF_LSHIFT_K(a, bits) vreinterpretq_f64_u64(vshlq_n_u64(vreinterpretq_u64_f64(a), bits))
#define SSMF_VF_ADD(a, b) vaddq_f64(a, b)
#define SSMF_VF_SUB(a, b) vsubq_f64(a, b)
#define SSMF_VF_MUL(a, b) vmulq_f64(a, b)
#define SSMF_VF_DIV(a, b) vdivq_f64(a, b)
#define SSMF_VF_MAX(a, b) vmaxq_f64(a, b)
#define SSMF_VF_MIN(a, b) vminq_f64(a, b)
#define SSMF_VF_SQRT(v) vsqrtq_f64(v)
#define SSMF_VF_ABS(v) vabsq_f64(v)
#define SSMF_VF_NEG(v) vnegq_f64(v)
#define SSMF_VF_COPYSIGN(a, b) vbslq_f64(vdupq_n_u64(0x8000000000000000ull), b, a)
// Skipping RSQRT_EST: it's only 8 bits, so it takes 3 refinement steps
#define SSMF_VF_FMA(a, b, c) vfmaq_f64(c, a, b)
#define SSMF_VF_FMAK(a, b, k) vfmaq_f64(vdupq_n_f64(double(k)), a, b)
#define SSMF_VF_ROUND(v) vrndnq_f64(v)
#define SSMF_VF_FLOOR(v) vrndmq_f64(v)
#if defined(__clang__) || defined(__GNUC__)
inline float64x2_t asmBarrier(float64x2_t v) {
	__asm__("" : "+w"(v));
	return v;
}
#	define SSMF_VF_BARRIER(v) asmBarrier(v)
#endif
#define SSMF_NAME(fn) ssmf_##fn##d
#include "../signalsmith-mathfun.h"
#endif // SIGNALSMITH_LINEAR_NEON_A64

#undef SIGNALSMITH_NEON_BITS32
#undef SIGNALSMITH_NEON_BITS64

// Everything specific to NEON: vector types, memory access, complex (de-)interleaving and arithmetic, plus the SSMF functions above.
// The generic code (`SimdFill` in ./fill.h) works out lane counts and alignment from `sizeof(Float)`.
#ifdef SIGNALSMITH_LINEAR_NEON_A64
struct Arch {
	using Float = float32x4_t;
	using Double = float64x2_t;
#else
// 32-bit NEON has no double vectors, so `double` goes through the scalar functions (one lane)
struct Arch : public simd_scalar::Arch {
	using Float = float32x4_t;
	using Double = double;
#	define SIGNALSMITH_USING_SCALAR(name) using simd_scalar::Arch::name;
	SIGNALSMITH_USING_SCALAR(splat)
	SIGNALSMITH_USING_SCALAR(load)
	SIGNALSMITH_USING_SCALAR(loadu)
	SIGNALSMITH_USING_SCALAR(store)
	SIGNALSMITH_USING_SCALAR(storeu)
	SIGNALSMITH_USING_SCALAR(deinterleave)
	SIGNALSMITH_USING_SCALAR(interleave)
	SIGNALSMITH_USING_SCALAR(ifNegative)
	SIGNALSMITH_USING_SCALAR(add)
	SIGNALSMITH_USING_SCALAR(sub)
	SIGNALSMITH_USING_SCALAR(mul)
	SIGNALSMITH_USING_SCALAR(div)
	SIGNALSMITH_USING_SCALAR(max)
	SIGNALSMITH_USING_SCALAR(min)
	SIGNALSMITH_USING_SCALAR(sqrt)
	SIGNALSMITH_USING_SCALAR(fma)
	SIGNALSMITH_USING_SCALAR(fms)
	SIGNALSMITH_USING_SCALAR(fnma)
	SIGNALSMITH_USING_SCALAR(fnms)
	SIGNALSMITH_USING_SCALAR(abs)
	SIGNALSMITH_USING_SCALAR(neg)
	SIGNALSMITH_USING_SCALAR(pow2)
	SIGNALSMITH_USING_SCALAR(ceil)
	SIGNALSMITH_USING_SCALAR(floor)
	SIGNALSMITH_USING_SCALAR(trunc)
	SIGNALSMITH_USING_SCALAR(round)
	SIGNALSMITH_USING_SCALAR(cbrt)
	SIGNALSMITH_USING_SCALAR(rsqrt)
	SIGNALSMITH_USING_SCALAR(log)
	SIGNALSMITH_USING_SCALAR(log2)
	SIGNALSMITH_USING_SCALAR(log10)
	SIGNALSMITH_USING_SCALAR(exp)
	SIGNALSMITH_USING_SCALAR(exp2)
	SIGNALSMITH_USING_SCALAR(sin)
	SIGNALSMITH_USING_SCALAR(cos)
	SIGNALSMITH_USING_SCALAR(tan)
	SIGNALSMITH_USING_SCALAR(asin)
	SIGNALSMITH_USING_SCALAR(acos)
	SIGNALSMITH_USING_SCALAR(atan)
	SIGNALSMITH_USING_SCALAR(sinh)
	SIGNALSMITH_USING_SCALAR(cosh)
	SIGNALSMITH_USING_SCALAR(tanh)
	SIGNALSMITH_USING_SCALAR(erf)
	SIGNALSMITH_USING_SCALAR(erfc)
	SIGNALSMITH_USING_SCALAR(dim)
	SIGNALSMITH_USING_SCALAR(copysign)
	SIGNALSMITH_USING_SCALAR(pow)
	SIGNALSMITH_USING_SCALAR(atan2)
	SIGNALSMITH_USING_SCALAR(hypot)
	SIGNALSMITH_USING_SCALAR(sincos)
#	undef SIGNALSMITH_USING_SCALAR
#endif

	static void splat(float32x4_t &v, double k) {v = vdupq_n_f32(float(k));}
	// NEON loads/stores don't need alignment, so these are the same
	static float32x4_t load(const float *p) {return vld1q_f32(p);}
	static float32x4_t loadu(const float *p) {return vld1q_f32(p);}
	static void store(float *p, float32x4_t v) {vst1q_f32(p, v);}
	static void storeu(float *p, float32x4_t v) {vst1q_f32(p, v);}

	// Interleaved complex values: (a, b) are two consecutive vectors of [r0, i0, r1, i1, ...]
	static void deinterleave(float32x4_t a, float32x4_t b, float32x4_t &real, float32x4_t &imag) {
		float32x4x2_t ri = vuzpq_f32(a, b);
		real = ri.val[0];
		imag = ri.val[1];
	}
	static void interleave(float32x4_t real, float32x4_t imag, float32x4_t &a, float32x4_t &b) {
		float32x4x2_t ab = vzipq_f32(real, imag);
		a = ab.val[0];
		b = ab.val[1];
	}

	// v < 0 ? a : b, on the sign bit (so -0 counts as negative) to match the SSE4.1/AVX blends
	static float32x4_t ifNegative(float32x4_t v, float32x4_t a, float32x4_t b) {
		uint32x4_t mask = vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_f32(v), 31));
		return vbslq_f32(mask, a, b);
	}

	static float32x4_t add(float32x4_t a, float32x4_t b) {return vaddq_f32(a, b);}
	static float32x4_t sub(float32x4_t a, float32x4_t b) {return vsubq_f32(a, b);}
	static float32x4_t mul(float32x4_t a, float32x4_t b) {return vmulq_f32(a, b);}
#ifdef SIGNALSMITH_LINEAR_NEON_A64
	static float32x4_t div(float32x4_t a, float32x4_t b) {return vdivq_f32(a, b);}
	static float32x4_t sqrt(float32x4_t a) {return vsqrtq_f32(a);}
#else
	static float32x4_t div(float32x4_t a, float32x4_t b) {return divLanes(a, b);}
	static float32x4_t sqrt(float32x4_t a) {return ssmf_sqrtf(a);}
#endif
	static float32x4_t max(float32x4_t a, float32x4_t b) {return vmaxq_f32(a, b);}
	static float32x4_t min(float32x4_t a, float32x4_t b) {return vminq_f32(a, b);}
	// a*b + c, a*b - c, c - a*b and -(a*b + c)
#if defined(SIGNALSMITH_LINEAR_NEON_A64) || defined(__ARM_FEATURE_FMA)
	static float32x4_t fma(float32x4_t a, float32x4_t b, float32x4_t c) {return vfmaq_f32(c, a, b);}
	static float32x4_t fms(float32x4_t a, float32x4_t b, float32x4_t c) {return vnegq_f32(vfmsq_f32(c, a, b));}
	static float32x4_t fnma(float32x4_t a, float32x4_t b, float32x4_t c) {return vfmsq_f32(c, a, b);}
	static float32x4_t fnms(float32x4_t a, float32x4_t b, float32x4_t c) {return vnegq_f32(vfmaq_f32(c, a, b));}
#else
	// not fused
	static float32x4_t fma(float32x4_t a, float32x4_t b, float32x4_t c) {return vmlaq_f32(c, a, b);}
	static float32x4_t fms(float32x4_t a, float32x4_t b, float32x4_t c) {return vnegq_f32(vmlsq_f32(c, a, b));}
	static float32x4_t fnma(float32x4_t a, float32x4_t b, float32x4_t c) {return vmlsq_f32(c, a, b);}
	static float32x4_t fnms(float32x4_t a, float32x4_t b, float32x4_t c) {return vnegq_f32(vmlaq_f32(c, a, b));}
#endif

#ifdef SIGNALSMITH_LINEAR_NEON_A64
	static void splat(float64x2_t &v, double k) {v = vdupq_n_f64(k);}
	static float64x2_t load(const double *p) {return vld1q_f64(p);}
	static float64x2_t loadu(const double *p) {return vld1q_f64(p);}
	static void store(double *p, float64x2_t v) {vst1q_f64(p, v);}
	static void storeu(double *p, float64x2_t v) {vst1q_f64(p, v);}

	static void deinterleave(float64x2_t a, float64x2_t b, float64x2_t &real, float64x2_t &imag) {
		real = vzip1q_f64(a, b);
		imag = vzip2q_f64(a, b);
	}
	static void interleave(float64x2_t real, float64x2_t imag, float64x2_t &a, float64x2_t &b) {
		a = vzip1q_f64(real, imag);
		b = vzip2q_f64(real, imag);
	}

	static float64x2_t ifNegative(float64x2_t v, float64x2_t a, float64x2_t b) {
		uint64x2_t mask = vreinterpretq_u64_s64(vshrq_n_s64(vreinterpretq_s64_f64(v), 63));
		return vbslq_f64(mask, a, b);
	}

	static float64x2_t add(float64x2_t a, float64x2_t b) {return vaddq_f64(a, b);}
	static float64x2_t sub(float64x2_t a, float64x2_t b) {return vsubq_f64(a, b);}
	static float64x2_t mul(float64x2_t a, float64x2_t b) {return vmulq_f64(a, b);}
	static float64x2_t div(float64x2_t a, float64x2_t b) {return vdivq_f64(a, b);}
	static float64x2_t max(float64x2_t a, float64x2_t b) {return vmaxq_f64(a, b);}
	static float64x2_t min(float64x2_t a, float64x2_t b) {return vminq_f64(a, b);}
	static float64x2_t sqrt(float64x2_t a) {return vsqrtq_f64(a);}
	static float64x2_t fma(float64x2_t a, float64x2_t b, float64x2_t c) {return vfmaq_f64(c, a, b);}
	static float64x2_t fms(float64x2_t a, float64x2_t b, float64x2_t c) {return vnegq_f64(vfmsq_f64(c, a, b));}
	static float64x2_t fnma(float64x2_t a, float64x2_t b, float64x2_t c) {return vfmsq_f64(c, a, b);}
	static float64x2_t fnms(float64x2_t a, float64x2_t b, float64x2_t c) {return vnegq_f64(vfmaq_f64(c, a, b));}
#	define SIGNALSMITH_SSMF1(name) \
	static float32x4_t name(float32x4_t a) {return ssmf_##name##f(a);} \
	static float64x2_t name(float64x2_t a) {return ssmf_##name##d(a);}
#	define SIGNALSMITH_SSMF2(name) \
	static float32x4_t name(float32x4_t a, float32x4_t b) {return ssmf_##name##f(a, b);} \
	static float64x2_t name(float64x2_t a, float64x2_t b) {return ssmf_##name##d(a, b);}
	static void sincos(float64x2_t v, float64x2_t &sinOut, float64x2_t &cosOut) {ssmf_sincosd(v, &sinOut, &cosOut);}
#else
#	define SIGNALSMITH_SSMF1(name) \
	static float32x4_t name(float32x4_t a) {return ssmf_##name##f(a);}
#	define SIGNALSMITH_SSMF2(name) \
	static float32x4_t name(float32x4_t a, float32x4_t b) {return ssmf_##name##f(a, b);}
#endif
	SIGNALSMITH_SSMF1(abs)
	SIGNALSMITH_SSMF1(neg)
	SIGNALSMITH_SSMF1(pow2)
	SIGNALSMITH_SSMF1(ceil)
	SIGNALSMITH_SSMF1(floor)
	SIGNALSMITH_SSMF1(trunc)
	SIGNALSMITH_SSMF1(round)
	SIGNALSMITH_SSMF1(cbrt)
	SIGNALSMITH_SSMF1(rsqrt)
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
	static void sincos(float32x4_t v, float32x4_t &sinOut, float32x4_t &cosOut) {ssmf_sincosf(v, &sinOut, &cosOut);}
};

#include "./fill.h"

}}}; // namespace

#undef SIGNALSMITH_LINEAR_NEON_A64
