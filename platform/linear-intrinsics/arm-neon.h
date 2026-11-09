// 128-bit ARM NEON.  This covers both the AArch64 version (which has everything, including sqrt/rounding/...) and the 32-bit version (which needs to explicitly target NEON, and is float-only, so double falls back to scalar)

#include <arm_neon.h>

#if defined(__aarch64__) || defined(_M_ARM64)
#	define SIGNALSMITH_LINEAR_NEON_A64
#endif

namespace signalsmith { namespace linear { namespace simd_neon {

#define SIGNALSMITH_NEON_BITS32(op, a, b) vreinterpretq_f32_u32(op(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)))
#define SIGNALSMITH_NEON_BITS64(op, a, b) vreinterpretq_f64_u64(op(vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b)))

#if defined(__clang__) || defined(__GNUC__)
inline float32x4_t asmBarrier(float32x4_t v) {
	__asm__("" : "+w"(v)); // empty ASM block which the compiler doesn't re-associate through
	return v;
}
#endif

#if defined(SIGNALSMITH_LINEAR_NEON_A64) && (defined(__clang__) || defined(__GNUC__))
inline float64x2_t asmBarrier(float64x2_t v) {
	__asm__("" : "+w"(v));
	return v;
}
#endif

// Everything specific to NEON: vector types, memory access, complex (de-)interleaving and arithmetic, plus the SSMF functions (included as static members).
// The generic code (`SimdFill` in ./fill.h) works out lane counts and alignment from `sizeof(V)`.
struct ArchFloat {
	using V = float32x4_t;
	using Scalar = float;

	static V splat(double k) {return vdupq_n_f32(float(k));}
	// NEON loads/stores don't need alignment, so these are the same
	static V load(const float *p) {return vld1q_f32(p);}
	static V loadu(const float *p) {return vld1q_f32(p);}
	static void store(float *p, V v) {vst1q_f32(p, v);}
	static void storeu(float *p, V v) {vst1q_f32(p, v);}

	// Interleaved complex values: (a, b) are two consecutive vectors of [r0, i0, r1, i1, ...]
	static void deinterleave(V a, V b, V &real, V &imag) {
		float32x4x2_t ri = vuzpq_f32(a, b);
		real = ri.val[0];
		imag = ri.val[1];
	}
	static void interleave(V real, V imag, V &a, V &b) {
		float32x4x2_t ab = vzipq_f32(real, imag);
		a = ab.val[0];
		b = ab.val[1];
	}

	// v < 0 ? a : b, on the sign bit (so -0 counts as negative) to match the SSE4.1/AVX blends
	static V ifNegative(V v, V a, V b) {
		uint32x4_t mask = vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_f32(v), 31));
		return vbslq_f32(mask, a, b);
	}

	static V add(V a, V b) {return vaddq_f32(a, b);}
	static V sub(V a, V b) {return vsubq_f32(a, b);}
	static V mul(V a, V b) {return vmulq_f32(a, b);}
#ifdef SIGNALSMITH_LINEAR_NEON_A64
	static V div(V a, V b) {return vdivq_f32(a, b);}
#else
	// 32-bit NEON doesn't have division - write to the stack, do as calar, and read back out
	static V div(V a, V b) {
		float fa[4], fb[4];
		vst1q_f32(fa, a);
		vst1q_f32(fb, b);
		for (int i = 0; i < 4; ++i) fa[i] /= fb[i];
		return vld1q_f32(fa);
	}
#endif
	static V max(V a, V b) {return vmaxq_f32(a, b);}
	static V min(V a, V b) {return vminq_f32(a, b);}
	// a*b + c, a*b - c, c - a*b and -(a*b + c)
#if defined(SIGNALSMITH_LINEAR_NEON_A64) || defined(__ARM_FEATURE_FMA)
	static V fma(V a, V b, V c) {return vfmaq_f32(c, a, b);}
	static V fms(V a, V b, V c) {return vnegq_f32(vfmsq_f32(c, a, b));}
	static V fnma(V a, V b, V c) {return vfmsq_f32(c, a, b);}
	static V fnms(V a, V b, V c) {return vnegq_f32(vfmaq_f32(c, a, b));}
#else
	// not fused
	static V fma(V a, V b, V c) {return vmlaq_f32(c, a, b);}
	static V fms(V a, V b, V c) {return vnegq_f32(vmlsq_f32(c, a, b));}
	static V fnma(V a, V b, V c) {return vmlsq_f32(c, a, b);}
	static V fnms(V a, V b, V c) {return vnegq_f32(vmlaq_f32(c, a, b));}
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
#	define SSMF_VF_DIV(a, b) div(a, b)
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
#	define SSMF_VF_BARRIER(v) asmBarrier(v)
#endif
#define SSMF_NAME(fn) fn // static member functions
#define SSMF_INLINE static inline
#include "../signalsmith-mathfun.h"
};

#ifdef SIGNALSMITH_LINEAR_NEON_A64
struct ArchDouble {
	using V = float64x2_t;
	using Scalar = double;

	static V splat(double k) {return vdupq_n_f64(k);}
	static V load(const double *p) {return vld1q_f64(p);}
	static V loadu(const double *p) {return vld1q_f64(p);}
	static void store(double *p, V v) {vst1q_f64(p, v);}
	static void storeu(double *p, V v) {vst1q_f64(p, v);}

	static void deinterleave(V a, V b, V &real, V &imag) {
		real = vzip1q_f64(a, b);
		imag = vzip2q_f64(a, b);
	}
	static void interleave(V real, V imag, V &a, V &b) {
		a = vzip1q_f64(real, imag);
		b = vzip2q_f64(real, imag);
	}

	static V ifNegative(V v, V a, V b) {
		uint64x2_t mask = vreinterpretq_u64_s64(vshrq_n_s64(vreinterpretq_s64_f64(v), 63));
		return vbslq_f64(mask, a, b);
	}

	static V add(V a, V b) {return vaddq_f64(a, b);}
	static V sub(V a, V b) {return vsubq_f64(a, b);}
	static V mul(V a, V b) {return vmulq_f64(a, b);}
	static V div(V a, V b) {return vdivq_f64(a, b);}
	static V max(V a, V b) {return vmaxq_f64(a, b);}
	static V min(V a, V b) {return vminq_f64(a, b);}
	static V fma(V a, V b, V c) {return vfmaq_f64(c, a, b);}
	static V fms(V a, V b, V c) {return vnegq_f64(vfmsq_f64(c, a, b));}
	static V fnma(V a, V b, V c) {return vfmsq_f64(c, a, b);}
	static V fnms(V a, V b, V c) {return vnegq_f64(vfmaq_f64(c, a, b));}

#	define SSMF_BITS 64
#	if defined(SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY64)
#		define SSMF_ACCURACY SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY64
#	elif defined(SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY)
#		define SSMF_ACCURACY SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY
#	endif
#	define SSMF_VF float64x2_t
#	define SSMF_VF_0 vdupq_n_f64(0.0)
#	define SSMF_VF_F(v) vdupq_n_f64(double(v))
#	define SSMF_VF_I(v) vreinterpretq_f64_s64(vdupq_n_s64(v))
#	define SSMF_VF_AND(a, b) SIGNALSMITH_NEON_BITS64(vandq_u64, a, b)
#	define SSMF_VF_ANDNOT(a, b) SIGNALSMITH_NEON_BITS64(vbicq_u64, b, a)
#	define SSMF_VF_OR(a, b) SIGNALSMITH_NEON_BITS64(vorrq_u64, a, b)
#	define SSMF_VF_XOR(a, b) SIGNALSMITH_NEON_BITS64(veorq_u64, a, b)
#	define SSMF_VF_NOT(a) vreinterpretq_f64_u32(vmvnq_u32(vreinterpretq_u32_f64(a)))
#	define SSMF_VF_RSHIFT_K(a, bits) vreinterpretq_f64_u64(vshrq_n_u64(vreinterpretq_u64_f64(a), bits))
#	define SSMF_VF_LSHIFT_K(a, bits) vreinterpretq_f64_u64(vshlq_n_u64(vreinterpretq_u64_f64(a), bits))
#	define SSMF_VF_ADD(a, b) vaddq_f64(a, b)
#	define SSMF_VF_SUB(a, b) vsubq_f64(a, b)
#	define SSMF_VF_MUL(a, b) vmulq_f64(a, b)
#	define SSMF_VF_DIV(a, b) vdivq_f64(a, b)
#	define SSMF_VF_MAX(a, b) vmaxq_f64(a, b)
#	define SSMF_VF_MIN(a, b) vminq_f64(a, b)
#	define SSMF_VF_SQRT(v) vsqrtq_f64(v)
#	define SSMF_VF_ABS(v) vabsq_f64(v)
#	define SSMF_VF_NEG(v) vnegq_f64(v)
#	define SSMF_VF_COPYSIGN(a, b) vbslq_f64(vdupq_n_u64(0x8000000000000000ull), b, a)
// Skipping RSQRT_EST: it's only 8 bits, so it takes 3 refinement steps
#	define SSMF_VF_FMA(a, b, c) vfmaq_f64(c, a, b)
#	define SSMF_VF_FMAK(a, b, k) vfmaq_f64(vdupq_n_f64(double(k)), a, b)
#	define SSMF_VF_ROUND(v) vrndnq_f64(v)
#	define SSMF_VF_FLOOR(v) vrndmq_f64(v)
#	if defined(__clang__) || defined(__GNUC__)
#		define SSMF_VF_BARRIER(v) asmBarrier(v)
#	endif
#	define SSMF_NAME(fn) fn // static member functions
#	define SSMF_INLINE static inline
#	include "../signalsmith-mathfun.h"
};
#else
// 32-bit NEON has no double vectors, so `double` uses the scalar version (one lane)
using ArchDouble = simd_scalar::ArchDouble;
#endif

#undef SIGNALSMITH_NEON_BITS32
#undef SIGNALSMITH_NEON_BITS64

#include "./fill.h"

}}}; // namespace

#undef SIGNALSMITH_LINEAR_NEON_A64
