// WebAssembly SIMD128 (Wasm 2.0), with relaxed SIMD (Wasm 3.0) for FMA, min/max and lane-select.  I'd recommend compiling with `-mrelaxed-simd` since it's widely supported
// This *should* work on wasm64 (memory64) too - it compiles OK but hasn't been tested.

#include <wasm_simd128.h>

// Use Relaxed SIMD if it's enabled
#if defined(__wasm_relaxed_simd__)
#	define SIGNALSMITH_LINEAR_WASM_RELAXED
#endif

namespace signalsmith { namespace linear { namespace simd_wasm_simd128 {

// Everything specific to WASM: vector types, memory access, complex (de-)interleaving and arithmetic, plus the SSMF functions (included as static members).
// There's only one vector type (`v128_t`), so float/double are only distinguished by which struct they're in.
struct ArchFloat {
	using V = v128_t;
	using Scalar = float;

	static V splat(double k) {return wasm_f32x4_splat(float(k));}
	// WASM loads/stores don't need alignment, so these are the same
	static V load(const float *p) {return wasm_v128_load(p);}
	static V loadu(const float *p) {return wasm_v128_load(p);}
	static void store(float *p, V v) {wasm_v128_store(p, v);}
	static void storeu(float *p, V v) {wasm_v128_store(p, v);}

	// Interleaved complex values: (a, b) are two consecutive vectors of [r0, i0, r1, i1, ...]
	static void deinterleave(V a, V b, V &real, V &imag) {
		real = wasm_i32x4_shuffle(a, b, 0, 2, 4, 6);
		imag = wasm_i32x4_shuffle(a, b, 1, 3, 5, 7);
	}
	static void interleave(V real, V imag, V &a, V &b) {
		a = wasm_i32x4_shuffle(real, imag, 0, 4, 1, 5);
		b = wasm_i32x4_shuffle(real, imag, 2, 6, 3, 7);
	}

	// v < 0 ? a : b, on the sign bit (so -0 counts as negative) to match the SSE4.1/AVX blends
	static V ifNegative(V v, V a, V b) {
		V mask = wasm_i32x4_shr(v, 31);
#ifdef SIGNALSMITH_LINEAR_WASM_RELAXED
		return wasm_i32x4_relaxed_laneselect(a, b, mask); // deterministic, since every mask lane is all-0 or all-1
#else
		return wasm_v128_bitselect(a, b, mask);
#endif
	}

	static V add(V a, V b) {return wasm_f32x4_add(a, b);}
	static V sub(V a, V b) {return wasm_f32x4_sub(a, b);}
	static V mul(V a, V b) {return wasm_f32x4_mul(a, b);}
	static V div(V a, V b) {return wasm_f32x4_div(a, b);}
	// Strict min/max handle NaN and ±0 exactly, which is slow on x86/ARM.  We don't care about those (same as -ffast-math), so pmin/pmax are fine without relaxed SIMD.
#ifdef SIGNALSMITH_LINEAR_WASM_RELAXED
	static V max(V a, V b) {return wasm_f32x4_relaxed_max(a, b);}
	static V min(V a, V b) {return wasm_f32x4_relaxed_min(a, b);}
#else
	static V max(V a, V b) {return wasm_f32x4_pmax(a, b);}
	static V min(V a, V b) {return wasm_f32x4_pmin(a, b);}
#endif
	// a*b + c, a*b - c, c - a*b and -(a*b + c)
#ifdef SIGNALSMITH_LINEAR_WASM_RELAXED
	// fused where the hardware supports it
	static V fma(V a, V b, V c) {return wasm_f32x4_relaxed_madd(a, b, c);}
	static V fms(V a, V b, V c) {return wasm_f32x4_neg(wasm_f32x4_relaxed_nmadd(a, b, c));}
	static V fnma(V a, V b, V c) {return wasm_f32x4_relaxed_nmadd(a, b, c);}
	static V fnms(V a, V b, V c) {return wasm_f32x4_neg(wasm_f32x4_relaxed_madd(a, b, c));}
#else
	// not fused
	static V fma(V a, V b, V c) {return wasm_f32x4_add(wasm_f32x4_mul(a, b), c);}
	static V fms(V a, V b, V c) {return wasm_f32x4_sub(wasm_f32x4_mul(a, b), c);}
	static V fnma(V a, V b, V c) {return wasm_f32x4_sub(c, wasm_f32x4_mul(a, b));}
	static V fnms(V a, V b, V c) {return wasm_f32x4_neg(wasm_f32x4_add(wasm_f32x4_mul(a, b), c));}
#endif

#define SSMF_BITS 32
#if defined(SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY32)
#	define SSMF_ACCURACY SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY32
#elif defined(SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY)
#	define SSMF_ACCURACY SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY
#endif
#define SSMF_VF v128_t
#define SSMF_VF_0 wasm_f32x4_const_splat(0.0f)
#define SSMF_VF_F(v) wasm_f32x4_splat(float(v))
#define SSMF_VF_I(v) wasm_i32x4_splat(v)
#define SSMF_VF_AND(a, b) wasm_v128_and(a, b)
#define SSMF_VF_ANDNOT(a, b) wasm_v128_andnot(b, a) // v128.andnot is (first & ~second)
#define SSMF_VF_OR(a, b) wasm_v128_or(a, b)
#define SSMF_VF_XOR(a, b) wasm_v128_xor(a, b)
#define SSMF_VF_NOT(a) wasm_v128_not(a)
#define SSMF_VF_RSHIFT_K(a, bits) wasm_u32x4_shr(a, bits)
#define SSMF_VF_LSHIFT_K(a, bits) wasm_i32x4_shl(a, bits)
#define SSMF_VF_ADD(a, b) wasm_f32x4_add(a, b)
#define SSMF_VF_SUB(a, b) wasm_f32x4_sub(a, b)
#define SSMF_VF_MUL(a, b) wasm_f32x4_mul(a, b)
#define SSMF_VF_DIV(a, b) wasm_f32x4_div(a, b)
#define SSMF_VF_MAX(a, b) max(a, b)
#define SSMF_VF_MIN(a, b) min(a, b)
#define SSMF_VF_SQRT(v) wasm_f32x4_sqrt(v)
#define SSMF_VF_ABS(v) wasm_f32x4_abs(v)
#define SSMF_VF_NEG(v) wasm_f32x4_neg(v)
#define SSMF_VF_COPYSIGN(a, b) wasm_v128_bitselect(b, a, wasm_i32x4_const_splat(int32_t(0x80000000u)))
#define SSMF_VF_ROUND(v) wasm_f32x4_nearest(v)
#define SSMF_VF_FLOOR(v) wasm_f32x4_floor(v)
#ifdef SIGNALSMITH_LINEAR_WASM_RELAXED
#	define SSMF_VF_FMA(a, b, c) wasm_f32x4_relaxed_madd(a, b, c)
#	define SSMF_VF_FMAK(a, b, k) wasm_f32x4_relaxed_madd(a, b, wasm_f32x4_splat(float(k)))
#endif
#define SSMF_NAME(fn) fn // static member functions
#define SSMF_INLINE static inline
#include "../signalsmith-mathfun.h"
};

struct ArchDouble {
	using V = v128_t;
	using Scalar = double;

	static V splat(double k) {return wasm_f64x2_splat(k);}
	static V load(const double *p) {return wasm_v128_load(p);}
	static V loadu(const double *p) {return wasm_v128_load(p);}
	static void store(double *p, V v) {wasm_v128_store(p, v);}
	static void storeu(double *p, V v) {wasm_v128_store(p, v);}

	static void deinterleave(V a, V b, V &real, V &imag) {
		real = wasm_i64x2_shuffle(a, b, 0, 2);
		imag = wasm_i64x2_shuffle(a, b, 1, 3);
	}
	static void interleave(V real, V imag, V &a, V &b) {
		a = wasm_i64x2_shuffle(real, imag, 0, 2);
		b = wasm_i64x2_shuffle(real, imag, 1, 3);
	}

	static V ifNegative(V v, V a, V b) {
		V mask = wasm_i64x2_shr(v, 63);
#ifdef SIGNALSMITH_LINEAR_WASM_RELAXED
		return wasm_i64x2_relaxed_laneselect(a, b, mask);
#else
		return wasm_v128_bitselect(a, b, mask);
#endif
	}

	static V add(V a, V b) {return wasm_f64x2_add(a, b);}
	static V sub(V a, V b) {return wasm_f64x2_sub(a, b);}
	static V mul(V a, V b) {return wasm_f64x2_mul(a, b);}
	static V div(V a, V b) {return wasm_f64x2_div(a, b);}
#ifdef SIGNALSMITH_LINEAR_WASM_RELAXED
	static V max(V a, V b) {return wasm_f64x2_relaxed_max(a, b);}
	static V min(V a, V b) {return wasm_f64x2_relaxed_min(a, b);}
	static V fma(V a, V b, V c) {return wasm_f64x2_relaxed_madd(a, b, c);}
	static V fms(V a, V b, V c) {return wasm_f64x2_neg(wasm_f64x2_relaxed_nmadd(a, b, c));}
	static V fnma(V a, V b, V c) {return wasm_f64x2_relaxed_nmadd(a, b, c);}
	static V fnms(V a, V b, V c) {return wasm_f64x2_neg(wasm_f64x2_relaxed_madd(a, b, c));}
#else
	static V max(V a, V b) {return wasm_f64x2_pmax(a, b);}
	static V min(V a, V b) {return wasm_f64x2_pmin(a, b);}
	static V fma(V a, V b, V c) {return wasm_f64x2_add(wasm_f64x2_mul(a, b), c);}
	static V fms(V a, V b, V c) {return wasm_f64x2_sub(wasm_f64x2_mul(a, b), c);}
	static V fnma(V a, V b, V c) {return wasm_f64x2_sub(c, wasm_f64x2_mul(a, b));}
	static V fnms(V a, V b, V c) {return wasm_f64x2_neg(wasm_f64x2_add(wasm_f64x2_mul(a, b), c));}
#endif

#define SSMF_BITS 64
#if defined(SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY64)
#	define SSMF_ACCURACY SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY64
#elif defined(SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY)
#	define SSMF_ACCURACY SIGNALSMITH_LINEAR_INTRINSICS_ACCURACY
#endif
#define SSMF_VF v128_t
#define SSMF_VF_0 wasm_f64x2_const_splat(0.0)
#define SSMF_VF_F(v) wasm_f64x2_splat(double(v))
#define SSMF_VF_I(v) wasm_i64x2_splat(v)
#define SSMF_VF_AND(a, b) wasm_v128_and(a, b)
#define SSMF_VF_ANDNOT(a, b) wasm_v128_andnot(b, a)
#define SSMF_VF_OR(a, b) wasm_v128_or(a, b)
#define SSMF_VF_XOR(a, b) wasm_v128_xor(a, b)
#define SSMF_VF_NOT(a) wasm_v128_not(a)
#define SSMF_VF_RSHIFT_K(a, bits) wasm_u64x2_shr(a, bits)
#define SSMF_VF_LSHIFT_K(a, bits) wasm_i64x2_shl(a, bits)
#define SSMF_VF_ADD(a, b) wasm_f64x2_add(a, b)
#define SSMF_VF_SUB(a, b) wasm_f64x2_sub(a, b)
#define SSMF_VF_MUL(a, b) wasm_f64x2_mul(a, b)
#define SSMF_VF_DIV(a, b) wasm_f64x2_div(a, b)
#define SSMF_VF_MAX(a, b) max(a, b)
#define SSMF_VF_MIN(a, b) min(a, b)
#define SSMF_VF_SQRT(v) wasm_f64x2_sqrt(v)
#define SSMF_VF_ABS(v) wasm_f64x2_abs(v)
#define SSMF_VF_NEG(v) wasm_f64x2_neg(v)
#define SSMF_VF_COPYSIGN(a, b) wasm_v128_bitselect(b, a, wasm_i64x2_const_splat(int64_t(0x8000000000000000ull)))
#define SSMF_VF_ROUND(v) wasm_f64x2_nearest(v)
#define SSMF_VF_FLOOR(v) wasm_f64x2_floor(v)
#ifdef SIGNALSMITH_LINEAR_WASM_RELAXED
#	define SSMF_VF_FMA(a, b, c) wasm_f64x2_relaxed_madd(a, b, c)
#	define SSMF_VF_FMAK(a, b, k) wasm_f64x2_relaxed_madd(a, b, wasm_f64x2_splat(double(k)))
#endif
#define SSMF_NAME(fn) fn // static member functions
#define SSMF_INLINE static inline
#include "../signalsmith-mathfun.h"
};

#include "./fill.h"

}}}; // namespace

#undef SIGNALSMITH_LINEAR_WASM_RELAXED
