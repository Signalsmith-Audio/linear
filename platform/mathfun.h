/* Copyright 2026 Geraint Luff / Signalsmith Audio Ltd.

This header defines floating-point functions, given #defines for the basic operations (constants, arithmetic, bitwise).

It's a "function header" - explicitly designed to be included multiple times if needed.  You define various `SSMF_...` macros, and these are un-defined by this header.  The functions are defined using a naming macro, so you can define your own prefix/suffix, e.g. `SSMF_NAME(abs)` -> `prefix_abs_suffix`.

Some of these fundamental operations are required (and they'll error), but others are optional and the implementation will work around that.
*/

#ifndef SSMF_ACCURACY
#	if SSMF_BITS == 32
#		define SSMF_ACCURACY 22
#	elif SSMF_BITS == 64
#		define SSMF_ACCURACY 48
#	else
#		error "need SSMF_ACCURACY"
#	endif
#endif

// make constants from scalar literals
#ifndef SSMF_F
#if SSMF_BITS == 32
#	define SSMF_F(v) float(v)
#elif SSMF_BITS == 64
#	define SSMF_F(v) double(v)
#else
#	error "if SSMF_F() isn't defined, SSMF_BITS must be 32 (float) or 64 (double)"
#endif
#endif
// types for the SIMD vector
#ifndef SSMF_VF
#	error "SSMF_VF not defined"
#endif
// evaluates to a vector full of 0s
#ifndef SSMF_VF_0
#	error "SSMF_VF_0 not defined"
#endif
// create a vector from a (float) scalar
#ifndef SSMF_VF_F
#	error "SSMF_VF_F() not defined"
#endif
// create a vector from an (int) scalar
#ifndef SSMF_VF_I
#	error "SSMF_VF_I() not defined"
#endif
// a & b
#ifndef SSMF_VF_AND
#	error "SSMF_VF_AND() not defined"
#endif
// ~a & b
#ifndef SSMF_VF_ANDNOT
#	error "SSMF_VF_ANDNOT() not defined"
#endif
// a | b
#ifndef SSMF_VF_OR
#	error "SSMF_VF_OR() not defined"
#endif
// a ^ b
#ifndef SSMF_VF_XOR
#	error "SSMF_VF_XOR() not defined"
#endif
// a + b
#ifndef SSMF_VF_ADD
#	error "SSMF_VF_ADD() not defined"
#endif
// a - b
#ifndef SSMF_VF_SUB
#	error "SSMF_VF_SUB() not defined"
#endif
// a * b
#ifndef SSMF_VF_MUL
#	error "SSMF_VF_MUL() not defined"
#endif
// a / b
#ifndef SSMF_VF_DIV
#	error "SSMF_VF_DIV() not defined"
#endif
// max(a, b)
#ifndef SSMF_VF_MAX
#	error "SSMF_VF_MAX() not defined"
#endif
// min(a, b)
#ifndef SSMF_VF_MIN
#	error "SSMF_VF_MIN() not defined"
#endif
// a >> bits
#ifndef SSMF_VF_RSHIFT_K
#	error "SSMF_VF_RSHIFT_K() not defined"
#endif

// If we don't have "interpret this as integers, and then convert to float", we can implement it
#ifndef SSMF_VF_ITOF
#	if SSMF_BITS == 32
#		define SSMF_VF_ITOF_MAGIC 16777216.0f
#elif SSMF_BITS == 64
#		define SSMF_VF_ITOF_MAGIC 4503599627370496.0
#else
#		error "SSMF_VF_ITOF is required unless it's 32/64-bit IEEE float"
#endif
// The "magic number" is a float such that the mantissa ends up incrementing by integers
#	define SSMF_VF_ITOF(v) SSMF_VF_SUB( \
		SSMF_VF_OR(SSMF_VF_F(SSMF_VF_ITOF_MAGIC), v), \
		SSMF_VF_F(SSMF_VF_ITOF_MAGIC) \
	)
#endif

// Optional: a*b + k, for constant literal `k`
#ifndef SSMF_FMAK
#	define SSMF_FMAK(a, b, k) SSMF_VF_ADD(SSMF_VF_MUL(a, b), SSMF_VF_F(SSMF_F(k)))
#endif

// Polynomials (Horner's method) with constant coefficients
#define SSMF_VF_POLY0(x, c0) SSMF_VF_F(SSMF_F(c0))
#define SSMF_VF_POLY1(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY0(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY2(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY1(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY3(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY2(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY4(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY3(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY5(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY4(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY6(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY5(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY7(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY6(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY8(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY7(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY9(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY8(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY10(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY9(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY11(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY10(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY12(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY11(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY13(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY12(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY14(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY13(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY15(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY14(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY16(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY15(x, __VA_ARGS__), x, c0)
// Monic variants, where the highest-order coefficient is 1 (so we can skip the multiplication)
#define SSMF_VF_POLY1M(x, c0) SSMF_VF_ADD(x, SSMF_VF_F(SSMF_F(c0)))
#define SSMF_VF_POLY2M(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY1M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY3M(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY2M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY4M(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY3M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY5M(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY4M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY6M(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY5M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY7M(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY6M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY8M(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY7M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY9M(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY8M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY10M(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY9M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY11M(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY10M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY12M(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY11M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY13M(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY12M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY14M(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY13M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY15M(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY14M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY16M(x, c0, ...) SSMF_FMAK(SSMF_VF_POLY15M(x, __VA_ARGS__), x, c0)

// Optional
#ifndef SSMF_NAME
#	if SSMF_BITS == 32
#		define SSMF_NAME(name) mathfun_##name##f
#elif SSMF_BITS == 64
#		define SSMF_NAME(name) mathfun_##name##d
#	else
#		define SSMF_NAME(name) mathfun_##name##
#	endif
#endif

// abs(): bitwise AND to clear the sign bit
inline SSMF_VF SSMF_NAME(abs)(SSMF_VF v) {
#ifdef SSMF_VF_ABS
	return SSMF_VF_ABS(v);
#elif SSMF_BITS == 32
	SSMF_VF mask = SSMF_VF_I(0x7FFFFFFF);
#elif SSMF_BITS == 64
	SSMF_VF mask = SSMF_VF_I(0x7FFFFFFFFFFFFFFFLL);
#else
#	error "need SSMF_VF_ABS() unless it's 32/64-bit IEEE floats"
#endif
	return SSMF_VF_AND(v, mask);
}

// neg(): bitwise XOR to flip the sign bit
inline SSMF_VF SSMF_NAME(neg)(SSMF_VF v) {
#ifdef SSMF_VF_NEG
	return SSMF_VF_NEG(v);
#elif SSMF_BITS == 32
	SSMF_VF mask = SSMF_VF_F(-0.0f);
#elif SSMF_BITS == 64
	SSMF_VF mask = SSMF_VF_F(-0.0);
#else
#	error "need SSMF_VF_NEG() unless it's 32/64-bit IEEE floats"
#endif
	return SSMF_VF_XOR(v, mask);
}

// pow2(): multiply with itself
inline SSMF_VF SSMF_NAME(pow2)(SSMF_VF v) {
	return SSMF_VF_MUL(v, v);
}

// dim()
inline SSMF_VF SSMF_NAME(dim)(SSMF_VF a, SSMF_VF b) {
	SSMF_VF diff = SSMF_VF_SUB(a, b);
	return SSMF_VF_MAX(SSMF_VF_0, diff);
}

// copysign()
inline SSMF_VF SSMF_NAME(copysign)(SSMF_VF a, SSMF_VF b) {
#ifdef SSMF_VF_COPYSIGN
	return SSMF_VF_COPYSIGN(a, b);
#elif SSMF_BITS == 32
	SSMF_VF mask = SSMF_VF_F(-0.0f);
#elif SSMF_BITS == 64
	SSMF_VF mask = SSMF_VF_F(-0.0);
#else
#	error "need SSMF_VF_COPYSIGN() unless it's 32/64-bit IEEE floats"
#endif
	return SSMF_VF_OR(SSMF_VF_ANDNOT(mask, a), SSMF_VF_AND(mask, b));
}

// log(): Pulls out exponent and mantissa, converting both to float
inline SSMF_VF SSMF_NAME(log2)(SSMF_VF v) {
#ifdef SSMF_VF_GET_EM
SSMF_VF exponent, mantissa;
SSMF_VF_GET_EM(v, exponent, mantissa);
#else
#	if SSMF_BITS == 32
	static constexpr int mantissaMaskInt = 0x007FFFFF, mantissaBits = 23;
	static constexpr float mantissaScale = 1.1920928955078125e-7f; // 2^-23
	static constexpr float exponentBase = 127;
#	elif SSMF_BITS == 64
	static constexpr long long mantissaMaskInt = 0x000FFFFFFFFFFFFFLL;
	static constexpr int mantissaBits = 52;
	static constexpr double mantissaScale = 2.220446049250313e-16; // 2^-52
	static constexpr double exponentBase = 1023;
#	else
#		error "SSMF_VF_GET_EM(v, e, m) required unless it's 32/64-bit IEEE"
#	endif
	// Get the mantissa by masking, converting to float, and scaling
	SSMF_VF mantissaMask = SSMF_VF_I(mantissaMaskInt);
	SSMF_VF mantissaInt = SSMF_VF_ITOF(SSMF_VF_AND(mantissaMask, v));
	SSMF_VF mantissa = SSMF_VF_MUL(mantissaInt, SSMF_VF_F(mantissaScale));
	
	SSMF_VF exponentOffset = SSMF_VF_ITOF(SSMF_VF_RSHIFT_K(v, mantissaBits));
	SSMF_VF exponent = SSMF_VF_SUB(exponentOffset, SSMF_VF_F(exponentBase));
#endif

#if SSMF_ACCURACY < 4
	SSMF_VF curve = mantissa;
#else
	// Cubic which matches value/gradient of log2(1 + x) at 0 and 1
	SSMF_VF curve = SSMF_VF_MUL(mantissa, SSMF_VF_POLY2(mantissa, 1.4426950408889634, -0.6067376022224086, 0.16404256133344516));
#endif	
	return SSMF_VF_ADD(curve, exponent);
}

// log(): multiplies log2()
inline SSMF_VF SSMF_NAME(log)(SSMF_VF v) {
	SSMF_VF resultLog2 = SSMF_NAME(log2)(v);
	SSMF_VF mult = SSMF_VF_F(SSMF_F(0.6931471805599453));
	return SSMF_VF_MUL(resultLog2, mult);
}

// Clean up all our arguments unless explicitly told not to
#ifndef SSMF_NO_CLEANUP
#	undef SSMF_BITS
#	undef SSMF_F
#	undef SSMF_VF
#	undef SSMF_VF_0
#	undef SSMF_VF_F
#	undef SSMF_VF_I
#	undef SSMF_VF_AND
#	undef SSMF_VF_ANDNOT
#	undef SSMF_VF_OR
#	undef SSMF_VF_XOR
#	undef SSMF_VF_ADD
#	undef SSMF_VF_SUB
#	undef SSMF_VF_MUL
#	undef SSMF_VF_DIV
#	undef SSMF_VF_MAX
#	undef SSMF_VF_MIN
#	undef SSMF_VF_RSHIFT_K
#	undef SSMF_VF_ITOF
#	undef SSMF_FMAK

#	undef SSMF_NAME
#endif
#undef SSMF_VF_POLY0
#undef SSMF_VF_POLY1
#undef SSMF_VF_POLY2
#undef SSMF_VF_POLY3
#undef SSMF_VF_POLY4
#undef SSMF_VF_POLY5
#undef SSMF_VF_POLY6
#undef SSMF_VF_POLY7
#undef SSMF_VF_POLY8
#undef SSMF_VF_POLY9
#undef SSMF_VF_POLY10
#undef SSMF_VF_POLY11
#undef SSMF_VF_POLY12
#undef SSMF_VF_POLY13
#undef SSMF_VF_POLY14
#undef SSMF_VF_POLY15
#undef SSMF_VF_POLY16
#undef SSMF_VF_POLY1M
#undef SSMF_VF_POLY2M
#undef SSMF_VF_POLY3M
#undef SSMF_VF_POLY4M
#undef SSMF_VF_POLY5M
#undef SSMF_VF_POLY6M
#undef SSMF_VF_POLY7M
#undef SSMF_VF_POLY8M
#undef SSMF_VF_POLY9M
#undef SSMF_VF_POLY10M
#undef SSMF_VF_POLY11M
#undef SSMF_VF_POLY12M
#undef SSMF_VF_POLY13M
#undef SSMF_VF_POLY14M
#undef SSMF_VF_POLY15M
#undef SSMF_VF_POLY16M
