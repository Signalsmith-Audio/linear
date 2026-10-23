/* Copyright 2026 Geraint Luff / Signalsmith Audio Ltd.

This header defines floating-point functions, given #defines for the basic operations (constants, arithmetic, bitwise).

It's a "function header" - explicitly designed to be included multiple times if needed.  You define various `SIGNALSMITH_MATHFUN_...` macros, and these are un-defined by this header.  The functions are defined using a naming macro, so you can define your own prefix/suffix, e.g. `SIGNALSMITH_MATHFUN_NAME(abs)` -> `prefix_abs_suffix`.

Some of these fundamental operations are required (and they'll error), but others are optional and the implementation will work around that.
*/

#if SIGNALSMITH_MATHFUN_BITS == 32
#	define SIGNALSMITH_MATHFUN_F float
#elif SIGNALSMITH_MATHFUN_BITS == 64
#	define SIGNALSMITH_MATHFUN_F double
#else
#	error "SIGNALSMITH_MATHFUN_BITS must be defined as 32 (float) or 64 (double)"
#endif
// types for the SIMD vector
#ifndef SIGNALSMITH_MATHFUN_VF
#	error "SIGNALSMITH_MATHFUN_VF not defined"
#endif
// evaluates to a vector full of 0s
#ifndef SIGNALSMITH_MATHFUN_VF_0
#	error "SIGNALSMITH_MATHFUN_VF_0 not defined"
#endif
// create a vector from a (float) scalar
#ifndef SIGNALSMITH_MATHFUN_VF_F
#	error "SIGNALSMITH_MATHFUN_VF_F() not defined"
#endif
// create a vector from an (int) scalar
#ifndef SIGNALSMITH_MATHFUN_VF_I
#	error "SIGNALSMITH_MATHFUN_VF_I() not defined"
#endif
// a & b
#ifndef SIGNALSMITH_MATHFUN_VF_AND
#	error "SIGNALSMITH_MATHFUN_VF_AND() not defined"
#endif
// ~a & b
#ifndef SIGNALSMITH_MATHFUN_VF_ANDNOT
#	error "SIGNALSMITH_MATHFUN_VF_ANDNOT() not defined"
#endif
// a | b
#ifndef SIGNALSMITH_MATHFUN_VF_OR
#	error "SIGNALSMITH_MATHFUN_VF_OR() not defined"
#endif
// a ^ b
#ifndef SIGNALSMITH_MATHFUN_VF_XOR
#	error "SIGNALSMITH_MATHFUN_VF_XOR() not defined"
#endif
// a + b
#ifndef SIGNALSMITH_MATHFUN_VF_ADD
#	error "SIGNALSMITH_MATHFUN_VF_ADD() not defined"
#endif
// a - b
#ifndef SIGNALSMITH_MATHFUN_VF_SUB
#	error "SIGNALSMITH_MATHFUN_VF_SUB() not defined"
#endif
// a * b
#ifndef SIGNALSMITH_MATHFUN_VF_MUL
#	error "SIGNALSMITH_MATHFUN_VF_MUL() not defined"
#endif
// a / b
#ifndef SIGNALSMITH_MATHFUN_VF_DIV
#	error "SIGNALSMITH_MATHFUN_VF_DIV() not defined"
#endif
// max(a, b)
#ifndef SIGNALSMITH_MATHFUN_VF_MAX
#	error "SIGNALSMITH_MATHFUN_VF_MAX() not defined"
#endif
// min(a, b)
#ifndef SIGNALSMITH_MATHFUN_VF_MIN
#	error "SIGNALSMITH_MATHFUN_VF_MIN() not defined"
#endif
// a >> bits
#ifndef SIGNALSMITH_MATHFUN_VF_RSHIFT_K
#	error "SIGNALSMITH_MATHFUN_VF_RSHIFT_K() not defined"
#endif

// If we don't have "interpret this as integers, and then convert to float", we can implement it
#ifndef SIGNALSMITH_MATHFUN_VF_ITOF
#	if SIGNALSMITH_MATHFUN_BITS == 32
#		define SIGNALSMITH_MATHFUN_VF_ITOF_MAGIC 16777216.0f
#	else
#		define SIGNALSMITH_MATHFUN_VF_ITOF_MAGIC 4503599627370496.0
#	endif
// The "magic number" is a float such that the mantissa ends up incrementing by integers
#	define SIGNALSMITH_MATHFUN_VF_ITOF(v) SIGNALSMITH_MATHFUN_VF_SUB( \
		SIGNALSMITH_MATHFUN_VF_OR(SIGNALSMITH_MATHFUN_VF_F(SIGNALSMITH_MATHFUN_VF_ITOF_MAGIC), v), \
		SIGNALSMITH_MATHFUN_VF_F(SIGNALSMITH_MATHFUN_VF_ITOF_MAGIC) \
	)
#endif

// Optional
#ifndef SIGNALSMITH_MATHFUN_NAME
#	if SIGNALSMITH_MATHFUN_BITS == 32
#		define SIGNALSMITH_MATHFUN_NAME(name) mathfun_##name##f
#	else
#		define SIGNALSMITH_MATHFUN_NAME(name) mathfun_##name##d
#	endif
#endif

// abs(): bitwise AND to clear the sign bit
inline SIGNALSMITH_MATHFUN_VF SIGNALSMITH_MATHFUN_NAME(abs)(SIGNALSMITH_MATHFUN_VF v) {
#if SIGNALSMITH_MATHFUN_BITS == 32
	SIGNALSMITH_MATHFUN_VF mask = SIGNALSMITH_MATHFUN_VF_I(0x7FFFFFFF);
#else
	SIGNALSMITH_MATHFUN_VF mask = SIGNALSMITH_MATHFUN_VF_I(0x7FFFFFFFFFFFFFFFLL);
#endif
	return SIGNALSMITH_MATHFUN_VF_AND(v, mask);
}

// neg(): bitwise XOR to flip the sign bit
inline SIGNALSMITH_MATHFUN_VF SIGNALSMITH_MATHFUN_NAME(neg)(SIGNALSMITH_MATHFUN_VF v) {
#if SIGNALSMITH_MATHFUN_BITS == 32
	SIGNALSMITH_MATHFUN_VF mask = SIGNALSMITH_MATHFUN_VF_F(-0.0f);
#else
	SIGNALSMITH_MATHFUN_VF mask = SIGNALSMITH_MATHFUN_VF_F(-0.0);
#endif
	return SIGNALSMITH_MATHFUN_VF_XOR(v, mask);
}

// pow2(): multiply with itself
inline SIGNALSMITH_MATHFUN_VF SIGNALSMITH_MATHFUN_NAME(pow2)(SIGNALSMITH_MATHFUN_VF v) {
	return SIGNALSMITH_MATHFUN_VF_MUL(v, v);
}

// dim()
inline SIGNALSMITH_MATHFUN_VF SIGNALSMITH_MATHFUN_NAME(dim)(SIGNALSMITH_MATHFUN_VF a, SIGNALSMITH_MATHFUN_VF b) {
	SIGNALSMITH_MATHFUN_VF diff = SIGNALSMITH_MATHFUN_VF_SUB(a, b);
	return SIGNALSMITH_MATHFUN_VF_MAX(SIGNALSMITH_MATHFUN_VF_0, diff);
}

// copysign()
inline SIGNALSMITH_MATHFUN_VF SIGNALSMITH_MATHFUN_NAME(copysign)(SIGNALSMITH_MATHFUN_VF a, SIGNALSMITH_MATHFUN_VF b) {
#if SIGNALSMITH_MATHFUN_BITS == 32
	SIGNALSMITH_MATHFUN_VF mask = SIGNALSMITH_MATHFUN_VF_F(-0.0f);
#else
	SIGNALSMITH_MATHFUN_VF mask = SIGNALSMITH_MATHFUN_VF_F(-0.0);
#endif
	return SIGNALSMITH_MATHFUN_VF_OR(SIGNALSMITH_MATHFUN_VF_ANDNOT(mask, a), SIGNALSMITH_MATHFUN_VF_AND(mask, b));
}

// log(): Pulls out exponent and mantissa, converting both to float
inline SIGNALSMITH_MATHFUN_VF SIGNALSMITH_MATHFUN_NAME(log2)(SIGNALSMITH_MATHFUN_VF v) {
#if SIGNALSMITH_MATHFUN_BITS == 32
	static constexpr int mantissaMaskInt = 0x007FFFFF, mantissaBits = 23;
	static constexpr float mantissaScale = 1.1920928955078125e-7f; // 2^-23
	static constexpr float exponentBase = 127;
#else
	static constexpr long long mantissaMaskInt = 0x000FFFFFFFFFFFFFLL;
	static constexpr int mantissaBits = 52;
	static constexpr double mantissaScale = 2.220446049250313e-16; // 2^-52
	static constexpr double exponentBase = 1023;
#endif
	// Get the mantissa by masking, converting to float, and scaling
	SIGNALSMITH_MATHFUN_VF mantissaMask = SIGNALSMITH_MATHFUN_VF_I(mantissaMaskInt);
	SIGNALSMITH_MATHFUN_VF mantissaInt = SIGNALSMITH_MATHFUN_VF_ITOF(SIGNALSMITH_MATHFUN_VF_AND(mantissaMask, v));
	SIGNALSMITH_MATHFUN_VF mantissa = SIGNALSMITH_MATHFUN_VF_MUL(mantissaInt, SIGNALSMITH_MATHFUN_VF_F(mantissaScale));

	// Cubic which matches value/gradient of log2(1 + x) at 0 and 1
	SIGNALSMITH_MATHFUN_VF c1 = SIGNALSMITH_MATHFUN_VF_F(SIGNALSMITH_MATHFUN_F(1.4426950408889634));
	SIGNALSMITH_MATHFUN_VF c2 = SIGNALSMITH_MATHFUN_VF_F(SIGNALSMITH_MATHFUN_F(-0.6067376022224086));
	SIGNALSMITH_MATHFUN_VF c3 = SIGNALSMITH_MATHFUN_VF_F(SIGNALSMITH_MATHFUN_F(0.16404256133344516));
	SIGNALSMITH_MATHFUN_VF curve = SIGNALSMITH_MATHFUN_VF_MUL(mantissa, SIGNALSMITH_MATHFUN_VF_ADD(c1, SIGNALSMITH_MATHFUN_VF_MUL(mantissa, SIGNALSMITH_MATHFUN_VF_ADD(c2, SIGNALSMITH_MATHFUN_VF_MUL(mantissa, c3)))));
	
	SIGNALSMITH_MATHFUN_VF exponentOffset = SIGNALSMITH_MATHFUN_VF_ITOF(SIGNALSMITH_MATHFUN_VF_RSHIFT_K(v, mantissaBits));
	SIGNALSMITH_MATHFUN_VF exponent = SIGNALSMITH_MATHFUN_VF_SUB(exponent, SIGNALSMITH_MATHFUN_VF_F(exponentBase));
	
	return SIGNALSMITH_MATHFUN_VF_ADD(curve, exponent);
}

// log(): multiplies log2()
inline SIGNALSMITH_MATHFUN_VF SIGNALSMITH_MATHFUN_NAME(log)(SIGNALSMITH_MATHFUN_VF v) {
	SIGNALSMITH_MATHFUN_VF resultLog2 = SIGNALSMITH_MATHFUN_NAME(log2)(v);
	SIGNALSMITH_MATHFUN_VF mult = SIGNALSMITH_MATHFUN_VF_F(SIGNALSMITH_MATHFUN_F(0.6931471805599453));
	return SIGNALSMITH_MATHFUN_VF_MUL(resultLog2, mult);
}

// Clean up all our arguments unless explicitly told not to
#ifndef SIGNALSMITH_MATHFUN_NO_CLEANUP
#	undef SIGNALSMITH_MATHFUN_BITS
#	undef SIGNALSMITH_MATHFUN_VF
#	undef SIGNALSMITH_MATHFUN_VF_0
#	undef SIGNALSMITH_MATHFUN_VF_F
#	undef SIGNALSMITH_MATHFUN_VF_I
#	undef SIGNALSMITH_MATHFUN_VF_AND
#	undef SIGNALSMITH_MATHFUN_VF_ANDNOT
#	undef SIGNALSMITH_MATHFUN_VF_OR
#	undef SIGNALSMITH_MATHFUN_VF_XOR
#	undef SIGNALSMITH_MATHFUN_VF_ADD
#	undef SIGNALSMITH_MATHFUN_VF_SUB
#	undef SIGNALSMITH_MATHFUN_VF_MUL
#	undef SIGNALSMITH_MATHFUN_VF_DIV
#	undef SIGNALSMITH_MATHFUN_VF_MAX
#	undef SIGNALSMITH_MATHFUN_VF_MIN
#	undef SIGNALSMITH_MATHFUN_VF_RSHIFT_K
#	undef SIGNALSMITH_MATHFUN_VF_ITOF

#	undef SIGNALSMITH_MATHFUN_NAME
#endif
