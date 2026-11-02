/* Copyright 2026 Geraint Luff / Signalsmith Audio Ltd.

SIGNALSMITH MATHFUN
-------------------

This header defines common floating-point functions, given #defines for the basic operations (constants, arithmetic, bitwise).  The functions are defined using a naming macro, so you can define your own prefix/suffix, e.g. `SSMF_NAME(abs)` -> `prefix_abs_suffix`.

It's a "function header" - explicitly designed to be included multiple times if needed, for different value types (float/double/SIMD) or accuracy levels.  You define various `SSMF_...` macros, and these are un-defined by this header.

It's focused on audio DSP use, which means:
-	approximate results - you can choose the accuracy/complexity tradeoff with `SSMF_ACCURACY`, up to 50 bits.
-	no NaN/Inf support - anything which would produce these outputs will give nonsense
-	limited range - some functions (rounding, sin/cos/etc.) only give proper results for "small number" range (±2^22 for float, ±2^51 for double).  Outside this range their result should still be sensibly bounded, but not necessarily correct.
-	inaccurate results for subnormals - they won't be _accurate_, but should still be between f(0) and f(smallestNormal)

Some fundamental operations are required (and they'll error if missing), but others are optional and the implementation will work around that.  Defining them should always be better if they exist simply (e.g. as one or two instructions).  If you set `SSMF_BITS` to either 32 or 64, then it assumes IEEE floating-point, and can cover more things.

General config:
	SSMF_NAME(fn): defaults to ssmf_fn
	SSMF_NO_CLEANUP: if defined, the arguments don't get #undef'd at the end

Constants and accuracy:
	SSMF_BITS: if 32 or 64, it can guess values for the others in this group
	SSMF_ACCURACY: target accuracy (in mantissa bits) for the approximation - actual results might be 1 bit worse than this
	SSMF_EPSILON: difference between 1 and the next representable value
	SSMF_ROUND_LIMIT: values below this magnitude get accurately rounded

Types and constants:
	SSMF_VF: type for all function arguments/results, could be scalar or SIMD
	SSMF_VF_0: expression which gives a SSMF_VF with value 0
	SSMF_VF_F(c): create a vector from a scalar literal, including converting to the appropriate type (e.g. float) since literals might be in any form, including non-bracketed expressions
	SSMF_VF_I(c): same as previous, but using an integer literal which gets re-interpreted as a float

Mandatory bitwise ops:
	SSMF_VF_AND(a, b): a & b
	SSMF_VF_ANDNOT(a, b): ~a & b
	SSMF_VF_OR(a, b): a | b
	SSMF_VF_XOR(a, b): a ^ b
	SSMF_VF_RSHIFT_K(a, k): a >> k, for integer constant k
	SSMF_VF_LSHIFT_K(a, k): a << k, for integer constant k

Mandatory arithmetic:
	SSMF_VF_ADD(a, b): a + b
	SSMF_VF_SUB(a, b): a - b
	SSMF_VF_MUL(a, b): a * b
	SSMF_VF_DIV(a, b): a / b
	SSMF_VF_MAX(a, b): max(a, b)
	SSMF_VF_MIN(a, b): min(a, b)

Optional:
	SSMF_VF_BARRIER(v): returns v unchanged, but the compiler can't reassociate maths through it (e.g. an empty `asm` block, with a register constraint for the target)
	SSMF_ATTRIBUTES: attributes given to every defined function
	SSMF_INLINE: correct way to specify inline functions
	SSMF_VF_FMA(a, b, c): a*b + c
	SSMF_VF_FMAK(a, b, k): a*b + k, for float constant k
	SSMF_VF_ABS(v)
	SSMF_VF_NEG(v)
	SSMF_VF_COPYSIGN(a, b): magnitude of a, sign of b
	SSMF_VF_ROUND(v)
	SSMF_VF_FLOOR(v)
	SSMF_VF_LOG2(v)
	SSMF_VF_LOG(v)
	SSMF_VF_EXP2(v)
	SSMF_VF_EXP(v)

*/

#ifndef SSMF_NAME
#	define SSMF_NAME(name) ssmf_##name
#endif

#ifndef SSMF_ACCURACY
#	if SSMF_BITS == 32
#		define SSMF_ACCURACY 20
#	elif SSMF_BITS == 64
#		define SSMF_ACCURACY 48
#	else
#		error "need SSMF_ACCURACY"
#	endif
#endif

#ifndef SSMF_EPSILON
#	if SSMF_BITS == 32
#		define SSMF_EPSILON 1.1920928955078125e-07 // 2^-23
#	elif SSMF_BITS == 64
#		define SSMF_EPSILON 2.220446049250313080847263336181640625e-16 // 2^-52
#	else
#		error "need SSMF_EPSILON if it's not 32-/64-bit IEEE"
#	endif
#endif

#ifndef SSMF_ROUND_LIMIT
#	if SSMF_BITS == 32
#		define SSMF_ROUND_LIMIT 4194304 // 2^22
#	elif SSMF_BITS == 64
#		define SSMF_ROUND_LIMIT 2251799813685248 // 2^51
#	else
#		error "need SSMF_ROUND_LIMIT if it's not 32-/64-bit IEEE"
#	endif
#endif

#ifndef SSMF_VF
#	error "SSMF_VF not defined"
#endif
#ifndef SSMF_VF_0
#	error "SSMF_VF_0 not defined"
#endif
#ifndef SSMF_VF_F
#	error "SSMF_VF_F() not defined"
#endif
#ifndef SSMF_VF_I
#	error "SSMF_VF_I() not defined"
#endif
#ifndef SSMF_VF_AND
#	error "SSMF_VF_AND() not defined"
#endif
#ifndef SSMF_VF_ANDNOT
#	error "SSMF_VF_ANDNOT() not defined"
#endif
#ifndef SSMF_VF_OR
#	error "SSMF_VF_OR() not defined"
#endif
#ifndef SSMF_VF_XOR
#	error "SSMF_VF_XOR() not defined"
#endif
#ifndef SSMF_VF_ADD
#	error "SSMF_VF_ADD() not defined"
#endif
#ifndef SSMF_VF_SUB
#	error "SSMF_VF_SUB() not defined"
#endif
#ifndef SSMF_VF_MUL
#	error "SSMF_VF_MUL() not defined"
#endif
#ifndef SSMF_VF_DIV
#	error "SSMF_VF_DIV() not defined"
#endif
#ifndef SSMF_VF_MAX
#	error "SSMF_VF_MAX() not defined"
#endif
#ifndef SSMF_VF_MIN
#	error "SSMF_VF_MIN() not defined"
#endif
#ifndef SSMF_VF_RSHIFT_K
#	error "SSMF_VF_RSHIFT_K() not defined"
#endif
#ifndef SSMF_VF_LSHIFT_K
#	error "SSMF_VF_LSHIFT_K() not defined"
#endif

// If we don't have a proper barrier, we wipe the lowest bit to prevent reassociation (only valid for values where that bit is already zero, e.g. small integers)
#ifdef SSMF_VF_BARRIER
#	define SSMF_VF_BARRIER_LOWBIT(v) SSMF_VF_BARRIER(v)
#elif SSMF_BITS == 32
#	define SSMF_VF_BARRIER_LOWBIT(v) SSMF_VF_AND(v, SSMF_VF_I(0xFFFFFFFE))
#elif SSMF_BITS == 64
#	define SSMF_VF_BARRIER_LOWBIT(v) SSMF_VF_AND(v, SSMF_VF_I(0xFFFFFFFFFFFFFFFE))
#else
#	error "SSMF_VF_BARRIER is required unless it's 32/64-bit IEEE float"
#endif

// Optional: a*b + c
#ifndef SSMF_VF_FMA
#	define SSMF_VF_FMA(a, b, c) SSMF_VF_ADD(SSMF_VF_MUL(a, b), c)
#endif
// Optional: a*b + k, for constant literal `k`
#ifndef SSMF_VF_FMAK
#	define SSMF_VF_FMAK(a, b, k) SSMF_VF_ADD(SSMF_VF_MUL(a, b), SSMF_VF_F(k))
#endif

// Optional: attributes added before every function definition (e.g. `__attribute__((always_inline))`)
#ifndef SSMF_ATTRIBUTES
#	define SSMF_ATTRIBUTES
#endif
// Optional: how functions are declared (`static` in C, so a single translation unit doesn't need an external definition)
#ifndef SSMF_INLINE
#	ifdef __cplusplus
#		define SSMF_INLINE inline
#	else
#		define SSMF_INLINE static inline
#	endif
#endif

// Polynomials (Horner's method) with constant coefficients
#define SSMF_VF_POLY0(x, c0) SSMF_VF_F(c0)
#define SSMF_VF_POLY1(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY0(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY2(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY1(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY3(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY2(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY4(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY3(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY5(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY4(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY6(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY5(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY7(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY6(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY8(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY7(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY9(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY8(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY10(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY9(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY11(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY10(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY12(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY11(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY13(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY12(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY14(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY13(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY15(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY14(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY16(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY15(x, __VA_ARGS__), x, c0)
// Monic variants, where the highest-order coefficient is 1 (so we can skip the multiplication)
#define SSMF_VF_POLY1M(x, c0) SSMF_VF_ADD(x, SSMF_VF_F(c0))
#define SSMF_VF_POLY2M(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY1M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY3M(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY2M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY4M(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY3M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY5M(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY4M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY6M(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY5M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY7M(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY6M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY8M(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY7M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY9M(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY8M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY10M(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY9M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY11M(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY10M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY12M(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY11M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY13M(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY12M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY14M(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY13M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY15M(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY14M(x, __VA_ARGS__), x, c0)
#define SSMF_VF_POLY16M(x, c0, ...) SSMF_VF_FMAK(SSMF_VF_POLY15M(x, __VA_ARGS__), x, c0)
// SSMF_VF_ROUND_SMALL - a helper valid within (at least) the SSMF_ROUND_LIMIT
#if defined(SSMF_VF_ROUND)
#	define SSMF_VF_ROUND_SMALL(v) SSMF_VF_ROUND(v)
#elif defined(SSMF_VF_FLOOR)
// Add the largest value below 0.5, so that e.g. 0.49999997f doesn't round up (since 0.49999997f + 0.5f -> 1.0f)
#	define SSMF_VF_ROUND_SMALL(v) SSMF_VF_FLOOR(SSMF_VF_ADD((v), SSMF_VF_F(0.5 - SSMF_EPSILON*0.25)))
#else
#	if SSMF_BITS == 32
#		define SSMF_VF_ROUND_MAGIC 12582912 // 1.5 * 2^23
#	elif SSMF_BITS == 64
#		define SSMF_VF_ROUND_MAGIC 6755399441055744 // 1.5 * 2^52
#	else
#		error "SSMF_VF_ROUND() or SSMF_VF_FLOOR() required unless it's 32/64-bit IEEE float"
#	endif
#	ifdef SSMF_VF_BARRIER
			// Add, barrier, subtract, barrier (so the integer result can't be reassociated with whatever comes next, e.g. `v - round(v)`)
#		define SSMF_VF_ROUND_SMALL(v) SSMF_VF_BARRIER(SSMF_VF_SUB(SSMF_VF_BARRIER(SSMF_VF_ADD((v), SSMF_VF_F(SSMF_VF_ROUND_MAGIC))), SSMF_VF_F(SSMF_VF_ROUND_MAGIC)))
#	else
			// (v + magic) is always positive, so abs() is a no-op which acts as a barrier, but the compiler can't prove it, so it stops the add/subtract being folded.
			// The result is an integer, so its lowest mantissa bit is 0, and BARRIER_LOWBIT fences it from whatever comes next.
#		define SSMF_VF_ROUND_SMALL(v) SSMF_VF_BARRIER_LOWBIT(SSMF_VF_SUB(SSMF_NAME(abs)(SSMF_VF_ADD((v), SSMF_VF_F(SSMF_VF_ROUND_MAGIC))), SSMF_VF_F(SSMF_VF_ROUND_MAGIC)))
#	endif
#endif
// abs(): bitwise AND to clear the sign bit
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(abs)(SSMF_VF v) {
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
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(neg)(SSMF_VF v) {
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

// round(): nearest integer, only valid for small values
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(round)(SSMF_VF v) {
	return SSMF_VF_ROUND_SMALL(v);
}

// remainder1(): v - round(v), in [-0.5, 0.5].  Large inputs (which are already integers) give 0.
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(remainder1)(SSMF_VF v) {
	v = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-SSMF_ROUND_LIMIT)), SSMF_VF_F(SSMF_ROUND_LIMIT));
	return SSMF_VF_SUB(v, SSMF_VF_ROUND_SMALL(v));
}

// pow2(): multiply with itself
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(pow2)(SSMF_VF v) {
	return SSMF_VF_MUL(v, v);
}

// dim()
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(dim)(SSMF_VF a, SSMF_VF b) {
	SSMF_VF diff = SSMF_VF_SUB(a, b);
	return SSMF_VF_MAX(SSMF_VF_0, diff);
}

// copysign()
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(copysign)(SSMF_VF a, SSMF_VF b) {
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

// log2(): splits into exponent and mantissa using bitwise ops, then applies a curve to the mantissa
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(log2)(SSMF_VF v) {
#if defined(SSMF_VF_LOG2)
	return SSMF_VF_LOG2(v);
#elif defined(SSMF_VF_LOG)
	return SSMF_VF_MUL(SSMF_VF_LOG(v), SSMF_VF_F(1.4426950408889634));
#else
#	if SSMF_BITS == 32
	const int mantissaMaskInt = 0x007FFFFF;
	const int exponentBits = 8, exponentMagic = 256, exponentRoundingMask = 511;
	const float exponentBase = 127;
#	elif SSMF_BITS == 64
	const long long mantissaMaskInt = 0x000FFFFFFFFFFFFFLL;
	const int exponentBits = 11, exponentMagic = 2048, exponentRoundingMask = 4095;
	const double exponentBase = 1023;
#	else
#		error "log2() only currently implemented for 32/64-bit IEEE"
#	endif
	// Trim to just the mantissa bits
	SSMF_VF mantissaMasked = SSMF_VF_AND(v, SSMF_VF_I(mantissaMaskInt));
	// Left-shift by 1 - this moves one bit into the exponent, and doubles the remainder
	SSMF_VF mantissaShifted = SSMF_VF_LSHIFT_K(mantissaMasked, 1);
	// XOR with 1.0 sets the exponent to 0b011...11 (2^0), or 0b011...10 (2^-1) depending on the top mantissa bit
	// This maps [1.5, 2) -> [0.5, 1), and [1, 1.5) -> [1, 2), so it's centered(-ish) on 1
	SSMF_VF mantissa = SSMF_VF_XOR(mantissaShifted, SSMF_VF_F(1));
	SSMF_VF mantissaM1 = SSMF_VF_SUB(mantissa, SSMF_VF_F(1)); // We multiply by this to get exact results near 0
	
	// Shift everything so the exponent is in the mantissa, and mask in an exponent which puts the exponent at an integer position.
	SSMF_VF exponentShifted = SSMF_VF_RSHIFT_K(v, exponentBits);
	SSMF_VF exponentOffset = SSMF_VF_OR(exponentShifted, SSMF_VF_F(exponentMagic));
	// Add 0.5 and then truncate to an integer value (with a bit-mask)
	SSMF_VF exponentRounded = SSMF_VF_AND(SSMF_VF_ADD(exponentOffset, SSMF_VF_F(0.5)), SSMF_VF_F(exponentRoundingMask));
	// Subtract the 'magic offset' and the exponent base at the same time
	SSMF_VF exponent = SSMF_VF_SUB(exponentRounded, SSMF_VF_F(exponentBase + exponentMagic));

	// Minimax fits of log2((1+m)/2)/(m-1) for m in [0.5, 2).  We multiply by (m-1) to get good results near 0.
#	if SSMF_ACCURACY <= 2 // 2.3 bits
	SSMF_VF curve = SSMF_VF_MUL(mantissaM1, SSMF_VF_F(0.6666666666666666));
#	elif SSMF_ACCURACY <= 5 // 5.14 bits
	SSMF_VF curve = SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY1(mantissa, 0.91177916450319807, -0.16340833189102097));
#	elif SSMF_ACCURACY <= 8 // 8.46 bits
	SSMF_VF curve = SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY2(mantissa, 0.96241418483320151, -0.28999588271602933, 0.05063502033000334));
#	elif SSMF_ACCURACY <= 11 // 11.49 bits
	SSMF_VF curve = SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY3(mantissa, 0.98369960241357313, -0.3607029285420576, 0.11565419259812297, -0.017493501875099196));
#	elif SSMF_ACCURACY <= 14 // 14.38 bits
	SSMF_VF curve = SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY4(mantissa, 0.9927714667919143, -0.39917168495103311, 0.17062675075462577, -0.049352485139110357, 0.0064279551203554903));
#	elif SSMF_ACCURACY <= 17 // 17.21 bits
	SSMF_VF curve = SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY5(mantissa, 0.99673421699056985, -0.41976757516362784, 0.21040570606352918, -0.085232649549313533, 0.021664070674094597, -0.0024569788933524986));
#	elif SSMF_ACCURACY <= 20 // 21.05 bits
	SSMF_VF curve = SSMF_VF_DIV(SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY2M(mantissa, 87.919282444833087, 43.198765569390631)), SSMF_VF_POLY2(mantissa, 87.940924478146329, 81.98046715472374, 13.233186867851849));
#	elif SSMF_ACCURACY <= 25 // 25.86 bits
	SSMF_VF curve = SSMF_VF_DIV(SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY2M(mantissa, 12.849423098943026, 9.3665014702745513)), SSMF_VF_POLY3(mantissa, 12.849923750627394, 15.051003529697775, 4.0971980158496795, 0.18597952684601951));
#	elif SSMF_ACCURACY <= 30 // 30.25 bits
	SSMF_VF curve = SSMF_VF_DIV(SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY3M(mantissa, 615.07015959602086, 560.02796626919769, 97.774089410506335)), SSMF_VF_POLY3(mantissa, 615.07481238180026, 832.27477563489219, 295.01130521376649, 23.600973955242399));
#	elif SSMF_ACCURACY <= 34 // 34.98 bits
	SSMF_VF curve = SSMF_VF_DIV(SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY3M(mantissa, 59.874988428042606, 68.198456234160986, 18.8326304136726)), SSMF_VF_POLY4(mantissa, 59.875061680986427, 94.704066242849095, 44.077134984989968, 6.221219353247025, 0.16387559422677625));
#	elif SSMF_ACCURACY <= 39 // 39.39 bits
	SSMF_VF curve = SSMF_VF_DIV(SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY4M(mantissa, 4167.7301872918406, 5522.5148934114723, 1980.6031106095261, 178.08851861856616)), SSMF_VF_POLY4(mantissa, 4167.7311549300921, 7367.5372709525045, 4080.8900698751668, 774.38647159574089, 36.955473282984883));
#	elif SSMF_ACCURACY <= 43 // 44.08 bits
	SSMF_VF curve = SSMF_VF_DIV(SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY4M(mantissa, 297.67500491434294, 460.80363178588181, 213.87328780984984, 31.568450793030927)), SSMF_VF_POLY5(mantissa, 297.67501625201004, 592.58273558781423, 393.25976268431577, 100.73798113118333, 8.7102964327181933, 0.1496575691326113));
#	elif SSMF_ACCURACY <= 48 // 48.50 bits
	SSMF_VF curve = SSMF_VF_DIV(SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY5M(mantissa, 27669.209715649555, 48061.607127308169, 26564.217866364506, 5218.706787092744, 286.09318187352375)), SSMF_VF_POLY5(mantissa, 27669.20991276007, 60310.626293978567, 45553.358309032737, 14175.149115843604, 1682.0498613576738, 53.295745555738556));
#	else // 53.16 bits
	SSMF_VF curve = SSMF_VF_DIV(SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY5M(mantissa, 1539.949449031581, 3012.7926448644621, 1985.4447244334174, 516.40343770651327, 47.614385556181887)), SSMF_VF_POLY6(mantissa, 1539.9494508493094, 3694.5206013958505, 3191.8801514383294, 1211.3930558551092, 197.70059821180519, 11.54910988714642, 0.13957288228854356));
#	endif
	// Include a barrier so that exponent doesn't get reassociated, which would lose precision in our curve
	return SSMF_VF_ADD(SSMF_VF_BARRIER_LOWBIT(exponent), curve);
#endif
}

// log(): multiplies log2() if not otherwise defined
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(log)(SSMF_VF v) {
#if defined(SSMF_VF_LOG)
	return SSMF_VF_LOG(v);
#else
	return SSMF_VF_MUL(SSMF_NAME(log2)(v), SSMF_VF_F(0.6931471805599453));
#endif
}

// Minimax fits (relative error, exact at 0) of 2^x for x in [-0.5, 0.5], some of which are R(x)^2
#if SSMF_ACCURACY <= 4 // 4.13 bits
#	define SSMF_VF_EXP2_SMALL(fraction) SSMF_VF_POLY1(fraction, 1, 0.66666666666666663)
#elif SSMF_ACCURACY <= 8 // 8.99 bits
#	define SSMF_VF_EXP2_SMALL(fraction) SSMF_VF_POLY2(fraction, 1, 0.70294179397838641, 0.23986402898051107)
#elif SSMF_ACCURACY <= 13 // 13.27 bits
#	define SSMF_VF_EXP2_SMALL(fraction) SSMF_VF_POLY3(fraction, 1, 0.69328292709619788, 0.24221095927271177, 0.055008931130349735)
#elif SSMF_ACCURACY <= 18 // 18.44 bits
#	define SSMF_VF_EXP2_SMALL(fraction) SSMF_VF_POLY4(fraction, 1, 0.69312419341797848, 0.24024098609879685, 0.0559064245957012, 0.0095828530962366633)
#elif SSMF_ACCURACY <= 23 // 23.38 bits
#	define SSMF_VF_EXP2_SMALL(fraction) SSMF_VF_POLY5(fraction, 1, 0.6931469775991671, 0.2402224208537713, 0.055507337432473138, 0.0096715126395556482, 0.0013264727198609666)
#elif SSMF_ACCURACY <= 28 // 28.92 bits
#	define SSMF_VF_EXP2_SMALL(fraction) SSMF_VF_POLY6(fraction, 1, 0.69314720285504527, 0.24022647913632003, 0.055503324711590182, 0.0096184373574484706, 0.0013398874403644236, 0.00015353361944104771)
#elif SSMF_ACCURACY <= 34 // 34.95 bits
#	define SSMF_VF_EXP2_SMALL(fraction) SSMF_NAME(pow2)(SSMF_VF_POLY6(fraction, 1, 0.34657359046976205, 0.060056626631417712, 0.0069380072448574932, 0.00060113425550469848, 4.1719000831718447e-05, 2.4049139025015073e-06))
#elif SSMF_ACCURACY <= 41 // 41.31 bits
#	define SSMF_VF_EXP2_SMALL(fraction) SSMF_NAME(pow2)(SSMF_VF_POLY7(fraction, 1, 0.34657359028025697, 0.060056626754494323, 0.0069380135732753376, 0.00060113271094624718, 4.1667443127800065e-05, 2.4092783208205937e-06, 1.1905096297808618e-07))
#elif SSMF_ACCURACY <= 48 // 48.19 bits
#	define SSMF_VF_EXP2_SMALL(fraction) SSMF_NAME(pow2)(SSMF_VF_POLY8(fraction, 1, 0.34657359027994744, 0.060056626739796927, 0.0069380135845075594, 0.00060113306874660336, 4.1667348593792191e-05, 2.4068044831461723e-06, 1.1927308276889719e-07, 5.1588071720172589e-09))
#else // 54.90 bits
#	define SSMF_VF_EXP2_SMALL(fraction) SSMF_NAME(pow2)(SSMF_VF_POLY9(fraction, 1, 0.34657359027997264, 0.060056626739773314, 0.0069380135831044612, 0.00060113306929947643, 4.1667369182049283e-05, 2.4068007441213058e-06, 1.1916210833453763e-07, 5.1664940928876514e-09, 1.9864502507589705e-10))
#endif

// exp2(): splits into an integer part (which becomes the exponent bits) and a fractional part in [-0.5, 0.5], which is curve-fitted
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(exp2)(SSMF_VF v) {
#if defined(SSMF_VF_EXP2)
	return SSMF_VF_EXP2(v);
#elif defined(SSMF_VF_EXP)
	return SSMF_VF_EXP(SSMF_VF_MUL(v, SSMF_VF_F(0.6931471805599453)));
#else
#	if SSMF_BITS == 32
	const int mantissaBits = 23;
	// round(v) must stay within the normal exponent range, so it doesn't wrap around the exponent bits
	const float minInput = -126, maxInput = 127.49609375; // 127.5 - 2^-8
	// When added to an integer, the lowest mantissa bits hold that integer plus the exponent bias (127)
	const float exponentMagic = 12582912 + 127; // 1.5 * 2^23 + 127
#	elif SSMF_BITS == 64
	const int mantissaBits = 52;
	const double minInput = -1022, maxInput = 1023.4990234375; // 1023.5 - 2^-10
	const double exponentMagic = 6755399441056767LL; // 1.5 * 2^52 + 1023
#	else
#		error "exp2() only currently implemented for 32/64-bit IEEE"
#	endif
	v = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(minInput)), SSMF_VF_F(maxInput));
#	if defined(SSMF_VF_ROUND) || defined(SSMF_VF_FLOOR)
	SSMF_VF integer = SSMF_VF_ROUND_SMALL(v);
	SSMF_VF scale = SSMF_VF_LSHIFT_K(SSMF_VF_ADD(integer, SSMF_VF_F(exponentMagic)), mantissaBits);
#	else
	// Our magic number includes the exponent bias, and we've clamped so that abs() is a no-op (but still a barrier)
	SSMF_VF withMagic = SSMF_NAME(abs)(SSMF_VF_ADD(v, SSMF_VF_F(exponentMagic)));
	SSMF_VF scale = SSMF_VF_LSHIFT_K(withMagic, mantissaBits);
	// The integer's lowest mantissa bit is 0, so BARRIER_LOWBIT fences it from the subtraction below
	SSMF_VF integer = SSMF_VF_BARRIER_LOWBIT(SSMF_VF_SUB(withMagic, SSMF_VF_F(exponentMagic)));
#	endif
	SSMF_VF fraction = SSMF_VF_SUB(v, integer);

	SSMF_VF curve = SSMF_VF_EXP2_SMALL(fraction);
	return SSMF_VF_MUL(curve, scale);
#endif
}

// exp(): multiplies exp2() if not otherwise defined
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(exp)(SSMF_VF v) {
#if defined(SSMF_VF_EXP)
	return SSMF_VF_EXP(v);
#else
	return SSMF_NAME(exp2)(SSMF_VF_MUL(v, SSMF_VF_F(1.4426950408889634)));
#endif
}

// Clean up all our arguments unless explicitly told not to
#ifndef SSMF_NO_CLEANUP
#	undef SSMF_BITS
#	undef SSMF_ACCURACY
#	undef SSMF_ROUND_LIMIT
#	undef SSMF_EPSILON
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
#	undef SSMF_VF_LSHIFT_K
#	undef SSMF_VF_BARRIER
#	undef SSMF_VF_BARRIER_LOWBIT
#	undef SSMF_VF_FMA
#	undef SSMF_VF_FMAK
#	undef SSMF_VF_ABS
#	undef SSMF_VF_NEG
#	undef SSMF_VF_COPYSIGN
#	undef SSMF_VF_ROUND
#	undef SSMF_VF_FLOOR
#	undef SSMF_VF_LOG2
#	undef SSMF_VF_LOG
#	undef SSMF_VF_EXP2
#	undef SSMF_VF_EXP
#	undef SSMF_NAME
#	undef SSMF_INLINE
#	undef SSMF_ATTRIBUTES
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
#undef SSMF_VF_ROUND_MAGIC
#undef SSMF_VF_ROUND_SMALL
#undef SSMF_VF_EXP2_SMALL
