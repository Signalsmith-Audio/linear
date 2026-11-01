/* Copyright 2026 Geraint Luff / Signalsmith Audio Ltd.

This header defines common floating-point functions, given #defines for the basic operations (constants, arithmetic, bitwise).  The functions are defined using a naming macro, so you can define your own prefix/suffix, e.g. `SSMF_NAME(abs)` -> `prefix_abs_suffix`.

It's a "function header" - explicitly designed to be included multiple times if needed, for different value types (float/double/SIMD) or accuracy levels.  You define various `SSMF_...` macros, and these are un-defined by this header.

It's focused on audio DSP use, which means:
-	approximate results - you can choose the accuracy/complexity tradeoff with `SSMF_ACCURACY`.
-	no NaN/Inf support - anything which would produce these outputs will give nonsense
-	limited range - some functions (rounding, sin/cos/etc.) only give proper results for "small number" range (±2^22 for float, ±2^51 for double).  Outside this range their result should still be sensibly bounded, but not necessarily correct.
-	inaccurate results for subnormals - they won't be _accurate_, but should still be between f(0) and f(smallestNormal)

Some fundamental operations are required (and they'll error if missing), but others are optional and the implementation will work around that.  Defining them should always be better if they exist simply (e.g. as one or two instructions).  If you set `SSMF_BITS` to either 32 or 64, then it assumes IEEE floating-point, and can cover more things.
*/

#ifndef SSMF_ACCURACY
#	if SSMF_BITS == 32
#		define SSMF_ACCURACY 20
#	elif SSMF_BITS == 64
#		define SSMF_ACCURACY 48
#	else
#		error "need SSMF_ACCURACY"
#	endif
#endif

#ifndef SSMF_SMALL_LIMIT
#	if SSMF_BITS == 32
#		define SSMF_SMALL_LIMIT 4194304 // 2^22
#	elif SSMF_BITS == 64
#		define SSMF_SMALL_LIMIT 2251799813685248 // 2^51
#	else
#		error "need SSMF_SMALL_LIMIT if it's not 32-/64-bit IEEE"
#	endif
#endif

// types for the SIMD vector
#ifndef SSMF_VF
#	error "SSMF_VF not defined"
#endif
// evaluates to a vector full of 0s
#ifndef SSMF_VF_0
#	error "SSMF_VF_0 not defined"
#endif
// create a vector from a scalar - this must include converting to the appropriate type (e.g. float) since literals might be anything
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
// a << bits
#ifndef SSMF_VF_LSHIFT_K
#	error "SSMF_VF_LSHIFT_K() not defined"
#endif

// Optional: SSMF_VF_BARRIER(v) returns v unchanged, but the compiler can't reassociate maths through it (e.g. an empty `asm` block, with a register constraint for the target)
// If we don't have one, we wipe the lowest bit as protection (only valid for values where that bit is already zero, e.g. small integers)
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

// Optional
#ifndef SSMF_NAME
#	if SSMF_BITS == 32
#		define SSMF_NAME(name) mathfun_##name##f
#	elif SSMF_BITS == 64
#		define SSMF_NAME(name) mathfun_##name##d
#	else
#		define SSMF_NAME(name) mathfun_##name##
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

// SSMF_VF_ROUND_SMALL - a helper valid in the SSMF_SMALL_LIMIT
#ifdef SSMF_VF_ROUND_SMALL
// already exists, great
#elif defined(SSMF_VF_ROUND)
#	define SSMF_VF_ROUND_SMALL(v) SSMF_VF_ROUND(v)
#elif defined(SSMF_VF_FLOOR)
#	define SSMF_VF_ROUND_SMALL(v) SSMF_VF_FLOOR(SSMF_VF_ADD((v), SSMF_VF_F(0.5)))
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
			// (v + magic) is always positive, so abs() is a no-op, but the compiler can't prove it, so it stops the add/subtract being folded.
			// The result is an integer, so its lowest mantissa bit is 0, and BARRIER_LOWBIT fences it from whatever comes next.
#		define SSMF_VF_ROUND_SMALL(v) SSMF_VF_BARRIER_LOWBIT(SSMF_VF_SUB(SSMF_NAME(abs)(SSMF_VF_ADD((v), SSMF_VF_F(SSMF_VF_ROUND_MAGIC))), SSMF_VF_F(SSMF_VF_ROUND_MAGIC)))
#	endif
#endif

// round(): nearest integer, only valid for small values
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(round)(SSMF_VF v) {
	return SSMF_VF_ROUND_SMALL(v);
}

// remainder1(): v - round(v), in [-0.5, 0.5].  Large inputs (which are already integers) give 0.
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(remainder1)(SSMF_VF v) {
	v = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-SSMF_SMALL_LIMIT)), SSMF_VF_F(SSMF_SMALL_LIMIT));
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
#elif SSMF_BITS == 32
	const int mantissaMaskInt = 0x007FFFFF;
	const int exponentBits = 8, exponentMagic = 256, exponentRoundingMask = 511;
	const float exponentBase = 127;
#elif SSMF_BITS == 64
	const long long mantissaMaskInt = 0x000FFFFFFFFFFFFFLL;
	const int exponentBits = 11, exponentMagic = 2048, exponentRoundingMask = 4095;
	const double exponentBase = 1023;
#else
#	error "log2() only currently implemented for 32/64-bit IEEE"
#endif
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
#if SSMF_ACCURACY <= 2 // 2.3 bits
	SSMF_VF curve = SSMF_VF_MUL(mantissaM1, SSMF_VF_F(0.6666666666666666));
#else // 30.25 bits (29.2 bits measured for double)
	SSMF_VF curve = SSMF_VF_DIV(SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY3M(mantissa, 615.07015959602082, 560.02796626919764, 97.774089410506336)), SSMF_VF_POLY3(mantissa, 615.07481238180024, 832.27477563489221, 295.01130521376648, 23.600973955242398));
#endif
	// Include a barrier so that exponent doesn't get reassociated, which would lose precision in our curve
	return SSMF_VF_ADD(SSMF_VF_BARRIER_LOWBIT(exponent), curve);
}

// log(): multiplies log2() if not otherwise defined
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(log)(SSMF_VF v) {
#if defined(SSMF_VF_LOG)
	return SSMF_VF_LOG(v);
#else
	return SSMF_VF_MUL(SSMF_NAME(log2)(v), SSMF_VF_F(0.6931471805599453));
#endif
}

// Clean up all our arguments unless explicitly told not to
#ifndef SSMF_NO_CLEANUP
#	undef SSMF_BITS
#	undef SSMF_ACCURACY
#	undef SSMF_SMALL_LIMIT
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
#	undef SSMF_VF_LOG2
#	undef SSMF_VF_LOG
#	undef SSMF_VF_ROUND
#	undef SSMF_VF_ROUND_MAGIC
#	undef SSMF_VF_ROUND_SMALL
#	undef SSMF_VF_FLOOR

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
