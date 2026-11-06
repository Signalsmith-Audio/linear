// Copyright 2026 Geraint Luff / Signalsmith Audio Ltd.
// SPDX-License-Identifier: 0BSD

/*
SIGNALSMITH MATHFUN
===================

This header defines common floating-point functions, given #defines for the basic operations (constants, arithmetic, bitwise).  The functions are defined using a naming macro, so you can define your own prefix/suffix, e.g. `SSMF_NAME(abs)` -> `prefix_abs_suffix`.

It's focused on audio DSP use, which means:
-	approximate results - you can choose the accuracy/complexity tradeoff with `SSMF_ACCURACY`, up to 50 bits.
-	no NaN/Inf support - anything which is given (or would produce) these values will give nonsense
-	inaccurate results for subnormals - they should still be plausible (i.e. between f(0) and f(smallestNormal)) just not accurate

FUNCTION HEADER
---------------

It's a "function header" - explicitly designed to be included multiple times if needed, for different value types (float/double/SIMD) or accuracy levels.  You define various `SSMF_...` macros, and these are un-defined by this header.

Some fundamental operations are required (and they'll error if missing), but others are optional and the implementation will work around that.  Defining them should always be better if they exist simply (e.g. as one or two instructions).  If you set `SSMF_BITS` to either 32 or 64, then it assumes IEEE floating-point, and can cover more things.

If you supply a rounding method (SSMF_VF_ROUND/SSMF_VF_FLOOR) which only works within a given range (e.g. by casting to a fixed-size integer and back), then various rounding functions and periodic functions (sin/cos/etc.) can return nonsense values outside of that range.

ARGUMENTS
---------

General config:
	SSMF_NAME(fn): defaults to ssmf_fn
	SSMF_NO_CLEANUP: if defined, the arguments don't get #undef'd at the end

Constants and accuracy:
	SSMF_BITS: if 32 or 64, it assumes standard IEEE floats, and provides defaults for SSMF_ACCURACY (20 or 48) and SSMF_EPSILON
	SSMF_ACCURACY: target accuracy (in mantissa bits) for the approximation - actual results might be 1 bit worse than this
	SSMF_EPSILON: difference between 1 and the next representable value
	SSMF_FULL_RANGE: if defined, adds extra guards against overflows (x² -> Inf) or underflows (x² -> subnormal, might get rounded to 0), and exp() results outside that "safe" range are also slightly more approximate.

Types and constants (mandatory):
	SSMF_VF: type for all function arguments/results, could be scalar or SIMD
	SSMF_VF_0: expression which gives a SSMF_VF with value 0
	SSMF_VF_F(c): create a vector from a scalar literal, including converting to the appropriate type (e.g. float) since literals might be in any form, including non-bracketed expressions
	SSMF_VF_I(c): same as previous, but using an integer literal which gets re-interpreted as a float

Bitwise ops (mandatory):
	SSMF_VF_AND(a, b): a & b
	SSMF_VF_ANDNOT(a, b): ~a & b
	SSMF_VF_OR(a, b): a | b
	SSMF_VF_XOR(a, b): a ^ b
	SSMF_VF_RSHIFT_K(a, k): a >> k, for integer constant k
	SSMF_VF_LSHIFT_K(a, k): a << k, for integer constant k

Arithmetic (mandatory):
	SSMF_VF_ADD(a, b): a + b
	SSMF_VF_SUB(a, b): a - b
	SSMF_VF_MUL(a, b): a * b
	SSMF_VF_DIV(a, b): a / b
	SSMF_VF_MAX(a, b): max(a, b)
	SSMF_VF_MIN(a, b): min(a, b)

Optional:
	SSMF_VF_BARRIER(v): returns v unchanged, but the compiler can't reassociate arithmetic through it (e.g. an empty `asm` block, with a register constraint for the target)
	SSMF_ATTRIBUTES: attributes given to every defined function
	SSMF_INLINE: correct way to specify inline functions
	SSMF_VF_FMA(a, b, c): a*b + c
	SSMF_VF_FMAK(a, b, k): a*b + k, for float constant k
	SSMF_VF_ABS(v)
	SSMF_VF_NEG(v)
	SSMF_VF_COPYSIGN(a, b): magnitude of a, sign of b
	SSMF_VF_ROUND(v)
	SSMF_VF_FLOOR(v)
	SSMF_VF_SQRT(v)
	SSMF_VF_RSQRT(v) // reciprocal, if that's faster
	SSMF_VF_HYPOT(a, b)
	SSMF_VF_LOG2(v)
	SSMF_VF_LOG(v)
	SSMF_VF_EXP2(v)
	SSMF_VF_EXP(v)
	SSMF_VF_EXPM1(v)
	SSMF_VF_SINH(v)
	SSMF_VF_COSH(v)
	SSMF_VF_POW(a, b)
	SSMF_VF_CBRT(v)
	SSMF_VF_ERF(v)
	SSMF_VF_ERFC(v)
	SSMF_VF_SIN(v)
	SSMF_VF_COS(v)
	SSMF_VF_SINCOS(v, sinOut, cosOut): writes to two pointers
	SSMF_VF_TAN(v)
	SSMF_VF_TANH(v)
	SSMF_VF_ASIN(v)
	SSMF_VF_ACOS(v)
	SSMF_VF_ATAN(v)
	SSMF_VF_ATAN2(y, x)

APPROXIMATIONS
---------

The approximations were found through brute-force search of different shapes of approximation.  At its core, each is a minimax Padé or polynomial approximation, with optional modifications of the input or output.

Each candidate shape is given a rule-of-thumb "complexity" metric:
* N for polynomials
* N + M + 2 for Padé: the +2 is for division, which is often slower than 2 FMAs, but also enables parallel evaluation of the top/bottom polynomials (improves pipelining)
* +1 for |x| or x² on the input
* +1 for y² on the output, or multiplying by x
* +2 for multiplying the output by |x|

The best fit for each "complexity" is kept, and then they are chosen at compile-time based on the requested accuracy in bits (i.e. `log2(error)`).

Sometimes the function actually fitted is some variant which already handles some features (like poles, or zeros where we want to maintain accuracy), for example instead of matching tan() directly, we match `tan(πf)(1 - 2f)(1 + 2f)/f` which is finite and non-negative everywhere.  These are noted in the comments.

If you re-use any of these approximations, attribution would be nice, but isn't strictly required.

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
#ifndef SSMF_VF_NOT
#	error "SSMF_VF_NOT() not defined"
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
#	define SSMF_VF_BARRIER_HIGHBIT_MASK(v) SSMF_VF_BARRIER(v)
#	define SSMF_VF_BARRIER_HIGHBIT(v, knownLessThan2) SSMF_VF_BARRIER(v)
#elif SSMF_BITS == 32
#	define SSMF_VF_BARRIER_LOWBIT(v) SSMF_VF_AND(SSMF_VF_I(0xFFFFFFFE), v) // For small integers
#	define SSMF_VF_BARRIER_HIGHBIT_MASK(v) SSMF_VF_AND(SSMF_VF_I(0x40000000), v) // Always 0 for things with magnitude < 2
#	define SSMF_VF_BARRIER_HIGHBIT(v, knownSmall) SSMF_VF_XOR(v, SSMF_VF_BARRIER_HIGHBIT_MASK(knownSmall))
#elif SSMF_BITS == 64
#	define SSMF_VF_BARRIER_LOWBIT(v) SSMF_VF_AND(SSMF_VF_I(0xFFFFFFFFFFFFFFFELL), v)
#	define SSMF_VF_BARRIER_HIGHBIT_MASK(v) SSMF_VF_AND(SSMF_VF_I(0x4000000000000000LL), v)
#	define SSMF_VF_BARRIER_HIGHBIT(v, knownSmall) SSMF_VF_XOR(v, SSMF_VF_BARRIER_HIGHBIT_MASK(knownSmall))
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
// SSMF_VF_ROUND_INNER - uses either a supplied ROUND or FLOAT, otherwise undefined
#if defined(SSMF_VF_ROUND)
#	define SSMF_VF_ROUND_INNER(v) SSMF_VF_ROUND(v)
#elif defined(SSMF_VF_FLOOR)
// Add the largest value below 0.5, so that e.g. 0.49999997f doesn't round up (since 0.49999997f + 0.5f -> 1.0f)
#	define SSMF_VF_ROUND_INNER(v) SSMF_VF_FLOOR(SSMF_VF_ADD((v), SSMF_VF_F(0.5 - SSMF_EPSILON*0.25)))
#endif
// abs(): bitwise AND to clear the sign bit
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(abs)(SSMF_VF v) {
#ifdef SSMF_VF_ABS
	return SSMF_VF_ABS(v);
#else
#	if SSMF_BITS == 32
	SSMF_VF mask = SSMF_VF_I(0x7FFFFFFF);
#	elif SSMF_BITS == 64
	SSMF_VF mask = SSMF_VF_I(0x7FFFFFFFFFFFFFFFLL);
#	else
#		error "need SSMF_VF_ABS() unless it's 32/64-bit IEEE floats"
#	endif
	return SSMF_VF_AND(v, mask);
#endif
}

// neg(): bitwise XOR to flip the sign bit
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(neg)(SSMF_VF v) {
#ifdef SSMF_VF_NEG
	return SSMF_VF_NEG(v);
#else
#	if SSMF_BITS == 32
	SSMF_VF mask = SSMF_VF_F(-0.0f);
#	elif SSMF_BITS == 64
	SSMF_VF mask = SSMF_VF_F(-0.0);
#	else
#		error "need SSMF_VF_NEG() unless it's 32/64-bit IEEE floats"
#	endif
	return SSMF_VF_XOR(v, mask);
#endif
}

// remainder1(): v - round(v), in [-0.5, 0.5].  Large inputs (which are already integers) give 0.
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(remainder1)(SSMF_VF v) {
#ifdef SSMF_VF_ROUND_INNER
#	ifdef SSMF_VF_BARRIER
	return SSMF_VF_SUB(SSMF_VF_BARRIER(v), SSMF_VF_ROUND_INNER(v));
#	else
	SSMF_VF sign = SSMF_VF_AND(v, SSMF_VF_F(-0.0));
	SSMF_VF absV = SSMF_VF_XOR(v, sign);
	return SSMF_VF_XOR(sign, SSMF_VF_SUB(absV, SSMF_VF_ROUND_INNER(absV)));
#	endif
#else
#	if SSMF_BITS == 32
#		define SSMF_ROUNDING_K 8388608 // 2^23
#		define SSMF_ROUNDING_SAFEBIT 0x20000000 // never set for values in the clamped+offset range
#	elif SSMF_BITS == 64
#		define SSMF_ROUNDING_K 4503599627370496LL // 2^52
#		define SSMF_ROUNDING_SAFEBIT 0x2000000000000000LL
#	else
#		error "need SSMF_VF_ROUND() or SSMF_VF_FLOOR() unless it's 32/64-bit IEEE float"
#	endif
	SSMF_VF sign = SSMF_VF_AND(v, SSMF_VF_F(-0.0));
	SSMF_VF absV = SSMF_VF_MIN(SSMF_VF_XOR(v, sign), SSMF_VF_F(SSMF_ROUNDING_K));
	SSMF_VF absVOffset = SSMF_VF_ADD(absV, SSMF_VF_F(SSMF_ROUNDING_K));
#	ifdef SSMF_VF_BARRIER
	absVOffset = SSMF_VF_BARRIER(absVOffset);
#	else
	absVOffset = SSMF_VF_ANDNOT(SSMF_VF_I(SSMF_ROUNDING_SAFEBIT), absVOffset);
#	endif
	SSMF_VF absVRounded = SSMF_VF_SUB(absVOffset, SSMF_VF_F(SSMF_ROUNDING_K));
	return SSMF_VF_XOR(SSMF_VF_SUB(absV, SSMF_VF_BARRIER_LOWBIT(absVRounded)), sign);
#	undef SSMF_ROUNDING_K
#	undef SSMF_ROUNDING_SAFEBIT
#endif
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
#else
#	if SSMF_BITS == 32
	SSMF_VF mask = SSMF_VF_F(-0.0f);
#	elif SSMF_BITS == 64
	SSMF_VF mask = SSMF_VF_F(-0.0);
#	else
#		error "need SSMF_VF_COPYSIGN() unless it's 32/64-bit IEEE floats"
#	endif
	return SSMF_VF_OR(SSMF_VF_ANDNOT(mask, a), SSMF_VF_AND(mask, b));
#endif
}

// round(): nearest integer
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(round)(SSMF_VF v) {
#if defined(SSMF_VF_ROUND_INNER)
	return SSMF_VF_ROUND_INNER(v);
#elif defined(SSMF_VF_BARRIER)
	return SSMF_VF_BARRIER(SSMF_VF_SUB(v, SSMF_NAME(remainder1)(v)));
#else
	return SSMF_VF_SUB(v, SSMF_NAME(remainder1)(v));
#endif
}

SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(ceil)(SSMF_VF v) {
#if defined(SSMF_VF_FLOOR)
	return SSMF_NAME(neg)(SSMF_VF_FLOOR(SSMF_NAME(neg)(v)));
#else
#	if SSMF_BITS == 32
	SSMF_VF huge = SSMF_VF_F(1.7014118e+38); // 2^127
#	elif SSMF_BITS == 64
	SSMF_VF huge = SSMF_VF_F(1.4119048864730642e+308); // 2^1023
#	else
#		error "need SSMF_VF_FLOOR() unless it's 32/64-bit IEEE floats"
#	endif
	SSMF_VF remainder = SSMF_NAME(remainder1)(v);
	SSMF_VF roundedDown = SSMF_VF_MIN(SSMF_VF_MUL(SSMF_VF_MAX(remainder, SSMF_VF_0), huge), SSMF_VF_F(1));
#	ifdef SSMF_VF_BARRIER
	return SSMF_VF_ADD(SSMF_VF_BARRIER(SSMF_VF_SUB(v, SSMF_VF_BARRIER(remainder))), roundedDown);
#	else
	SSMF_VF zeroBit = SSMF_VF_BARRIER_HIGHBIT_MASK(remainder); // always zero because remainder is small
	return SSMF_VF_ADD(SSMF_VF_XOR(zeroBit, SSMF_VF_SUB(v, SSMF_VF_XOR(zeroBit, remainder))), roundedDown);
#	endif
#endif
}
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(floor)(SSMF_VF v) {
#if defined(SSMF_VF_FLOOR)
	return SSMF_VF_FLOOR(v);
#else
	// Using ceil() avoids (v - rounded) being -0 when v = -0, which would give -1
	return SSMF_NAME(neg)(SSMF_NAME(ceil)(SSMF_NAME(neg)(v)));
#endif
}
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(trunc)(SSMF_VF v) {
	return SSMF_NAME(copysign)(SSMF_NAME(floor)(SSMF_NAME(abs)(v)), v);
}

// rsqrt()
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(rsqrt)(SSMF_VF v) {
#ifdef SSMF_VF_RSQRT
	return SSMF_VF_RSQRT(v);
#else
#	if SSMF_BITS == 32
	SSMF_VF invSqrt = SSMF_VF_RSHIFT_K(SSMF_VF_NOT(SSMF_VF_MAX(v, SSMF_VF_F(2.3509887e-38))), 1);
	// exponent bias is 127, right-shift approximately halves that (but with truncation)
	invSqrt = SSMF_VF_MUL(invSqrt, SSMF_VF_F(1.83523129466333e-20)); // 2^-65.5626 (minimax scale for this seed: 4.6 bits)
#	elif SSMF_BITS == 64
	SSMF_VF invSqrt = SSMF_VF_RSHIFT_K(SSMF_VF_NOT(SSMF_VF_MAX(v, SSMF_VF_F(4.450147717014403e-308))), 1);
	invSqrt = SSMF_VF_MUL(invSqrt, SSMF_VF_F(2.524949804293825e-155)); // 2^-513.5626
#	else
#		error "need SSMF_VF_SQRT() unless it's 32/64-bit IEEE floats"
#	endif
	// Prevent fast-math reassociation (v*invSqrt) into ((v*scale)*bits), which can underflow
#	ifdef SSMF_VF_BARRIER
	invSqrt = SSMF_VF_BARRIER(invSqrt);
#	else
	invSqrt = SSMF_NAME(abs)(invSqrt);
#	endif
	// Each step takes e -> 1.5e^2 (bits double, minus 0.6): 4.6 -> 8.5 -> 16.4 -> 32.3 -> 64
	const int stepCount = (SSMF_ACCURACY >= 5) + (SSMF_ACCURACY >= 9) + (SSMF_ACCURACY >= 17) + (SSMF_ACCURACY >= 33);
	for (int step = 0; step < stepCount; ++step) {
		// How wrong we are, squared.  We want this computed in a particular order so that huge inputs don't make invSqrt*invSqrt subnormal (which could be flushed to 0)
#	ifdef SSMF_VF_BARRIER
		SSMF_VF adj2 = SSMF_VF_MUL(SSMF_VF_BARRIER(SSMF_VF_MUL(v, invSqrt)), invSqrt);
#	else
		SSMF_VF adj2 = SSMF_VF_MUL(SSMF_NAME(abs)(SSMF_VF_MUL(v, invSqrt)), invSqrt);
#	endif
		invSqrt = SSMF_VF_MUL(invSqrt, SSMF_VF_SUB(SSMF_VF_F(1.5), SSMF_VF_MUL(SSMF_VF_F(0.5), adj2))); // close to 1, 1/sqrt(x) ~= 3/2 - x/2
	}
	return invSqrt;
#endif
}

SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(sqrt)(SSMF_VF v) {
#ifdef SSMF_VF_SQRT
	return SSMF_VF_SQRT(v);
#else
	return SSMF_VF_MUL(v, SSMF_NAME(rsqrt)(v));
#endif
}

// cbrt(): similar to sqrt(), but we have to use actual multiplication (not shifts) to divide the exponent by 3.  We estimate 1/cbrt(v) iteratively with Newton steps, then finish with v*y².
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(cbrt)(SSMF_VF v) {
#if defined(SSMF_VF_CBRT)
	return SSMF_VF_CBRT(v);
#else
	SSMF_VF absV = SSMF_VF_MAX(SSMF_NAME(abs)(v), SSMF_VF_F(SSMF_BITS == 32 ? 1.1754943508222875e-38 : 2.2250738585072014e-308));
	// The estimate: shift |v|'s exponent and mantissa down into the mantissa of 3.0 (into the space left by the 0 sign bit), so this is 3 + log2(|v|)/2^k (approximately, with bias)
#	if SSMF_BITS == 32
	absV = SSMF_VF_MAX(absV, SSMF_VF_F(1.1754943e-38f)); // smallest normal
	SSMF_VF y = SSMF_VF_OR(SSMF_VF_F(3), SSMF_VF_RSHIFT_K(absV, 9)); // shift right into mantissa, one down from MSB because we're using the top for 3 = 2*1.5
	y = SSMF_VF_FMAK(y, SSMF_VF_F(-0.33333334f), 2.6612005f); // scale by -1/3, add back the bias, plus a bit more because of the warped mantissa shape
	y = SSMF_NAME(abs)(SSMF_VF_LSHIFT_K(y, 8)); // shift back into the exponent (and clear sign bit)
#	elif SSMF_BITS == 64
	absV = SSMF_VF_MAX(absV, SSMF_VF_F(2.2250738585072014e-308));
	SSMF_VF y = SSMF_VF_OR(SSMF_VF_F(3), SSMF_VF_RSHIFT_K(absV, 12));
	y = SSMF_VF_FMAK(y, SSMF_VF_F(-0.3333333333333333), 2.6659833984);
	y = SSMF_NAME(abs)(SSMF_VF_LSHIFT_K(y, 11));
#	else
#		error "need SSMF_VF_CBRT() unless it's 32/64-bit IEEE floats"
#	endif
	// Newton steps: y *= 1 + (1 - absV*y³)/3
	const int stepCount = (SSMF_ACCURACY >= 7) + (SSMF_ACCURACY >= 15) + (SSMF_ACCURACY >= 30);
	for (int step = 0; step < stepCount; ++step) {
		SSMF_VF wrong = SSMF_VF_FMAK(SSMF_NAME(neg)(SSMF_NAME(abs)(SSMF_VF_MUL(v, y))), SSMF_VF_MUL(y, y), 1);
		y = SSMF_VF_FMA(y, SSMF_VF_MUL(wrong, SSMF_VF_F(0.3333333333333333)), y);
	}
	// Final step has different coefficients because we're doing it upside-down
	SSMF_VF vy = SSMF_VF_MUL(v, y);
	SSMF_VF wrong = SSMF_VF_FMAK(SSMF_NAME(neg)(SSMF_NAME(abs)(vy)), SSMF_VF_MUL(y, y), 1);
	SSMF_VF result = SSMF_VF_MUL(vy, y);
	return SSMF_VF_FMA(result, SSMF_VF_MUL(wrong, SSMF_VF_F(0.6666666666666666)), result);
#endif
}

// hypot()
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(hypot)(SSMF_VF a, SSMF_VF b) {
#if defined(SSMF_VF_HYPOT)
	return SSMF_VF_HYPOT(a, b);
#elif !defined(SSMF_FULL_RANGE)
	// This also might produce denormals (which could get flushed to zero), but that's OK
	return SSMF_NAME(sqrt)(SSMF_VF_FMA(a, a, SSMF_NAME(pow2)(b)));
#else
	// To ensure that neither a^2 nor b^2 overflow into Inf, we scale by some 2^N factor first and by its inverse afterwards
#	if SSMF_BITS == 32
	SSMF_VF exponentMask = SSMF_VF_I(0x7F800000);
	SSMF_VF minExp = SSMF_VF_F(2.3509887e-38); // 2^-125, since we halve it later
#	elif SSMF_BITS == 64
	SSMF_VF exponentMask = SSMF_VF_I(0x7FF0000000000000LL);
	SSMF_VF minExp = SSMF_VF_F(4.450147717014403e-308); // 2^-1021
#	else
#		error "need SSMF_VF_HYPOT() or (not SSMF_FULL_RANGE) unless it's 32/64-bit IEEE floats"
#	endif
	// 2^e for the larger of the two, e minimum 2
	SSMF_VF scaleA = SSMF_VF_AND(a, exponentMask), scaleB = SSMF_VF_AND(b, exponentMask);
	SSMF_VF scaleAfter = SSMF_VF_MAX(SSMF_VF_MAX(scaleA, scaleB), SSMF_VF_F(minExp));
	// Flipping the exponent bits gives 2^(1 - e)
	SSMF_VF scaleBefore = SSMF_VF_ANDNOT(scaleAfter, (exponentMask));
	// So we add an extra factor of 2 when scaling afterwards
	scaleAfter = SSMF_VF_MUL(scaleAfter, SSMF_VF_F(0.5));
	a = SSMF_VF_MUL(a, scaleBefore);
	b = SSMF_VF_MUL(b, scaleBefore);
	SSMF_VF knownSmall = SSMF_VF_MUL(scaleBefore, scaleAfter); // always == 1, but we constructed it with bitwise shenanigans
	a = SSMF_VF_BARRIER_HIGHBIT(a, knownSmall); // barrier so it can't factor the scaling out in `sum`
	SSMF_VF sum = SSMF_VF_FMA(a, a, SSMF_NAME(pow2)(b));
	SSMF_VF scaledResult = SSMF_NAME(sqrt)(sum);
	scaledResult = SSMF_VF_BARRIER_HIGHBIT(scaledResult, knownSmall);
	return SSMF_VF_MUL(scaledResult, scaleAfter);
#endif
}

// Minimax fits of log2((1+m)/2)/(m-1) for m in [0.5, 2).  We multiply by (m-1) to get good results near 0.
#if SSMF_ACCURACY <= 2 // 2.3 bits
#	define SSMF_VF_LOG2_CURVE(mantissa, mantissaM1) SSMF_VF_MUL(mantissaM1, SSMF_VF_F(0.6666666666666666))
#elif SSMF_ACCURACY <= 5 // 5.14 bits
#	define SSMF_VF_LOG2_CURVE(mantissa, mantissaM1) SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY1(mantissa, 0.91177916450319807, -0.16340833189102097))
#elif SSMF_ACCURACY <= 8 // 8.46 bits
#	define SSMF_VF_LOG2_CURVE(mantissa, mantissaM1) SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY2(mantissa, 0.96241418483320151, -0.28999588271602933, 0.05063502033000334))
#elif SSMF_ACCURACY <= 11 // 11.49 bits
#	define SSMF_VF_LOG2_CURVE(mantissa, mantissaM1) SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY3(mantissa, 0.98369960241357313, -0.3607029285420576, 0.11565419259812297, -0.017493501875099196))
#elif SSMF_ACCURACY <= 14 // 14.38 bits
#	define SSMF_VF_LOG2_CURVE(mantissa, mantissaM1) SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY4(mantissa, 0.9927714667919143, -0.39917168495103311, 0.17062675075462577, -0.049352485139110357, 0.0064279551203554903))
#elif SSMF_ACCURACY <= 17 // 17.21 bits
#	define SSMF_VF_LOG2_CURVE(mantissa, mantissaM1) SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY5(mantissa, 0.99673421699056985, -0.41976757516362784, 0.21040570606352918, -0.085232649549313533, 0.021664070674094597, -0.0024569788933524986))
#elif SSMF_ACCURACY <= 20 // 21.05 bits
#	define SSMF_VF_LOG2_CURVE(mantissa, mantissaM1) SSMF_VF_DIV(SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY2M(mantissa, 87.919282444833087, 43.198765569390631)), SSMF_VF_POLY2(mantissa, 87.940924478146329, 81.98046715472374, 13.233186867851849))
#elif SSMF_ACCURACY <= 25 // 25.86 bits
#	define SSMF_VF_LOG2_CURVE(mantissa, mantissaM1) SSMF_VF_DIV(SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY2M(mantissa, 12.849423098943026, 9.3665014702745513)), SSMF_VF_POLY3(mantissa, 12.849923750627394, 15.051003529697775, 4.0971980158496795, 0.18597952684601951))
#elif SSMF_ACCURACY <= 30 // 30.25 bits
#	define SSMF_VF_LOG2_CURVE(mantissa, mantissaM1) SSMF_VF_DIV(SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY3M(mantissa, 615.07015959602086, 560.02796626919769, 97.774089410506335)), SSMF_VF_POLY3(mantissa, 615.07481238180026, 832.27477563489219, 295.01130521376649, 23.600973955242399))
#elif SSMF_ACCURACY <= 34 // 34.98 bits
#	define SSMF_VF_LOG2_CURVE(mantissa, mantissaM1) SSMF_VF_DIV(SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY3M(mantissa, 59.874988428042606, 68.198456234160986, 18.8326304136726)), SSMF_VF_POLY4(mantissa, 59.875061680986427, 94.704066242849095, 44.077134984989968, 6.221219353247025, 0.16387559422677625))
#elif SSMF_ACCURACY <= 39 // 39.39 bits
#	define SSMF_VF_LOG2_CURVE(mantissa, mantissaM1) SSMF_VF_DIV(SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY4M(mantissa, 4167.7301872918406, 5522.5148934114723, 1980.6031106095261, 178.08851861856616)), SSMF_VF_POLY4(mantissa, 4167.7311549300921, 7367.5372709525045, 4080.8900698751668, 774.38647159574089, 36.955473282984883))
#elif SSMF_ACCURACY <= 43 // 44.08 bits
#	define SSMF_VF_LOG2_CURVE(mantissa, mantissaM1) SSMF_VF_DIV(SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY4M(mantissa, 297.67500491434294, 460.80363178588181, 213.87328780984984, 31.568450793030927)), SSMF_VF_POLY5(mantissa, 297.67501625201004, 592.58273558781423, 393.25976268431577, 100.73798113118333, 8.7102964327181933, 0.1496575691326113))
#elif SSMF_ACCURACY <= 48 // 48.50 bits
#	define SSMF_VF_LOG2_CURVE(mantissa, mantissaM1) SSMF_VF_DIV(SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY5M(mantissa, 27669.209715649555, 48061.607127308169, 26564.217866364506, 5218.706787092744, 286.09318187352375)), SSMF_VF_POLY5(mantissa, 27669.20991276007, 60310.626293978567, 45553.358309032737, 14175.149115843604, 1682.0498613576738, 53.295745555738556))
#else // 53.16 bits
#	define SSMF_VF_LOG2_CURVE(mantissa, mantissaM1) SSMF_VF_DIV(SSMF_VF_MUL(mantissaM1, SSMF_VF_POLY5M(mantissa, 1539.949449031581, 3012.7926448644621, 1985.4447244334174, 516.40343770651327, 47.614385556181887)), SSMF_VF_POLY6(mantissa, 1539.9494508493094, 3694.5206013958505, 3191.8801514383294, 1211.3930558551092, 197.70059821180519, 11.54910988714642, 0.13957288228854356))
#endif

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

	SSMF_VF curve = SSMF_VF_LOG2_CURVE(mantissa, mantissaM1);
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

// log10(): multiplies log2() if not otherwise defined
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(log10)(SSMF_VF v) {
#if defined(SSMF_VF_LOG10)
	return SSMF_VF_LOG10(v);
#else
	return SSMF_VF_MUL(SSMF_NAME(log2)(v), SSMF_VF_F(0.30102999566398120)); // log10(2)
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

// If it can, exp() uses exp2() (or vice-versa if native exp() is provided), but absolute errors in v*log2(e) become relative errors in the result.
// For "small range" output values, this is mostly OK, but we need a more accurate split otherwise.
#if SSMF_BITS == 32
#	if (SSMF_ACCURACY > 20) || (defined(SSMF_FULL_RANGE) && SSMF_ACCURACY > 18)
#		define SSMF_EXP_SPLIT
#	endif
#elif SSMF_BITS == 64 && SSMF_ACCURACY > 44
#	define SSMF_EXP_SPLIT
#endif

// exp2(): splits into an integer part (which becomes the exponent bits) and a fractional part in [-0.5, 0.5], which is curve-fitted
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(exp2)(SSMF_VF v) {
#if defined(SSMF_VF_EXP2)
	return SSMF_VF_EXP2(v);
#elif defined(SSMF_VF_EXP) && !defined(SSMF_EXP_SPLIT)
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
#		error "Need SSMF_VF_EXP2 (or SSMF_VF_EXP with a slightly approximate resolution) if it's not 32/64-bit IEEE"
#	endif
	v = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(minInput)), SSMF_VF_F(maxInput));
#	if defined(SSMF_VF_ROUND_INNER)
	SSMF_VF integer = SSMF_VF_ROUND_INNER(v);
	SSMF_VF scale = SSMF_VF_LSHIFT_K(SSMF_VF_ADD(integer, SSMF_VF_F(exponentMagic)), mantissaBits);
#	else
	// Our magic number includes the exponent bias, and we've clamped so that abs() is a no-op (but still a barrier)
	SSMF_VF withMagic = SSMF_NAME(abs)(SSMF_VF_ADD(v, SSMF_VF_F(exponentMagic)));
	SSMF_VF scale = SSMF_VF_LSHIFT_K(withMagic, mantissaBits);
	// The integer's lowest mantissa bit is 0, so BARRIER_LOWBIT fences it from the subtraction below
	SSMF_VF integer = SSMF_VF_BARRIER_LOWBIT(SSMF_VF_SUB(withMagic, SSMF_VF_F(exponentMagic)));
#	endif
	SSMF_VF fraction = SSMF_VF_SUB(v, integer);
#	if defined(SSMF_VF_EXP)
	// If we have a (presumably fast) built-in exp, we might as well use it for the fractional part instead of our fit
	SSMF_VF curve = SSMF_VF_EXP(SSMF_VF_MUL(fraction, SSMF_VF_F(0.6931471805599453)));
#	else
	SSMF_VF curve = SSMF_VF_EXP2_SMALL(fraction);
#	endif
	return SSMF_VF_MUL(curve, scale);
#endif
}

// Minimax fits (relative error, exact at 0) of 2^x for |x| < 0.0339 (float) or 7.5e-6 (double), written as (2^x - 1)/x so that expm1() can use them too
#ifdef SSMF_EXP_SPLIT
#	if SSMF_BITS == 32
#		if SSMF_ACCURACY <= 20 // 20.81 bits
#			define SSMF_VF_EXP2M1_RATIO_TINY(x) SSMF_VF_POLY1(x, 0.69319487647193889, 0.24022640853079075)
#		else // 28.77 bits
#			define SSMF_VF_EXP2M1_RATIO_TINY(x) SSMF_VF_POLY2(x, 0.69314718347733817, 0.24023566361042048, 0.055501787862308895)
#		endif
#	elif SSMF_BITS == 64 // only needed for SSMF_ACCURACY > 44
#		define SSMF_VF_EXP2M1_RATIO_TINY(x) SSMF_VF_POLY1(x, 0.69314718056228686, 0.24022650695910072) // 57.25 bits
#	else
#		error "SSMF_EXP_SPLIT is somehow set even though it's not 32-/64-bit IEEE"
#	endif
#endif

// exp()
// For float this costs up to ~2 bits (18 bits near ±88), and for double ~4 bits (44 bits near ±709), or ~1 bit less within SSMF_FULL_RANGE's small-input range (|x| up to ~35 / ~345)
// When that would miss SSMF_ACCURACY, we split both v and log2(e) so the main product is exact, and correct for the (small) remainder with a separate polynomial
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(exp)(SSMF_VF v) {
#if defined(SSMF_VF_EXP)
	return SSMF_VF_EXP(v);
#elif !defined(SSMF_EXP_SPLIT)
	return SSMF_NAME(exp2)(SSMF_VF_MUL(v, SSMF_VF_F(1.4426950408889634)));
#else
	// The best thing would be "Cody-Waite argument reduction" but I'm still wrapping my head around that.
	// Instead, we start by zeroing out half the mantissa of the two numbers (v, log2e), which we can then multiply exactly.
	// We're left with the remainder terms, so the full sum would be (vTrunc + vRem)*(log2eTrunc + log2eRem).
#	if SSMF_BITS == 32
	const float log2eTrunc = 1.44287109375f, log2eRem = -1.76052861e-04f;
	v = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-87.3258819580078125)), SSMF_VF_F(88.3627700805664062));
	SSMF_VF vTrunc = SSMF_VF_AND(v, SSMF_VF_I(0xFFFFF000)); // clear 12, keep 12
#	elif SSMF_BITS == 64
	const double log2eTrunc = 1.4426950514316559, log2eRem = -1.0542692476429138e-08; // log2(e) keeps 26 mantissa bits
	v = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-708.39641335555979)), SSMF_VF_F(709.43545721726324));
	SSMF_VF vTrunc = SSMF_VF_AND(v, SSMF_VF_I(0xFFFFFFFFFC000000LL)); // drop 26, keep 27
#	else
#		error "SSMF_EXP_SPLIT is somehow set even though it's not 32-/64-bit IEEE"
#	endif
	// Take exp2() of the exact part
	SSMF_VF expCoarse = SSMF_NAME(exp2)(SSMF_VF_MUL(vTrunc, SSMF_VF_F(log2eTrunc)));

	// We compute rem = vRem*log2e + vTrunc*log2eRem, which rounds but it's already much smaller so it's fine
	SSMF_VF vRem = SSMF_VF_SUB(v, vTrunc);
	SSMF_VF rem = SSMF_VF_FMA(vRem, SSMF_VF_F(1.4426950408889634), SSMF_VF_MUL(vTrunc, SSMF_VF_F(log2eRem)));
	// 2^rem, using the fits above (since |rem| is bounded)
	SSMF_VF correction = SSMF_VF_FMAK(SSMF_VF_EXP2M1_RATIO_TINY(rem), rem, 1);
	// Barrier because exp2() ends with a multiplication, and we don't want that reassociating (since the power *could* be subnormal or 0)
	expCoarse = SSMF_VF_BARRIER_HIGHBIT(expCoarse, rem);
	return SSMF_VF_MUL(expCoarse, correction);
#endif
}

// Approximations for (2^x - 1)/x, over [-0.5, 0.5]
#if SSMF_ACCURACY <= 6 // 6.65 bits
#	define SSMF_VF_EXPM1_SMALL(fraction) SSMF_VF_POLY1(fraction, 0.7000442826638025, 0.24021722076323884)
#elif SSMF_ACCURACY <= 11 // 11.17 bits
#	define SSMF_VF_EXPM1_SMALL(fraction) SSMF_VF_POLY2(fraction, 0.6931783036650985, 0.24202885767082458, 0.055503963410521724)
#elif SSMF_ACCURACY <= 16 // 16.03 bits
#	define SSMF_VF_EXPM1_SMALL(fraction) SSMF_VF_POLY3(fraction, 0.6931368577797392, 0.24023252161982106, 0.055837288260277985, 0.009618110776386968)
#elif SSMF_ACCURACY <= 21 // 21.14 bits
#	define SSMF_VF_EXPM1_SMALL(fraction) SSMF_VF_POLY4(fraction, 0.6931471435061761, 0.2402235134314774, 0.05550500213506492, 0.00966624542474763, 0.0013333538697077286)
#elif SSMF_ACCURACY <= 26 // 26.48 bits
#	define SSMF_VF_EXPM1_SMALL(fraction) SSMF_VF_POLY5(fraction, 0.6931471879266923, 0.24022649794966477, 0.05550357433634894, 0.009618237493735582, 0.0013390735478205204, 0.00015403512759642408)
#elif SSMF_ACCURACY <= 32 // 32.00 bits
#	define SSMF_VF_EXPM1_SMALL(fraction) SSMF_VF_POLY6(fraction, 0.6931471805815692, 0.24022650920051655, 0.055504107275970826, 0.009618057037010458, 0.0013333669451609847, 0.00015461329895107015, 1.5252719788364698e-05)
#elif SSMF_ACCURACY <= 37 // 37.70 bits
#	define SSMF_VF_EXPM1_SMALL(fraction) SSMF_VF_POLY7(fraction, 0.6931471805568766, 0.24022650696489165, 0.05550410906007004, 0.00961812894252873, 0.0013333478814173772, 0.00015403629590569136, 1.5303609855524417e-05, 1.3215476850803238e-06)
#elif SSMF_ACCURACY <= 43 // 43.55 bits
#	define SSMF_VF_EXPM1_SMALL(fraction) SSMF_VF_POLY8(fraction, 0.6931471805599377, 0.24022650695814085, 0.05550410866573463, 0.009618129159051157, 0.0013333557983831008, 0.0001540345613381229, 1.5252811940269293e-05, 1.3255160531929068e-06, 1.017807963809425e-07)
#else // 54.91 bits
#	define SSMF_VF_EXPM1_SMALL(fraction) SSMF_NAME(pow2)(SSMF_VF_POLY8(fraction, 0.8325546111576978, 0.14427072034653474, 0.020833508967823437, 0.002166103200344167, 0.0001647400524524726, 9.756663689790118e-06, 5.292758407174255e-07, 2.9195119014317305e-08, 1.1392525047316455e-09))
#endif

// expm1(): exp(v) - 1, split into an integer + fraction, then computed as [2^k(2^f - 1)] + [2^k - 1]
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(expm1)(SSMF_VF v) {
#if defined(SSMF_VF_EXPM1)
	return SSMF_VF_EXPM1(v);
#else
#	if SSMF_BITS == 32
	const int mantissaBits = 23;
	const float exponentMagic = 12582912 + 127; // 1.5 * 2^23 + 127
#	elif SSMF_BITS == 64
	const int mantissaBits = 52;
	const double exponentMagic = 6755399441056767LL; // 1.5 * 2^52 + 1023
#	else
#		error "Need SSMF_VF_EXPM1 if it's not 32/64-bit IEEE"
#	endif

#	if defined(SSMF_EXP_SPLIT) // Split into two terms for increased accuracy - see exp() for what this is actually doing
#		if SSMF_BITS == 32
	const float log2eTrunc = 1.44287109375f, log2eRem = -1.76052861e-04f;
	v = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-87.3258819580078125)), SSMF_VF_F(88.3627700805664062));
	SSMF_VF vTrunc = SSMF_VF_AND(v, SSMF_VF_I(0xFFFFF000));
#		else // already checked SSMF_BITS above
	const double log2eTrunc = 1.4426950514316559, log2eRem = -1.0542692476429138e-08;
	v = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-708.39641335555979)), SSMF_VF_F(709.43545721726324));
	SSMF_VF vTrunc = SSMF_VF_AND(v, SSMF_VF_I(0xFFFFFFFFFC000000LL));
#		endif
	SSMF_VF vLog2 = SSMF_VF_MUL(vTrunc, SSMF_VF_F(log2eTrunc));
	SSMF_VF vRem = SSMF_VF_SUB(v, vTrunc);
	SSMF_VF rem = SSMF_VF_FMA(vRem, SSMF_VF_F(1.4426950408889634), SSMF_VF_MUL(vTrunc, SSMF_VF_F(log2eRem)));
#	else
#		if SSMF_BITS == 32
	const float minInput = -126, maxInput = 127.49609375; // same as exp2()
#		else // already checked SSMF_BITS above
	const double minInput = -1022, maxInput = 1023.4990234375;
#		endif
	SSMF_VF vLog2 = SSMF_VF_MUL(v, SSMF_VF_F(1.4426950408889634)); // log2(e)
	vLog2 = SSMF_VF_MIN(SSMF_VF_MAX(vLog2, SSMF_VF_F(minInput)), SSMF_VF_F(maxInput));
#	endif

	// separate vLog2 (which is the exact part, if we're doing a split) into integer/fraction parts
#	if defined(SSMF_VF_ROUND_INNER)
	SSMF_VF integer = SSMF_VF_ROUND_INNER(vLog2);
	SSMF_VF scale = SSMF_VF_LSHIFT_K(SSMF_VF_ADD(integer, SSMF_VF_F(exponentMagic)), mantissaBits);
#	else
	SSMF_VF withMagic = SSMF_NAME(abs)(SSMF_VF_ADD(vLog2, SSMF_VF_F(exponentMagic)));
	SSMF_VF scale = SSMF_VF_LSHIFT_K(withMagic, mantissaBits);
	SSMF_VF integer = SSMF_VF_BARRIER_LOWBIT(SSMF_VF_SUB(withMagic, SSMF_VF_F(exponentMagic)));
#	endif
	SSMF_VF fraction = SSMF_VF_SUB(vLog2, integer);
	
	// Slightly funky recombination because the whole point of expm1() is to keep accuracy near 0
	SSMF_VF curveM1 = SSMF_VF_MUL(fraction, SSMF_VF_EXPM1_SMALL(fraction)); // 2^f - 1
	// Stop fast-math from refactoring into 2^k(2^f - 1 + 1) - 1, which loses our small-input precision
	SSMF_VF scaleM1 = SSMF_VF_BARRIER_HIGHBIT(SSMF_VF_SUB(scale, SSMF_VF_F(1)), fraction /* known to be <= 0.5 */);
#	if defined(SSMF_EXP_SPLIT)
	SSMF_VF result = SSMF_VF_ADD(SSMF_VF_MUL(curveM1, scale), scaleM1); // 2^vLog2 - 1
	// 2^(vLog2 + rem) - 1 = result*(1 + d) + d, where d = 2^rem - 1
	SSMF_VF d = SSMF_VF_MUL(rem, SSMF_VF_EXP2M1_RATIO_TINY(rem));
	return SSMF_VF_FMA(SSMF_VF_ADD(result, SSMF_VF_F(1)), d, result);
#	else
	return SSMF_VF_ADD(SSMF_VF_MUL(curveM1, scale), scaleM1);
#	endif
#endif
}

// cosh() = (e^x + e^-x)/2 = h + 1/(4h), for h = e^|x|/2
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(cosh)(SSMF_VF v) {
#if defined(SSMF_VF_COSH)
	return SSMF_VF_COSH(v);
#else
	SSMF_VF h = SSMF_VF_MUL(SSMF_NAME(exp)(SSMF_NAME(abs)(v)), SSMF_VF_F(0.5));
	return SSMF_VF_ADD(h, SSMF_VF_DIV(SSMF_VF_F(0.25), h));
#endif
}

// sinh() = (e^x - e^-x)/2 = (m + m/(m + 1))/2 for m = expm1(|x|), with the sign of x.  This avoids cancellation near 0, where m ≈ m/(m + 1) ≈ |x|
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(sinh)(SSMF_VF v) {
#if defined(SSMF_VF_SINH)
	return SSMF_VF_SINH(v);
#else
	SSMF_VF m = SSMF_NAME(expm1)(SSMF_NAME(abs)(v));
	SSMF_VF sum = SSMF_VF_ADD(m, SSMF_VF_DIV(m, SSMF_VF_ADD(m, SSMF_VF_F(1))));
	return SSMF_NAME(copysign)(SSMF_VF_MUL(sum, SSMF_VF_F(0.5)), v);
#endif
}

// pow(a, b): ostensibly = exp2(b*log2(a))
// However (similar to using exp2() for exp() or vice-versa) absolute errors in b*log2(a) become relative errors in the output.  This is OK when our requirements aren't very strict, so we check SSMF_EXP_SPLIT.
// Otherwise, we pull the same "split into exact + remainder" as in exp().  In the end, the accuracy is measured as if we rounded `a`, rather than the result.
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(pow)(SSMF_VF a, SSMF_VF b) {
#if defined(SSMF_VF_POW)
	return SSMF_VF_POW(a, b);
#elif !defined(SSMF_EXP_SPLIT)
#	if defined(SSMF_VF_EXP) && defined(SSMF_VF_LOG) // if we have both of these supplied, use them
	return SSMF_NAME(exp)(SSMF_VF_MUL(b, SSMF_NAME(log)(a)));
#	else // otherwise, exp2/log2 are the more natural fit for floating-point
	return SSMF_NAME(exp2)(SSMF_VF_MUL(b, SSMF_NAME(log2)(a)));
#	endif
#else
	// The truncated parts of b and log2(a) have few enough bits that their product is exact
#	if SSMF_BITS == 32
	SSMF_VF bMask = SSMF_VF_I(0xFFFFF000), log2aMask = bMask; // keep 12 and 12
	const float remLimit = 0.125, maxB = 1e30f;
#	else
	SSMF_VF bMask = SSMF_VF_I(0xFFFFFFFFFC000000LL), log2aMask = SSMF_VF_I(0xFFFFFFFFF8000000LL); // keep 27 and 26
	const double remLimit = 1.0/16384, maxB = 1e300;
#	endif
	// See log2() for what this is doing
#	if SSMF_BITS == 32
	const int mantissaMaskInt = 0x007FFFFF;
	const int exponentBits = 8, exponentMagic = 256, exponentRoundingMask = 511;
	const float exponentBase = 127;
#	else
	const long long mantissaMaskInt = 0x000FFFFFFFFFFFFFLL;
	const int exponentBits = 11, exponentMagic = 2048, exponentRoundingMask = 4095;
	const double exponentBase = 1023;
#	endif
	SSMF_VF mantissa = SSMF_VF_XOR(SSMF_VF_LSHIFT_K(SSMF_VF_AND(a, SSMF_VF_I(mantissaMaskInt)), 1), SSMF_VF_F(1));
	SSMF_VF mantissaM1 = SSMF_VF_SUB(mantissa, SSMF_VF_F(1));
	SSMF_VF exponentOffset = SSMF_VF_OR(SSMF_VF_RSHIFT_K(a, exponentBits), SSMF_VF_F(exponentMagic));
	SSMF_VF exponentRounded = SSMF_VF_AND(SSMF_VF_ADD(exponentOffset, SSMF_VF_F(0.5)), SSMF_VF_F(exponentRoundingMask));
	SSMF_VF exponent = SSMF_VF_BARRIER_LOWBIT(SSMF_VF_SUB(exponentRounded, SSMF_VF_F(exponentBase + exponentMagic)));
	// A supplied log() is only used for the mantissa, so that its rounding error is relative to the curve, not to log2(a).  We need to get the actual mantissa with (1 + m)/2 though.
#	if defined(SSMF_VF_LOG2)
	SSMF_VF curve = SSMF_VF_LOG2(SSMF_VF_MUL(SSMF_VF_ADD(mantissa, SSMF_VF_F(1)), SSMF_VF_F(0.5)));
#	elif defined(SSMF_VF_LOG)
	SSMF_VF curve = SSMF_VF_MUL(SSMF_VF_LOG(SSMF_VF_MUL(SSMF_VF_ADD(mantissa, SSMF_VF_F(1)), SSMF_VF_F(0.5))), SSMF_VF_F(1.4426950408889634));
#	else
	SSMF_VF curve = SSMF_VF_LOG2_CURVE(mantissa, mantissaM1);
#	endif

	SSMF_VF log2a = SSMF_VF_ADD(exponent, curve);
	SSMF_VF log2aTrunc = SSMF_VF_AND(log2a, log2aMask);
	// (exponent - log2aTrunc) is exact and less than 2, so we fence to protect its accuracy from reassociation
	SSMF_VF log2aRem = SSMF_VF_ADD(SSMF_VF_BARRIER_HIGHBIT(SSMF_VF_SUB(exponent, log2aTrunc), curve), curve);
	// Now the same as exp(), but the remainder can be larger, so we use expm1()'s polynomial for 2^rem - 1
#	ifdef SSMF_FULL_RANGE
	b = SSMF_VF_MIN(SSMF_VF_MAX(b, SSMF_VF_F(-maxB)), SSMF_VF_F(maxB)); // so b*log2(a) can't overflow
#	endif
	SSMF_VF bTrunc = SSMF_VF_AND(b, bMask);
	SSMF_VF bRem = SSMF_VF_SUB(b, bTrunc);
	SSMF_VF rem = SSMF_VF_FMA(bRem, log2a, SSMF_VF_MUL(bTrunc, log2aRem));
	// Only out-of-range results have a remainder this big, so this is just to prevent overflows
	rem = SSMF_VF_MIN(SSMF_VF_MAX(rem, SSMF_VF_F(-remLimit)), SSMF_VF_F(remLimit));
	SSMF_VF result = SSMF_VF_BARRIER_HIGHBIT(SSMF_NAME(exp2)(SSMF_VF_MUL(bTrunc, log2aTrunc)), rem);
	return SSMF_VF_MUL(result, SSMF_VF_FMAK(SSMF_VF_EXPM1_SMALL(rem), rem, 1));
#endif
}

// erf(): like tanh(), each accuracy level has its own fit range based on when the function gets within accuracy of ±1.  The target is scaled (input and output) so it ends at ±1 on those limits, while keeping the same gradient.  This scaling adds an extra layer of error, so we use fits with a reported error at least 0.5 bits above SSMF_ACCURACY.
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(erf)(SSMF_VF v) {
#if defined(SSMF_VF_ERF)
	return SSMF_VF_ERF(v);
#else
#	if SSMF_ACCURACY <= 3 // 4.20 bits, clamped at 1.572149
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-1.57214906012589)), SSMF_VF_F(1.57214906012589));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_MUL(x, SSMF_VF_POLY1(x2, 1.0671706121376627, -0.17441714846233314));
#	elif SSMF_ACCURACY <= 9 // 9.86 bits, clamped at 2.194386
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-2.1943858782722683)), SSMF_VF_F(2.1943858782722683));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_MUL(x, SSMF_VF_POLY4(x2, 1.1271673143870384, -0.36317621432826197, 0.09250451238284009, -0.013630116240415439, 0.0008449376854663133));
#	elif SSMF_ACCURACY <= 11 // 12.20 bits, clamped at 2.716987
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-2.7169869040075127)), SSMF_VF_F(2.7169869040075127));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_MUL(x, SSMF_VF_POLY6(x2, 1.1281391852383258, -0.3731535821823677, 0.10661896488417438, -0.021529038030736202, 0.002825613868945154, -0.00021207909274319325, 6.8173131252539295e-06));
#	elif SSMF_ACCURACY <= 17 // 17.53 bits, clamped at 3.164559
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-3.164559220780709)), SSMF_VF_F(3.164559220780709));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_MUL(x, SSMF_VF_DIV(SSMF_VF_POLY2M(x2, 30.528673233428872, 3.6418548953504843), SSMF_VF_POLY4(x2, 27.055474937792138, 12.243757753920422, 2.2668459267255363, 0.17137611711239722, -0.0019620096972315245)));
#	elif SSMF_ACCURACY <= 19 // 20.05 bits, clamped at 3.560571
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-3.5605714048479147)), SSMF_VF_F(3.5605714048479147));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_MUL(x, SSMF_VF_DIV(SSMF_VF_POLY5M(x2, -3106865.866273123, -368020.2994943098, -111234.39291108915, -880.2060260247853, -44.36168164833076), SSMF_VF_POLY3(x2, -2753385.6428314648, -1244006.10043064, -237659.8131511405, -21549.134908123164)));
#	elif SSMF_ACCURACY <= 25 // 25.56 bits, clamped at 3.919206
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-3.9192059604381617)), SSMF_VF_F(3.9192059604381617));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_MUL(x, SSMF_VF_DIV(SSMF_VF_POLY5M(x2, 555505.7779158001, 94173.72606877037, 26016.604727788574, 1870.0224574320414, 140.1708827640445), SSMF_VF_POLY5(x2, 492304.1875848896, 247560.36131451675, 56347.958704888566, 7402.084237453606, 575.4321470818073, 18.6529481119521)));
#	elif SSMF_ACCURACY <= 28 // 29.07 bits, clamped at 4.249255
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-4.2492546646492)), SSMF_VF_F(4.2492546646492));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_MUL(x, SSMF_VF_DIV(SSMF_VF_POLY5M(x2, 244709.33249175613, 39809.19179985701, 11707.591716179486, 818.5879882123378, 80.85897779346756), SSMF_VF_POLY7(x2, 216867.9989791474, 107569.3257460856, 24545.12439904256, 3314.013461369649, 278.62103182800513, 13.349622004673733, 0.012730711288459696, 0.00018935856200199728)));
#	elif SSMF_ACCURACY <= 34 // 37.91 bits, clamped at 4.630326
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-4.630325675299484)), SSMF_VF_F(4.630325675299484));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_MUL(x, SSMF_VF_DIV(SSMF_VF_POLY7M(x2, 84451758.12951213, 15117027.267969407, 4476111.263979304, 395033.584481442, 44086.624161564796, 2061.648180987379, 98.53483161862765), SSMF_VF_POLY8(x2, 74843421.95648743, 38344923.90158103, 9264149.509441251, 1385632.8753251638, 141010.74315242804, 10018.963085682055, 480.7217974187914, 13.287303669715348, 0.03193815854546952)));
#	elif SSMF_ACCURACY <= 37 // 38.18 bits, clamped at 4.983477
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-4.9834766432666395)), SSMF_VF_F(4.9834766432666395));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_MUL(x, SSMF_VF_DIV(SSMF_VF_POLY9M(x2, 10429216507965.994, 1841799730724.7407, 552358978534.035, 48105330129.091446, 5526400628.349616, 255183401.28030628, 13142682.2449327, 118617.77145230168, -363.03812995852843), SSMF_VF_POLY7(x2, 9242652480706.217, 4713136674530.709, 1136295694350.8665, 170147011394.84467, 17411124627.960976, 1251870573.9337568, 61315228.29996819, 1791439.2678924624)));
#	elif SSMF_ACCURACY <= 44 // 48.32 bits, clamped at 5.314017
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-5.314017270170049)), SSMF_VF_F(5.314017270170049));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_MUL(x, SSMF_VF_DIV(SSMF_VF_POLY9M(x2, 41084478862.17889, 7707660143.58946, 2316575712.134541, 226354991.94654047, 27435170.49436063, 1637053.7547770154, 103388.9152244387, 3556.13411912258, 111.03419512389004), SSMF_VF_POLY11(x2, 36410171385.85752, 18967459746.78011, 4734481214.52578, 748925161.6997828, 83547960.96769665, 6904499.753785638, 430297.08095463953, 20032.584686584345, 661.972592741042, 13.412985451255395, 0.036349463540466795, -2.9416960484193842e-05)));
#	elif SSMF_ACCURACY <= 47 // 48.15 bits, clamped at 5.625769
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-5.625768509906709)), SSMF_VF_F(5.625768509906709));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_MUL(x, SSMF_VF_DIV(SSMF_VF_POLY8M(x2, 650342104.9175574, 110759958.63670222, 34550401.54982963, 2954768.8027221244, 373167.39640365687, 18773.592641065272, 1220.7475998629427, 32.36629038869807), SSMF_VF_POLY13(x2, 576350684.1335638, 290275352.3169575, 69742878.49868727, 10561321.93846909, 1119891.5066582405, 87109.28936969487, 5035.137217425853, 212.08131859432478, 6.027709466903211, 0.08804367310100476, -0.0005214952934476771, 3.8841777001908255e-06, -2.0907132627026673e-08, 5.6186888546069345e-11)));
#	else // 58.79 bits, clamped at 5.921587
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-5.921587195794507)), SSMF_VF_F(5.921587195794507));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_MUL(x, SSMF_VF_DIV(SSMF_VF_POLY14(x2, 1.1283791670955126, 0.20998622564334127, 0.06437738459666888, 0.006341606380571237, 0.0008079461846980791, 5.077197149628065e-05, 3.5748864703648773e-06, 1.4692278587922402e-07, 6.064646042909775e-09, 1.453044478354801e-10, 3.0346846450608735e-12, 1.4734695233403275e-14, -3.1866818019754073e-17, 8.866943081863075e-20, -1.4480120285727068e-22), SSMF_VF_POLY11(x2, 1.0, 0.519428780472661, 0.1301958984440131, 0.02088538090203082, 0.0023959497391376652, 0.0002078225400688703, 1.4028596415341336e-05, 7.452377673870384e-07, 3.0997530211822836e-08, 9.827610988913585e-10, 2.213744572126085e-11, 2.914478141663195e-13)));
#	endif
	return SSMF_VF_MIN(SSMF_VF_MAX(y, SSMF_VF_F(-1)), SSMF_VF_F(1)); // clamp to avoid rounding errors
#endif
}

// erfc(): an erf() variant which gets accurate results for the x>0 tail, not at x=0.  We fit erfc(x)*exp(x²) for x >= 0, and use symmetry to get the x < 0 curve.
// The fits were generated for [0, 26.54], which is when erfc() reaches the smallest normal double.  However, the curve is still pretty good after that (positive and monotonically decreasing) so we only clamp to x <= 100.
// Because of the multiplication inside exp(x²), relative error increases for larger x, and we never get more than 17 bits (float) or 43 bits (double) of accuracy, but honestly that's fine.  The main thing for erfc() is that it keeps decreasing towards 0, rather than `1 - erf()` which has naff resolution from the cancellation.
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(erfc)(SSMF_VF v) {
#if defined(SSMF_VF_ERFC)
	return SSMF_VF_ERFC(v);
#else
	SSMF_VF x = SSMF_VF_MIN(SSMF_NAME(abs)(v), SSMF_VF_F(100));
#	if SSMF_ACCURACY <= 2 // 3.37 bits
	SSMF_VF ratio = SSMF_VF_DIV(SSMF_VF_F(1), SSMF_VF_POLY1(x, 1.0, 1.5792223416705684));
#	elif SSMF_ACCURACY <= 3 // 4.09 bits
	SSMF_VF ratio = SSMF_VF_DIV(SSMF_VF_F(1), SSMF_VF_POLY2(x, 1.0, 1.4415837994682252, 0.015250882189492613));
#	elif SSMF_ACCURACY <= 8 // 8.81 bits
	SSMF_VF ratio = SSMF_VF_DIV(SSMF_VF_POLY1M(x, 1.7885427216229213), SSMF_VF_POLY2(x, 1.7885427216229213, 2.9634212389295636, 1.783256750179798));
#	elif SSMF_ACCURACY <= 9 // 9.55 bits
	SSMF_VF ratio = SSMF_VF_DIV(SSMF_VF_POLY2M(x, 3508.6058660233703, 1764.6261525053897), SSMF_VF_POLY2(x, 3508.6058660233703, 5653.89102030055, 3188.932586153108));
#	elif SSMF_ACCURACY <= 13 // 14.33 bits
	SSMF_VF ratio = SSMF_VF_DIV(SSMF_VF_POLY2M(x, 3.6764192516952323, 3.0296951396929006), SSMF_VF_POLY3(x, 3.6764192516952323, 7.182784226747743, 5.391814328046256, 1.7717989862062038));
#	elif SSMF_ACCURACY <= 14 // 15.13 bits
	SSMF_VF ratio = SSMF_VF_DIV(SSMF_VF_POLY2M(x, 4.05178751940303, 3.1658254416805245), SSMF_VF_POLY4(x, 4.05178751940303, 7.740926831612598, 5.656679348301302, 1.7696196136359552, 5.9604224417670225e-05));
#	elif SSMF_ACCURACY <= 19 // 19.91 bits
	SSMF_VF ratio = SSMF_VF_DIV(SSMF_VF_POLY3M(x, 9.073908976949511, 9.731974192701692, 4.704873428953177), SSMF_VF_POLY4(x, 9.073908976949511, 19.97039915435147, 18.169924781531925, 8.337137386314694, 1.7724967242083165));
#	elif SSMF_ACCURACY <= 20 // 20.76 bits
	SSMF_VF ratio = SSMF_VF_DIV(SSMF_VF_POLY4M(x, 4690012.897102576, 4849755.815086881, 2282985.5055254702, 466042.3089363557), SSMF_VF_POLY4(x, 4690012.897102576, 10141753.203303427, 9038224.21423225, 4043965.0766810966, 826147.4206019717));
#	elif SSMF_ACCURACY <= 25 // 25.53 bits
	SSMF_VF ratio = SSMF_VF_DIV(SSMF_VF_POLY4M(x, 25.519356766489253, 33.058804952502875, 20.228880392659836, 6.605391972290597), SSMF_VF_POLY5(x, 25.519356766489253, 61.85434683189682, 64.50412080363658, 36.73662647330119, 11.707933255065262, 1.7724509376166488));
#	elif SSMF_ACCURACY <= 30 // 31.20 bits
	SSMF_VF ratio = SSMF_VF_DIV(SSMF_VF_POLY5M(x, 79.67977600943041, 119.46515908617441, 86.4057047873009, 36.13541440771226, 8.724168307007595), SSMF_VF_POLY6(x, 79.67977600943041, 209.3741557956329, 242.97942619463916, 160.87299989236016, 64.9350814353176, 15.463169943454702, 1.7724540548746264));
#	elif SSMF_ACCURACY <= 31 // 32.13 bits
	SSMF_VF ratio = SSMF_VF_DIV(SSMF_VF_POLY6M(x, 9154502295.44563, 13422061324.750505, 9534763698.962118, 3911074326.9803414, 923487124.0998269, 102798512.1796882), SSMF_VF_POLY6(x, 9154502295.44563, 23751810839.86024, 27181313925.513668, 17716557216.046066, 7023434411.256254, 1636832100.2836833, 182205780.77654034));
#	elif SSMF_ACCURACY <= 36 // 36.90 bits
	SSMF_VF ratio = SSMF_VF_DIV(SSMF_VF_POLY6M(x, 271.4098059348937, 458.16292172385283, 377.33319021844204, 187.3227519029718, 58.653441014136654, 11.041705239881674), SSMF_VF_POLY7(x, 271.4098059348937, 764.4160927018057, 968.4745716312567, 719.8820411064983, 341.8076077440001, 104.84669023426022, 19.570914338086865, 1.7724538362446827));
#	elif SSMF_ACCURACY <= 37 // 37.87 bits
	SSMF_VF ratio = SSMF_VF_DIV(SSMF_VF_POLY6M(x, 304.3175231305502, 504.15920622587043, 408.78809636025727, 199.6965825400279, 61.4368293070561, 11.332309368720297), SSMF_VF_POLY8(x, 304.3175231305502, 847.5447596355855, 1060.8224191337204, 778.0854874897682, 363.99875146519105, 109.78001080490182, 20.086001415971193, 1.7724537206230708, 1.2147649916174267e-09));
#	elif SSMF_ACCURACY <= 42 // 42.63 bits
	SSMF_VF ratio = SSMF_VF_DIV(SSMF_VF_POLY7M(x, 996.1116541878035, 1857.2979292030113, 1700.9450072520667, 962.6903998559569, 360.10831788325544, 88.97105499045799, 13.54439190163487), SSMF_VF_POLY8(x, 996.1116541878035, 2981.2895678712116, 4068.858393233097, 3321.943626569206, 1784.2895025551882, 650.2786224468455, 158.58332160466466, 24.006809466144865, 1.7724538519846111));
#	elif SSMF_ACCURACY <= 43 // 43.64 bits
	SSMF_VF ratio = SSMF_VF_DIV(SSMF_VF_POLY8M(x, 22609105356789.105, 41477917948701.01, 37465397174936.18, 20911269584216.68, 7707337475935.573, 1873330154291.8206, 279855756829.60693, 20184269786.931793), SSMF_VF_POLY8(x, 22609105356789.105, 66989561419755.61, 90445917345498.94, 72986759244954.94, 38706690537754.66, 13908907552605.994, 3338279472810.9624, 496031401937.60236, 35775686930.65413));
#	else // 48.40 bits
	SSMF_VF ratio = SSMF_VF_DIV(SSMF_VF_POLY8M(x, 3902.660360084558, 7925.332400341852, 7939.464336920553, 4996.126884385436, 2138.2383998191312, 635.3686276557452, 128.29515207552387, 16.22118506909692), SSMF_VF_POLY9(x, 3902.660360084558, 12329.013046912334, 17948.605449746636, 15855.733404285023, 9404.116328077822, 3902.740848457157, 1140.5372410012203, 228.28346269874996, 28.751301952215574, 1.7724538508243493));
#	endif
	// Barrier because exp() ends by multiplying by a power of 2, which could be subnormal (and flushed to 0) if fast-math multiplies by `ratio` first
	SSMF_VF expPart = SSMF_VF_BARRIER_HIGHBIT(SSMF_NAME(exp)(SSMF_VF_MUL(SSMF_NAME(neg)(x), x)), ratio /* <= 1 */);
	SSMF_VF r = SSMF_VF_MUL(expPart, ratio);
	// For negative inputs: 2 - r
#	if SSMF_BITS == 32 || SSMF_BITS == 64
	// The sign bit shifted down one place is 2 (or 0).  Writing this as `1 - copysign(1, v)` lets fast-math reassociate to `(copysign(r, v) + 1) - copysign(1, v)` (GCC does), which loses small r
	SSMF_VF offset = SSMF_VF_RSHIFT_K(SSMF_VF_AND(v, SSMF_VF_F(-0.0)), 1);
#	else
	SSMF_VF offset = SSMF_VF_SUB(SSMF_VF_F(1), SSMF_NAME(copysign)(SSMF_VF_F(1), v));
#	endif
	return SSMF_VF_ADD(offset, SSMF_NAME(copysign)(r, v));
#endif
}

// Minimax fits (relative error, exact at 0 and 0.5) of sin(πx) for x in [-0.5, 0.5], as x*R(x^2)
#if SSMF_ACCURACY <= 12 // 12.90 bits
#	define SSMF_VF_SINPI_SMALL(x) SSMF_VF_MUL(x, SSMF_VF_POLY2(SSMF_NAME(pow2)(x), 3.1411817285078589, -5.1429482766351837, 2.3128854504149916))
#elif SSMF_ACCURACY <= 19 // 19.81 bits
#	define SSMF_VF_SINPI_SMALL(x) SSMF_VF_MUL(x, SSMF_VF_POLY3(SSMF_NAME(pow2)(x), 3.1415892427046948, -5.1673288792131951, 2.5431271472385792, -0.55695805464366632))
#elif SSMF_ACCURACY <= 27 // 27.32 bits
#	define SSMF_VF_SINPI_SMALL(x) SSMF_VF_MUL(x, SSMF_VF_POLY4(SSMF_NAME(pow2)(x), 3.1415926348152059, -5.1677093833321717, 2.5500629779986133, -0.59819434984554865, 0.077455771970641532))
#elif SSMF_ACCURACY <= 35 // 35.32 bits
#	define SSMF_VF_SINPI_SMALL(x) SSMF_VF_MUL(x, SSMF_VF_POLY5(SSMF_NAME(pow2)(x), 3.1415926535164558, -5.1677127605906739, 2.5501631866993848, -0.59925067544734356, 0.082043239676703253, -0.0070165899481767621))
#elif SSMF_ACCURACY <= 43 // 43.74 bits
#	define SSMF_VF_SINPI_SMALL(x) SSMF_VF_MUL(x, SSMF_VF_POLY6(SSMF_NAME(pow2)(x), 3.1415926535895795, -5.1677127799718461, 2.5501640351428065, -0.59926442064757834, 0.082144690016877714, -0.00736362525866892, 0.0004469739032907512))
#else // 52.54 bits
#	define SSMF_VF_SINPI_SMALL(x) SSMF_VF_MUL(x, SSMF_VF_POLY7(SSMF_NAME(pow2)(x), 3.1415926535897927, -5.1677127800497376, 2.5501640398587706, -0.59926452874991076, 0.082145877942630272, -0.0073703591181299528, 0.00046597160939952696, -2.1115717814228518e-05))
#endif

// sincos(): reduces to half-turns t = k + f (k an integer, |f| <= 0.5), so sin(πt) = (-1)^k sin(πf) and cos(πt) = (-1)^k sin(π(0.5 - |f|))
// This is only backward-accurate: it computes sin/cos of (v/π rounded), so e.g. sin(float(π)) = 0
SSMF_ATTRIBUTES SSMF_INLINE void SSMF_NAME(sincos)(SSMF_VF v, SSMF_VF *sinOut, SSMF_VF *cosOut) {
#if defined(SSMF_VF_SINCOS)
	SSMF_VF_SINCOS(v, sinOut, cosOut);
#elif defined(SSMF_VF_SIN) && defined(SSMF_VF_COS)
	*sinOut = SSMF_VF_SIN(v);
	*cosOut = SSMF_VF_COS(v);
#else
#	if SSMF_BITS == 32
	const int signShift = 31;
	const float parityMagic = 8388608; // 2^23
#	elif SSMF_BITS == 64
	const int signShift = 63;
	const double parityMagic = 4503599627370496; // 2^52
#	else
#		error "need SSMF_VF_SINCOS (or _SIN/_COS pair) if not 32/64-bit IEEE"
#	endif
	SSMF_VF halfTurns = SSMF_VF_MUL(v, SSMF_VF_F(0.31830988618379067)); // 1/π
	SSMF_VF frac = SSMF_NAME(remainder1)(halfTurns);
	SSMF_VF rounded = SSMF_VF_BARRIER_LOWBIT(SSMF_VF_SUB(halfTurns, frac));
	// Add to a magic constant, then bit-shift to move final mantissa bit (1s unit) -> sign bit
	SSMF_VF sign = SSMF_VF_LSHIFT_K(SSMF_VF_ADD(SSMF_NAME(abs)(rounded), SSMF_VF_F(parityMagic)), signShift);
	// Intermediate rounding can push it slightly outside the correct range, so we clamp again
	SSMF_VF sinCurve = SSMF_VF_SINPI_SMALL(frac);
	*sinOut = SSMF_VF_XOR(SSMF_VF_MIN(SSMF_VF_MAX(sinCurve, SSMF_VF_F(-1)), SSMF_VF_F(1)), sign);
	// 0.5 - |f| keeps relative accuracy for cos()
	SSMF_VF fCos = SSMF_VF_SUB(SSMF_VF_F(0.5), SSMF_NAME(abs)(frac));
	SSMF_VF cosCurve = SSMF_VF_SINPI_SMALL(fCos); // fCos >= 0, so this can't be below 0
	*cosOut = SSMF_VF_XOR(SSMF_VF_MIN(cosCurve, SSMF_VF_F(1)), sign);
#endif
}

// sin(): sincos() with the cosine part unused (and optimised away)
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(sin)(SSMF_VF v) {
#if defined(SSMF_VF_SIN)
	return SSMF_VF_SIN(v);
#else
	SSMF_VF s, c;
	SSMF_NAME(sincos)(v, &s, &c);
	return s;
#endif
}

// cos(): sincos() with the sine part unused (and optimised away)
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(cos)(SSMF_VF v) {
#if defined(SSMF_VF_COS)
	return SSMF_VF_COS(v);
#else
	SSMF_VF s, c;
	SSMF_NAME(sincos)(v, &s, &c);
	return c;
#endif
}

// tan(): reduce to half-turns like sincos(), but doesn't need a parity flip.  We fit to `tan(πf)(1 - 2f)(1 + 2f)/f` (exact at 0 and 0.5), and the rational fits share the division.
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(tan)(SSMF_VF v) {
#if defined(SSMF_VF_TAN)
	return SSMF_VF_TAN(v);
#else
	SSMF_VF halfTurns = SSMF_VF_MUL(v, SSMF_VF_F(0.31830988618379067)); // 1/π
	SSMF_VF frac = SSMF_NAME(remainder1)(halfTurns);
	SSMF_VF frac2 = SSMF_NAME(pow2)(frac);
	// (1 ± 2f) is exact near f = ∓0.5, keeping relative accuracy near the poles.  We give a minimum of SSMF_EPSILON to avoid divide-by-0.
	SSMF_VF poles = SSMF_VF_MAX(SSMF_VF_MUL(SSMF_VF_FMAK(frac, SSMF_VF_F(-2), 1), SSMF_VF_FMAK(frac, SSMF_VF_F(2), 1)), SSMF_VF_F(SSMF_EPSILON*0.5));
#	if SSMF_ACCURACY <= 4 // 4.26 bits
	return SSMF_VF_DIV(SSMF_VF_MUL(frac, SSMF_VF_POLY1(SSMF_NAME(abs)(frac), 3.141592653589793, -1.1902271282389358)), poles);
#	elif SSMF_ACCURACY <= 9 // 9.51 bits
	return SSMF_VF_DIV(SSMF_VF_MUL(frac, SSMF_VF_POLY2(SSMF_NAME(abs)(frac), 3.141592653589793, 0.06876973987865644, -2.5179937362351845)), poles);
#	elif SSMF_ACCURACY <= 15 // 15.24 bits
	return SSMF_VF_DIV(SSMF_VF_MUL(frac, SSMF_VF_POLY1M(frac2, -1.0358644665125778)), SSMF_VF_MUL(poles, SSMF_VF_POLY1(frac2, -0.32972590043745165, 0.0844705843932185)));
#	elif SSMF_ACCURACY <= 22 // 22.75 bits
	return SSMF_VF_DIV(SSMF_VF_MUL(frac, SSMF_VF_POLY2M(frac2, 8.990633036485976, -9.962601387845622)), SSMF_VF_MUL(poles, SSMF_VF_POLY1(frac2, 2.8618073785640794, -1.1389058108960277)));
#	elif SSMF_ACCURACY <= 30 // 30.70 bits
	return SSMF_VF_DIV(SSMF_VF_MUL(frac, SSMF_VF_POLY2M(frac2, 4.938299711552962, -5.937347165813664)), SSMF_VF_MUL(poles, SSMF_VF_POLY2(frac2, 1.5719096191258697, -0.7736532895299336, 0.03864740491711173)));
#	elif SSMF_ACCURACY <= 39 // 39.07 bits
	return SSMF_VF_DIV(SSMF_VF_MUL(frac, SSMF_VF_POLY3M(frac2, -133.51516080107467, 169.40843477392502, -36.89443557136697)), SSMF_VF_MUL(poles, SSMF_VF_POLY2(frac2, -42.499195638400586, 23.744346464276344, -2.1748414010178068)));
#	elif SSMF_ACCURACY <= 47 // 47.81 bits
	return SSMF_VF_DIV(SSMF_VF_MUL(frac, SSMF_VF_POLY3M(frac2, -57.08825965761233, 75.25156803084654, -19.16332542980464)), SSMF_VF_MUL(poles, SSMF_VF_POLY3(frac2, -18.171757434045272, 11.048974034413364, -1.3718070292209932, 0.022226653542121562)));
#	else // 56.88 bits
	return SSMF_VF_DIV(SSMF_VF_MUL(frac, SSMF_VF_POLY4M(frac2, 3493.7058877061922, -4737.95274624247, 1341.1706038516124, -97.92371710815077)), SSMF_VF_MUL(poles, SSMF_VF_POLY3(frac2, 1112.0811234753974, -718.4129557060872, 107.5660775727489, -3.528518630514403)));
#	endif
#endif
}

// tanh(): we fit `tanh(x*t)/t`, where t = tanh(1 - 2^-SSMF_ACCURACY/2).  This scaling keeps the gradient at 0, but means it hits ±1 at ±L/t.  Each accuracy level has its own range, so does its own clamping.  Because of the scaling introducing another layer of approximation, we subtract 0.5 from the reported fit accuracy when choosing an approximation.
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(tanh)(SSMF_VF v) {
#if defined(SSMF_VF_TANH)
	return SSMF_VF_TANH(v);
#else
#	if SSMF_ACCURACY <= 1 // 2.13 bits, clamped at 1.297273
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-1.2972734327035422)), SSMF_VF_F(1.2972734327035422));
	SSMF_VF y = SSMF_VF_MUL(x, SSMF_VF_F(0.770847513554626));
#	elif SSMF_ACCURACY <= 4 // 4.90 bits, clamped at 2.138392
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-2.1383921168472426)), SSMF_VF_F(2.1383921168472426));
	SSMF_VF y = SSMF_VF_MUL(x, SSMF_VF_POLY1(SSMF_NAME(abs)(x), 1.0286193462938031, -0.2623364823504241));
#	elif SSMF_ACCURACY <= 5 // 6.39 bits, clamped at 2.460539
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-2.4605394724869036)), SSMF_VF_F(2.4605394724869036));
	SSMF_VF y = SSMF_VF_MUL(x, SSMF_VF_POLY3(SSMF_NAME(pow2)(x), 0.9881042303582034, -0.25301082548556936, 0.04356474318729903, -0.002914297189246735));
#	elif SSMF_ACCURACY <= 7 // 8.02 bits, clamped at 3.130412
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-3.1304129707689183)), SSMF_VF_F(3.1304129707689183));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_DIV(SSMF_VF_MUL(x, SSMF_VF_POLY1M(x2, 21.385579840357426)), SSMF_VF_POLY1(x2, 21.468222408594873, 7.771215258369642));
#	elif SSMF_ACCURACY <= 10 // 10.87 bits, clamped at 4.160792
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-4.160792635172477)), SSMF_VF_F(4.160792635172477));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_DIV(SSMF_VF_MUL(x, SSMF_VF_POLY1M(x2, 12.374844179118714)), SSMF_VF_POLY2(x2, 12.381446379503116, 5.090253718979554, 0.07679603752083053));
#	elif SSMF_ACCURACY <= 13 // 13.70 bits, clamped at 5.198905
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-5.198905911212312)), SSMF_VF_F(5.198905911212312));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_DIV(SSMF_VF_MUL(x, SSMF_VF_POLY2M(x2, 1568.4558753007798, 158.91442416315468)), SSMF_VF_POLY2(x2, 1568.5740539686826, 681.0214729274979, 19.584228576433027));
#	elif SSMF_ACCURACY <= 16 // 16.50 bits, clamped at 6.238370
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-6.238370312675585)), SSMF_VF_F(6.238370312675585));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_DIV(SSMF_VF_MUL(x, SSMF_VF_POLY2M(x2, 680.4652945010945, 77.65910496218123)), SSMF_VF_POLY3(x2, 680.4726191411667, 304.43187455998833, 11.802759743248542, 0.03636442846137611));
#	elif SSMF_ACCURACY <= 18 // 19.30 bits, clamped at 7.278052
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-7.278052098352149)), SSMF_VF_F(7.278052098352149));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_DIV(SSMF_VF_MUL(x, SSMF_VF_POLY3M(x2, 264118.7298104796, 32414.367741225553, 593.9219937428668)), SSMF_VF_POLY3(x2, 264119.13865628233, 120450.96425929989, 5532.027522641219, 36.7532894329216));
#	elif SSMF_ACCURACY <= 21 // 22.09 bits, clamped at 8.317767
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-8.31776712847213)), SSMF_VF_F(8.31776712847213));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_DIV(SSMF_VF_MUL(x, SSMF_VF_POLY3M(x2, 90905.91902537193, 11716.301242541722, 264.63482566569536)), SSMF_VF_POLY4(x2, 90905.93934730142, 42018.11497185256, 2150.0930303178666, 21.18495227273804, 0.021120490815682704));
#	elif SSMF_ACCURACY <= 24 // 25.39 bits, clamped at 9.010913
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-9.01091360837486)), SSMF_VF_F(9.01091360837486));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_DIV(SSMF_VF_MUL(x, SSMF_VF_POLY4M(x2, 75852229.20093732, 10145006.832901722, 264578.3464463952, 1553.8737889224292)), SSMF_VF_POLY4(x2, 75852230.9253997, 35429067.514753, 1960658.4793682683, 24829.113656453366, 58.45559274665155));
#	elif SSMF_ACCURACY <= 27 // 28.17 bits, clamped at 10.05063
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-10.050634154629416)), SSMF_VF_F(10.050634154629416));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_DIV(SSMF_VF_MUL(x, SSMF_VF_POLY4M(x2, 21881614.976043366, 3004454.7924150936, 86189.75340911633, 658.6357171137932)), SSMF_VF_POLY5(x2, 21881615.048543848, 10298325.762940474, 601417.349929116, 8932.842361395085, 32.90293424718516, 0.013957995736543752));
#	elif SSMF_ACCURACY <= 30 // 30.95 bits, clamped at 11.09035
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-11.090354894007058)), SSMF_VF_F(11.090354894007058));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_DIV(SSMF_VF_MUL(x, SSMF_VF_POLY5M(x2, 35813616250.14216, 5019851788.608733, 154589491.3608934, 1414984.5247691604, 3437.6358602981327)), SSMF_VF_POLY5(x2, 35813616267.44826, 16957723701.432825, 1032015489.2147915, 17188468.276952215, 84224.957513531, 86.17134110830318));
#	elif SSMF_ACCURACY <= 33 // 33.72 bits, clamped at 12.13007
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-12.130075660490554)), SSMF_VF_F(12.130075660490554));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_DIV(SSMF_VF_MUL(x, SSMF_VF_POLY5M(x2, 8782370547.890427, 1251640070.0749154, 40717111.806368455, 424976.02913639403, 1399.4364314276759)), SSMF_VF_POLY6(x2, 8782370548.509718, 4179096913.050446, 262766686.74165633, 4770146.373831896, 28505.402667033482, 47.55259927974734, 0.009800160958476613));
#	elif SSMF_ACCURACY <= 35 // 36.50 bits, clamped at 13.16979
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-13.169796430732964)), SSMF_VF_F(13.169796430732964));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_DIV(SSMF_VF_MUL(x, SSMF_VF_POLY6M(x2, 24375023543279.51, 3521811131100.0176, 119679450527.07112, 1380072597.8104298, 5628837.871175621, 6663.734422722501)), SSMF_VF_POLY6(x2, 24375023543530.44, 11646818976212.834, 751949308013.0352, 14598102700.055195, 100943602.22539833, 231918.3943970393, 119.24699005167751));
#	elif SSMF_ACCURACY <= 38 // 39.27 bits, clamped at 14.20951
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-14.209517201491574)), SSMF_VF_F(14.209517201491574));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_DIV(SSMF_VF_MUL(x, SSMF_VF_POLY6M(x2, 5196626032196.513, 759510535889.4419, 26744763551.12627, 333499849.2374566, 1594420.791894098, 2635.6223331706333)), SSMF_VF_POLY7(x2, 5196626032204.325, 2491719213203.2695, 164434363810.6736, 3368559389.4832377, 26047649.81806326, 75129.17470121612, 64.8806849049633, 0.007257160443439178));
#	elif SSMF_ACCURACY <= 41 // 42.05 bits, clamped at 15.24923
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-15.249237972320502)), SSMF_VF_F(15.249237972320502));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_DIV(SSMF_VF_MUL(x, SSMF_VF_POLY7M(x2, 2.2595120995529864e+16, 3334822454016408.5, 120952685124760.19, 1606329899852.8523, 8681107500.57382, 18229160.85086503, 11755.173140387267)), SSMF_VF_POLY7(x2, 2.2595120995534828e+16, 1.0866529452470764e+16, 730446369980511.1, 15637087810387.191, 132076084486.56212, 451656334.86497664, 551176.9893787411, 157.6830345920514));
#	elif SSMF_ACCURACY <= 44 // 45.33 bits, clamped at 15.94238
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-15.942385152879188)), SSMF_VF_F(15.942385152879188));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_DIV(SSMF_VF_MUL(x, SSMF_VF_POLY7M(x2, 4106093795549719.0, 611387943556189.9, 22761601913154.227, 319106492505.79645, 1908640790.1638255, 4830956.471294937, 4490.890867537267)), SSMF_VF_POLY8(x2, 4106093795549812.0, 1980085875404961.2, 135311054310294.83, 3010053982529.8115, 27392273884.167435, 107466268.13128597, 169664.2500951529, 84.29510983517976, 0.005634135244167599));
#	elif SSMF_ACCURACY <= 47 // 48.10 bits, clamped at 16.98210
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-16.98210592371872)), SSMF_VF_F(16.98210592371872));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_DIV(SSMF_VF_MUL(x, SSMF_VF_POLY8M(x2, 2.6173777728412635e+19, 3.925660033327721e+18, 1.492778510564893e+17, 2184758515913912.0, 14135508448240.094, 41148726049.00651, 49473039.83502154, 19049.312795251404)), SSMF_VF_POLY8(x2, 2.617377772841272e+19, 1.2650252609464187e+19, 8.761916904249545e+17, 2.010139114947822e+16, 193954113345072.72, 844998609933.895, 1610057040.8692753, 1150678.079441395, 199.97108369985762));
#	else // 50.87 bits, clamped at 18.02182
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-18.021826694558584)), SSMF_VF_F(18.021826694558584));
	SSMF_VF x2 = SSMF_NAME(pow2)(x);
	SSMF_VF y = SSMF_VF_DIV(SSMF_VF_MUL(x, SSMF_VF_POLY8M(x2, 4.4549310241037107e+18, 6.72453386129387e+17, 2.604397470101691e+16, 395352023305102.3, 2730240088677.983, 8892770763.804386, 13010008.817682644, 7266.840500257768)), SSMF_VF_POLY9(x2, 4.454931024103713e+18, 2.1574303941639299e+18, 1.511966362085536e+17, 3561693790800137.5, 36098479313038.35, 171339183779.39172, 378457336.73686033, 352758.8643212594, 106.89547240980683, 0.004468418538439141));
#	endif
	// Rounding near the ends of the range can give slightly more than 1
	return SSMF_VF_MIN(SSMF_VF_MAX(y, SSMF_VF_F(-1)), SSMF_VF_F(1));
#endif
}

// Minimax fits (relative error, exact at 0 and 1) of acos(x)/sqrt(1 - x^2) for x in [0, 1].  This is θ/sin(θ) for θ = acos(x), so it's smooth at both ends.
#if SSMF_ACCURACY <= 3 // 3.99 bits
#	define SSMF_VF_INVCOS_HELPER(x) SSMF_VF_POLY1(x, 1.5707963267948966, -0.5707963267948967)
#elif SSMF_ACCURACY <= 7 // 7.15 bits
#	define SSMF_VF_INVCOS_HELPER(x) SSMF_VF_POLY2(x, 1.5707963267948966, -0.8779311254333566, 0.30713479863846)
#elif SSMF_ACCURACY <= 10 // 10.04 bits
#	define SSMF_VF_INVCOS_HELPER(x) SSMF_VF_POLY3(x, 1.5707963267948966, -0.9697047424955487, 0.5779905341597728, -0.17908211845912067)
#elif SSMF_ACCURACY <= 12 // 12.82 bits
#	define SSMF_VF_INVCOS_HELPER(x) SSMF_VF_POLY4(x, 1.5707963267948966, -0.9930327142231299, 0.7115754590612406, -0.3980167987364173, 0.10867772710340992)
#elif SSMF_ACCURACY <= 16 // 16.06 bits
#	define SSMF_VF_INVCOS_HELPER(x) SSMF_VF_DIV(SSMF_VF_POLY1M(x, 2.5308212215416286), SSMF_VF_POLY2(x, 1.6111708299608758, 1.661657635024418, 0.25799275655633486))
#elif SSMF_ACCURACY <= 20 // 20.41 bits
#	define SSMF_VF_INVCOS_HELPER(x) SSMF_VF_DIV(SSMF_VF_POLY2M(x, 60.42602891672438, 34.84898865925272), SSMF_VF_POLY2(x, 38.468404774041964, 46.67408476817494, 11.13252803376019))
#elif SSMF_ACCURACY <= 25 // 25.26 bits
#	define SSMF_VF_INVCOS_HELPER(x) SSMF_VF_DIV(SSMF_VF_POLY2M(x, 8.51468127716079, 7.213100055642805), SSMF_VF_POLY3(x, 5.420614456448609, 8.042863678357916, 3.0467328604483255, 0.2175703375487433))
#elif SSMF_ACCURACY <= 29 // 29.64 bits
#	define SSMF_VF_INVCOS_HELPER(x) SSMF_VF_DIV(SSMF_VF_POLY3M(x, 343.73987146227984, 358.61709825975106, 77.86800478579288), SSMF_VF_POLY3(x, 218.83159872397826, 367.6152348759922, 174.1882660393982, 20.589874868455134))
#elif SSMF_ACCURACY <= 34 // 34.39 bits
#	define SSMF_VF_INVCOS_HELPER(x) SSMF_VF_DIV(SSMF_VF_POLY3M(x, 32.059793096185665, 41.73458715818428, 14.096932616121988), SSMF_VF_POLY4(x, 20.40989818304547, 39.56240800606454, 23.955651901456076, 4.768240688074001, 0.1951140918518503))
#elif SSMF_ACCURACY <= 38 // 38.79 bits
#	define SSMF_VF_INVCOS_HELPER(x) SSMF_VF_DIV(SSMF_VF_POLY4M(x, 1874.6713918117464, 2822.335896531385, 1214.6517918180828, 140.6013827008484), SSMF_VF_POLY4(x, 1193.452874719211, 2556.530533060466, 1804.082812588087, 466.2760132543957, 32.918229239902715))
#elif SSMF_ACCURACY <= 43 // 43.49 bits
#	define SSMF_VF_INVCOS_HELPER(x) SSMF_VF_DIV(SSMF_VF_POLY4M(x, 128.13413011506793, 225.17576515012445, 124.3081013270411, 23.216671732053154), SSMF_VF_POLY5(x, 81.5727207463726, 195.28215126432258, 162.67111355029164, 55.31930054053809, 6.808985321537674, 0.18039690122406574))
#elif SSMF_ACCURACY <= 47 // 47.91 bits
#	define SSMF_VF_INVCOS_HELPER(x) SSMF_VF_DIV(SSMF_VF_POLY5M(x, 9969.473609497229, 19597.857715389615, 12755.5731505381, 3107.7659987026477, 224.4032678889407), SSMF_VF_POLY5(x, 6346.764019902735, 16516.859183279532, 15462.02719940948, 6257.118228955638, 1025.1874898617004, 48.11762060744914))
#elif SSMF_ACCURACY <= 52 // 52.58 bits
#	define SSMF_VF_INVCOS_HELPER(x) SSMF_VF_DIV(SSMF_VF_POLY5M(x, 531.3017577121003, 1175.972027742726, 908.0275927344302, 289.543841195255, 34.597238875263514), SSMF_VF_POLY6(x, 338.2372040531732, 963.9755364627823, 1022.6356039554214, 496.9239393157272, 109.34406280079091, 9.156305170196749, 0.1698065016833537))
#else // 57.01 bits
#	define SSMF_VF_INVCOS_HELPER(x) SSMF_VF_DIV(SSMF_VF_POLY6M(x, 52135.75623136687, 126430.42984404074, 110555.01719836377, 42179.17337040722, 6688.577331622413, 330.36692859118085), SSMF_VF_POLY6(x, 33190.653264224486, 101617.93759345052, 118478.17155001643, 65555.2243060056, 17434.24400766037, 1977.9021174511472, 66.18806558363059))
#endif

// asin() = x*H(sqrt(1 - x^2)), for the helper H
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(asin)(SSMF_VF v) {
#if defined(SSMF_VF_ASIN)
	return SSMF_VF_ASIN(v);
#else
	SSMF_VF x = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-1)), SSMF_VF_F(1));
	SSMF_VF absX = SSMF_NAME(abs)(x);
	// calculate 1-x^2 as (1 - |x|)(1 + |x|), to keep relative accuracy near ±1
	SSMF_VF s = SSMF_NAME(sqrt)(SSMF_VF_MUL(SSMF_VF_SUB(SSMF_VF_F(1), absX), SSMF_VF_ADD(SSMF_VF_F(1), absX)));
	return SSMF_VF_MUL(x, SSMF_VF_INVCOS_HELPER(s));
#endif
}

// acos() = sqrt(1 - x^2)*H(|x|), but we for x >= 0, acos(x) = s*H(x) (the same s and H as asin()), which keeps relative accuracy near x = 1.  Negative x use acos(x) = π - acos(-x).
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(acos)(SSMF_VF v) {
#if defined(SSMF_VF_ACOS)
	return SSMF_VF_ACOS(v);
#else
	SSMF_VF absV = SSMF_NAME(abs)(v);
	SSMF_VF s = SSMF_NAME(sqrt)(SSMF_VF_MUL(SSMF_VF_SUB(SSMF_VF_F(1), absV), SSMF_VF_ADD(SSMF_VF_F(1), absV)));
	SSMF_VF acosAbs = SSMF_VF_MUL(s, SSMF_VF_INVCOS_HELPER(absV));
	// Awkwardly reflect around π/2 for v < 0, protecting against reassociation
	SSMF_VF offset = SSMF_VF_SUB(SSMF_VF_F(1.5707963267948966), SSMF_NAME(copysign)(SSMF_VF_F(1.5707963267948966), v));
	offset = SSMF_VF_BARRIER_HIGHBIT(offset, absV); // absV <= 1, or we're already in trouble
	return SSMF_VF_ADD(offset, SSMF_NAME(copysign)(acosAbs, v));
#endif
}

// atan() = x*s*H(s), for s = 1/sqrt(1 + x^2).  You can work this out by expanding asin(x/sqrt(1 + x^2))
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(atan)(SSMF_VF v) {
#if defined(SSMF_VF_ATAN)
	return SSMF_VF_ATAN(v);
#else
#	ifdef SSMF_FULL_RANGE
	// Clamp so that x^2 can't overflow
#		if SSMF_BITS == 32
		const float maxInput = 1073741824; // 2^60 - this is already well into the region where it returns π/2 anyway
#		elif SSMF_BITS == 64
	const double maxInput = 1152921504606846976.0; // 2^500
#		else
#			error "need SSMF_VF_ATAN() (or not SSMF_FULL_RANGE) unless it's 32/64-bit IEEE floats"
#		endif
	v = SSMF_VF_MIN(SSMF_VF_MAX(v, SSMF_VF_F(-maxInput)), SSMF_VF_F(maxInput));
#	endif
	SSMF_VF s = SSMF_NAME(rsqrt)(SSMF_VF_FMAK(v, v, 1));
	return SSMF_VF_MUL(SSMF_VF_MUL(v, s), SSMF_VF_INVCOS_HELPER(s));
#endif
}

// atan2(y, x): the angle for (|x|, y) is 2*atan(y/(hypot(x, y) + |x|)), where atan()'s input stays within [-1, 1].  For x < 0, we negate that and add ±π.  Only (0, 0) gives nonsense.
SSMF_ATTRIBUTES SSMF_INLINE SSMF_VF SSMF_NAME(atan2)(SSMF_VF y, SSMF_VF x) {
#if defined(SSMF_VF_ATAN2)
	return SSMF_VF_ATAN2(y, x);
#else
#	ifdef SSMF_FULL_RANGE
	// hypot(x, y) + |x| can overflow, so we halve the inputs first
	SSMF_VF xS = SSMF_VF_MUL(x, SSMF_VF_F(0.5)), yS = SSMF_VF_MUL(y, SSMF_VF_F(0.5));
#	else
	SSMF_VF xS = x, yS = y;
#	endif
	SSMF_VF half = SSMF_NAME(atan)(SSMF_VF_DIV(yS, SSMF_VF_ADD(SSMF_NAME(hypot)(xS, yS), SSMF_NAME(abs)(xS))));
#	if SSMF_BITS == 32 || SSMF_BITS == 64
	// 2 for x < 0 (otherwise 0) from the sign bit, like in erfc(), so fast-math can't reassociate it with the final sum
	SSMF_VF offsetScale = SSMF_VF_RSHIFT_K(SSMF_VF_AND(x, SSMF_VF_F(-0.0)), 1);
#	else
	SSMF_VF offsetScale = SSMF_VF_SUB(SSMF_VF_F(1), SSMF_NAME(copysign)(SSMF_VF_F(1), x));
#	endif
	SSMF_VF offset = SSMF_VF_MUL(offsetScale, SSMF_NAME(copysign)(SSMF_VF_F(1.5707963267948966), y));
	return SSMF_VF_FMA(half, SSMF_NAME(copysign)(SSMF_VF_F(2), x), offset);
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
#	undef SSMF_VF_NOT
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
#	undef SSMF_VF_FMA
#	undef SSMF_VF_FMAK
#	undef SSMF_VF_ABS
#	undef SSMF_VF_NEG
#	undef SSMF_VF_COPYSIGN
#	undef SSMF_VF_ROUND
#	undef SSMF_VF_FLOOR
#	undef SSMF_VF_SQRT
#	undef SSMF_VF_LOG2
#	undef SSMF_VF_LOG
#	undef SSMF_VF_LOG10
#	undef SSMF_VF_EXP2
#	undef SSMF_VF_EXP
#	undef SSMF_VF_EXPM1
#	undef SSMF_VF_SIN
#	undef SSMF_VF_COS
#	undef SSMF_VF_SINCOS
#	undef SSMF_VF_TAN
#	undef SSMF_VF_TANH
#	undef SSMF_VF_SINH
#	undef SSMF_VF_COSH
#	undef SSMF_VF_POW
#	undef SSMF_VF_CBRT
#	undef SSMF_VF_ERF
#	undef SSMF_VF_ERFC
#	undef SSMF_VF_ASIN
#	undef SSMF_VF_ACOS
#	undef SSMF_VF_ATAN
#	undef SSMF_VF_ATAN2
#	undef SSMF_VF_HYPOT
#	undef SSMF_NAME
#	undef SSMF_INLINE
#	undef SSMF_ATTRIBUTES
#	undef SSMF_FULL_RANGE
#endif
#undef SSMF_VF_BARRIER_LOWBIT
#undef SSMF_VF_BARRIER_HIGHBIT_MASK
#undef SSMF_VF_BARRIER_HIGHBIT
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
#undef SSMF_VF_ROUND_INNER
#undef SSMF_VF_EXP2_SMALL
#undef SSMF_VF_LOG2_CURVE
#undef SSMF_VF_EXPM1_SMALL
#undef SSMF_VF_EXP2M1_RATIO_TINY
#undef SSMF_VF_SINPI_SMALL
#undef SSMF_VF_INVCOS_HELPER
#undef SSMF_VF_SIGN_TO_ONE
#undef SSMF_EXP_SPLIT
