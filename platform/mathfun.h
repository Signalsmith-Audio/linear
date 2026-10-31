/* Copyright 2026 Geraint Luff / Signalsmith Audio Ltd.

This header defines floating-point functions, given #defines for the basic operations (constants, arithmetic, bitwise).

It's a "function header" - explicitly designed to be included multiple times if needed.  You define various `SSMF_...` macros, and these are un-defined by this header.  The functions are defined using a naming macro, so you can define your own prefix/suffix, e.g. `SSMF_NAME(abs)` -> `prefix_abs_suffix`.

Some of these fundamental operations are required (and they'll error), but others are optional and the implementation will work around that.
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
#		define SSMF_VF_ITOF_MAGIC 8388608.0f // 2^23
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

// Optional: a*b + c
#ifndef SSMF_VF_FMA
#	define SSMF_VF_FMA(a, b, c) SSMF_VF_ADD(SSMF_VF_MUL(a, b), c)
#endif
// Optional: a*b + k, for constant literal `k`
#ifndef SSMF_VF_FMAK
#	define SSMF_VF_FMAK(a, b, k) SSMF_VF_ADD(SSMF_VF_MUL(a, b), SSMF_VF_F(SSMF_F(k)))
#endif

// Polynomials (Horner's method) with constant coefficients
#define SSMF_VF_POLY0(x, c0) SSMF_VF_F(SSMF_F(c0))
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
#define SSMF_VF_POLY1M(x, c0) SSMF_VF_ADD(x, SSMF_VF_F(SSMF_F(c0)))
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
#elif SSMF_BITS == 64
#		define SSMF_NAME(name) mathfun_##name##d
#	else
#		define SSMF_NAME(name) mathfun_##name##
#	endif
#endif

// abs(): bitwise AND to clear the sign bit
SSMF_INLINE SSMF_VF SSMF_NAME(abs)(SSMF_VF v) {
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
SSMF_INLINE SSMF_VF SSMF_NAME(neg)(SSMF_VF v) {
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
SSMF_INLINE SSMF_VF SSMF_NAME(pow2)(SSMF_VF v) {
	return SSMF_VF_MUL(v, v);
}

// dim()
SSMF_INLINE SSMF_VF SSMF_NAME(dim)(SSMF_VF a, SSMF_VF b) {
	SSMF_VF diff = SSMF_VF_SUB(a, b);
	return SSMF_VF_MAX(SSMF_VF_0, diff);
}

// copysign()
SSMF_INLINE SSMF_VF SSMF_NAME(copysign)(SSMF_VF a, SSMF_VF b) {
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
SSMF_INLINE SSMF_VF SSMF_NAME(log2)(SSMF_VF v) {
#ifdef SSMF_VF_GET_EM
SSMF_VF exponent, mantissa;
SSMF_VF_GET_EM(v, exponent, mantissa);
#else
#	if SSMF_BITS == 32
	const int mantissaMaskInt = 0x007FFFFF, mantissaBits = 23;
	const float mantissaScale = 1.1920928955078125e-7f; // 2^-23
	const float exponentBase = 127;
#	elif SSMF_BITS == 64
	const long long mantissaMaskInt = 0x000FFFFFFFFFFFFFLL;
	const int mantissaBits = 52;
	const double mantissaScale = 2.220446049250313e-16; // 2^-52
	const double exponentBase = 1023;
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

	// Minimax fits of log2(1 + x) on [0, 1], with relative error, since it passes through (0, 0)
#if SSMF_ACCURACY <= 2
	// 1.7 bits
	SSMF_VF curve = mantissa;
#elif SSMF_ACCURACY <= 6
	// 5.2 bits
	SSMF_VF curve = SSMF_VF_MUL(mantissa, SSMF_VF_POLY1(mantissa, 1.40284708138686, -0.40284708138686015));
#elif SSMF_ACCURACY <= 9
	// 8.2 bits
	SSMF_VF curve = SSMF_VF_MUL(mantissa, SSMF_VF_POLY2(mantissa, 1.437907097573697, -0.632288225018458, 0.1943811274447612));
#elif SSMF_ACCURACY <= 12
	// 11.1 bits
	SSMF_VF curve = SSMF_VF_MUL(mantissa, SSMF_VF_POLY3(mantissa, 1.4420568158435028, -0.7001075467817148, 0.36118973101190066, -0.10313900007368867));
#elif SSMF_ACCURACY <= 14
	// 14.0 bits
	SSMF_VF curve = SSMF_VF_MUL(mantissa, SSMF_VF_POLY4(mantissa, 1.442605180900105, -0.7166753272258335, 0.44006828983482676, -0.22376892590948255, 0.05777078240038419));
#elif SSMF_ACCURACY <= 17
	// 16.7 bits
	SSMF_VF curve = SSMF_VF_MUL(mantissa, SSMF_VF_POLY5(mantissa, 1.4426819457470073, -0.720368921663955, 0.46864874228752906, -0.30137974806503814, 0.1439313799137711, -0.033513398219314185));
#elif SSMF_ACCURACY <= 21
	// 20.7 bits
	SSMF_VF curve = SSMF_VF_MUL(mantissa, SSMF_VF_DIV(SSMF_VF_POLY2M(mantissa, 52.940112296839175, 31.417871392488177), SSMF_VF_POLY2(mantissa, 36.69531093891793, 40.1233976204252, 8.539275129984224)));
#elif SSMF_ACCURACY <= 26
	// 25.5 bits
	SSMF_VF curve = SSMF_VF_MUL(mantissa, SSMF_VF_DIV(SSMF_VF_POLY2M(mantissa, 8.139581261783208, 7.039958806751733), SSMF_VF_POLY3(mantissa, 5.641927921297033, 7.70067959317588, 2.6630383802596174, 0.17389417380241037)));
#elif SSMF_ACCURACY <= 30
	// 29.9 bits
	SSMF_VF curve = SSMF_VF_MUL(mantissa, SSMF_VF_DIV(SSMF_VF_POLY3M(mantissa, 295.5483426799958, 314.56888567679255, 70.06537865918303), SSMF_VF_POLY3(mantissa, 204.85850065805658, 320.47175770570755, 140.51596951559216, 15.336379136615063)));
#elif SSMF_ACCURACY <= 35
	// 34.6 bits
	SSMF_VF curve = SSMF_VF_MUL(mantissa, SSMF_VF_DIV(SSMF_VF_POLY3M(mantissa, 30.243854450251835, 40.025064217078885, 13.769831477949054), SSMF_VF_POLY4(mantissa, 20.963442442273042, 38.224981487539054, 21.66920077103305, 4.026898855566918, 0.15422658886771418)));
#elif SSMF_ACCURACY <= 39
	// 39.0 bits
	SSMF_VF curve = SSMF_VF_MUL(mantissa, SSMF_VF_DIV(SSMF_VF_POLY4M(mantissa, 1595.0806204318492, 2439.0020100209417, 1068.383285258221, 126.60372926443013), SSMF_VF_POLY4(mantissa, 1105.6256348201953, 2243.4001835675954, 1493.7050928077551, 363.2136618696486, 24.125071910247318)));
#elif SSMF_ACCURACY <= 44
	// 43.7 bits
	SSMF_VF curve = SSMF_VF_MUL(mantissa, SSMF_VF_DIV(SSMF_VF_POLY4M(mantissa, 119.74289340140004, 213.29408659875025, 119.49328117018479, 22.69550774381113), SSMF_VF_POLY5(mantissa, 82.99944895327651, 189.34391923096965, 149.83190764842223, 48.28250211613345, 5.6265321930894485, 0.14145877225492806)));
#elif SSMF_ACCURACY <= 49
	// 48.1 bits
	SSMF_VF curve = SSMF_VF_MUL(mantissa, SSMF_VF_DIV(SSMF_VF_POLY5M(mantissa, 8426.336808061997, 16776.60921143971, 11072.581396939431, 2741.4654742215916, 202.35511756573848), SSMF_VF_POLY5(mantissa, 5840.691600956682, 14549.005174736994, 13002.533964237698, 5012.010546583294, 781.2022317764904, 34.90448993731307)));
#else
	// 52.7 bits
	SSMF_VF curve = SSMF_VF_MUL(mantissa, SSMF_VF_DIV(SSMF_VF_POLY5M(mantissa, 492.99263760776006, 1103.7172303574434, 862.7345277878262, 278.8257574322056, 33.8438837714279), SSMF_VF_POLY6(mantissa, 341.7164567946298, 935.8967147549896, 952.0448772883889, 442.75326885669534, 93.1180223914255, 7.45235088709869, 0.1323459834352492)));
#endif	
	return SSMF_VF_ADD(curve, exponent);
}

// log(): multiplies log2()
SSMF_INLINE SSMF_VF SSMF_NAME(log)(SSMF_VF v) {
	SSMF_VF resultLog2 = SSMF_NAME(log2)(v);
	SSMF_VF mult = SSMF_VF_F(SSMF_F(0.6931471805599453));
	return SSMF_VF_MUL(resultLog2, mult);
}

// Clean up all our arguments unless explicitly told not to
#ifndef SSMF_NO_CLEANUP
#	undef SSMF_BITS
#	undef SSMF_ACCURACY
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
#	undef SSMF_VF_ITOF_MAGIC
#	undef SSMF_VF_FMA
#	undef SSMF_VF_FMAK

#	undef SSMF_NAME
#	undef SSMF_INLINE
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
