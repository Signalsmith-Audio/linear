/* Copyright 2026 Geraint Luff / Signalsmith Audio Ltd.

Tests mathfun.h using plain scalars, checking each function is no more than 1 bit less accurate than SSMF_ACCURACY.  By default only mandatory ops are defined, so fallbacks get tested too.

	gcc -std=c99 mathfun-test.c -o mathfun-test && ./mathfun-test

Optional ops can be switched on to test their routes:
	TEST_ROUND: uses rint for SSMF_VF_ROUND
	TEST_ROUND_INT32: bounces through int32 for SSMF_VF_ROUND (like SSE2's cvtps2dq/cvtdq2ps) - actually only worth using for float, but we test double as well
	TEST_FLOOR: SSMF_VF_FLOOR (floor)
	TEST_BARRIER: SSMF_VF_BARRIER (empty `asm` block, GCC/Clang only)
	TEST_FMA: SSMF_VF_FMA (fma)
	TEST_LOGEXP / TEST_LOGEXP2: uses a reference log/exp, to test conversion between bases
	TEST_SMALL_INPUTS: skips SSMF_FULL_RANGE, and only tests inputs within SMALL_INPUTS_MAX
*/
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#ifdef TEST_ROUTE_INDEX
// Start with an arbitrary mask
#	define PRNG1 ((TEST_ROUTE_INDEX^0x9c8f2d35)&0xFFFFFFFF)
// Bit-mixing with "lowbias32" constants from https://github.com/skeeto/hash-prospector
#	define PRNG2 (((PRNG1^(PRNG1>>16))*0x7feb352d)&0xFFFFFFFF)
#	define PRNG3 (((PRNG2^(PRNG2>>15))*0x846ca68b)&0xFFFFFFFF)
#	define PRNG4 (PRNG3^(PRNG3>>16))
// Use these bits to turn on/off route options
#	if (PRNG4&1)
#		define TEST_ROUND
#	endif
#	if (PRNG4&2)
#		define TEST_ROUND_INT32
#	endif
#	if (PRNG4&4)
#		define TEST_FLOOR
#	endif
#	if (PRNG4&8)
#		define TEST_BARRIER
#	endif
#	if (PRNG4&16)
#		define TEST_FMA
#	endif
#	if (PRNG4&32)
#		define TEST_SQRT
#	endif
#	if (PRNG4&64)
#		define TEST_LOGEXP
#	endif
#	if (PRNG4&128)
#		define TEST_LOGEXP2
#	endif
#	if (PRNG4&256)
#		define TEST_SMALL_INPUTS
#	endif

#	undef PRNG1
#	undef PRNG2
#	undef PRNG3
#	undef PRNG4
#endif

#ifdef TEST_BARRIER
#	if defined(__GNUC__)
// Returns the value unchanged, but the compiler can't see through it
static inline float testBarrierF(float v) {
	__asm__ volatile("" : "+g"(v));
	return v;
}
static inline double testBarrierD(double v) {
	__asm__ volatile("" : "+g"(v));
	return v;
}
#	else
#		error "TEST_BARRIER needs GCC/Clang"
#	endif
#endif

#ifndef TEST_ACCURACY
#	define TEST_ACCURACY 20
#endif
#ifndef TEST_STEP
#	define TEST_STEP 0.001
#endif
// Rounding limits float results to about 21.6 bits, so only test floats where we can meet the 1-bit tolerance
#ifndef TEST_FLOAT
#	define TEST_FLOAT (TEST_ACCURACY <= 22)
#endif
#define SWEEP_MIN(T) ((T)(sizeof(T) == 4 ? 1e-30 : 1e-300))
#define SWEEP_MAX(T) ((T)(sizeof(T) == 4 ? 1e30 : 1e300))
#define LARGE_MAX(T) ((T)(sizeof(T) == 4 ? 3e38 : 1.7e308)) // close to the largest value, but (1 + TEST_STEP) times this doesn't overflow
// Past this, the ULP of x/pi approaches 1, so sin/cos accuracy is meaningless (but results should still be bounded)
#define TRIG_MAX(T) ((T)(sizeof(T) == 4 ? 1e6 : 1e15))
// A limited SSMF_VF_ROUND also limits the periodic functions (approximately, since the limit applies to x/pi)
#define TRIG_ACCURATE_MAX(T) (TRIG_MAX(T) < (T)ROUNDING_MAX(T)*3 ? TRIG_MAX(T) : (T)ROUNDING_MAX(T)*3)
// With SSMF_SMALL_INPUTS, squaring shouldn't overflow/underflow, so we stay well inside sqrt() of the exponent range
#ifdef TEST_SMALL_INPUTS
#	define TEST_SMALL_INPUTS_ON 1
#	define SMALL_INPUTS_MIN(T) ((T)(sizeof(T) == 4 ? 1e-15 : 1e-150))
#	define SMALL_INPUTS_MAX(T) ((T)(sizeof(T) == 4 ? 1e15 : 1e150))
#else
#	define TEST_SMALL_INPUTS_ON 0
#	define SMALL_INPUTS_MIN(T) SWEEP_MIN(T)
#	define SMALL_INPUTS_MAX(T) ((T)(LARGE_MAX(T)/2)) // so hypot(x, x) can't overflow
#endif
#ifdef TEST_ROUND_INT32
#	define ROUNDING_MAX(T) 1073741824 // largest float below 2^31 (otherwise it rounds up)
#else
#	define ROUNDING_MAX(T) SWEEP_MAX(T)
#endif

#if TEST_FLOAT
#	define SSMF_ACCURACY TEST_ACCURACY
#	define SSMF_NAME(fn) float_test_##fn
#	define SSMF_BITS 32
#	define SSMF_VF float
#	define SSMF_VF_0 0.0f
#	define SSMF_VF_F(v) ((float)(v))
static float intAsFloat(uint32_t i) {
	float f;
	memcpy(&f, &i, 4);
	return f;
}
static uint32_t floatAsInt(float f) {
	uint32_t i;
	memcpy(&i, &f, 4);
	return i;
}
#	define SSMF_VF_I(v) intAsFloat((uint32_t)(v))
#	define SSMF_VF_NOT(a) intAsFloat(~floatAsInt(a))
#	define SSMF_VF_AND(a, b) intAsFloat(floatAsInt(a)&floatAsInt(b))
#	define SSMF_VF_ANDNOT(a, b) intAsFloat(~floatAsInt(a)&floatAsInt(b))
#	define SSMF_VF_OR(a, b) intAsFloat(floatAsInt(a)|floatAsInt(b))
#	define SSMF_VF_XOR(a, b) intAsFloat(floatAsInt(a)^floatAsInt(b))
#	define SSMF_VF_ADD(a, b) ((a) + (b))
#	define SSMF_VF_SUB(a, b) ((a) - (b))
#	define SSMF_VF_MUL(a, b) ((a)*(b))
#	define SSMF_VF_DIV(a, b) ((a)/(b))
#	define SSMF_VF_MAX(a, b) fmaxf(a, b)
#	define SSMF_VF_MIN(a, b) fminf(a, b)
#	define SSMF_VF_RSHIFT_K(a, bits) intAsFloat(floatAsInt(a) >> (bits))
#	define SSMF_VF_LSHIFT_K(a, bits) intAsFloat(floatAsInt(a) << (bits))
#	if defined(TEST_ROUND_INT32)
#		define SSMF_VF_ROUND(v) ((float)(int32_t)lrintf(v))
#	elif defined(TEST_ROUND)
#		define SSMF_VF_ROUND(v) rintf(v)
#	endif
#	ifdef TEST_FLOOR
#		define SSMF_VF_FLOOR(v) floorf(v)
#	endif
#	ifdef TEST_BARRIER
#		define SSMF_VF_BARRIER(v) testBarrierF(v)
#	endif
#	ifdef TEST_LOGEXP2
#		define SSMF_VF_LOG2(v) log2f(v)
#		define SSMF_VF_EXP2(v) exp2f(v)
#	endif
#	ifdef TEST_LOGEXP
#		define SSMF_VF_LOG(v) logf(v)
#		define SSMF_VF_EXP(v) expf(v)
#	endif
#	ifdef TEST_FMA
#		define SSMF_VF_FMA(a, b, c) fmaf(a, b, c)
#	endif
#	ifdef TEST_SQRT
#		define SSMF_VF_SQRT(v) sqrtf(v)
#	endif
#	ifndef TEST_SMALL_INPUTS
#		define SSMF_FULL_RANGE
#	endif
#	include "./signalsmith-mathfun.h"
#endif // TEST_FLOAT

#define SSMF_ACCURACY TEST_ACCURACY
#define SSMF_NAME(fn) double_test_##fn
#define SSMF_BITS 64
#define SSMF_VF double
#define SSMF_VF_0 0.0
#define SSMF_VF_F(v) ((double)(v))
static double intAsDouble(uint64_t i) {
	double d;
	memcpy(&d, &i, 8);
	return d;
}
static uint64_t doubleAsInt(double d) {
	uint64_t i;
	memcpy(&i, &d, 8);
	return i;
}
#define SSMF_VF_I(v) intAsDouble((uint64_t)(v))
#define SSMF_VF_NOT(a) intAsDouble(~doubleAsInt(a))
#define SSMF_VF_AND(a, b) intAsDouble(doubleAsInt(a)&doubleAsInt(b))
#define SSMF_VF_ANDNOT(a, b) intAsDouble(~doubleAsInt(a)&doubleAsInt(b))
#define SSMF_VF_OR(a, b) intAsDouble(doubleAsInt(a)|doubleAsInt(b))
#define SSMF_VF_XOR(a, b) intAsDouble(doubleAsInt(a)^doubleAsInt(b))
#define SSMF_VF_ADD(a, b) ((a) + (b))
#define SSMF_VF_SUB(a, b) ((a) - (b))
#define SSMF_VF_MUL(a, b) ((a)*(b))
#define SSMF_VF_DIV(a, b) ((a)/(b))
#define SSMF_VF_MAX(a, b) fmax(a, b)
#define SSMF_VF_MIN(a, b) fmin(a, b)
#define SSMF_VF_RSHIFT_K(a, bits) intAsDouble(doubleAsInt(a) >> (bits))
#define SSMF_VF_LSHIFT_K(a, bits) intAsDouble(doubleAsInt(a) << (bits))
#if defined(TEST_ROUND_INT32)
#	define SSMF_VF_ROUND(v) ((double)(int32_t)lrint(v))
#elif defined(TEST_ROUND)
#	define SSMF_VF_ROUND(v) rint(v)
#endif
#ifdef TEST_FLOOR
#	define SSMF_VF_FLOOR(v) floor(v)
#endif
#ifdef TEST_BARRIER
#	define SSMF_VF_BARRIER(v) testBarrierD(v)
#endif
#ifdef TEST_LOGEXP2
#	define SSMF_VF_LOG2(v) log2l(v)
#	define SSMF_VF_EXP2(v) exp2l(v)
#endif
#ifdef TEST_LOGEXP
#	define SSMF_VF_LOG(v) logl(v)
#	define SSMF_VF_EXP(v) expl(v)
#endif
#ifdef TEST_FMA
#	define SSMF_VF_FMA(a, b, c) fma(a, b, c)
#endif
#ifdef TEST_SQRT
#	define SSMF_VF_SQRT(v) sqrt(v)
#endif
#ifndef TEST_SMALL_INPUTS
#	define SSMF_FULL_RANGE
#endif
#include "./signalsmith-mathfun.h"

static int returnCode = 0;

// Each function records its worst relative error, and passes if that's within 1 bit of TEST_ACCURACY
typedef struct {
	const char *name;
	double worst, worstX, limit;
	double expected, actual;
} Check;
static Check check;

static void startCheck(const char *name) {
	Check newCheck = {name, 0, 0, ldexp(1, 1 - TEST_ACCURACY)};
	check = newCheck;
}
static void assertCheck(int success, double x, long double expected, long double actual) {
	if (!success && check.worst == 0) {
		check.worst = 1000;
		check.worstX = x;
		check.expected = expected;
		check.actual = actual;
	}
}
static void test(double x, long double expected, long double actual) {
	long double error = fabsl(actual - expected);
	if (expected != 0) error /= fabsl(expected);
	if (!(error <= check.worst)) {
		check.worst = (double)error;
		check.worstX = x;
		check.expected = expected;
		check.actual = actual;
	}
}
static void endCheck() {
	int ok = (check.worst <= check.limit);
	if (check.worst == 0) {
		printf("- \x1b[32mPASS\x1b[0m \x1b[36m%16s\x1b[0m: matched exactly\n", check.name);
	} else if (ok) {
		printf("- \x1b[32mPASS\x1b[0m \x1b[36m%16s\x1b[0m: worst (%.1f bits at x = %.17g) >= %.0f bits\n", check.name, -log2(check.worst), check.worstX, -log2(check.limit));
	} else {
		printf("- \x1b[31mFAIL\x1b[0m \x1b[36m%16s\x1b[0m: worst (%.1f bits: x = %.17g, f(x) = %.17g, expected %.17g) < %.0f bits\n", check.name, -log2(check.worst), check.worstX, check.actual, check.expected, -log2(check.limit));
	}
	if (!ok) returnCode = 1;
}

// asin/acos references which avoid 1 - x^2 (gcc's fast-math asinl()/acosl() cancel near ±1)
static long double refAsin(long double x) {return atan2l(x, sqrtl((1 - x)*(1 + x)));}
static long double refAcos(long double x) {return 2*atan2l(sqrtl(1 - x), sqrtl(1 + x));}

// Neighbouring values, for testing edge-cases
#define SSMF_TEST_PI 3.141592653589793238462643383279502884L
#define SSMF_TEST_EPSILON(T) (sizeof(T) == 4 ? 1.1920928955078125e-07 : 2.220446049250313080847263336181640625e-16)
// x/pi rounded to T (volatile so fast-math can't keep extra precision), which is what sin()/cos() are accurate for
static inline float float_halfTurns(float x) {volatile float t = x*0.31830988618379067f; return t;}
static inline double double_halfTurns(double x) {volatile double t = x*0.31830988618379067; return t;}
// sin(pi*t) and cos(pi*t), reduced exactly first so that integer/half-integer t gives exact zeros
// (volatile steps, so fast-math can't reassociate the exact reduction)
static long double refSinPi(long double t) {
	volatile long double k = roundl(t), f = t - k;
	long double r = sinl(SSMF_TEST_PI*f);
	return (fmodl(k, 2) == 0) ? r : -r;
}
static long double refCosPi(long double t) {
	volatile long double k = roundl(t), f = t - k;
	volatile long double fCos = 0.5L - fabsl(f); // cosl() near pi/2 would lose relative accuracy
	long double r = sinl(SSMF_TEST_PI*fCos);
	return (fmodl(k, 2) == 0) ? r : -r;
}
// tan(pi*t), reduced exactly first, using 1/tan(pi*(0.5 - |f|)) near the poles to keep relative accuracy
static long double refTanPi(long double t) {
	volatile long double k = roundl(t), f = t - k;
	if (fabsl(f) < 0.25L) return tanl(SSMF_TEST_PI*f);
	volatile long double fPole = 0.5L - fabsl(f);
	long double r = 1/tanl(SSMF_TEST_PI*fPole);
	return (f < 0) ? -r : r;
}
static inline int float_isSubnormal(float x) {uint32_t i; memcpy(&i, &x, 4); return (i & 0x7F800000u) == 0 && (i & 0x007FFFFFu) != 0;}
static inline int double_isSubnormal(double x) {uint64_t i; memcpy(&i, &x, 8); return (i & 0x7FF0000000000000ull) == 0 && (i & 0x000FFFFFFFFFFFFFull) != 0;}
static inline float float_nextDown(float x) {return nextafterf(x, -SWEEP_MAX(float));}
static inline float float_nextUp(float x) {return nextafterf(x, SWEEP_MAX(float));}
static inline double double_nextDown(double x) {return nextafter(x, -SWEEP_MAX(double));}
static inline double double_nextUp(double x) {return nextafter(x, SWEEP_MAX(double));}

#define RANGE_EXP(T, x, low, high) for (T x = (T)low; x <= (T)high; x *= (T)(1 + TEST_STEP))
#define RANGE_LINEAR(T, x, low, high) for (T x = low; x <= (T)high; x += (T)(TEST_STEP*(high - low)))

// When x/pi lands exactly a pole, we just check it's large and finite
#define TEST_TAN(T, x) { \
		T t = T##_halfTurns(x), y = T##_test_tan(x); \
		if (fabsl(t - roundl(t)) == 0.5L) { \
			assertCheck(T##_test_abs(y) >= (T)1e6 && T##_test_abs(y) <= (T)(4/SSMF_TEST_EPSILON(T)), x, refTanPi(T##_nextDown(t)), y); \
		} else { \
			test(x, refTanPi(t), y); \
		} \
	}

// Same tests for both types
#define TEST_TYPE(T) \
	{ \
		startCheck("abs"); \
		RANGE_EXP(T, x, SWEEP_MIN(T), SWEEP_MAX(T)) { \
			test(x, x, T##_test_abs(x)); \
			test(-x, x, T##_test_abs(-x)); \
		} \
		endCheck(); \
		startCheck("neg"); \
		RANGE_EXP(T, x, SWEEP_MIN(T), SWEEP_MAX(T)) { \
			test(x, -x, T##_test_neg(x)); \
			test(-x, x, T##_test_neg(-x)); \
		} \
		endCheck(); \
		startCheck("round"); \
		RANGE_EXP(T, x, SWEEP_MIN(T), ROUNDING_MAX(T)) { \
			T y = T##_test_round(x); \
			assertCheck(T##_test_abs(x - y) <= 0.5, x, roundl(x), y); \
			T yN = T##_test_round(-x); \
			assertCheck(T##_test_abs(-x - yN) <= 0.5, -x, roundl(-x), yN); \
		} \
		for (T x = -1000; x <= 1000; ++x) { \
			test(x, roundl(x), T##_test_round(x)); \
		} \
		for (T k = -1000; k <= 1000; ++k) { /* either side of k + 0.5 */ \
			T below = T##_nextDown(k + (T)0.5), above = T##_nextUp(k + (T)0.5); \
			test(below, k, T##_test_round(below)); \
			test(above, k + 1, T##_test_round(above)); \
		} \
		endCheck(); \
		startCheck("floor"); \
		for (T k = -1000; k <= 1000; k += (T)0.25) { \
			T xs[5] = {k, T##_nextDown(k), T##_nextUp(k), k + (T)0.125, (T)-0.0}; \
			for (int i = 0; i < 5; ++i) { \
				T x = xs[i]; \
				if (T##_isSubnormal(x)) continue; /* fast-math may flush these to 0 */ \
				assertCheck(T##_test_floor(x) == floorl(x), x, floorl(x), T##_test_floor(x)); \
			} \
		} \
		RANGE_EXP(T, x, SWEEP_MIN(T), ROUNDING_MAX(T)) { \
			T xs[2] = {x, -x}; \
			for (int i = 0; i < 2; ++i) { \
				T y = xs[i]; \
				assertCheck(T##_test_floor(y) == floorl(y), y, floorl(y), T##_test_floor(y)); \
			} \
		} \
		endCheck(); \
		startCheck("ceil"); \
		for (T k = -1000; k <= 1000; k += (T)0.25) { \
			T xs[5] = {k, T##_nextDown(k), T##_nextUp(k), k + (T)0.125, (T)-0.0}; \
			for (int i = 0; i < 5; ++i) { \
				T x = xs[i]; \
				if (T##_isSubnormal(x)) continue; /* fast-math may flush these to 0 */ \
				assertCheck(T##_test_ceil(x) == ceill(x), x, ceill(x), T##_test_ceil(x)); \
			} \
		} \
		RANGE_EXP(T, x, SWEEP_MIN(T), ROUNDING_MAX(T)) { \
			T xs[2] = {x, -x}; \
			for (int i = 0; i < 2; ++i) { \
				T y = xs[i]; \
				assertCheck(T##_test_ceil(y) == ceill(y), y, ceill(y), T##_test_ceil(y)); \
			} \
		} \
		endCheck(); \
		startCheck("trunc"); \
		for (T k = -1000; k <= 1000; k += (T)0.25) { \
			T xs[5] = {k, T##_nextDown(k), T##_nextUp(k), k + (T)0.125, (T)-0.0}; \
			for (int i = 0; i < 5; ++i) { \
				T x = xs[i]; \
				if (T##_isSubnormal(x)) continue; /* fast-math may flush these to 0 */ \
				assertCheck(T##_test_trunc(x) == truncl(x), x, truncl(x), T##_test_trunc(x)); \
			} \
		} \
		RANGE_EXP(T, x, SWEEP_MIN(T), ROUNDING_MAX(T)) { \
			T xs[2] = {x, -x}; \
			for (int i = 0; i < 2; ++i) { \
				T y = xs[i]; \
				assertCheck(T##_test_trunc(y) == truncl(y), y, truncl(y), T##_test_trunc(y)); \
			} \
		} \
		endCheck(); \
		startCheck("remainder1"); \
		RANGE_EXP(T, x, SWEEP_MIN(T), ROUNDING_MAX(T)) { \
			T r = T##_test_remainder1(x); \
			long double k = (long double)x - r; \
			assertCheck(T##_test_abs(r) <= 0.5 && k == roundl(k), x, x - roundl(x), r); \
			r = T##_test_remainder1(-x); \
			k = -(long double)x - r; \
			assertCheck(T##_test_abs(r) <= 0.5 && k == roundl(k), -x, -x - roundl(-x), r); \
		} \
		endCheck(); \
		startCheck("pow2"); \
		RANGE_EXP(T, x, sqrtl(SWEEP_MIN(T)), sqrtl(SWEEP_MAX(T))) { \
			test(x, (long double)x*x, T##_test_pow2(x)); \
			test(-x, (long double)x*x, T##_test_pow2(-x)); \
		} \
		endCheck(); \
		startCheck("dim"); \
		RANGE_EXP(T, x, SWEEP_MIN(T), SWEEP_MAX(T)) { \
			T y = x*(T)0.75; \
			test(x, x - y, T##_test_dim(x, y)); \
			test(y, 0, T##_test_dim(y, x)); \
		} \
		endCheck(); \
		startCheck("copysign"); \
		RANGE_EXP(T, x, SWEEP_MIN(T), SWEEP_MAX(T)) { \
			test(x, -x, T##_test_copysign(x, (T)-1)); \
			test(-x, x, T##_test_copysign(-x, (T)2)); \
			test(x, x, T##_test_copysign(x, (T)0)); \
		} \
		endCheck(); \
		startCheck("sqrt"); \
		test(0, 0, T##_test_sqrt(0)); \
		RANGE_EXP(T, x, SWEEP_MIN(T), LARGE_MAX(T)) { \
			test(x, sqrtl(x), T##_test_sqrt(x)); \
		} \
		endCheck(); \
		startCheck("hypot"); \
		test(0, 0, T##_test_hypot(0, 0)); \
		RANGE_EXP(T, x, SMALL_INPUTS_MIN(T), SMALL_INPUTS_MAX(T)) { \
			const T ratios[] = {0, (T)1e-6, (T)0.001, (T)0.3, (T)0.75, 1}; \
			for (int r = 0; r < 6; ++r) { \
				T y = x*ratios[r]; \
				long double expected = hypotl(x, y); \
				test(x, expected, T##_test_hypot(x, y)); \
				test(x, expected, T##_test_hypot(-y, x)); \
				test(x, expected, T##_test_hypot(y, -x)); \
			} \
		} \
		endCheck(); \
		startCheck("log2"); \
		RANGE_EXP(T, x, SWEEP_MIN(T), SWEEP_MAX(T)) { \
			test(x, log2l(x), T##_test_log2(x)); \
		} \
		RANGE_LINEAR(T, x, 0.9, 1.1) { \
			test(x, log2l(x), T##_test_log2(x)); \
		} \
		RANGE_LINEAR(T, x, 0.999, 1.111) { \
			test(x, log2l(x), T##_test_log2(x)); \
		} \
		endCheck(); \
		startCheck("log"); \
		RANGE_EXP(T, x, SWEEP_MIN(T), SWEEP_MAX(T)) { \
			test(x, logl(x), T##_test_log(x)); \
		} \
		RANGE_LINEAR(T, x, 0.9, 1.1) { \
			test(x, logl(x), T##_test_log(x)); \
		} \
		RANGE_LINEAR(T, x, 0.999, 1.111) { \
			test(x, logl(x), T##_test_log(x)); \
		} \
		endCheck(); \
		startCheck("log10"); \
		RANGE_EXP(T, x, SWEEP_MIN(T), SWEEP_MAX(T)) { \
			test(x, log10l(x), T##_test_log10(x)); \
		} \
		RANGE_LINEAR(T, x, 0.999, 1.111) { \
			test(x, log10l(x), T##_test_log10(x)); \
		} \
		endCheck(); \
		startCheck("exp2"); \
		RANGE_LINEAR(T, x, (sizeof(T) == 4 ? -126 : -1022), (sizeof(T) == 4 ? 127 : 1023)) { \
			test(x, exp2l(x), T##_test_exp2(x)); \
		} \
		RANGE_LINEAR(T, x, -2, 2) { \
			test(x, exp2l(x), T##_test_exp2(x)); \
		} \
		for (T k = -100; k <= 100; ++k) { \
			test(k, exp2l(k), T##_test_exp2(k)); \
			T below = T##_nextDown(k + (T)0.5), above = T##_nextUp(k + (T)0.5); \
			test(below, exp2l(below), T##_test_exp2(below)); \
			test(above, exp2l(above), T##_test_exp2(above)); \
		} \
		endCheck(); \
		startCheck("exp"); /* small (epsilon-level) errors when scaling the argument compound into large relative errors in the result very quickly. */ \
		RANGE_LINEAR(T, x, -4, 4) { \
			test(x, expl(x), T##_test_exp(x)); \
		} \
		endCheck(); \
		startCheck("expm1"); /* relative accuracy near 0 as well */ \
		RANGE_LINEAR(T, x, -4, 4) { \
			test(x, expm1l(x), T##_test_expm1(x)); \
		} \
		RANGE_EXP(T, x, SWEEP_MIN(T), 4) { \
			test(x, expm1l(x), T##_test_expm1(x)); \
			test(-x, expm1l(-x), T##_test_expm1(-x)); \
		} \
		RANGE_LINEAR(T, x, -80, -4) { /* approaches -1 */ \
			test(x, expm1l(x), T##_test_expm1(x)); \
		} \
		endCheck(); \
		/* sin/cos are backward-accurate: compared against the exact value for (x/pi rounded to T) */ \
		startCheck("sin"); \
		RANGE_LINEAR(T, x, -10, 10) { \
			test(x, refSinPi(T##_halfTurns(x)), T##_test_sin(x)); \
		} \
		RANGE_EXP(T, x, SWEEP_MIN(T), TRIG_ACCURATE_MAX(T)) { \
			test(x, refSinPi(T##_halfTurns(x)), T##_test_sin(x)); \
			test(-x, refSinPi(T##_halfTurns(-x)), T##_test_sin(-x)); \
		} \
		endCheck(); \
		startCheck("cos"); \
		RANGE_LINEAR(T, x, -10, 10) { \
			test(x, refCosPi(T##_halfTurns(x)), T##_test_cos(x)); \
		} \
		RANGE_EXP(T, x, SWEEP_MIN(T), TRIG_ACCURATE_MAX(T)) { \
			test(x, refCosPi(T##_halfTurns(x)), T##_test_cos(x)); \
			test(-x, refCosPi(T##_halfTurns(-x)), T##_test_cos(-x)); \
		} \
		for (T k = -100; k <= 100; ++k) { /* either side of cos()'s zeros */ \
			T edge = (T)((k + 0.5L)*SSMF_TEST_PI); \
			T below = T##_nextDown(edge), above = T##_nextUp(edge); \
			test(below, refCosPi(T##_halfTurns(below)), T##_test_cos(below)); \
			test(above, refCosPi(T##_halfTurns(above)), T##_test_cos(above)); \
		} \
		endCheck(); \
		startCheck("sincos"); /* same as sin() and cos() where they're accurate, and bounded for all inputs (within the rounding range) */ \
		RANGE_EXP(T, x, SWEEP_MIN(T), SWEEP_MAX(T)) { \
			T s, c; \
			int inRange = (x <= TRIG_ACCURATE_MAX(T)); \
			if (x > (T)ROUNDING_MAX(T)*3) break; \
			T##_test_sincos(x, &s, &c); \
			assertCheck((!inRange || (s == T##_test_sin(x) && c == T##_test_cos(x))) && T##_test_abs(s) <= 1 && T##_test_abs(c) <= 1, x, sinl(x), s); \
			T##_test_sincos(-x, &s, &c); \
			assertCheck((!inRange || (s == T##_test_sin(-x) && c == T##_test_cos(-x))) && T##_test_abs(s) <= 1 && T##_test_abs(c) <= 1, -x, sinl(-x), s); \
		} \
		endCheck(); \
		startCheck("tanh"); /* also checks it never goes outside [-1, 1] */ \
		RANGE_LINEAR(T, x, -20, 20) { \
			T y = T##_test_tanh(x); \
			test(x, tanhl(x), y); \
			assertCheck(T##_test_abs(y) <= 1, x, tanhl(x), y); \
		} \
		RANGE_EXP(T, x, SWEEP_MIN(T), SWEEP_MAX(T)) { \
			T y = T##_test_tanh(x), yNeg = T##_test_tanh(-x); \
			test(x, tanhl(x), y); \
			test(-x, tanhl(-x), yNeg); \
			assertCheck(T##_test_abs(y) <= 1 && T##_test_abs(yNeg) <= 1, x, tanhl(x), y); \
		} \
		endCheck(); \
		startCheck("asin"); /* relative accuracy near 0 and ±1, and bounded (inputs are clamped to [-1, 1]) */ \
		RANGE_LINEAR(T, x, -1, 1) { \
			test(x, refAsin(x), T##_test_asin(x)); \
		} \
		RANGE_EXP(T, x, SWEEP_MIN(T), 1) { \
			test(x, refAsin(x), T##_test_asin(x)); \
			test(-x, refAsin(-x), T##_test_asin(-x)); \
			test(1 - x, refAsin(1 - x), T##_test_asin(1 - x)); \
			test(x - 1, refAsin(x - 1), T##_test_asin(x - 1)); \
		} \
		test(1, refAsin(1), T##_test_asin(1)); \
		test(-1, refAsin(-1), T##_test_asin(-1)); \
		assertCheck(T##_test_abs(T##_test_asin(2)) <= (T)1.5707964 && T##_test_abs(T##_test_asin(-2)) <= (T)1.5707964, 2, refAsin(1), T##_test_asin(2)); \
		endCheck(); \
		startCheck("acos"); /* relative accuracy near 1, and bounded (inputs are clamped to [-1, 1]) */ \
		RANGE_LINEAR(T, x, -1, 1) { \
			test(x, refAcos(x), T##_test_acos(x)); \
		} \
		RANGE_EXP(T, x, SWEEP_MIN(T), 1) { \
			test(x, refAcos(x), T##_test_acos(x)); \
			test(-x, refAcos(-x), T##_test_acos(-x)); \
			test(1 - x, refAcos(1 - x), T##_test_acos(1 - x)); \
			test(x - 1, refAcos(x - 1), T##_test_acos(x - 1)); \
		} \
		test(1, refAcos(1), T##_test_acos(1)); \
		test(-1, refAcos(-1), T##_test_acos(-1)); \
		test(0, refAcos(0), T##_test_acos(0)); \
		assertCheck(T##_test_acos(2) >= 0 && T##_test_acos(-2) <= (T)3.1415927, 2, refAcos(1), T##_test_acos(2)); \
		endCheck(); \
		startCheck("atan"); /* relative accuracy near 0, and bounded */ \
		RANGE_LINEAR(T, x, -10, 10) { \
			test(x, atanl(x), T##_test_atan(x)); \
		} \
		RANGE_EXP(T, x, SWEEP_MIN(T), (TEST_SMALL_INPUTS_ON ? SMALL_INPUTS_MAX(T) : LARGE_MAX(T))) { \
			T y = T##_test_atan(x), yNeg = T##_test_atan(-x); \
			test(x, atanl(x), y); \
			test(-x, atanl(-x), yNeg); \
			assertCheck(y <= (T)1.5707963267948966 && yNeg >= -(T)1.5707963267948966, x, atanl(x), y); \
		} \
		endCheck(); \
		startCheck("tan"); /* backward-accurate like sin/cos, and bounded for all inputs (within the rounding range) */ \
		RANGE_LINEAR(T, x, -10, 10) TEST_TAN(T, x) \
		RANGE_EXP(T, x, SWEEP_MIN(T), TRIG_ACCURATE_MAX(T)) { \
			TEST_TAN(T, x) \
			TEST_TAN(T, -x) \
		} \
		for (T k = -100; k <= 100; ++k) { /* either side of tan()'s poles */ \
			T edge = (T)((k + 0.5L)*SSMF_TEST_PI); \
			T below = T##_nextDown(edge), above = T##_nextUp(edge); \
			TEST_TAN(T, edge) \
			TEST_TAN(T, below) \
			TEST_TAN(T, above) \
		} \
		RANGE_EXP(T, x, TRIG_ACCURATE_MAX(T), SWEEP_MAX(T)) { \
			if (x > (T)ROUNDING_MAX(T)*3) break; \
			T y = T##_test_tan(x), yNeg = T##_test_tan(-x); \
			assertCheck(T##_test_abs(y) <= (T)(4/SSMF_TEST_EPSILON(T)) && T##_test_abs(yNeg) <= (T)(4/SSMF_TEST_EPSILON(T)), x, tanl(x), y); \
		} \
		endCheck(); \
	}

int main(void) {
	printf("TEST_ACCURACY = %d, TEST_STEP = %g\n", TEST_ACCURACY, (double)TEST_STEP);
	printf("config:"
#ifdef TEST_ROUND_INT32
		" ROUND_INT32"
#endif
#ifdef TEST_ROUND
		" ROUND"
#endif
#ifdef TEST_FLOOR
		" FLOOR"
#endif
#ifdef TEST_BARRIER
		" BARRIER"
#endif
#ifdef TEST_FMA
		" FMA"
#endif
#ifdef TEST_LOGEXP
		" LOG/EXP"
#endif
#ifdef TEST_LOGEXP2
		" LOG2/EXP2"
#endif
#ifdef TEST_SMALL_INPUTS
		" !FULL_RANGE"
#endif
#ifdef __FAST_MATH__
		" (fast-math)"
#endif
#ifdef __OPTIMIZE__
		" (optimise)"
#endif
		"\n");
#if TEST_FLOAT
	printf("\n\x1b[4m\x1b[1mFloat:\x1b[0m\n");
	TEST_TYPE(float)
#endif
	printf("\n\x1b[4m\x1b[1mDouble:\x1b[0m\n");
	TEST_TYPE(double)
	return returnCode;
}
