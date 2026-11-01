/* Copyright 2026 Geraint Luff / Signalsmith Audio Ltd.

Tests mathfun.h using plain scalars, checking each function is no more than 1 bit less accurate than SSMF_ACCURACY.  By default only mandatory ops are defined, so fallbacks get tested too.

	gcc -std=c99 mathfun-test.c -o mathfun-test && ./mathfun-test

Optional ops can be switched on to test their routes:
	TEST_ROUND: uses rint for SSMF_VF_ROUND
	TEST_ROUND_INT32: bounces through int32 for SSMF_VF_ROUND (like SSE2's cvtps2dq/cvtdq2ps) - actually only worth using for float, but we test double as well
	TEST_FLOOR: SSMF_VF_FLOOR (floor)
	TEST_BARRIER: SSMF_VF_BARRIER (empty `asm` block, GCC/Clang only)
	TEST_FMA: SSMF_VF_FMA (fma)
*/
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

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
#		define SSMF_ROUND_LIMIT 2147483520 // largest float below 2^31 (2^31 - 1 rounds up to 2^31 as a float, which overflows int32)
#	elif TEST_ROUND
#		define SSMF_VF_ROUND(v) rintf(v)
#	endif
#	ifdef TEST_FLOOR
#		define SSMF_VF_FLOOR(v) floorf(v)
#	endif
#	ifdef TEST_BARRIER
#		define SSMF_VF_BARRIER(v) testBarrierF(v)
#	endif
#	ifdef TEST_FMA
#		define SSMF_VF_FMA(a, b, c) fmaf(a, b, c)
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
#	define SSMF_ROUND_LIMIT 2147483647
#elif TEST_ROUND
#	define SSMF_VF_ROUND(v) rint(v)
#endif
#ifdef TEST_FLOOR
#	define SSMF_VF_FLOOR(v) floor(v)
#endif
#ifdef TEST_BARRIER
#	define SSMF_VF_BARRIER(v) testBarrierD(v)
#endif
#ifdef TEST_FMA
#	define SSMF_VF_FMA(a, b, c) fma(a, b, c)
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

// Neighbouring values, for testing edge-cases
static inline float float_nextDown(float x) {return nextafterf(x, -1e30f);}
static inline float float_nextUp(float x) {return nextafterf(x, 1e30f);}
static inline double double_nextDown(double x) {return nextafter(x, -1e300);}
static inline double double_nextUp(double x) {return nextafter(x, 1e300);}

#define RANGE_EXP(T, x, low, high) for (T x = (T)low; x <= (T)high; x *= (T)(1 + TEST_STEP))
#define RANGE_LINEAR(T, x, low, high) for (T x = low; x <= (T)high; x += (T)(TEST_STEP*(high - low)))

// Same tests for both types
#define TEST_TYPE(T) \
	{ \
		startCheck("abs"); \
		RANGE_EXP(T, x, 1e-30, 1e30) { \
			test(x, x, T##_test_abs(x)); \
			test(-x, x, T##_test_abs(-x)); \
		} \
		endCheck(); \
		startCheck("neg"); \
		RANGE_EXP(T, x, 1e-30, 1e30) { \
			test(x, -x, T##_test_neg(x)); \
			test(-x, x, T##_test_neg(-x)); \
		} \
		endCheck(); \
		startCheck("round"); \
		RANGE_EXP(T, x, 1e-30, 4194304) { \
			T y = T##_test_round(x); \
			assertCheck(T##_test_abs(x - y) <= 0.5, x, roundl(x), y); \
		} \
		RANGE_EXP(T, x, 1e-30, 4194304) { \
			T y = T##_test_round(-x); \
			assertCheck(T##_test_abs(-x - y) <= 0.5, -x, roundl(-x), y); \
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
		startCheck("remainder1"); /* exact below 2^22, only bounded above that */ \
		RANGE_EXP(T, x, 1e-30, 1e30) { \
			T r = T##_test_remainder1(x); \
			long double k = (long double)x - r; \
			assertCheck(T##_test_abs(r) <= 0.5 && (x >= 4194304 || k == roundl(k)), x, x - roundl(x), r); \
			r = T##_test_remainder1(-x); \
			k = -(long double)x - r; \
			assertCheck(T##_test_abs(r) <= 0.5 && (x >= 4194304 || k == roundl(k)), -x, -x - roundl(-x), r); \
		} \
		endCheck(); \
		startCheck("pow2"); \
		RANGE_EXP(T, x, 1e-15, 1e15) { \
			test(x, (long double)x*x, T##_test_pow2(x)); \
			test(-x, (long double)x*x, T##_test_pow2(-x)); \
		} \
		endCheck(); \
		startCheck("dim"); \
		RANGE_EXP(T, x, 1e-30, 1e30) { \
			T y = x*(T)0.75; \
			test(x, x - y, T##_test_dim(x, y)); \
			test(y, 0, T##_test_dim(y, x)); \
		} \
		endCheck(); \
		startCheck("copysign"); \
		RANGE_EXP(T, x, 1e-30, 1e30) { \
			test(x, -x, T##_test_copysign(x, (T)-1)); \
			test(-x, x, T##_test_copysign(-x, (T)2)); \
			test(x, x, T##_test_copysign(x, (T)0)); \
		} \
		endCheck(); \
		startCheck("log2"); \
		RANGE_EXP(T, x, 1e-30, 1e30) { \
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
		RANGE_EXP(T, x, 1e-30, 1e30) { \
			test(x, logl(x), T##_test_log(x)); \
		} \
		RANGE_LINEAR(T, x, 0.9, 1.1) { \
			test(x, logl(x), T##_test_log(x)); \
		} \
		RANGE_LINEAR(T, x, 0.999, 1.111) { \
			test(x, logl(x), T##_test_log(x)); \
		} \
		endCheck(); \
	}

int main(void) {
	printf("TEST_ACCURACY = %d, TEST_STEP = %g\n", TEST_ACCURACY, (double)TEST_STEP);
	printf("optional ops:"
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
