/* Copyright 2026 Geraint Luff / Signalsmith Audio Ltd.

Tests mathfun.h using plain scalars, checking each function is no more than 1 bit less accurate than SSMF_ACCURACY.  Only mandatory ops are defined, so fallbacks get tested too.

	gcc -std=c99 mathfun-test.c -o mathfun-test && ./mathfun-test

*/
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

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
#define SSMF_ACCURACY TEST_ACCURACY
#define SSMF_NAME(fn) float_test_##fn
#define SSMF_BITS 32
#define SSMF_F(v) ((float)(v))
#define SSMF_VF float
#define SSMF_VF_0 0.0f
#define SSMF_VF_F(v) ((float)(v))
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
#define SSMF_VF_I(v) intAsFloat((uint32_t)(v))
#define SSMF_VF_AND(a, b) intAsFloat(floatAsInt(a)&floatAsInt(b))
#define SSMF_VF_ANDNOT(a, b) intAsFloat(~floatAsInt(a)&floatAsInt(b))
#define SSMF_VF_OR(a, b) intAsFloat(floatAsInt(a)|floatAsInt(b))
#define SSMF_VF_XOR(a, b) intAsFloat(floatAsInt(a)^floatAsInt(b))
#define SSMF_VF_ADD(a, b) ((a) + (b))
#define SSMF_VF_SUB(a, b) ((a) - (b))
#define SSMF_VF_MUL(a, b) ((a)*(b))
#define SSMF_VF_DIV(a, b) ((a)/(b))
#define SSMF_VF_MAX(a, b) fmaxf(a, b)
#define SSMF_VF_MIN(a, b) fminf(a, b)
#define SSMF_VF_RSHIFT_K(a, bits) intAsFloat(floatAsInt(a) >> (bits))
#include "./mathfun.h"
#endif

#define SSMF_ACCURACY TEST_ACCURACY
#define SSMF_NAME(fn) double_test_##fn
#define SSMF_BITS 64
#define SSMF_F(v) ((double)(v))
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
#include "./mathfun.h"

static int returnCode = 0;

// Each function records its worst relative error, and passes if that's within 1 bit of TEST_ACCURACY
typedef struct {
	const char *name;
	double worst, worstX, limit;
} Check;
static Check check;

static void startCheck(const char *name) {
	Check newCheck = {name, 0, 0, ldexp(1, 1 - TEST_ACCURACY)};
	check = newCheck;
}
static void test(double x, long double expected, long double actual) {
	long double error = fabsl(actual - expected);
	if (expected != 0) error /= fabsl(expected);
	if (!(error <= check.worst)) {
		check.worst = (double)error;
		check.worstX = x;
	}
}
static void endCheck() {
	int ok = (check.worst <= check.limit);
	if (check.worst == 0) {
		printf("- \x1b[32mPASS\x1b[0m \x1b[36m%16s\x1b[0m: matched exactly\n", check.name);
	} else if (ok) {
		printf("- \x1b[32mPASS\x1b[0m \x1b[36m%16s\x1b[0m: worst (%.1f bits at x = %.17g) >= %.0f bits\n", check.name, -log2(check.worst), check.worstX, -log2(check.limit));
	} else {
		printf("- \x1b[31mFAIL\x1b[0m \x1b[36m%16s\x1b[0m: worst (%.1f bits at x = %.17g) < %.0f bits\n", check.name, -log2(check.worst), check.worstX, -log2(check.limit));
	}
	if (!ok) returnCode = 1;
}

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
		endCheck(); \
		startCheck("log"); \
		RANGE_EXP(T, x, 1e-30, 1e30) { \
			test(x, logl(x), T##_test_log(x)); \
		} \
		RANGE_LINEAR(T, x, 0.9, 1.1) { \
			test(x, logl(x), T##_test_log(x)); \
		} \
		endCheck(); \
	}

int main(void) {
	printf("TEST_ACCURACY = %d, TEST_STEP = %g\n", TEST_ACCURACY, (double)TEST_STEP);
#if TEST_FLOAT
	printf("\n\x1b[4m\x1b[1mFloat:\x1b[0m\n");
	TEST_TYPE(float)
#endif
	printf("\n\x1b[4m\x1b[1mDouble:\x1b[0m\n");
	TEST_TYPE(double)
	return returnCode;
}
