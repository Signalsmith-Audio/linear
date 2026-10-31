#if defined(__clang__) || defined(__GNUC__)
#	pragma GCC push_options
// Here's where we would put `#pragma GCC target("foo,bar")`, but SSE2 is always supported
#endif

namespace signalsmith { namespace linear {

namespace sse2 {

	struct ComplexFloat {
		__m128 real, imag;
	};
	struct ComplexDouble {
		__m128d real, imag;
	};

#define SSMF_BITS 32
#define SSMF_VF __m128
#define SSMF_VF_0 _mm_setzero_ps()
#define SSMF_VF_F(v) _mm_set1_ps(v)
#define SSMF_VF_I(v) _mm_castsi128_ps(_mm_set1_epi32(v))
#define SSMF_VF_AND(a, b) _mm_and_ps(a, b)
#define SSMF_VF_ANDNOT(a, b) _mm_andnot_ps(a, b)
#define SSMF_VF_OR(a, b) _mm_or_ps(a, b)
#define SSMF_VF_XOR(a, b) _mm_xor_ps(a, b)
#define SSMF_VF_ADD(a, b) _mm_add_ps(a, b)
#define SSMF_VF_SUB(a, b) _mm_sub_ps(a, b)
#define SSMF_VF_MUL(a, b) _mm_mul_ps(a, b)
#define SSMF_VF_DIV(a, b) _mm_div_ps(a, b)
#define SSMF_VF_MAX(a, b) _mm_max_ps(a, b)
#define SSMF_VF_MIN(a, b) _mm_min_ps(a, b)
#define SSMF_VF_RSHIFT_K(a, bits) _mm_castsi128_ps(_mm_srli_epi32(_mm_castps_si128(a), bits))
#define SSMF_VF_ITOF(a) _mm_cvtepi32_ps(_mm_castps_si128(a))
#include "../mathfun.h"

#define SSMF_BITS 64
#define SSMF_VF __m128d
#define SSMF_VF_0 _mm_setzero_pd()
#define SSMF_VF_F(v) _mm_set1_pd(v)
#define SSMF_VF_I(v) _mm_castsi128_pd(_mm_set1_epi64x(v))
#define SSMF_VF_AND(a, b) _mm_and_pd(a, b)
#define SSMF_VF_ANDNOT(a, b) _mm_andnot_pd(a, b)
#define SSMF_VF_OR(a, b) _mm_or_pd(a, b)
#define SSMF_VF_XOR(a, b) _mm_xor_pd(a, b)
#define SSMF_VF_ADD(a, b) _mm_add_pd(a, b)
#define SSMF_VF_SUB(a, b) _mm_sub_pd(a, b)
#define SSMF_VF_MUL(a, b) _mm_mul_pd(a, b)
#define SSMF_VF_DIV(a, b) _mm_div_pd(a, b)
#define SSMF_VF_MAX(a, b) _mm_max_pd(a, b)
#define SSMF_VF_MIN(a, b) _mm_min_pd(a, b)
#define SSMF_VF_RSHIFT_K(a, b) _mm_castsi128_pd(_mm_srli_epi64(_mm_castpd_si128(a), b))
#include "../mathfun.h"

//#ifndef DEV_NO_FALLBACK
	// Fallback
	template<class Expr>
	bool isAligned(Expr, size_t) {
		return false;
	}
	template<bool aligned, class Expr>
	__m128 getFloat(Expr expr, size_t i) {
		return _mm_setr_ps(expr.get(i), expr.get(i + 1), expr.get(i + 2), expr.get(i + 3));
	}
	template<bool aligned, class Expr>
	__m128d getDouble(Expr expr, size_t i) {
		return _mm_setr_pd(expr.get(i), expr.get(i + 1));
	}
//#endif

	// ReadableReal
	template<class V>
	bool isAligned(expression::ReadableReal<V> expr, size_t i) {
		return (size_t(expr.pointer + i)&15) == 0;
	}
	template<bool aligned>
	__m128 getFloat(expression::ReadableReal<float> expr, size_t i) {
		if (aligned) {
			return _mm_load_ps(expr.pointer + i);
		} else {
			return _mm_loadu_ps(expr.pointer + i);
		}
	}
	template<bool aligned>
	__m128d getDouble(expression::ReadableReal<double> expr, size_t i) {
		if (aligned) {
			return _mm_load_pd(expr.pointer + i);
		} else {
			return _mm_loadu_pd(expr.pointer + i);
		}
	}
	
	// Constant
	template<class V>
	bool isAligned(expression::ConstantExpr<V>, size_t) {
		return true;
	}
	template<bool aligned, class V>
	__m128 getFloat(expression::ConstantExpr<V> expr, size_t) {
		return _mm_set1_ps(float(expr.value));
	}
	template<bool aligned, class V>
	__m128d getDouble(expression::ConstantExpr<V> expr, size_t) {
		return _mm_set1_pd(double(expr.value));
	}

	// Pre-declare expressions so the implementations can reference each other
#define SIGNALSMITH_EXPR_UNARY(Name) \
	template<class A> \
	bool isAligned(expression::Name<A> expr, size_t i); \
	template<bool aligned, class A, class=ItemTypeF<A>> \
	__m128 getFloat(expression::Name<A> expr, size_t i); \
	template<bool aligned, class A, class=ItemTypeD<A>> \
	__m128d getDouble(expression::Name<A> expr, size_t i);
	SIGNALSMITH_EXPR_UNARY(Abs)
	SIGNALSMITH_EXPR_UNARY(Norm)
	SIGNALSMITH_EXPR_UNARY(Neg)
	SIGNALSMITH_EXPR_UNARY(Sqrt)
	SIGNALSMITH_EXPR_UNARY(Log)
#undef SIGNALSMITH_EXPR_UNARY
#define SIGNALSMITH_EXPR_BINARY(Name) \
	template<class A, class B> \
	bool isAligned(expression::Name<A, B> expr, size_t i); \
	template<bool aligned, class A, class B> \
	__m128 getFloat(expression::Name<A, B> expr, size_t i); \
	template<bool aligned, class A, class B> \
	__m128d getDouble(expression::Name<A, B> expr, size_t i);
	SIGNALSMITH_EXPR_BINARY(Add)
	SIGNALSMITH_EXPR_BINARY(Sub)
	SIGNALSMITH_EXPR_BINARY(Mul)
	SIGNALSMITH_EXPR_BINARY(Div)
	SIGNALSMITH_EXPR_BINARY(Max)
	SIGNALSMITH_EXPR_BINARY(Min)
	SIGNALSMITH_EXPR_BINARY(Dim)
	SIGNALSMITH_EXPR_BINARY(Copysign)
#undef SIGNALSMITH_EXPR_BINARY

#define SIGNALSMITH_EXPR_BINARY(Name) \
	template<class A, class B> \
	bool isAligned(expression::Name<A, B> expr, size_t i) { \
		return isAligned(expr.a, i) && isAligned(expr.b, i); \
	}
	SIGNALSMITH_EXPR_BINARY(Add)
	SIGNALSMITH_EXPR_BINARY(Sub)
	SIGNALSMITH_EXPR_BINARY(Mul)
	SIGNALSMITH_EXPR_BINARY(Div)
	SIGNALSMITH_EXPR_BINARY(Max)
	SIGNALSMITH_EXPR_BINARY(Min)
	SIGNALSMITH_EXPR_BINARY(Dim)
	SIGNALSMITH_EXPR_BINARY(Copysign)
#undef SIGNALSMITH_EXPR_BINARY

	// Expressions which map neatly to a single intrinsic
#define SIGNALSMITH_EXPR_UNARY(Name, mmFloat, mmDouble) \
	template<class A> \
	bool isAligned(expression::Name<A> expr, size_t i) { \
		return isAligned(expr.a, i); \
	} \
	template<bool aligned, class A, class> \
	__m128 getFloat(expression::Name<A> expr, size_t i) { \
		return mmFloat(getFloat<aligned>(expr.a, i)); \
	} \
	template<bool aligned, class A, class B> \
	__m128d getDouble(expression::Name<A> expr, size_t i) { \
		return mmDouble(getDouble<aligned>(expr.a, i)); \
	}
	SIGNALSMITH_EXPR_UNARY(Norm, mathfun_pow2f, mathfun_pow2d)
	SIGNALSMITH_EXPR_UNARY(Abs, mathfun_absf, mathfun_absd)
	SIGNALSMITH_EXPR_UNARY(Neg, mathfun_negf, mathfun_negd)
	SIGNALSMITH_EXPR_UNARY(Sqrt, _mm_sqrt_ps, _mm_sqrt_pd)
	SIGNALSMITH_EXPR_UNARY(Log, mathfun_logf, mathfun_logd)
#undef SIGNALSMITH_EXPR_UNARY
#define SIGNALSMITH_EXPR_BINARY(Name, mmFloat, mmDouble) \
	template<bool aligned, class A, class B> \
	__m128 getFloat(expression::Name<A, B> expr, size_t i) { \
		return mmFloat(getFloat<aligned>(expr.a, i), getFloat<aligned>(expr.b, i)); \
	} \
	template<bool aligned, class A, class B> \
	__m128d getDouble(expression::Name<A, B> expr, size_t i) { \
		return mmDouble(getDouble<aligned>(expr.a, i), getDouble<aligned>(expr.b, i)); \
	}
	SIGNALSMITH_EXPR_BINARY(Add, _mm_add_ps, _mm_add_pd)
	SIGNALSMITH_EXPR_BINARY(Sub, _mm_sub_ps, _mm_sub_pd)
	SIGNALSMITH_EXPR_BINARY(Mul, _mm_mul_ps, _mm_mul_pd)
	SIGNALSMITH_EXPR_BINARY(Div, _mm_div_ps, _mm_div_pd)
	SIGNALSMITH_EXPR_BINARY(Max, _mm_max_ps, _mm_max_pd)
	SIGNALSMITH_EXPR_BINARY(Min, _mm_min_ps, _mm_min_pd)
	SIGNALSMITH_EXPR_BINARY(Dim, mathfun_dimf, mathfun_dimd)
	SIGNALSMITH_EXPR_BINARY(Copysign, mathfun_copysignf, mathfun_copysignd)
#undef SIGNALSMITH_EXPR_BINARY
}

}}; // namespace

#if defined(__clang__) || defined(__GNUC__)
#	pragma GCC pop_options
#endif
