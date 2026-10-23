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

	// Define `isAligned` since they're always predictable
#define SIGNALSMITH_EXPR_UNARY(Name) \
	template<class A> \
	bool isAligned(expression::Abs<A> expr, size_t i) { \
		return isAligned(expr.a, i); \
	}
	SIGNALSMITH_EXPR_UNARY(Abs)
	SIGNALSMITH_EXPR_UNARY(Norm)
	SIGNALSMITH_EXPR_UNARY(Neg)
	SIGNALSMITH_EXPR_UNARY(Sqrt)
#undef SIGNALSMITH_EXPR_UNARY
#define SIGNALSMITH_EXPR_BINARY(Name) \
	template<class A, class B> \
	bool isAligned(expression::Add<A, B> expr, size_t i) { \
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

	// Abs
	template<bool aligned, class A, class>
	__m128 getFloat(expression::Abs<A> expr, size_t i) {
		__m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)); // clears the sign bit
		__m128 a = getFloat<aligned>(expr.a, i);
		return _mm_and_ps(a, mask);
	}
	template<bool aligned, class A, class>
	__m128d getDouble(expression::Abs<A> expr, size_t i) {
		__m128d mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)); // clears the sign bit
		__m128d a = getDouble<aligned>(expr.a, i);
		return _mm_and_pd(a, mask);
	}
	// Norm (real only, complex values fall back)
	template<bool aligned, class A, class>
	__m128 getFloat(expression::Norm<A> expr, size_t i) {
		__m128 a = getFloat<aligned>(expr.a, i);
		return _mm_mul_ps(a, a);
	}
	template<bool aligned, class A, class>
	__m128d getDouble(expression::Norm<A> expr, size_t i) {
		__m128d a = getDouble<aligned>(expr.a, i);
		return _mm_mul_pd(a, a);
	}
	// Neg
	template<bool aligned, class A>
	__m128 getFloat(expression::Neg<A> expr, size_t i) {
		__m128 signBit = _mm_set1_ps(-0.0f);
		return _mm_xor_ps(getFloat<aligned>(expr.a, i), signBit);
	}
	template<bool aligned, class A>
	__m128d getDouble(expression::Neg<A> expr, size_t i) {
		__m128d signBit = _mm_set1_pd(-0.0);
		return _mm_xor_pd(getDouble<aligned>(expr.a, i), signBit);
	}
	// Sqrt
	template<bool aligned, class A>
	__m128 getFloat(expression::Sqrt<A> expr, size_t i) {
		return _mm_sqrt_ps(getFloat<aligned>(expr.a, i));
	}
	template<bool aligned, class A>
	__m128d getDouble(expression::Sqrt<A> expr, size_t i) {
		return _mm_sqrt_pd(getDouble<aligned>(expr.a, i));
	}

	// Expressions which map neatly to a single intrinsic
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
#undef SIGNALSMITH_EXPR_BINARY
	
	// Dim - implemented as max(0, a - b)
	template<bool aligned, class A, class B>
	__m128 getFloat(expression::Dim<A, B> expr, size_t i) {
		__m128 diff = _mm_sub_ps(getFloat<aligned>(expr.a, i), getFloat<aligned>(expr.b, i));
		return _mm_max_ps(_mm_setzero_ps(), diff);
	}
	template<bool aligned, class A, class B>
	__m128d getDouble(expression::Dim<A, B> expr, size_t i) {
		__m128d diff = _mm_sub_pd(getDouble<aligned>(expr.a, i), getDouble<aligned>(expr.b, i));
		return _mm_max_pd(_mm_setzero_pd(), diff);
	}
	// Copysign
	template<bool aligned, class A, class B>
	__m128 getFloat(expression::Copysign<A, B> expr, size_t i) {
		__m128 signBit = _mm_set1_ps(-0.0f);
		__m128 a = getFloat<aligned>(expr.a, i), b = getFloat<aligned>(expr.b, i);
		return _mm_or_ps(_mm_andnot_ps(signBit, a), _mm_and_ps(signBit, b));
	}
	template<bool aligned, class A, class B>
	__m128d getDouble(expression::Copysign<A, B> expr, size_t i) {
		__m128d signBit = _mm_set1_pd(-0.0);
		__m128d a = getDouble<aligned>(expr.a, i), b = getDouble<aligned>(expr.b, i);
		return _mm_or_pd(_mm_andnot_pd(signBit, a), _mm_and_pd(signBit, b));
	}
}

}}; // namespace

#if defined(__clang__) || defined(__GNUC__)
#	pragma GCC pop_options
#endif
