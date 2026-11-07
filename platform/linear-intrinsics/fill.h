/* Generic SIMD expression evaluation, given a class of basic operations (types, memory access, arithmetic and SSMF functions) such as `Sse2`.

GCC/Clang need #pragma wrapped around all the functions which we want to inline with a particular architecture inside, which means we can't just do it with a template.  Instead, we have this (unguarded, namespace-agnostic) fill class which assumes the architecture is present as `Arch`, and can be included from with different namespaces.
*/
struct SimdFill {
	using F = typename Arch::Float;
	using D = typename Arch::Double;
	struct ComplexFloat {
		F real, imag;
	};
	struct ComplexDouble {
		D real, imag;
	};
	static constexpr size_t floatLanes = sizeof(F)/sizeof(float);
	static constexpr size_t doubleLanes = sizeof(D)/sizeof(double);
	static constexpr size_t alignmentSize = sizeof(F);

	template<class V>
	static bool pointerAligned(const V *ptr) {
		return (size_t(ptr)&(alignmentSize - 1)) == 0;
	}

	template<class V>
	static V k(double v) {
		V result;
		Arch::splat(result, v);
		return result;
	}
	static float scalarOf(F);
	static double scalarOf(D);
	static F smallestNormal(F) {return k<F>(std::numeric_limits<float>::min());}
	static D smallestNormal(D) {return k<D>(std::numeric_limits<double>::min());}

	template<bool aligned, class S>
	static auto load(const S *pointer) -> decltype(Arch::load(pointer)) {
		return aligned ? Arch::load(pointer) : Arch::loadu(pointer);
	}
	template<bool aligned, class S, class V>
	static void store(S *pointer, V v) {
		if (aligned) {
			Arch::store(pointer, v);
		} else {
			Arch::storeu(pointer, v);
		}
	}

	// Choose getFloat()/getDouble() etc. from the scalar type
	template<bool aligned, class Expr>
	static F getReal(Expr expr, size_t i, const float *) {return getFloat<aligned>(expr, i);}
	template<bool aligned, class Expr>
	static D getReal(Expr expr, size_t i, const double *) {return getDouble<aligned>(expr, i);}
	template<bool aligned, class Expr>
	static ComplexFloat getComplex(Expr expr, size_t i, const float *) {return getCFloat<aligned>(expr, i);}
	template<bool aligned, class Expr>
	static ComplexDouble getComplex(Expr expr, size_t i, const double *) {return getCDouble<aligned>(expr, i);}

	template<bool aligned, class S, class Expr>
	static void fillRealSimd(S *pointer, Expr expr, size_t simdSize) {
		constexpr size_t lanes = alignmentSize/sizeof(S);
		for (size_t i = 0; i < simdSize; i += lanes) {
			store<aligned>(pointer + i, getReal<aligned>(expr, i, pointer));
		}
	}
	template<class S, class Expr>
	static void fillReal(S *pointer, Expr expr, size_t size) {
		size_t simdSize = size - size%(alignmentSize/sizeof(S));
		// Don't explode the code-paths - have one for "everything is aligned", one otherwise
		if (pointerAligned(pointer) && isAligned(expr)) {
			fillRealSimd<true>(pointer, expr, simdSize);
		} else {
			fillRealSimd<false>(pointer, expr, simdSize);
		}
		for (size_t i = simdSize; i < size; ++i) {
			pointer[i] = expr.get(i);
		}
	}
	template<bool aligned, class S, class Expr>
	static void fillComplexSimd(std::complex<S> *pointer, Expr expr, size_t simdSize) {
		constexpr size_t lanes = alignmentSize/sizeof(S);
		for (size_t i = 0; i < simdSize; i += lanes) {
			auto c = getComplex<aligned>(expr, i, (const S *)nullptr);
			S *s = reinterpret_cast<S *>(pointer + i);
			decltype(c.real) a, b;
			Arch::interleave(c.real, c.imag, a, b);
			store<aligned>(s, a);
			store<aligned>(s + lanes, b);
		}
	}
	template<class S, class Expr>
	static void fillComplex(std::complex<S> *pointer, Expr expr, size_t size) {
		size_t simdSize = size - size%(alignmentSize/sizeof(S));
		if (pointerAligned(pointer) && isAligned(expr)) {
			fillComplexSimd<true>(pointer, expr, simdSize);
		} else {
			fillComplexSimd<false>(pointer, expr, simdSize);
		}
		for (size_t i = simdSize; i < size; ++i) {
			pointer[i] = expr.get(i);
		}
	}
	template<bool aligned, class S, class Expr>
	static void fillSplitSimd(S *real, S *imag, Expr expr, size_t simdSize) {
		constexpr size_t lanes = alignmentSize/sizeof(S);
		for (size_t i = 0; i < simdSize; i += lanes) {
			auto c = getComplex<aligned>(expr, i, real);
			store<aligned>(real + i, c.real);
			store<aligned>(imag + i, c.imag);
		}
	}
	template<class S, class Expr>
	static void fillSplit(S *real, S *imag, Expr expr, size_t size) {
		size_t simdSize = size - size%(alignmentSize/sizeof(S));
		if (pointerAligned(real) && pointerAligned(imag) && isAligned(expr)) {
			fillSplitSimd<true>(real, imag, expr, simdSize);
		} else {
			fillSplitSimd<false>(real, imag, expr, simdSize);
		}
		for (size_t i = simdSize; i < size; ++i) {
			std::complex<S> c = expr.get(i);
			real[i] = c.real();
			imag[i] = c.imag();
		}
	}
	// ReadableReal
	template<class V>
	static bool isAligned(expression::ReadableReal<V> expr) {
		return pointerAligned(expr.pointer);
	}
	template<bool aligned>
	static F getFloat(expression::ReadableReal<float> expr, size_t i) {
		return load<aligned>(expr.pointer + i);
	}
	template<bool aligned>
	static D getDouble(expression::ReadableReal<double> expr, size_t i) {
		return load<aligned>(expr.pointer + i);
	}
	// ReadableComplex: two interleaved vectors
	template<class V>
	static bool isAligned(expression::ReadableComplex<V> expr) {
		return pointerAligned(expr.pointer);
	}
	template<bool aligned, class S, class C>
	static C loadComplex(const std::complex<S> *pointer) {
		const S *s = reinterpret_cast<const S *>(pointer);
		C c;
		Arch::deinterleave(load<aligned>(s), load<aligned>(s + alignmentSize/sizeof(S)), c.real, c.imag);
		return c;
	}
	template<bool aligned>
	static ComplexFloat getCFloat(expression::ReadableComplex<float> expr, size_t i) {
		return loadComplex<aligned, float, ComplexFloat>(expr.pointer + i);
	}
	template<bool aligned>
	static ComplexDouble getCDouble(expression::ReadableComplex<double> expr, size_t i) {
		return loadComplex<aligned, double, ComplexDouble>(expr.pointer + i);
	}
	// ReadableSplit
	template<class V>
	static bool isAligned(expression::ReadableSplit<V> expr) {
		return pointerAligned(expr.pointer.real) && pointerAligned(expr.pointer.imag);
	}
	template<bool aligned>
	static ComplexFloat getCFloat(expression::ReadableSplit<float> expr, size_t i) {
		return {load<aligned>(expr.pointer.real + i), load<aligned>(expr.pointer.imag + i)};
	}
	template<bool aligned>
	static ComplexDouble getCDouble(expression::ReadableSplit<double> expr, size_t i) {
		return {load<aligned>(expr.pointer.real + i), load<aligned>(expr.pointer.imag + i)};
	}

	// Constant
	template<class V>
	static bool isAligned(expression::ConstantExpr<V>) {
		return true;
	}
	template<bool aligned, class V>
	static F getFloat(expression::ConstantExpr<V> expr, size_t) {
		return k<F>(float(expr.value));
	}
	template<bool aligned, class V>
	static D getDouble(expression::ConstantExpr<V> expr, size_t) {
		return k<D>(double(expr.value));
	}
	template<bool aligned, class V>
	static ComplexFloat getCFloat(expression::ConstantExpr<V> expr, size_t) {
		std::complex<float> value = expr.value;
		return {k<F>(value.real()), k<F>(value.imag())};
	}
	template<bool aligned, class V>
	static ComplexDouble getCDouble(expression::ConstantExpr<V> expr, size_t) {
		std::complex<double> value = expr.value;
		return {k<D>(value.real()), k<D>(value.imag())};
	}

	// Functions which mathfun doesn't define, composed from ones it does
	template<class V>
	static V asinh(V x) {
		// log(|x| + sqrt(x² + 1)): absolute (not relative) accuracy near 0, and x² overflows for |x| > ~1e19 (or ~1e154 for double)
		V ax = Arch::abs(x);
		V r = Arch::log(Arch::add(ax, Arch::sqrt(Arch::add(Arch::pow2(ax), k<V>(1)))));
		return Arch::copysign(r, x);
	}
	template<class V>
	static V acosh(V x) {
		// log(x + sqrt((x - 1)(x + 1))): absolute (not relative) accuracy near 1, and overflows for x > ~1e19 (or ~1e154 for double)
		V s = Arch::sqrt(Arch::mul(Arch::sub(x, k<V>(1)), Arch::add(x, k<V>(1))));
		return Arch::log(Arch::add(x, s));
	}
	template<class V>
	static V atanh(V x) {
		// log((1 + x)/(1 - x))/2: absolute (not relative) accuracy near 0
		V ratio = Arch::div(Arch::add(k<V>(1), x), Arch::sub(k<V>(1), x));
		return Arch::mul(Arch::log(ratio), k<V>(0.5));
	}

	// Complex functions, from the real ones.  Unless noted, they have similar accuracy to the real ones (relative to |z| or |result|), and they work for any z which doesn't overflow or produce Inf/NaN.
	template<class C>
	static decltype(C::real) complexAbs(C z) {
		return Arch::hypot(z.real, z.imag);
	}
	template<class C>
	static decltype(C::real) complexNorm(C z) {
		return Arch::add(Arch::pow2(z.real), Arch::pow2(z.imag));
	}
	template<class C>
	static decltype(C::real) complexArg(C z) {
		return Arch::atan2(z.imag, z.real); // nonsense for z == 0
	}
	template<class C>
	static C complexConj(C z) {
		return {z.real, Arch::neg(z.imag)};
	}
	template<class C>
	static decltype(C::real) complexReal(C z) {
		return z.real;
	}
	template<class C>
	static decltype(C::real) complexImag(C z) {
		return z.imag;
	}
	template<class C>
	static C complexMul(C a, C b) {
		return {Arch::sub(Arch::mul(a.real, b.real), Arch::mul(a.imag, b.imag)), Arch::add(Arch::mul(a.real, b.imag), Arch::mul(a.imag, b.real))};
	}
	template<class C>
	static C complexExp(C z) {
		using V = decltype(z.real);
		V e = Arch::exp(z.real), s, c;
		Arch::sincos(z.imag, s, c);
		return {Arch::mul(e, c), Arch::mul(e, s)};
	}
	template<class C>
	static C complexLog(C z) {
		// Absolute (not relative) accuracy for the real part when |z| is close to 1, and nonsense for z == 0
		return {Arch::log(complexAbs(z)), complexArg(z)};
	}
	template<class C>
	static C complexLog10(C z) {
		using V = decltype(z.real);
		C l = complexLog(z);
		V scale = k<V>(0.43429448190325182); // 1/ln(10)
		return {Arch::mul(l.real, scale), Arch::mul(l.imag, scale)};
	}
	template<class C>
	static C complexSqrt(C z) {
		using V = decltype(z.real);
		// t = sqrt((|z| + |re|)/2) and h = im/(2t) are both accurate, and are (re, im) or (|im|, ±re) depending on the sign of re
		V w = Arch::mul(Arch::add(complexAbs(z), Arch::abs(z.real)), k<V>(0.5));
		// Clamped so that t (and h) are exactly 0 when z == 0
		V invT = Arch::div(k<V>(1), Arch::sqrt(Arch::max(w, smallestNormal(w))));
		V t = Arch::mul(w, invT), h = Arch::mul(Arch::mul(z.imag, k<V>(0.5)), invT);
		return {Arch::ifNegative(z.real, Arch::abs(h), t), Arch::ifNegative(z.real, Arch::copysign(t, z.imag), h)};
	}
	template<class C>
	static C complexSin(C z) {
		using V = decltype(z.real);
		V s, c;
		Arch::sincos(z.real, s, c);
		return {Arch::mul(s, Arch::cosh(z.imag)), Arch::mul(c, Arch::sinh(z.imag))};
	}
	template<class C>
	static C complexCos(C z) {
		using V = decltype(z.real);
		V s, c;
		Arch::sincos(z.real, s, c);
		return {Arch::mul(c, Arch::cosh(z.imag)), Arch::neg(Arch::mul(s, Arch::sinh(z.imag)))};
	}
	template<class C>
	static C complexSinh(C z) {
		using V = decltype(z.real);
		V s, c;
		Arch::sincos(z.imag, s, c);
		return {Arch::mul(Arch::sinh(z.real), c), Arch::mul(Arch::cosh(z.real), s)};
	}
	template<class C>
	static C complexCosh(C z) {
		using V = decltype(z.real);
		V s, c;
		Arch::sincos(z.imag, s, c);
		return {Arch::mul(Arch::cosh(z.real), c), Arch::mul(Arch::sinh(z.real), s)};
	}
	template<class C>
	static C complexTan(C z) {
		using V = decltype(z.real);
		// (sin(2a), sinh(2b))/(cos(2a) + cosh(2b)), with the denominator written as 2(cos²(a) + sinh²(b)) to avoid cancellation near the poles.  sinh²(b) overflows for |b| > ~44 (or ~355 for double)
		V s, c, sh = Arch::sinh(z.imag), ch = Arch::cosh(z.imag);
		Arch::sincos(z.real, s, c);
		V invD = Arch::div(k<V>(1), Arch::add(Arch::pow2(c), Arch::pow2(sh)));
		return {Arch::mul(Arch::mul(s, c), invD), Arch::mul(Arch::mul(sh, ch), invD)};
	}
	template<class C>
	static C complexTanh(C z) {
		using V = decltype(z.real);
		// (sinh(2a), sin(2b))/(cosh(2a) + cos(2b)), with the denominator written as 2(sinh²(a) + cos²(b)) to avoid cancellation near the poles.  sinh²(a) overflows for |a| > ~44 (or ~355 for double)
		V s, c, sh = Arch::sinh(z.real), ch = Arch::cosh(z.real);
		Arch::sincos(z.imag, s, c);
		V invD = Arch::div(k<V>(1), Arch::add(Arch::pow2(sh), Arch::pow2(c)));
		return {Arch::mul(Arch::mul(sh, ch), invD), Arch::mul(Arch::mul(s, c), invD)};
	}
	template<class C>
	static C complexAsinh(C z) {
		using V = decltype(z.real);
		// log(w + sqrt(w² + 1)) with w = ±z so that re(w) >= 0 (avoiding cancellation), then the sign put back.
		// Absolute (not relative) accuracy near 0 and near the branch points ±i, and w² overflows for |z| > ~1e19 (or ~1e154 for double)
		V sign = Arch::copysign(k<V>(1), z.real);
		C w{Arch::abs(z.real), Arch::mul(z.imag, sign)};
		C w2p1{Arch::add(Arch::mul(Arch::sub(w.real, w.imag), Arch::add(w.real, w.imag)), k<V>(1)), Arch::mul(Arch::add(w.real, w.real), w.imag)};
		C q = complexSqrt(w2p1);
		C l = complexLog(C{Arch::add(w.real, q.real), Arch::add(w.imag, q.imag)});
		return {Arch::mul(l.real, sign), Arch::mul(l.imag, sign)};
	}
	template<class C>
	static C complexAsin(C z) {
		// -i*asinh(i*z)
		C a = complexAsinh(C{Arch::neg(z.imag), z.real});
		return {a.imag, Arch::neg(a.real)};
	}
	template<class C>
	static C complexAcos(C z) {
		using V = decltype(z.real);
		// pi/2 - asin(z)
		C a = complexAsin(z);
		return {Arch::sub(k<V>(1.5707963267948966), a.real), Arch::neg(a.imag)};
	}
	template<class C>
	static C complexAcosh(C z) {
		using V = decltype(z.real);
		// log(z + sqrt(z + 1)*sqrt(z - 1)): absolute (not relative) accuracy near ±1, and z² overflows for |z| > ~1e19 (or ~1e154 for double)
		C s = complexMul(complexSqrt(C{Arch::add(z.real, k<V>(1)), z.imag}), complexSqrt(C{Arch::sub(z.real, k<V>(1)), z.imag}));
		return complexLog(C{Arch::add(z.real, s.real), Arch::add(z.imag, s.imag)});
	}
	template<class C>
	static C complexAtanh(C z) {
		using V = decltype(z.real);
		// (log(|1 + z|²/|1 - z|²)/4, atan2(2b, (1 - a)(1 + a) - b²)/2): absolute (not relative) accuracy near 0 and near the branch points ±1
		V b2 = Arch::pow2(z.imag);
		V onePlus = Arch::add(k<V>(1), z.real), oneMinus = Arch::sub(k<V>(1), z.real);
		V ratio = Arch::div(Arch::add(Arch::pow2(onePlus), b2), Arch::add(Arch::pow2(oneMinus), b2));
		V angle = Arch::atan2(Arch::add(z.imag, z.imag), Arch::sub(Arch::mul(oneMinus, onePlus), b2));
		return {Arch::mul(Arch::log(ratio), k<V>(0.25)), Arch::mul(angle, k<V>(0.5))};
	}
	template<class C>
	static C complexAtan(C z) {
		// -i*atanh(i*z)
		C a = complexAtanh(C{Arch::neg(z.imag), z.real});
		return {a.imag, Arch::neg(a.real)};
	}
	static ComplexFloat complexPolar(F r, F theta) {
		F s, c;
		Arch::sincos(theta, s, c);
		return {Arch::mul(r, c), Arch::mul(r, s)};
	}
	static ComplexDouble complexPolar(D r, D theta) {
		D s, c;
		Arch::sincos(theta, s, c);
		return {Arch::mul(r, c), Arch::mul(r, s)};
	}

	// No vector approximation: call the `std::` function on each lane
#define SIGNALSMITH_SCALAR_FALLBACK(fn) \
	template<class V> \
	static V fn(V v) { \
		using S = decltype(scalarOf(v)); \
		alignas(V) S array[sizeof(V)/sizeof(S)]; \
		Arch::store(array, v); \
		for (auto &x : array) x = std::fn(x); \
		return Arch::load(array); \
	}
	SIGNALSMITH_SCALAR_FALLBACK(tgamma)
	SIGNALSMITH_SCALAR_FALLBACK(lgamma)
#undef SIGNALSMITH_SCALAR_FALLBACK

	// Binary expressions are aligned if both sides are
#define SIGNALSMITH_EXPR_BINARY(Name) \
	template<class A, class B> \
	static bool isAligned(expression::Name<A, B> expr) { \
		return isAligned(expr.a) && isAligned(expr.b); \
	}
	SIGNALSMITH_EXPR_BINARY(Add)
	SIGNALSMITH_EXPR_BINARY(Sub)
	SIGNALSMITH_EXPR_BINARY(Mul)
	SIGNALSMITH_EXPR_BINARY(Div)
	SIGNALSMITH_EXPR_BINARY(Max)
	SIGNALSMITH_EXPR_BINARY(Min)
	SIGNALSMITH_EXPR_BINARY(Dim)
	SIGNALSMITH_EXPR_BINARY(Copysign)
	SIGNALSMITH_EXPR_BINARY(Pow)
	SIGNALSMITH_EXPR_BINARY(Atan2)
	SIGNALSMITH_EXPR_BINARY(Hypot)
	SIGNALSMITH_EXPR_BINARY(Polar)
#undef SIGNALSMITH_EXPR_BINARY

	// Expressions which map neatly to a single function
#define SIGNALSMITH_EXPR_UNARY(Name, fn) \
	template<class A> \
	static bool isAligned(expression::Name<A> expr) { \
		return isAligned(expr.a); \
	} \
	template<bool aligned, class A, class=ItemTypeF<A>> \
	static F getFloat(expression::Name<A> expr, size_t i) { \
		return fn(getFloat<aligned>(expr.a, i)); \
	} \
	template<bool aligned, class A, class=ItemTypeD<A>> \
	static D getDouble(expression::Name<A> expr, size_t i) { \
		return fn(getDouble<aligned>(expr.a, i)); \
	}
	SIGNALSMITH_EXPR_UNARY(Norm, Arch::pow2)
	SIGNALSMITH_EXPR_UNARY(Abs, Arch::abs)
	SIGNALSMITH_EXPR_UNARY(Neg, Arch::neg)
	SIGNALSMITH_EXPR_UNARY(Sqrt, Arch::sqrt)
	SIGNALSMITH_EXPR_UNARY(Cbrt, Arch::cbrt)
	SIGNALSMITH_EXPR_UNARY(Ceil, Arch::ceil)
	SIGNALSMITH_EXPR_UNARY(Floor, Arch::floor)
	SIGNALSMITH_EXPR_UNARY(Trunc, Arch::trunc)
	SIGNALSMITH_EXPR_UNARY(Round, Arch::round)
	SIGNALSMITH_EXPR_UNARY(Log, Arch::log)
	SIGNALSMITH_EXPR_UNARY(Log2, Arch::log2)
	SIGNALSMITH_EXPR_UNARY(Log10, Arch::log10)
	SIGNALSMITH_EXPR_UNARY(Exp, Arch::exp)
	SIGNALSMITH_EXPR_UNARY(Exp2, Arch::exp2)
	SIGNALSMITH_EXPR_UNARY(Sin, Arch::sin)
	SIGNALSMITH_EXPR_UNARY(Cos, Arch::cos)
	SIGNALSMITH_EXPR_UNARY(Tan, Arch::tan)
	SIGNALSMITH_EXPR_UNARY(Asin, Arch::asin)
	SIGNALSMITH_EXPR_UNARY(Acos, Arch::acos)
	SIGNALSMITH_EXPR_UNARY(Atan, Arch::atan)
	SIGNALSMITH_EXPR_UNARY(Sinh, Arch::sinh)
	SIGNALSMITH_EXPR_UNARY(Cosh, Arch::cosh)
	SIGNALSMITH_EXPR_UNARY(Tanh, Arch::tanh)
	SIGNALSMITH_EXPR_UNARY(Asinh, asinh)
	SIGNALSMITH_EXPR_UNARY(Acosh, acosh)
	SIGNALSMITH_EXPR_UNARY(Atanh, atanh)
	SIGNALSMITH_EXPR_UNARY(Erf, Arch::erf)
	SIGNALSMITH_EXPR_UNARY(Erfc, Arch::erfc)
	SIGNALSMITH_EXPR_UNARY(Tgamma, tgamma)
	SIGNALSMITH_EXPR_UNARY(Lgamma, lgamma)
#undef SIGNALSMITH_EXPR_UNARY
#define SIGNALSMITH_EXPR_BINARY(Name, fn) \
	template<bool aligned, class A, class B> \
	static F getFloat(expression::Name<A, B> expr, size_t i) { \
		return fn(getFloat<aligned>(expr.a, i), getFloat<aligned>(expr.b, i)); \
	} \
	template<bool aligned, class A, class B> \
	static D getDouble(expression::Name<A, B> expr, size_t i) { \
		return fn(getDouble<aligned>(expr.a, i), getDouble<aligned>(expr.b, i)); \
	}
	SIGNALSMITH_EXPR_BINARY(Add, Arch::add)
	SIGNALSMITH_EXPR_BINARY(Sub, Arch::sub)
	SIGNALSMITH_EXPR_BINARY(Mul, Arch::mul)
	SIGNALSMITH_EXPR_BINARY(Div, Arch::div)
	SIGNALSMITH_EXPR_BINARY(Max, Arch::max)
	SIGNALSMITH_EXPR_BINARY(Min, Arch::min)
	SIGNALSMITH_EXPR_BINARY(Dim, Arch::dim)
	SIGNALSMITH_EXPR_BINARY(Copysign, Arch::copysign)
	SIGNALSMITH_EXPR_BINARY(Pow, Arch::pow)
	SIGNALSMITH_EXPR_BINARY(Atan2, Arch::atan2)
	SIGNALSMITH_EXPR_BINARY(Hypot, Arch::hypot)
#undef SIGNALSMITH_EXPR_BINARY
	// a*b + c (linear.h puts the Mul on the left)
	template<bool aligned, class A, class B, class C>
	static F getFloat(expression::Add<expression::Mul<A, B>, C> expr, size_t i) {
		return Arch::fma(getFloat<aligned>(expr.a.a, i), getFloat<aligned>(expr.a.b, i), getFloat<aligned>(expr.b, i));
	}
	template<bool aligned, class A, class B, class C>
	static D getDouble(expression::Add<expression::Mul<A, B>, C> expr, size_t i) {
		return Arch::fma(getDouble<aligned>(expr.a.a, i), getDouble<aligned>(expr.a.b, i), getDouble<aligned>(expr.b, i));
	}

	// One-argument complex-to-complex
	template<bool aligned, class A>
	static ItemTypeCF<A, ComplexFloat> getCFloat(expression::Neg<A> expr, size_t i) {
		auto a = getCFloat<aligned>(expr.a, i);
		return {Arch::neg(a.real), Arch::neg(a.imag)};
	}
	template<bool aligned, class A>
	static ItemTypeCD<A, ComplexDouble> getCDouble(expression::Neg<A> expr, size_t i) {
		auto a = getCDouble<aligned>(expr.a, i);
		return {Arch::neg(a.real), Arch::neg(a.imag)};
	}

	// Add
	template<bool aligned, class A, class B>
	static ItemTypeCF<A, ItemTypeCF<B, ComplexFloat>> getCFloat(expression::Add<A, B> expr, size_t i) {
		auto a = getCFloat<aligned>(expr.a, i);
		auto b = getCFloat<aligned>(expr.b, i);
		return {Arch::add(a.real, b.real), Arch::add(a.imag, b.imag)};
	}
	template<bool aligned, class A, class B>
	static ItemTypeCD<A, ItemTypeCD<B, ComplexDouble>> getCDouble(expression::Add<A, B> expr, size_t i) {
		auto a = getCDouble<aligned>(expr.a, i);
		auto b = getCDouble<aligned>(expr.b, i);
		return {Arch::add(a.real, b.real), Arch::add(a.imag, b.imag)};
	}
	template<bool aligned, class A, class B>
	static ItemTypeCF<A, ItemTypeF<B, ComplexFloat>> getCFloat(expression::Add<A, B> expr, size_t i) {
		auto a = getCFloat<aligned>(expr.a, i);
		auto b = getFloat<aligned>(expr.b, i);
		return {Arch::add(a.real, b), a.imag};
	}
	template<bool aligned, class A, class B>
	static ItemTypeCD<A, ItemTypeD<B, ComplexDouble>> getCDouble(expression::Add<A, B> expr, size_t i) {
		auto a = getCDouble<aligned>(expr.a, i);
		auto b = getDouble<aligned>(expr.b, i);
		return {Arch::add(a.real, b), a.imag};
	}
	template<bool aligned, class A, class B>
	static ItemTypeF<A, ItemTypeCF<B, ComplexFloat>> getCFloat(expression::Add<A, B> expr, size_t i) {
		auto a = getFloat<aligned>(expr.a, i);
		auto b = getCFloat<aligned>(expr.b, i);
		return {Arch::add(a, b.real), b.imag};
	}
	template<bool aligned, class A, class B>
	static ItemTypeD<A, ItemTypeCD<B, ComplexDouble>> getCDouble(expression::Add<A, B> expr, size_t i) {
		auto a = getDouble<aligned>(expr.a, i);
		auto b = getCDouble<aligned>(expr.b, i);
		return {Arch::add(a, b.real), b.imag};
	}
	// Sub
	template<bool aligned, class A, class B>
	static ItemTypeCF<A, ItemTypeCF<B, ComplexFloat>> getCFloat(expression::Sub<A, B> expr, size_t i) {
		auto a = getCFloat<aligned>(expr.a, i);
		auto b = getCFloat<aligned>(expr.b, i);
		return {Arch::sub(a.real, b.real), Arch::sub(a.imag, b.imag)};
	}
	template<bool aligned, class A, class B>
	static ItemTypeCD<A, ItemTypeCD<B, ComplexDouble>> getCDouble(expression::Sub<A, B> expr, size_t i) {
		auto a = getCDouble<aligned>(expr.a, i);
		auto b = getCDouble<aligned>(expr.b, i);
		return {Arch::sub(a.real, b.real), Arch::sub(a.imag, b.imag)};
	}
	template<bool aligned, class A, class B>
	static ItemTypeCF<A, ItemTypeF<B, ComplexFloat>> getCFloat(expression::Sub<A, B> expr, size_t i) {
		auto a = getCFloat<aligned>(expr.a, i);
		auto b = getFloat<aligned>(expr.b, i);
		return {Arch::sub(a.real, b), a.imag};
	}
	template<bool aligned, class A, class B>
	static ItemTypeCD<A, ItemTypeD<B, ComplexDouble>> getCDouble(expression::Sub<A, B> expr, size_t i) {
		auto a = getCDouble<aligned>(expr.a, i);
		auto b = getDouble<aligned>(expr.b, i);
		return {Arch::sub(a.real, b), a.imag};
	}
	template<bool aligned, class A, class B>
	static ItemTypeF<A, ItemTypeCF<B, ComplexFloat>> getCFloat(expression::Sub<A, B> expr, size_t i) {
		auto a = getFloat<aligned>(expr.a, i);
		auto b = getCFloat<aligned>(expr.b, i);
		return {Arch::sub(a, b.real), Arch::neg(b.imag)};
	}
	template<bool aligned, class A, class B>
	static ItemTypeD<A, ItemTypeCD<B, ComplexDouble>> getCDouble(expression::Sub<A, B> expr, size_t i) {
		auto a = getDouble<aligned>(expr.a, i);
		auto b = getCDouble<aligned>(expr.b, i);
		return {Arch::sub(a, b.real), Arch::neg(b.imag)};
	}
	// Mul
	template<bool aligned, class A, class B>
	static ItemTypeCF<A, ItemTypeCF<B, ComplexFloat>> getCFloat(expression::Mul<A, B> expr, size_t i) {
		auto a = getCFloat<aligned>(expr.a, i);
		auto b = getCFloat<aligned>(expr.b, i);
		return {
			Arch::sub(Arch::mul(a.real, b.real), Arch::mul(a.imag, b.imag)),
			Arch::add(Arch::mul(a.real, b.imag), Arch::mul(a.imag, b.real)),
		};
	}
	template<bool aligned, class A, class B>
	static ItemTypeCD<A, ItemTypeCD<B, ComplexDouble>> getCDouble(expression::Mul<A, B> expr, size_t i) {
		auto a = getCDouble<aligned>(expr.a, i);
		auto b = getCDouble<aligned>(expr.b, i);
		return {
			Arch::sub(Arch::mul(a.real, b.real), Arch::mul(a.imag, b.imag)),
			Arch::add(Arch::mul(a.real, b.imag), Arch::mul(a.imag, b.real)),
		};
	}
	template<bool aligned, class A, class B>
	static ItemTypeCF<A, ItemTypeF<B, ComplexFloat>> getCFloat(expression::Mul<A, B> expr, size_t i) {
		auto a = getCFloat<aligned>(expr.a, i);
		auto b = getFloat<aligned>(expr.b, i);
		return {Arch::mul(a.real, b), Arch::mul(a.imag, b)};
	}
	template<bool aligned, class A, class B>
	static ItemTypeCD<A, ItemTypeD<B, ComplexDouble>> getCDouble(expression::Mul<A, B> expr, size_t i) {
		auto a = getCDouble<aligned>(expr.a, i);
		auto b = getDouble<aligned>(expr.b, i);
		return {Arch::mul(a.real, b), Arch::mul(a.imag, b)};
	}
	template<bool aligned, class A, class B>
	static ItemTypeF<A, ItemTypeCF<B, ComplexFloat>> getCFloat(expression::Mul<A, B> expr, size_t i) {
		auto a = getFloat<aligned>(expr.a, i);
		auto b = getCFloat<aligned>(expr.b, i);
		return {Arch::mul(a, b.real), Arch::mul(a, b.imag)};
	}
	template<bool aligned, class A, class B>
	static ItemTypeD<A, ItemTypeCD<B, ComplexDouble>> getCDouble(expression::Mul<A, B> expr, size_t i) {
		auto a = getDouble<aligned>(expr.a, i);
		auto b = getCDouble<aligned>(expr.b, i);
		return {Arch::mul(a, b.real), Arch::mul(a, b.imag)};
	}
	//Div
	template<bool aligned, class A, class B>
	static ItemTypeCF<A, ItemTypeCF<B, ComplexFloat>> getCFloat(expression::Div<A, B> expr, size_t i) {
		auto a = getCFloat<aligned>(expr.a, i);
		auto b = getCFloat<aligned>(expr.b, i);
		auto mag2B = Arch::add(Arch::pow2(b.real), Arch::pow2(b.imag));
		return {
			Arch::div(Arch::add(Arch::mul(a.real, b.real), Arch::mul(a.imag, b.imag)), mag2B),
			Arch::div(Arch::sub(Arch::mul(a.imag, b.real), Arch::mul(a.real, b.imag)), mag2B)
		};
	}
	template<bool aligned, class A, class B>
	static ItemTypeCD<A, ItemTypeCD<B, ComplexDouble>> getCDouble(expression::Div<A, B> expr, size_t i) {
		auto a = getCDouble<aligned>(expr.a, i);
		auto b = getCDouble<aligned>(expr.b, i);
		auto mag2B = Arch::add(Arch::pow2(b.real), Arch::pow2(b.imag));
		return {
			Arch::div(Arch::add(Arch::mul(a.real, b.real), Arch::mul(a.imag, b.imag)), mag2B),
			Arch::div(Arch::sub(Arch::mul(a.imag, b.real), Arch::mul(a.real, b.imag)), mag2B)
		};
	}
	template<bool aligned, class A, class B>
	static ItemTypeCF<A, ItemTypeF<B, ComplexFloat>> getCFloat(expression::Div<A, B> expr, size_t i) {
		auto a = getCFloat<aligned>(expr.a, i);
		auto b = getFloat<aligned>(expr.b, i);
		return {Arch::div(a.real, b), Arch::div(a.imag, b)};
	}
	template<bool aligned, class A, class B>
	static ItemTypeCD<A, ItemTypeD<B, ComplexDouble>> getCDouble(expression::Div<A, B> expr, size_t i) {
		auto a = getCDouble<aligned>(expr.a, i);
		auto b = getDouble<aligned>(expr.b, i);
		return {Arch::div(a.real, b), Arch::div(a.imag, b)};
	}
	template<bool aligned, class A, class B>
	static ItemTypeF<A, ItemTypeCF<B, ComplexFloat>> getCFloat(expression::Div<A, B> expr, size_t i) {
		auto a = getFloat<aligned>(expr.a, i);
		auto b = getCFloat<aligned>(expr.b, i);
		auto mag2B = Arch::add(Arch::pow2(b.real), Arch::pow2(b.imag));
		auto abNum = Arch::div(a, mag2B);
		return {Arch::mul(abNum, b.real), Arch::mul(abNum, Arch::neg(b.imag))};
	}
	template<bool aligned, class A, class B>
	static ItemTypeD<A, ItemTypeCD<B, ComplexDouble>> getCDouble(expression::Div<A, B> expr, size_t i) {
		auto a = getDouble<aligned>(expr.a, i);
		auto b = getCDouble<aligned>(expr.b, i);
		auto mag2B = Arch::add(Arch::pow2(b.real), Arch::pow2(b.imag));
		auto abNum = Arch::div(a, mag2B);
		return {Arch::mul(abNum, b.real), Arch::mul(abNum, Arch::neg(b.imag))};
	}

	 // Only two-argument float-to-complex
	template<bool aligned, class A, class B>
	static ItemTypeF<A, ComplexFloat> getCFloat(expression::Polar<A, B> expr, size_t i) {
		return complexPolar(getFloat<aligned>(expr.a, i), getFloat<aligned>(expr.b, i));
	}
	template<bool aligned, class A, class B>
	static ItemTypeD<A, ComplexDouble> getCDouble(expression::Polar<A, B> expr, size_t i) {
		return complexPolar(getDouble<aligned>(expr.a, i), getDouble<aligned>(expr.b, i));
	}

	// Complex -> real
#define SIGNALSMITH_EXPR_COMPLEX_REAL(Name, fn) \
	template<bool aligned, class A> \
	static auto getFloat(expression::Name<A> expr, size_t i) -> ItemTypeCF<A, decltype(fn(getCFloat<aligned>(expr.a, i)))> { \
		return fn(getCFloat<aligned>(expr.a, i)); \
	} \
	template<bool aligned, class A> \
	static auto getDouble(expression::Name<A> expr, size_t i) -> ItemTypeCD<A, decltype(fn(getCDouble<aligned>(expr.a, i)))> { \
		return fn(getCDouble<aligned>(expr.a, i)); \
	}
	SIGNALSMITH_EXPR_COMPLEX_REAL(Abs, complexAbs)
	SIGNALSMITH_EXPR_COMPLEX_REAL(Norm, complexNorm)

	template<class A>
	static bool isAligned(expression::Arg<A> expr) {
		return isAligned(expr.a);
	}
	template<class A>
	static bool isAligned(expression::Real<A> expr) {
		return isAligned(expr.a);
	}
	template<class A>
	static bool isAligned(expression::Imag<A> expr) {
		return isAligned(expr.a);
	}
	SIGNALSMITH_EXPR_COMPLEX_REAL(Arg, complexArg)
	SIGNALSMITH_EXPR_COMPLEX_REAL(Real, complexReal)
	SIGNALSMITH_EXPR_COMPLEX_REAL(Imag, complexImag)
#undef SIGNALSMITH_EXPR_COMPLEX_REAL

	// Complex -> complex
#define SIGNALSMITH_EXPR_COMPLEX(Name, fn) \
	template<bool aligned, class A> \
	static ItemTypeCF<A, ComplexFloat> getCFloat(expression::Name<A> expr, size_t i) { \
		return fn(getCFloat<aligned>(expr.a, i)); \
	} \
	template<bool aligned, class A> \
	static ItemTypeCD<A, ComplexDouble> getCDouble(expression::Name<A> expr, size_t i) { \
		return fn(getCDouble<aligned>(expr.a, i)); \
	}
	SIGNALSMITH_EXPR_COMPLEX(Conj, complexConj)
	SIGNALSMITH_EXPR_COMPLEX(Exp, complexExp)
	SIGNALSMITH_EXPR_COMPLEX(Log, complexLog)
	SIGNALSMITH_EXPR_COMPLEX(Log10, complexLog10)
	SIGNALSMITH_EXPR_COMPLEX(Sqrt, complexSqrt)
	SIGNALSMITH_EXPR_COMPLEX(Sin, complexSin)
	SIGNALSMITH_EXPR_COMPLEX(Cos, complexCos)
	SIGNALSMITH_EXPR_COMPLEX(Tan, complexTan)
	SIGNALSMITH_EXPR_COMPLEX(Asin, complexAsin)
	SIGNALSMITH_EXPR_COMPLEX(Acos, complexAcos)
	SIGNALSMITH_EXPR_COMPLEX(Atan, complexAtan)
	SIGNALSMITH_EXPR_COMPLEX(Sinh, complexSinh)
	SIGNALSMITH_EXPR_COMPLEX(Cosh, complexCosh)
	SIGNALSMITH_EXPR_COMPLEX(Tanh, complexTanh)
	SIGNALSMITH_EXPR_COMPLEX(Asinh, complexAsinh)
	SIGNALSMITH_EXPR_COMPLEX(Acosh, complexAcosh)
	SIGNALSMITH_EXPR_COMPLEX(Atanh, complexAtanh)
#undef SIGNALSMITH_EXPR_COMPLEX
	template<class A>
	static bool isAligned(expression::Conj<A> expr) {
		return isAligned(expr.a);
	}
};
