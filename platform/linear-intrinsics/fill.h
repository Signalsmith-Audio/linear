/* Generic SIMD expression evaluation, given basic operations (types, memory access, arithmetic and SSMF functions) for each scalar type, as `ArchFloat` and `ArchDouble`.

GCC/Clang need #pragma wrapped around all the functions which we want to inline with a particular architecture inside, which means we can't just do it with a template.  Instead, we have this (unguarded, namespace-agnostic) fill class which assumes `ArchFloat` and `ArchDouble` are present, and can be included from with different namespaces.

Each of those has a vector type `V` (which can be the same type for both, e.g. MSVC's NEON types are all `__n128`) and a `Scalar` type.  The vector type is never used to choose between them: float code calls `ArchFloat::` and double code calls `ArchDouble::`.
*/
struct SimdFill {
	// The elements left over after the last full vector
	using ScalarFill = ::signalsmith::linear::simd_scalar::SimdFill;

	using F = typename ArchFloat::V;
	using D = typename ArchDouble::V;
	struct ComplexFloat {
		using Arch = ArchFloat;
		F real, imag;
	};
	struct ComplexDouble {
		using Arch = ArchDouble;
		D real, imag;
	};
	template<class S>
	using ArchFor = typename std::conditional<std::is_same<S, float>::value, ArchFloat, ArchDouble>::type;
	// How many lanes for a scalar type.  These can differ between float and double (e.g. for the scalar `ArchDouble` used by 32-bit NEON)
	template<class S>
	static constexpr size_t lanes() {
		return sizeof(typename ArchFor<S>::V)/sizeof(S);
	}

	template<class S>
	static bool pointerAligned(const S *ptr) {
		return (size_t(ptr)&(sizeof(typename ArchFor<S>::V) - 1)) == 0;
	}
	template<class S>
	static bool pointerAligned(const std::complex<S> *ptr) {
		return pointerAligned(reinterpret_cast<const S *>(ptr));
	}

	template<bool aligned, class S>
	static typename ArchFor<S>::V load(const S *pointer) {
		return aligned ? ArchFor<S>::load(pointer) : ArchFor<S>::loadu(pointer);
	}
	template<bool aligned, class S>
	static void store(S *pointer, typename ArchFor<S>::V v) {
		if (aligned) {
			ArchFor<S>::store(pointer, v);
		} else {
			ArchFor<S>::storeu(pointer, v);
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
		constexpr size_t lanes = SimdFill::lanes<S>();
		for (size_t i = 0; i < simdSize; i += lanes) {
			store<aligned>(pointer + i, getReal<aligned>(expr, i, pointer));
		}
	}
	template<class S, class Expr>
	static void fillReal(S *pointer, Expr expr, size_t size) {
		size_t simdSize = size - size%lanes<S>();
		// Don't explode the code-paths - have one for "everything is aligned", one otherwise
		if (pointerAligned(pointer) && isAligned(expr)) {
			fillRealSimd<true>(pointer, expr, simdSize);
		} else {
			fillRealSimd<false>(pointer, expr, simdSize);
		}
		for (size_t i = simdSize; i < size; ++i) {
			pointer[i] = ScalarFill::getReal<false>(expr, i, pointer);
		}
	}
	template<bool aligned, class S, class Expr>
	static void fillComplexSimd(std::complex<S> *pointer, Expr expr, size_t simdSize) {
		constexpr size_t lanes = SimdFill::lanes<S>();
		for (size_t i = 0; i < simdSize; i += lanes) {
			auto c = getComplex<aligned>(expr, i, (const S *)nullptr);
			S *s = reinterpret_cast<S *>(pointer + i);
			decltype(c.real) a, b;
			ArchFor<S>::interleave(c.real, c.imag, a, b);
			store<aligned>(s, a);
			store<aligned>(s + lanes, b);
		}
	}
	template<class S, class Expr>
	static void fillComplex(std::complex<S> *pointer, Expr expr, size_t size) {
		size_t simdSize = size - size%lanes<S>();
		if (pointerAligned(pointer) && isAligned(expr)) {
			fillComplexSimd<true>(pointer, expr, simdSize);
		} else {
			fillComplexSimd<false>(pointer, expr, simdSize);
		}
		for (size_t i = simdSize; i < size; ++i) {
			auto c = ScalarFill::getComplex<false>(expr, i, (const S *)nullptr);
			pointer[i] = {c.real, c.imag};
		}
	}
	template<bool aligned, class S, class Expr>
	static void fillSplitSimd(S *real, S *imag, Expr expr, size_t simdSize) {
		constexpr size_t lanes = SimdFill::lanes<S>();
		for (size_t i = 0; i < simdSize; i += lanes) {
			auto c = getComplex<aligned>(expr, i, real);
			store<aligned>(real + i, c.real);
			store<aligned>(imag + i, c.imag);
		}
	}
	template<class S, class Expr>
	static void fillSplit(S *real, S *imag, Expr expr, size_t size) {
		size_t simdSize = size - size%lanes<S>();
		if (pointerAligned(real) && pointerAligned(imag) && isAligned(expr)) {
			fillSplitSimd<true>(real, imag, expr, simdSize);
		} else {
			fillSplitSimd<false>(real, imag, expr, simdSize);
		}
		for (size_t i = simdSize; i < size; ++i) {
			auto c = ScalarFill::getComplex<false>(expr, i, real);
			real[i] = c.real;
			imag[i] = c.imag;
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
		ArchFor<S>::deinterleave(load<aligned>(s), load<aligned>(s + lanes<S>()), c.real, c.imag);
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
		return ArchFloat::splat(float(expr.value));
	}
	template<bool aligned, class V>
	static D getDouble(expression::ConstantExpr<V> expr, size_t) {
		return ArchDouble::splat(double(expr.value));
	}
	template<bool aligned, class V>
	static ComplexFloat getCFloat(expression::ConstantExpr<V> expr, size_t) {
		std::complex<float> value = expr.value;
		return {ArchFloat::splat(value.real()), ArchFloat::splat(value.imag())};
	}
	template<bool aligned, class V>
	static ComplexDouble getCDouble(expression::ConstantExpr<V> expr, size_t) {
		std::complex<double> value = expr.value;
		return {ArchDouble::splat(value.real()), ArchDouble::splat(value.imag())};
	}

	// Functions which mathfun doesn't define, composed from ones it does
	template<class A>
	static typename A::V asinh(typename A::V x) {
		using V = typename A::V;
		// log(|x| + sqrt(x² + 1)): absolute (not relative) accuracy near 0, and x² overflows for |x| > ~1e19 (or ~1e154 for double)
		V ax = A::abs(x);
		V r = A::log(A::add(ax, A::sqrt(A::add(A::pow2(ax), A::splat(1)))));
		return A::copysign(r, x);
	}
	template<class A>
	static typename A::V acosh(typename A::V x) {
		using V = typename A::V;
		// log(x + sqrt((x - 1)(x + 1))): absolute (not relative) accuracy near 1, and overflows for x > ~1e19 (or ~1e154 for double)
		V s = A::sqrt(A::mul(A::sub(x, A::splat(1)), A::add(x, A::splat(1))));
		return A::log(A::add(x, s));
	}
	template<class A>
	static typename A::V atanh(typename A::V x) {
		using V = typename A::V;
		// log((1 + x)/(1 - x))/2: absolute (not relative) accuracy near 0
		V ratio = A::div(A::add(A::splat(1), x), A::sub(A::splat(1), x));
		return A::mul(A::log(ratio), A::splat(0.5));
	}

	// Complex functions, from the real ones.  Unless noted, they have similar accuracy to the real ones (relative to |z| or |result|), and they work for any z which doesn't overflow or produce Inf/NaN.
	template<class C>
	static decltype(C::real) complexAbs(C z) {
		using A = typename C::Arch;
		return A::hypot(z.real, z.imag);
	}
	template<class C>
	static decltype(C::real) complexNorm(C z) {
		using A = typename C::Arch;
		return A::fma(z.real, z.real, A::pow2(z.imag));
	}
	template<class C>
	static decltype(C::real) complexArg(C z) {
		using A = typename C::Arch;
		return A::atan2(z.imag, z.real); // nonsense for z == 0
	}
	template<class C>
	static C complexConj(C z) {
		using A = typename C::Arch;
		return {z.real, A::neg(z.imag)};
	}
	template<class C>
	static decltype(C::real) complexReal(C z) {
		return z.real;
	}
	template<class C>
	static decltype(C::real) complexImag(C z) {
		return z.imag;
	}
	// Complex multiply/divide use FMA, so e.g. z*conj(z) can have a non-zero imaginary part
	template<class C>
	static C complexMul(C a, C b) {
		using A = typename C::Arch;
		return {A::fms(a.real, b.real, A::mul(a.imag, b.imag)), A::fma(a.real, b.imag, A::mul(a.imag, b.real))};
	}
	// conj(a)*b
	template<class C>
	static C complexConjMul(C a, C b) {
		using A = typename C::Arch;
		return {A::fma(a.real, b.real, A::mul(a.imag, b.imag)), A::fms(a.real, b.imag, A::mul(a.imag, b.real))};
	}
	// a*b + c
	template<class C>
	static C complexMulAdd(C a, C b, C c) {
		using A = typename C::Arch;
		return {A::fma(a.real, b.real, A::fnma(a.imag, b.imag, c.real)), A::fma(a.real, b.imag, A::fma(a.imag, b.real, c.imag))};
	}
	// conj(a)*b + c
	template<class C>
	static C complexConjMulAdd(C a, C b, C c) {
		using A = typename C::Arch;
		return {A::fma(a.real, b.real, A::fma(a.imag, b.imag, c.real)), A::fma(a.real, b.imag, A::fnma(a.imag, b.real, c.imag))};
	}
	template<class C>
	static C complexDiv(C a, C b) {
		using A = typename C::Arch;
		auto mag2 = complexNorm(b);
		C n = complexConjMul(b, a);
		return {A::div(n.real, mag2), A::div(n.imag, mag2)};
	}
	template<class C>
	static C complexExp(C z) {
		using A = typename C::Arch;
		using V = typename A::V;
		V e = A::exp(z.real), s, c;
		A::sincos(z.imag, &s, &c);
		return {A::mul(e, c), A::mul(e, s)};
	}
	template<class C>
	static C complexLog(C z) {
		using A = typename C::Arch;
		// Absolute (not relative) accuracy for the real part when |z| is close to 1, and nonsense for z == 0
		return {A::log(complexAbs(z)), complexArg(z)};
	}
	template<class C>
	static C complexLog10(C z) {
		using A = typename C::Arch;
		using V = typename A::V;
		C l = complexLog(z);
		V scale = A::splat(0.43429448190325182); // 1/ln(10)
		return {A::mul(l.real, scale), A::mul(l.imag, scale)};
	}
	template<class C>
	static C complexSqrt(C z) {
		using A = typename C::Arch;
		using V = typename A::V;
		// t = sqrt((|z| + |re|)/2) and h = im/(2t) are both accurate, and are (re, im) or (|im|, ±re) depending on the sign of re
		V w = A::mul(A::add(complexAbs(z), A::abs(z.real)), A::splat(0.5));
		// Clamped so that t (and h) are exactly 0 when z == 0
		V invT = A::div(A::splat(1), A::sqrt(A::max(w, A::splat(std::numeric_limits<typename A::Scalar>::min()))));
		V t = A::mul(w, invT), h = A::mul(A::mul(z.imag, A::splat(0.5)), invT);
		return {A::ifNegative(z.real, A::abs(h), t), A::ifNegative(z.real, A::copysign(t, z.imag), h)};
	}
	template<class C>
	static C complexSin(C z) {
		using A = typename C::Arch;
		using V = typename A::V;
		V s, c;
		A::sincos(z.real, &s, &c);
		return {A::mul(s, A::cosh(z.imag)), A::mul(c, A::sinh(z.imag))};
	}
	template<class C>
	static C complexCos(C z) {
		using A = typename C::Arch;
		using V = typename A::V;
		V s, c;
		A::sincos(z.real, &s, &c);
		return {A::mul(c, A::cosh(z.imag)), A::neg(A::mul(s, A::sinh(z.imag)))};
	}
	template<class C>
	static C complexSinh(C z) {
		using A = typename C::Arch;
		using V = typename A::V;
		V s, c;
		A::sincos(z.imag, &s, &c);
		return {A::mul(A::sinh(z.real), c), A::mul(A::cosh(z.real), s)};
	}
	template<class C>
	static C complexCosh(C z) {
		using A = typename C::Arch;
		using V = typename A::V;
		V s, c;
		A::sincos(z.imag, &s, &c);
		return {A::mul(A::cosh(z.real), c), A::mul(A::sinh(z.real), s)};
	}
	template<class C>
	static C complexTan(C z) {
		using A = typename C::Arch;
		using V = typename A::V;
		// (sin(2a), sinh(2b))/(cos(2a) + cosh(2b)), with the denominator written as 2(cos²(a) + sinh²(b)) to avoid cancellation near the poles.  sinh²(b) overflows for |b| > ~44 (or ~355 for double)
		V s, c, sh = A::sinh(z.imag), ch = A::cosh(z.imag);
		A::sincos(z.real, &s, &c);
		V invD = A::div(A::splat(1), A::add(A::pow2(c), A::pow2(sh)));
		return {A::mul(A::mul(s, c), invD), A::mul(A::mul(sh, ch), invD)};
	}
	template<class C>
	static C complexTanh(C z) {
		using A = typename C::Arch;
		using V = typename A::V;
		// (sinh(2a), sin(2b))/(cosh(2a) + cos(2b)), with the denominator written as 2(sinh²(a) + cos²(b)) to avoid cancellation near the poles.  sinh²(a) overflows for |a| > ~44 (or ~355 for double)
		V s, c, sh = A::sinh(z.real), ch = A::cosh(z.real);
		A::sincos(z.imag, &s, &c);
		V invD = A::div(A::splat(1), A::add(A::pow2(sh), A::pow2(c)));
		return {A::mul(A::mul(sh, ch), invD), A::mul(A::mul(s, c), invD)};
	}
	template<class C>
	static C complexAsinh(C z) {
		using A = typename C::Arch;
		using V = typename A::V;
		// log(w + sqrt(w² + 1)) with w = ±z so that re(w) >= 0 (avoiding cancellation), then the sign put back.
		// Absolute (not relative) accuracy near 0 and near the branch points ±i, and w² overflows for |z| > ~1e19 (or ~1e154 for double)
		V sign = A::copysign(A::splat(1), z.real);
		C w{A::abs(z.real), A::mul(z.imag, sign)};
		C w2p1{A::add(A::mul(A::sub(w.real, w.imag), A::add(w.real, w.imag)), A::splat(1)), A::mul(A::add(w.real, w.real), w.imag)};
		C q = complexSqrt(w2p1);
		C l = complexLog(C{A::add(w.real, q.real), A::add(w.imag, q.imag)});
		return {A::mul(l.real, sign), A::mul(l.imag, sign)};
	}
	template<class C>
	static C complexAsin(C z) {
		using A = typename C::Arch;
		// -i*asinh(i*z)
		C a = complexAsinh(C{A::neg(z.imag), z.real});
		return {a.imag, A::neg(a.real)};
	}
	template<class C>
	static C complexAcos(C z) {
		using A = typename C::Arch;
		// pi/2 - asin(z)
		C a = complexAsin(z);
		return {A::sub(A::splat(1.5707963267948966), a.real), A::neg(a.imag)};
	}
	template<class C>
	static C complexAcosh(C z) {
		using A = typename C::Arch;
		// log(z + sqrt(z + 1)*sqrt(z - 1)): absolute (not relative) accuracy near ±1, and z² overflows for |z| > ~1e19 (or ~1e154 for double)
		C s = complexMul(complexSqrt(C{A::add(z.real, A::splat(1)), z.imag}), complexSqrt(C{A::sub(z.real, A::splat(1)), z.imag}));
		return complexLog(C{A::add(z.real, s.real), A::add(z.imag, s.imag)});
	}
	template<class C>
	static C complexAtanh(C z) {
		using A = typename C::Arch;
		using V = typename A::V;
		// (log(|1 + z|²/|1 - z|²)/4, atan2(2b, (1 - a)(1 + a) - b²)/2): absolute (not relative) accuracy near 0 and near the branch points ±1
		V b2 = A::pow2(z.imag);
		V onePlus = A::add(A::splat(1), z.real), oneMinus = A::sub(A::splat(1), z.real);
		V ratio = A::div(A::add(A::pow2(onePlus), b2), A::add(A::pow2(oneMinus), b2));
		V angle = A::atan2(A::add(z.imag, z.imag), A::sub(A::mul(oneMinus, onePlus), b2));
		return {A::mul(A::log(ratio), A::splat(0.25)), A::mul(angle, A::splat(0.5))};
	}
	template<class C>
	static C complexAtan(C z) {
		using A = typename C::Arch;
		// -i*atanh(i*z)
		C a = complexAtanh(C{A::neg(z.imag), z.real});
		return {a.imag, A::neg(a.real)};
	}
	template<class C>
	static C complexPolar(decltype(C::real) r, decltype(C::real) theta) {
		using A = typename C::Arch;
		decltype(C::real) s, c;
		A::sincos(theta, &s, &c);
		return {A::mul(r, c), A::mul(r, s)};
	}

	// No vector approximation: call the `std::` function on each lane
#define SIGNALSMITH_SCALAR_FALLBACK(fn) \
	template<class A> \
	static typename A::V fn(typename A::V v) { \
		using V = typename A::V; \
		using S = typename A::Scalar; \
		alignas(V) S array[sizeof(V)/sizeof(S)]; \
		A::store(array, v); \
		for (auto &x : array) x = std::fn(x); \
		return A::load(array); \
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
#define SIGNALSMITH_EXPR_UNARY(Name, fnFloat, fnDouble) \
	template<class A> \
	static bool isAligned(expression::Name<A> expr) { \
		return isAligned(expr.a); \
	} \
	template<bool aligned, class A, class=ItemTypeF<A>> \
	static F getFloat(expression::Name<A> expr, size_t i) { \
		return fnFloat(getFloat<aligned>(expr.a, i)); \
	} \
	template<bool aligned, class A, class=ItemTypeD<A>> \
	static D getDouble(expression::Name<A> expr, size_t i) { \
		return fnDouble(getDouble<aligned>(expr.a, i)); \
	}
	SIGNALSMITH_EXPR_UNARY(Norm, ArchFloat::pow2, ArchDouble::pow2)
	SIGNALSMITH_EXPR_UNARY(Abs, ArchFloat::abs, ArchDouble::abs)
	SIGNALSMITH_EXPR_UNARY(Neg, ArchFloat::neg, ArchDouble::neg)
	SIGNALSMITH_EXPR_UNARY(Sqrt, ArchFloat::sqrt, ArchDouble::sqrt)
	SIGNALSMITH_EXPR_UNARY(Cbrt, ArchFloat::cbrt, ArchDouble::cbrt)
	SIGNALSMITH_EXPR_UNARY(Ceil, ArchFloat::ceil, ArchDouble::ceil)
	SIGNALSMITH_EXPR_UNARY(Floor, ArchFloat::floor, ArchDouble::floor)
	SIGNALSMITH_EXPR_UNARY(Trunc, ArchFloat::trunc, ArchDouble::trunc)
	SIGNALSMITH_EXPR_UNARY(Round, ArchFloat::round, ArchDouble::round)
	SIGNALSMITH_EXPR_UNARY(Log, ArchFloat::log, ArchDouble::log)
	SIGNALSMITH_EXPR_UNARY(Log2, ArchFloat::log2, ArchDouble::log2)
	SIGNALSMITH_EXPR_UNARY(Log10, ArchFloat::log10, ArchDouble::log10)
	SIGNALSMITH_EXPR_UNARY(Exp, ArchFloat::exp, ArchDouble::exp)
	SIGNALSMITH_EXPR_UNARY(Exp2, ArchFloat::exp2, ArchDouble::exp2)
	SIGNALSMITH_EXPR_UNARY(Sin, ArchFloat::sin, ArchDouble::sin)
	SIGNALSMITH_EXPR_UNARY(Cos, ArchFloat::cos, ArchDouble::cos)
	SIGNALSMITH_EXPR_UNARY(Tan, ArchFloat::tan, ArchDouble::tan)
	SIGNALSMITH_EXPR_UNARY(Asin, ArchFloat::asin, ArchDouble::asin)
	SIGNALSMITH_EXPR_UNARY(Acos, ArchFloat::acos, ArchDouble::acos)
	SIGNALSMITH_EXPR_UNARY(Atan, ArchFloat::atan, ArchDouble::atan)
	SIGNALSMITH_EXPR_UNARY(Sinh, ArchFloat::sinh, ArchDouble::sinh)
	SIGNALSMITH_EXPR_UNARY(Cosh, ArchFloat::cosh, ArchDouble::cosh)
	SIGNALSMITH_EXPR_UNARY(Tanh, ArchFloat::tanh, ArchDouble::tanh)
	SIGNALSMITH_EXPR_UNARY(Asinh, asinh<ArchFloat>, asinh<ArchDouble>)
	SIGNALSMITH_EXPR_UNARY(Acosh, acosh<ArchFloat>, acosh<ArchDouble>)
	SIGNALSMITH_EXPR_UNARY(Atanh, atanh<ArchFloat>, atanh<ArchDouble>)
	SIGNALSMITH_EXPR_UNARY(Erf, ArchFloat::erf, ArchDouble::erf)
	SIGNALSMITH_EXPR_UNARY(Erfc, ArchFloat::erfc, ArchDouble::erfc)
	SIGNALSMITH_EXPR_UNARY(Tgamma, tgamma<ArchFloat>, tgamma<ArchDouble>)
	SIGNALSMITH_EXPR_UNARY(Lgamma, lgamma<ArchFloat>, lgamma<ArchDouble>)
#undef SIGNALSMITH_EXPR_UNARY
#define SIGNALSMITH_EXPR_BINARY(Name, fn) \
	template<bool aligned, class A, class B> \
	static F getFloat(expression::Name<A, B> expr, size_t i) { \
		return ArchFloat::fn(getFloat<aligned>(expr.a, i), getFloat<aligned>(expr.b, i)); \
	} \
	template<bool aligned, class A, class B> \
	static D getDouble(expression::Name<A, B> expr, size_t i) { \
		return ArchDouble::fn(getDouble<aligned>(expr.a, i), getDouble<aligned>(expr.b, i)); \
	}
	SIGNALSMITH_EXPR_BINARY(Add, add)
	SIGNALSMITH_EXPR_BINARY(Sub, sub)
	SIGNALSMITH_EXPR_BINARY(Mul, mul)
	SIGNALSMITH_EXPR_BINARY(Div, div)
	SIGNALSMITH_EXPR_BINARY(Max, max)
	SIGNALSMITH_EXPR_BINARY(Min, min)
	SIGNALSMITH_EXPR_BINARY(Dim, dim)
	SIGNALSMITH_EXPR_BINARY(Copysign, copysign)
	SIGNALSMITH_EXPR_BINARY(Pow, pow)
	SIGNALSMITH_EXPR_BINARY(Atan2, atan2)
	SIGNALSMITH_EXPR_BINARY(Hypot, hypot)
#undef SIGNALSMITH_EXPR_BINARY
	// Fused multiply-add shapes.  linear.h puts things in a consistent shape for better matching
#define SIGNALSMITH_EXPR_FMA(fn, a, b, c, ...) \
	template<bool aligned, class A, class B, class C> \
	static F getFloat(__VA_ARGS__ expr, size_t i) { \
		return ArchFloat::fn(getFloat<aligned>(a, i), getFloat<aligned>(b, i), getFloat<aligned>(c, i)); \
	} \
	template<bool aligned, class A, class B, class C> \
	static D getDouble(__VA_ARGS__ expr, size_t i) { \
		return ArchDouble::fn(getDouble<aligned>(a, i), getDouble<aligned>(b, i), getDouble<aligned>(c, i)); \
	}
	SIGNALSMITH_EXPR_FMA(fma, expr.a.a, expr.a.b, expr.b, expression::Add<expression::Mul<A, B>, C>) // A*B + C
	SIGNALSMITH_EXPR_FMA(fms, expr.a.a, expr.a.b, expr.b, expression::Sub<expression::Mul<A, B>, C>) // A*B - C
	SIGNALSMITH_EXPR_FMA(fnma, expr.b.a, expr.b.b, expr.a, expression::Sub<C, expression::Mul<A, B>>) // C - A*B
	SIGNALSMITH_EXPR_FMA(fnms, expr.a.a.a, expr.a.a.b, expr.a.b, expression::Neg<expression::Add<expression::Mul<A, B>, C>>) // -(A*B + C)
#undef SIGNALSMITH_EXPR_FMA
	// A/sqrt(B) -> A*rsqrt(B), which isn't exact (but is often faster)
	template<bool aligned, class A, class B>
	static F getFloat(expression::Div<A, expression::Sqrt<B>> expr, size_t i) {
		return ArchFloat::mul(getFloat<aligned>(expr.a, i), ArchFloat::rsqrt(getFloat<aligned>(expr.b.a, i)));
	}
	template<bool aligned, class A, class B>
	static D getDouble(expression::Div<A, expression::Sqrt<B>> expr, size_t i) {
		return ArchDouble::mul(getDouble<aligned>(expr.a, i), ArchDouble::rsqrt(getDouble<aligned>(expr.b.a, i)));
	}
	// A*B - C*D matches both of the Sub shapes above, so it needs its own
	template<bool aligned, class A, class B, class C, class D2>
	static F getFloat(expression::Sub<expression::Mul<A, B>, expression::Mul<C, D2>> expr, size_t i) {
		return ArchFloat::fms(getFloat<aligned>(expr.a.a, i), getFloat<aligned>(expr.a.b, i), getFloat<aligned>(expr.b, i));
	}
	template<bool aligned, class A, class B, class C, class D2>
	static D getDouble(expression::Sub<expression::Mul<A, B>, expression::Mul<C, D2>> expr, size_t i) {
		return ArchDouble::fms(getDouble<aligned>(expr.a.a, i), getDouble<aligned>(expr.a.b, i), getDouble<aligned>(expr.b, i));
	}

	// One-argument complex-to-complex
	template<bool aligned, class A>
	static ItemTypeCF<A, ComplexFloat> getCFloat(expression::Neg<A> expr, size_t i) {
		auto a = getCFloat<aligned>(expr.a, i);
		return {ArchFloat::neg(a.real), ArchFloat::neg(a.imag)};
	}
	template<bool aligned, class A>
	static ItemTypeCD<A, ComplexDouble> getCDouble(expression::Neg<A> expr, size_t i) {
		auto a = getCDouble<aligned>(expr.a, i);
		return {ArchDouble::neg(a.real), ArchDouble::neg(a.imag)};
	}

	// Add
	template<bool aligned, class A, class B>
	static ItemTypeCF<A, ItemTypeCF<B, ComplexFloat>> getCFloat(expression::Add<A, B> expr, size_t i) {
		auto a = getCFloat<aligned>(expr.a, i);
		auto b = getCFloat<aligned>(expr.b, i);
		return {ArchFloat::add(a.real, b.real), ArchFloat::add(a.imag, b.imag)};
	}
	template<bool aligned, class A, class B>
	static ItemTypeCD<A, ItemTypeCD<B, ComplexDouble>> getCDouble(expression::Add<A, B> expr, size_t i) {
		auto a = getCDouble<aligned>(expr.a, i);
		auto b = getCDouble<aligned>(expr.b, i);
		return {ArchDouble::add(a.real, b.real), ArchDouble::add(a.imag, b.imag)};
	}
	template<bool aligned, class A, class B>
	static ItemTypeCF<A, ItemTypeF<B, ComplexFloat>> getCFloat(expression::Add<A, B> expr, size_t i) {
		auto a = getCFloat<aligned>(expr.a, i);
		auto b = getFloat<aligned>(expr.b, i);
		return {ArchFloat::add(a.real, b), a.imag};
	}
	template<bool aligned, class A, class B>
	static ItemTypeCD<A, ItemTypeD<B, ComplexDouble>> getCDouble(expression::Add<A, B> expr, size_t i) {
		auto a = getCDouble<aligned>(expr.a, i);
		auto b = getDouble<aligned>(expr.b, i);
		return {ArchDouble::add(a.real, b), a.imag};
	}
	template<bool aligned, class A, class B>
	static ItemTypeF<A, ItemTypeCF<B, ComplexFloat>> getCFloat(expression::Add<A, B> expr, size_t i) {
		auto a = getFloat<aligned>(expr.a, i);
		auto b = getCFloat<aligned>(expr.b, i);
		return {ArchFloat::add(a, b.real), b.imag};
	}
	template<bool aligned, class A, class B>
	static ItemTypeD<A, ItemTypeCD<B, ComplexDouble>> getCDouble(expression::Add<A, B> expr, size_t i) {
		auto a = getDouble<aligned>(expr.a, i);
		auto b = getCDouble<aligned>(expr.b, i);
		return {ArchDouble::add(a, b.real), b.imag};
	}
	// Sub
	template<bool aligned, class A, class B>
	static ItemTypeCF<A, ItemTypeCF<B, ComplexFloat>> getCFloat(expression::Sub<A, B> expr, size_t i) {
		auto a = getCFloat<aligned>(expr.a, i);
		auto b = getCFloat<aligned>(expr.b, i);
		return {ArchFloat::sub(a.real, b.real), ArchFloat::sub(a.imag, b.imag)};
	}
	template<bool aligned, class A, class B>
	static ItemTypeCD<A, ItemTypeCD<B, ComplexDouble>> getCDouble(expression::Sub<A, B> expr, size_t i) {
		auto a = getCDouble<aligned>(expr.a, i);
		auto b = getCDouble<aligned>(expr.b, i);
		return {ArchDouble::sub(a.real, b.real), ArchDouble::sub(a.imag, b.imag)};
	}
	template<bool aligned, class A, class B>
	static ItemTypeCF<A, ItemTypeF<B, ComplexFloat>> getCFloat(expression::Sub<A, B> expr, size_t i) {
		auto a = getCFloat<aligned>(expr.a, i);
		auto b = getFloat<aligned>(expr.b, i);
		return {ArchFloat::sub(a.real, b), a.imag};
	}
	template<bool aligned, class A, class B>
	static ItemTypeCD<A, ItemTypeD<B, ComplexDouble>> getCDouble(expression::Sub<A, B> expr, size_t i) {
		auto a = getCDouble<aligned>(expr.a, i);
		auto b = getDouble<aligned>(expr.b, i);
		return {ArchDouble::sub(a.real, b), a.imag};
	}
	template<bool aligned, class A, class B>
	static ItemTypeF<A, ItemTypeCF<B, ComplexFloat>> getCFloat(expression::Sub<A, B> expr, size_t i) {
		auto a = getFloat<aligned>(expr.a, i);
		auto b = getCFloat<aligned>(expr.b, i);
		return {ArchFloat::sub(a, b.real), ArchFloat::neg(b.imag)};
	}
	template<bool aligned, class A, class B>
	static ItemTypeD<A, ItemTypeCD<B, ComplexDouble>> getCDouble(expression::Sub<A, B> expr, size_t i) {
		auto a = getDouble<aligned>(expr.a, i);
		auto b = getCDouble<aligned>(expr.b, i);
		return {ArchDouble::sub(a, b.real), ArchDouble::neg(b.imag)};
	}
	// Mul
	template<bool aligned, class A, class B>
	static ItemTypeCF<A, ItemTypeCF<B, ComplexFloat>> getCFloat(expression::Mul<A, B> expr, size_t i) {
		return complexMul(getCFloat<aligned>(expr.a, i), getCFloat<aligned>(expr.b, i));
	}
	template<bool aligned, class A, class B>
	static ItemTypeCD<A, ItemTypeCD<B, ComplexDouble>> getCDouble(expression::Mul<A, B> expr, size_t i) {
		return complexMul(getCDouble<aligned>(expr.a, i), getCDouble<aligned>(expr.b, i));
	}
	template<bool aligned, class A, class B>
	static ItemTypeCF<A, ItemTypeF<B, ComplexFloat>> getCFloat(expression::Mul<A, B> expr, size_t i) {
		auto a = getCFloat<aligned>(expr.a, i);
		auto b = getFloat<aligned>(expr.b, i);
		return {ArchFloat::mul(a.real, b), ArchFloat::mul(a.imag, b)};
	}
	template<bool aligned, class A, class B>
	static ItemTypeCD<A, ItemTypeD<B, ComplexDouble>> getCDouble(expression::Mul<A, B> expr, size_t i) {
		auto a = getCDouble<aligned>(expr.a, i);
		auto b = getDouble<aligned>(expr.b, i);
		return {ArchDouble::mul(a.real, b), ArchDouble::mul(a.imag, b)};
	}
	template<bool aligned, class A, class B>
	static ItemTypeF<A, ItemTypeCF<B, ComplexFloat>> getCFloat(expression::Mul<A, B> expr, size_t i) {
		auto a = getFloat<aligned>(expr.a, i);
		auto b = getCFloat<aligned>(expr.b, i);
		return {ArchFloat::mul(a, b.real), ArchFloat::mul(a, b.imag)};
	}
	template<bool aligned, class A, class B>
	static ItemTypeD<A, ItemTypeCD<B, ComplexDouble>> getCDouble(expression::Mul<A, B> expr, size_t i) {
		auto a = getDouble<aligned>(expr.a, i);
		auto b = getCDouble<aligned>(expr.b, i);
		return {ArchDouble::mul(a, b.real), ArchDouble::mul(a, b.imag)};
	}
	// conj(A)*B (linear.h puts the Conj on the left)
	template<bool aligned, class A, class B>
	static ItemTypeCF<A, ItemTypeCF<B, ComplexFloat>> getCFloat(expression::Mul<expression::Conj<A>, B> expr, size_t i) {
		return complexConjMul(getCFloat<aligned>(expr.a.a, i), getCFloat<aligned>(expr.b, i));
	}
	template<bool aligned, class A, class B>
	static ItemTypeCD<A, ItemTypeCD<B, ComplexDouble>> getCDouble(expression::Mul<expression::Conj<A>, B> expr, size_t i) {
		return complexConjMul(getCDouble<aligned>(expr.a.a, i), getCDouble<aligned>(expr.b, i));
	}
	// A*B + C (complex C)
	template<bool aligned, class A, class B, class C>
	static ItemTypeCF<A, ItemTypeCF<B, ItemTypeCF<C, ComplexFloat>>> getCFloat(expression::Add<expression::Mul<A, B>, C> expr, size_t i) {
		return complexMulAdd(getCFloat<aligned>(expr.a.a, i), getCFloat<aligned>(expr.a.b, i), getCFloat<aligned>(expr.b, i));
	}
	// A is conjugate
	template<bool aligned, class A, class B, class C>
	static ItemTypeCF<A, ItemTypeCF<B, ItemTypeCF<C, ComplexFloat>>> getCFloat(expression::Add<expression::Mul<expression::Conj<A>, B>, C> expr, size_t i) {
		return complexConjMulAdd(getCFloat<aligned>(expr.a.a.a, i), getCFloat<aligned>(expr.a.b, i), getCFloat<aligned>(expr.b, i));
	}
	// B is real
	template<bool aligned, class A, class B, class C>
	static ItemTypeCF<A, ItemTypeF<B, ItemTypeCF<C, ComplexFloat>>> getCFloat(expression::Add<expression::Mul<A, B>, C> expr, size_t i) {
		auto a = getCFloat<aligned>(expr.a.a, i);
		auto b = getFloat<aligned>(expr.a.b, i);
		auto c = getCFloat<aligned>(expr.b, i);
		return {ArchFloat::fma(a.real, b, c.real), ArchFloat::fma(a.imag, b, c.imag)};
	}
	// A is real
	template<bool aligned, class A, class B, class C>
	static ItemTypeF<A, ItemTypeCF<B, ItemTypeCF<C, ComplexFloat>>> getCFloat(expression::Add<expression::Mul<A, B>, C> expr, size_t i) {
		auto a = getFloat<aligned>(expr.a.a, i);
		auto b = getCFloat<aligned>(expr.a.b, i);
		auto c = getCFloat<aligned>(expr.b, i);
		return {ArchFloat::fma(a, b.real, c.real), ArchFloat::fma(a, b.imag, c.imag)};
	}
	// A and B are complex, C is real
	template<bool aligned, class A, class B, class C>
	static ItemTypeCF<A, ItemTypeCF<B, ItemTypeF<C, ComplexFloat>>> getCFloat(expression::Add<expression::Mul<A, B>, C> expr, size_t i) {
		auto a = getCFloat<aligned>(expr.a.a, i);
		auto b = getCFloat<aligned>(expr.a.b, i);
		auto c = getFloat<aligned>(expr.b, i);
		return {ArchFloat::fma(a.real, b.real, ArchFloat::fnma(a.imag, b.imag, c)), ArchFloat::fma(a.real, b.imag, ArchFloat::mul(a.imag, b.real))};
	}
	// Same but for doubles
	template<bool aligned, class A, class B, class C>
	static ItemTypeCD<A, ItemTypeCD<B, ItemTypeCD<C, ComplexDouble>>> getCDouble(expression::Add<expression::Mul<A, B>, C> expr, size_t i) {
		return complexMulAdd(getCDouble<aligned>(expr.a.a, i), getCDouble<aligned>(expr.a.b, i), getCDouble<aligned>(expr.b, i));
	}
	template<bool aligned, class A, class B, class C>
	static ItemTypeCD<A, ItemTypeCD<B, ItemTypeCD<C, ComplexDouble>>> getCDouble(expression::Add<expression::Mul<expression::Conj<A>, B>, C> expr, size_t i) {
		return complexConjMulAdd(getCDouble<aligned>(expr.a.a.a, i), getCDouble<aligned>(expr.a.b, i), getCDouble<aligned>(expr.b, i));
	}
	template<bool aligned, class A, class B, class C>
	static ItemTypeCD<A, ItemTypeD<B, ItemTypeCD<C, ComplexDouble>>> getCDouble(expression::Add<expression::Mul<A, B>, C> expr, size_t i) {
		auto a = getCDouble<aligned>(expr.a.a, i);
		auto b = getDouble<aligned>(expr.a.b, i);
		auto c = getCDouble<aligned>(expr.b, i);
		return {ArchDouble::fma(a.real, b, c.real), ArchDouble::fma(a.imag, b, c.imag)};
	}
	template<bool aligned, class A, class B, class C>
	static ItemTypeD<A, ItemTypeCD<B, ItemTypeCD<C, ComplexDouble>>> getCDouble(expression::Add<expression::Mul<A, B>, C> expr, size_t i) {
		auto a = getDouble<aligned>(expr.a.a, i);
		auto b = getCDouble<aligned>(expr.a.b, i);
		auto c = getCDouble<aligned>(expr.b, i);
		return {ArchDouble::fma(a, b.real, c.real), ArchDouble::fma(a, b.imag, c.imag)};
	}
	template<bool aligned, class A, class B, class C>
	static ItemTypeCD<A, ItemTypeCD<B, ItemTypeD<C, ComplexDouble>>> getCDouble(expression::Add<expression::Mul<A, B>, C> expr, size_t i) {
		auto a = getCDouble<aligned>(expr.a.a, i);
		auto b = getCDouble<aligned>(expr.a.b, i);
		auto c = getDouble<aligned>(expr.b, i);
		return {ArchDouble::fma(a.real, b.real, ArchDouble::fnma(a.imag, b.imag, c)), ArchDouble::fma(a.real, b.imag, ArchDouble::mul(a.imag, b.real))};
	}
	// Div
	template<bool aligned, class A, class B>
	static ItemTypeCF<A, ItemTypeCF<B, ComplexFloat>> getCFloat(expression::Div<A, B> expr, size_t i) {
		return complexDiv(getCFloat<aligned>(expr.a, i), getCFloat<aligned>(expr.b, i));
	}
	template<bool aligned, class A, class B>
	static ItemTypeCD<A, ItemTypeCD<B, ComplexDouble>> getCDouble(expression::Div<A, B> expr, size_t i) {
		return complexDiv(getCDouble<aligned>(expr.a, i), getCDouble<aligned>(expr.b, i));
	}
	template<bool aligned, class A, class B>
	static ItemTypeCF<A, ItemTypeF<B, ComplexFloat>> getCFloat(expression::Div<A, B> expr, size_t i) {
		auto a = getCFloat<aligned>(expr.a, i);
		auto b = getFloat<aligned>(expr.b, i);
		return {ArchFloat::div(a.real, b), ArchFloat::div(a.imag, b)};
	}
	template<bool aligned, class A, class B>
	static ItemTypeCD<A, ItemTypeD<B, ComplexDouble>> getCDouble(expression::Div<A, B> expr, size_t i) {
		auto a = getCDouble<aligned>(expr.a, i);
		auto b = getDouble<aligned>(expr.b, i);
		return {ArchDouble::div(a.real, b), ArchDouble::div(a.imag, b)};
	}
	template<bool aligned, class A, class B>
	static ItemTypeF<A, ItemTypeCF<B, ComplexFloat>> getCFloat(expression::Div<A, B> expr, size_t i) {
		auto a = getFloat<aligned>(expr.a, i);
		auto b = getCFloat<aligned>(expr.b, i);
		auto mag2B = complexNorm(b);
		auto abNum = ArchFloat::div(a, mag2B);
		return {ArchFloat::mul(abNum, b.real), ArchFloat::mul(abNum, ArchFloat::neg(b.imag))};
	}
	template<bool aligned, class A, class B>
	static ItemTypeD<A, ItemTypeCD<B, ComplexDouble>> getCDouble(expression::Div<A, B> expr, size_t i) {
		auto a = getDouble<aligned>(expr.a, i);
		auto b = getCDouble<aligned>(expr.b, i);
		auto mag2B = complexNorm(b);
		auto abNum = ArchDouble::div(a, mag2B);
		return {ArchDouble::mul(abNum, b.real), ArchDouble::mul(abNum, ArchDouble::neg(b.imag))};
	}

	 // Only two-argument float-to-complex
	template<bool aligned, class A, class B>
	static ItemTypeF<A, ComplexFloat> getCFloat(expression::Polar<A, B> expr, size_t i) {
		return complexPolar<ComplexFloat>(getFloat<aligned>(expr.a, i), getFloat<aligned>(expr.b, i));
	}
	template<bool aligned, class A, class B>
	static ItemTypeD<A, ComplexDouble> getCDouble(expression::Polar<A, B> expr, size_t i) {
		return complexPolar<ComplexDouble>(getDouble<aligned>(expr.a, i), getDouble<aligned>(expr.b, i));
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
