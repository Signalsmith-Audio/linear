//#define ACCELERATE_NEW_LAPACK

// If possible, only include vecLib, since JUCE has conflicts with vImage
#if defined(__has_include) && __has_include(<vecLib/vecLib.h>)
#	include <vecLib/vecLib.h>
#else
#	include <Accelerate/Accelerate.h>
#endif

#include <cstring> // std::memcpy
#include <limits>
#include <type_traits>

#include "./basic-fill-warnings.h"

namespace signalsmith { namespace linear {

template<>
struct LinearImpl<true> : public LinearImplBase<true> {
	using Base = LinearImplBase<true>;

	LinearImpl() : Base(this), cached(*this) {
		basicFillWarningReset();
	}

	template<class V>
	void reserve(size_t size) {
		// assume float/doubles if the right size
		if (sizeof(V) == sizeof(float)) {
			cached.reserveFloats(size*4);
		} else if (sizeof(V) == sizeof(double)) {
			cached.reserveDoubles(size*4);
		}
	}

	template<class Pointer, class Expr>
	void fill(Pointer pointer, Expr expr, size_t size) {
		fillExpr(pointer, expr, size);
	}

	template<class Pointer, class Expr>
	void fill(Pointer pointer, Expression<Expr> expr, size_t size) {
		return fillExpr(pointer, (Expr &)expr, size);
	}
	template<class Pointer, class Expr>
	void fill(Pointer pointer, WritableExpression<Expr> expr, size_t size) {
		return fillExpr(pointer, (Expr &)expr, size);
	}

private:

	// Most generic fill
	template<class Pointer, class Expr>
	void fillExpr(Pointer pointer, Expr expr, size_t size) {
		fillBasic(pointer, expr, size);
	}

	template<class Pointer, class Expr>
	void fillBasic(Pointer pointer, Expr expr, size_t size) {
		basicFillWarning<Expr>();
		for (size_t i = 0; i < size; ++i) {
			pointer[i] = expr.get(i);
		}
	}
	template<class V, class Expr>
	void fillBasic(SplitPointer<V> pointer, Expr expr, size_t size) {
		basicFillWarning<Expr>();
		using Complex = typename SplitPointer<V>::Complex;
		for (size_t i = 0; i < size; ++i) {
			Complex c = expr.get(i);
			pointer.real[i] = c.real();
			pointer.imag[i] = c.imag();
		}
	}
	
	template<typename V>
	void clear(V *v, size_t size) {
		for (size_t i = 0; i < size; ++i) v[i] = 0;
	}
	void clear(float *v, size_t size) {
		vDSP_vclr(v, 1, size);
	}
	void clear(double *v, size_t size) {
		vDSP_vclrD(v, 1, size);
	}
	// Filling a split-complex vector with real values won't hit the specialisations below, so we handle it here
	template<class Expr>
	ItemTypeF<Expr> fillBasic(SplitPointer<float> pointer, Expr expr, size_t size) {
		fillExpr(pointer.real, expr, size);
		clear(pointer.imag, size);
	}
	template<class Expr>
	ItemTypeD<Expr> fillBasic(SplitPointer<double> pointer, Expr expr, size_t size) {
		fillExpr(pointer.real, expr, size);
		clear(pointer.imag, size);
	}
	
	// Copying from existing pointer
	void fillExpr(RealPointer<float> pointer, expression::ReadableReal<float> expr, size_t size) {
		std::memcpy(pointer, expr.pointer, size*sizeof(float));
	}
	void fillExpr(RealPointer<float> pointer, WritableReal<float> expr, size_t size) {
		std::memcpy(pointer, expr.pointer, size*sizeof(float));
	}
	void fillExpr(RealPointer<double> pointer, expression::ReadableReal<double> expr, size_t size) {
		std::memcpy(pointer, expr.pointer, size*sizeof(double));
	}
	void fillExpr(RealPointer<double> pointer, WritableReal<double> expr, size_t size) {
		std::memcpy(pointer, expr.pointer, size*sizeof(double));
	}
	void fillExpr(ComplexPointer<float> pointer, expression::ReadableComplex<float> expr, size_t size) {
		std::memcpy(pointer, expr.pointer, size*sizeof(std::complex<float>));
	}
	void fillExpr(ComplexPointer<float> pointer, WritableComplex<float> expr, size_t size) {
		std::memcpy(pointer, expr.pointer, size*sizeof(std::complex<float>));
	}
	void fillExpr(ComplexPointer<double> pointer, expression::ReadableComplex<double> expr, size_t size) {
		std::memcpy(pointer, expr.pointer, size*sizeof(std::complex<double>));
	}
	void fillExpr(ComplexPointer<double> pointer, WritableComplex<double> expr, size_t size) {
		std::memcpy(pointer, expr.pointer, size*sizeof(std::complex<double>));
	}

	// Real part of a split pointer
	void fillExpr(RealPointer<float> pointer, expression::Real<expression::ReadableSplit<float>> expr, size_t size) {
		std::memcpy(pointer, expr.a.pointer.real, size*sizeof(float));
	}
	void fillExpr(RealPointer<double> pointer, expression::Real<expression::ReadableSplit<double>> expr, size_t size) {
		std::memcpy(pointer, expr.a.pointer.real, size*sizeof(double));
	}
	void fillExpr(RealPointer<float> pointer, expression::Real<WritableSplit<float>> expr, size_t size) {
		std::memcpy(pointer, expr.a.pointer.real, size*sizeof(float));
	}
	void fillExpr(RealPointer<double> pointer, expression::Real<WritableSplit<double>> expr, size_t size) {
		std::memcpy(pointer, expr.a.pointer.real, size*sizeof(double));
	}
	// Imaginary part
	void fillExpr(RealPointer<float> pointer, expression::Imag<expression::ReadableSplit<float>> expr, size_t size) {
		std::memcpy(pointer, expr.a.pointer.imag, size*sizeof(float));
	}
	void fillExpr(RealPointer<double> pointer, expression::Imag<expression::ReadableSplit<double>> expr, size_t size) {
		std::memcpy(pointer, expr.a.pointer.imag, size*sizeof(double));
	}
	void fillExpr(RealPointer<float> pointer, expression::Imag<WritableSplit<float>> expr, size_t size) {
		std::memcpy(pointer, expr.a.pointer.imag, size*sizeof(float));
	}
	void fillExpr(RealPointer<double> pointer, expression::Imag<WritableSplit<double>> expr, size_t size) {
		std::memcpy(pointer, expr.a.pointer.imag, size*sizeof(double));
	}
	
	// Filling with a constant
	template<class V>
	void fillExpr(RealPointer<float> pointer, expression::ConstantExpr<V> expr, size_t size) {
		float v = expr.value;
		vDSP_vfill(&v, pointer, 1, size);
	}
	template<class V>
	void fillExpr(RealPointer<double> pointer, expression::ConstantExpr<V> expr, size_t size) {
		double v = expr.value;
		vDSP_vfillD(&v, pointer, 1, size);
	}
	template<class V>
	void fillExpr(ComplexPointer<float> pointer, expression::ConstantExpr<V> expr, size_t size) {
		std::complex<float> v = expr.value;
		auto vSplit = dspSplit(&v);
		auto pSplit = dspSplit(pointer);
		vDSP_zvfill(&vSplit, &pSplit, 2, size);
	}
	template<class V>
	void fillExpr(ComplexPointer<double> pointer, expression::ConstantExpr<V> expr, size_t size) {
		std::complex<double> v = expr.value;
		auto vSplit = dspSplit(&v);
		auto pSplit = dspSplit(pointer);
		vDSP_zvfillD(&vSplit, &pSplit, 2, size);
	}
	template<class V>
	void fillExpr(SplitPointer<float> pointer, expression::ConstantExpr<V> expr, size_t size) {
		std::complex<float> v = expr.value;
		float vr = v.real(), vi = v.imag();
		auto vSplit = dspSplit({&vr, &vi});
		auto pSplit = dspSplit(pointer);
		vDSP_zvfill(&vSplit, &pSplit, 1, size);
	}
	template<class V>
	void fillExpr(SplitPointer<double> pointer, expression::ConstantExpr<V> expr, size_t size) {
		std::complex<double> v = expr.value;
		double vr = v.real(), vi = v.imag();
		auto vSplit = dspSplit({&vr, &vi});
		auto pSplit = dspSplit(pointer);
		vDSP_zvfillD(&vSplit, &pSplit, 1, size);
	}

	//---- Unary ops

	template<class A>
	ItemTypeF<A> fillExpr(RealPointer<float> pointer, expression::Abs<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		vDSP_vabs(a, 1, pointer, 1, size);
	}
	template<class A>
	ItemTypeD<A> fillExpr(RealPointer<double> pointer, expression::Abs<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		vDSP_vabsD(a, 1, pointer, 1, size);
	}
	template<class A>
	ItemType<A, std::complex<float>> fillExpr(RealPointer<float> pointer, expression::Abs<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a, size));
		vDSP_zvabs(&a, 1, pointer, 1, size);
	}
	template<class A>
	ItemType<A, std::complex<double>> fillExpr(RealPointer<double> pointer, expression::Abs<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a, size));
		vDSP_zvabsD(&a, 1, pointer, 1, size);
	}

	template<class A>
	ItemTypeF<A> fillExpr(RealPointer<float> pointer, expression::Neg<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		vDSP_vneg(a, 1, pointer, 1, size);
	}
	template<class A>
	ItemTypeD<A> fillExpr(RealPointer<double> pointer, expression::Neg<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		vDSP_vnegD(a, 1, pointer, 1, size);
	}
	template<class A>
	ItemType<A, std::complex<float>> fillExpr(ComplexPointer<float> pointer, expression::Neg<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.complex(expr.a, size));
		auto p = dspSplit(pointer);
		vDSP_zvneg(&a, 2, &p, 2, size);
	}
	template<class A>
	ItemType<A, std::complex<double>> fillExpr(ComplexPointer<double> pointer, expression::Neg<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.complex(expr.a, size));
		auto p = dspSplit(pointer);
		vDSP_zvnegD(&a, 2, &p, 2, size);
	}
	template<class A>
	ItemType<A, std::complex<float>> fillExpr(SplitPointer<float> pointer, expression::Neg<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a, size));
		auto p = dspSplit(pointer);
		vDSP_zvneg(&a, 1, &p, 1, size);
	}
	template<class A>
	ItemType<A, std::complex<double>> fillExpr(SplitPointer<double> pointer, expression::Neg<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a, size));
		auto p = dspSplit(pointer);
		vDSP_zvnegD(&a, 1, &p, 1, size);
	}

	//---- Unary real functions (vForce)
	template<class A>
	void fillExpr(RealPointer<float> pointer, expression::Exp<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		int intSize = int(size);
		vvexpf(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<double> pointer, expression::Exp<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		int intSize = int(size);
		vvexp(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<float> pointer, expression::Exp2<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		int intSize = int(size);
		vvexp2f(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<double> pointer, expression::Exp2<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		int intSize = int(size);
		vvexp2(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<float> pointer, expression::Log<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		int intSize = int(size);
		vvlogf(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<double> pointer, expression::Log<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		int intSize = int(size);
		vvlog(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<float> pointer, expression::Log2<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		int intSize = int(size);
		vvlog2f(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<double> pointer, expression::Log2<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		int intSize = int(size);
		vvlog2(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<float> pointer, expression::Log10<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		int intSize = int(size);
		vvlog10f(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<double> pointer, expression::Log10<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		int intSize = int(size);
		vvlog10(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<float> pointer, expression::Sqrt<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		int intSize = int(size);
		vvsqrtf(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<double> pointer, expression::Sqrt<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		int intSize = int(size);
		vvsqrt(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<float> pointer, expression::Ceil<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		int intSize = int(size);
		vvceilf(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<double> pointer, expression::Ceil<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		int intSize = int(size);
		vvceil(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<float> pointer, expression::Floor<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		int intSize = int(size);
		vvfloorf(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<double> pointer, expression::Floor<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		int intSize = int(size);
		vvfloor(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<float> pointer, expression::Trunc<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		int intSize = int(size);
		vvintf(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<double> pointer, expression::Trunc<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		int intSize = int(size);
		vvint(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<float> pointer, expression::Sin<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		int intSize = int(size);
		vvsinf(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<double> pointer, expression::Sin<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		int intSize = int(size);
		vvsin(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<float> pointer, expression::Cos<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		int intSize = int(size);
		vvcosf(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<double> pointer, expression::Cos<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		int intSize = int(size);
		vvcos(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<float> pointer, expression::Tan<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		int intSize = int(size);
		vvtanf(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<double> pointer, expression::Tan<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		int intSize = int(size);
		vvtan(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<float> pointer, expression::Asin<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		int intSize = int(size);
		vvasinf(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<double> pointer, expression::Asin<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		int intSize = int(size);
		vvasin(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<float> pointer, expression::Acos<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		int intSize = int(size);
		vvacosf(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<double> pointer, expression::Acos<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		int intSize = int(size);
		vvacos(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<float> pointer, expression::Atan<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		int intSize = int(size);
		vvatanf(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<double> pointer, expression::Atan<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		int intSize = int(size);
		vvatan(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<float> pointer, expression::Sinh<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		int intSize = int(size);
		vvsinhf(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<double> pointer, expression::Sinh<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		int intSize = int(size);
		vvsinh(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<float> pointer, expression::Cosh<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		int intSize = int(size);
		vvcoshf(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<double> pointer, expression::Cosh<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		int intSize = int(size);
		vvcosh(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<float> pointer, expression::Tanh<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		int intSize = int(size);
		vvtanhf(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<double> pointer, expression::Tanh<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		int intSize = int(size);
		vvtanh(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<float> pointer, expression::Asinh<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		int intSize = int(size);
		vvasinhf(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<double> pointer, expression::Asinh<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		int intSize = int(size);
		vvasinh(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<float> pointer, expression::Acosh<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		int intSize = int(size);
		vvacoshf(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<double> pointer, expression::Acosh<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		int intSize = int(size);
		vvacosh(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<float> pointer, expression::Atanh<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		int intSize = int(size);
		vvatanhf(pointer, a, &intSize);
	}
	template<class A>
	void fillExpr(RealPointer<double> pointer, expression::Atanh<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		int intSize = int(size);
		vvatanh(pointer, a, &intSize);
	}

	//---- Real results from real or complex values
	template<class A>
	ItemTypeF<A> fillExpr(RealPointer<float> pointer, expression::Norm<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		vDSP_vsq(a, 1, pointer, 1, size);
	}
	template<class A>
	ItemTypeD<A> fillExpr(RealPointer<double> pointer, expression::Norm<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		vDSP_vsqD(a, 1, pointer, 1, size);
	}
	template<class A>
	ItemTypeCF<A> fillExpr(RealPointer<float> pointer, expression::Norm<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a, size));
		vDSP_zvmags(&a, 1, pointer, 1, size);
	}
	template<class A>
	ItemTypeCD<A> fillExpr(RealPointer<double> pointer, expression::Norm<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a, size));
		vDSP_zvmagsD(&a, 1, pointer, 1, size);
	}
	template<class A>
	ItemTypeCF<A> fillExpr(RealPointer<float> pointer, expression::Arg<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a, size));
		vDSP_zvphas(&a, 1, pointer, 1, size);
	}
	template<class A>
	ItemTypeCD<A> fillExpr(RealPointer<double> pointer, expression::Arg<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a, size));
		vDSP_zvphasD(&a, 1, pointer, 1, size);
	}
	template<class A>
	ItemTypeCF<A> fillExpr(RealPointer<float> pointer, expression::Real<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = (float *)floats.complex(expr.a, size);
		vDSP_mmov(a, pointer, 1, size, 2, 1);
	}
	template<class A>
	ItemTypeCF<A> fillExpr(RealPointer<float> pointer, expression::Imag<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = (float *)floats.complex(expr.a, size);
		vDSP_mmov(a + 1, pointer, 1, size, 2, 1);
	}
	template<class A>
	ItemTypeCD<A> fillExpr(RealPointer<double> pointer, expression::Real<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = (double *)doubles.complex(expr.a, size);
		vDSP_mmovD(a, pointer, 1, size, 2, 1);
	}
	template<class A>
	ItemTypeCD<A> fillExpr(RealPointer<double> pointer, expression::Imag<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = (double *)doubles.complex(expr.a, size);
		vDSP_mmovD(a + 1, pointer, 1, size, 2, 1);
	}

	// Interleaved inputs can be read directly (stride 2) instead of converting to split
	void fillExpr(RealPointer<float> pointer, expression::Abs<expression::ReadableComplex<float>> expr, size_t size) {
		auto a = dspSplit(expr.a.pointer);
		vDSP_zvabs(&a, 2, pointer, 1, size);
	}
	void fillExpr(RealPointer<float> pointer, expression::Norm<expression::ReadableComplex<float>> expr, size_t size) {
		auto a = dspSplit(expr.a.pointer);
		vDSP_zvmags(&a, 2, pointer, 1, size);
	}
	void fillExpr(RealPointer<float> pointer, expression::Arg<expression::ReadableComplex<float>> expr, size_t size) {
		auto a = dspSplit(expr.a.pointer);
		vDSP_zvphas(&a, 2, pointer, 1, size);
	}
	void fillExpr(RealPointer<float> pointer, expression::Abs<WritableComplex<float>> expr, size_t size) {
		auto a = dspSplit(expr.a.pointer);
		vDSP_zvabs(&a, 2, pointer, 1, size);
	}
	void fillExpr(RealPointer<float> pointer, expression::Norm<WritableComplex<float>> expr, size_t size) {
		auto a = dspSplit(expr.a.pointer);
		vDSP_zvmags(&a, 2, pointer, 1, size);
	}
	void fillExpr(RealPointer<float> pointer, expression::Arg<WritableComplex<float>> expr, size_t size) {
		auto a = dspSplit(expr.a.pointer);
		vDSP_zvphas(&a, 2, pointer, 1, size);
	}
	void fillExpr(RealPointer<double> pointer, expression::Abs<expression::ReadableComplex<double>> expr, size_t size) {
		auto a = dspSplit(expr.a.pointer);
		vDSP_zvabsD(&a, 2, pointer, 1, size);
	}
	void fillExpr(RealPointer<double> pointer, expression::Norm<expression::ReadableComplex<double>> expr, size_t size) {
		auto a = dspSplit(expr.a.pointer);
		vDSP_zvmagsD(&a, 2, pointer, 1, size);
	}
	void fillExpr(RealPointer<double> pointer, expression::Arg<expression::ReadableComplex<double>> expr, size_t size) {
		auto a = dspSplit(expr.a.pointer);
		vDSP_zvphasD(&a, 2, pointer, 1, size);
	}
	void fillExpr(RealPointer<double> pointer, expression::Abs<WritableComplex<double>> expr, size_t size) {
		auto a = dspSplit(expr.a.pointer);
		vDSP_zvabsD(&a, 2, pointer, 1, size);
	}
	void fillExpr(RealPointer<double> pointer, expression::Norm<WritableComplex<double>> expr, size_t size) {
		auto a = dspSplit(expr.a.pointer);
		vDSP_zvmagsD(&a, 2, pointer, 1, size);
	}
	void fillExpr(RealPointer<double> pointer, expression::Arg<WritableComplex<double>> expr, size_t size) {
		auto a = dspSplit(expr.a.pointer);
		vDSP_zvphasD(&a, 2, pointer, 1, size);
	}

	//---- Binary real ops
	template<class A, class B>
	NotConstant<A, NotConstant<B>> fillExpr(RealPointer<float> pointer, expression::Add<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		auto *b = floats.real(expr.b, size);
		vDSP_vadd(a, 1, b, 1, pointer, 1, size);
	}
	template<class A, class B>
	NotConstant<A, NotConstant<B>> fillExpr(RealPointer<double> pointer, expression::Add<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		auto *b = doubles.real(expr.b, size);
		vDSP_vaddD(a, 1, b, 1, pointer, 1, size);
	}
	template<class A, class B>
	NotConstant<B> fillExpr(RealPointer<float> pointer, expression::Sub<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		auto *b = floats.real(expr.b, size);
		vDSP_vsub(b, 1, a, 1, pointer, 1, size);
	}
	template<class A, class B>
	NotConstant<B> fillExpr(RealPointer<double> pointer, expression::Sub<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		auto *b = doubles.real(expr.b, size);
		vDSP_vsubD(b, 1, a, 1, pointer, 1, size);
	}
	template<class A, class B>
	NotConstant<A, NotConstant<B>> fillExpr(RealPointer<float> pointer, expression::Mul<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		auto *b = floats.real(expr.b, size);
		vDSP_vmul(a, 1, b, 1, pointer, 1, size);
	}
	template<class A, class B>
	NotConstant<A, NotConstant<B>> fillExpr(RealPointer<double> pointer, expression::Mul<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		auto *b = doubles.real(expr.b, size);
		vDSP_vmulD(a, 1, b, 1, pointer, 1, size);
	}
	template<class A, class B>
	NotConstant<B> fillExpr(RealPointer<float> pointer, expression::Div<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		auto *b = floats.real(expr.b, size);
		int intSize = int(size);
		vvdivf(pointer, a, b, &intSize);
	}
	template<class A, class B>
	NotConstant<B> fillExpr(RealPointer<double> pointer, expression::Div<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		auto *b = doubles.real(expr.b, size);
		int intSize = int(size);
		vvdiv(pointer, a, b, &intSize);
	}
	template<class A, class B>
	NotConstant<A, NotConstant<B>> fillExpr(RealPointer<float> pointer, expression::Max<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		auto *b = floats.real(expr.b, size);
		vDSP_vmax(a, 1, b, 1, pointer, 1, size);
	}
	template<class A, class B>
	NotConstant<A, NotConstant<B>> fillExpr(RealPointer<double> pointer, expression::Max<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		auto *b = doubles.real(expr.b, size);
		vDSP_vmaxD(a, 1, b, 1, pointer, 1, size);
	}
	template<class A, class B>
	NotConstant<A, NotConstant<B>> fillExpr(RealPointer<float> pointer, expression::Min<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		auto *b = floats.real(expr.b, size);
		vDSP_vmin(a, 1, b, 1, pointer, 1, size);
	}
	template<class A, class B>
	NotConstant<A, NotConstant<B>> fillExpr(RealPointer<double> pointer, expression::Min<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		auto *b = doubles.real(expr.b, size);
		vDSP_vminD(a, 1, b, 1, pointer, 1, size);
	}
	template<class A, class B>
	void fillExpr(RealPointer<float> pointer, expression::Hypot<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		auto *b = floats.real(expr.b, size);
		vDSP_vdist(a, 1, b, 1, pointer, 1, size);
	}
	template<class A, class B>
	void fillExpr(RealPointer<double> pointer, expression::Hypot<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		auto *b = doubles.real(expr.b, size);
		vDSP_vdistD(a, 1, b, 1, pointer, 1, size);
	}
	template<class A, class B>
	void fillExpr(RealPointer<float> pointer, expression::Pow<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		auto *b = floats.real(expr.b, size);
		int intSize = int(size);
		vvpowf(pointer, b, a, &intSize); // exponent comes first
	}
	template<class A, class B>
	void fillExpr(RealPointer<double> pointer, expression::Pow<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		auto *b = doubles.real(expr.b, size);
		int intSize = int(size);
		vvpow(pointer, b, a, &intSize); // exponent comes first
	}
	template<class A, class B>
	void fillExpr(RealPointer<float> pointer, expression::Atan2<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		auto *b = floats.real(expr.b, size);
		int intSize = int(size);
		vvatan2f(pointer, a, b, &intSize);
	}
	template<class A, class B>
	void fillExpr(RealPointer<double> pointer, expression::Atan2<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		auto *b = doubles.real(expr.b, size);
		int intSize = int(size);
		vvatan2(pointer, a, b, &intSize);
	}

	//---- Binary real ops with a constant
	template<class A, class V>
	void fillExpr(RealPointer<float> pointer, expression::Add<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		float b = float(expr.b.value);
		vDSP_vsadd(a, 1, &b, pointer, 1, size);
	}
	template<class A, class V>
	void fillExpr(RealPointer<double> pointer, expression::Add<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		double b = double(expr.b.value);
		vDSP_vsaddD(a, 1, &b, pointer, 1, size);
	}
	template<class A, class V>
	void fillExpr(RealPointer<float> pointer, expression::Sub<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		float b = -float(expr.b.value);
		vDSP_vsadd(a, 1, &b, pointer, 1, size);
	}
	template<class A, class V>
	void fillExpr(RealPointer<double> pointer, expression::Sub<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		double b = -double(expr.b.value);
		vDSP_vsaddD(a, 1, &b, pointer, 1, size);
	}
	template<class A, class V>
	void fillExpr(RealPointer<float> pointer, expression::Mul<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		float b = float(expr.b.value);
		vDSP_vsmul(a, 1, &b, pointer, 1, size);
	}
	template<class A, class V>
	void fillExpr(RealPointer<double> pointer, expression::Mul<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		double b = double(expr.b.value);
		vDSP_vsmulD(a, 1, &b, pointer, 1, size);
	}
	template<class A, class V>
	void fillExpr(RealPointer<float> pointer, expression::Div<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		float b = float(expr.b.value);
		vDSP_vsdiv(a, 1, &b, pointer, 1, size);
	}
	template<class A, class V>
	void fillExpr(RealPointer<double> pointer, expression::Div<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		double b = double(expr.b.value);
		vDSP_vsdivD(a, 1, &b, pointer, 1, size);
	}
	template<class A, class V>
	void fillExpr(RealPointer<float> pointer, expression::Max<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		float b = float(expr.b.value);
		vDSP_vthr(a, 1, &b, pointer, 1, size);
	}
	template<class A, class V>
	void fillExpr(RealPointer<double> pointer, expression::Max<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		double b = double(expr.b.value);
		vDSP_vthrD(a, 1, &b, pointer, 1, size);
	}
	template<class A, class V>
	void fillExpr(RealPointer<float> pointer, expression::Min<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		float low = std::numeric_limits<float>::lowest(), b = float(expr.b.value);
		vDSP_vclip(a, 1, &low, &b, pointer, 1, size);
	}
	template<class A, class V>
	void fillExpr(RealPointer<double> pointer, expression::Min<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		double low = std::numeric_limits<double>::lowest(), b = double(expr.b.value);
		vDSP_vclipD(a, 1, &low, &b, pointer, 1, size);
	}
	template<class A, class V>
	void fillExpr(RealPointer<float> pointer, expression::Pow<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		float b = float(expr.b.value);
		int intSize = int(size);
		vvpowsf(pointer, &b, a, &intSize);
	}
	template<class A, class V>
	void fillExpr(RealPointer<double> pointer, expression::Pow<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		double b = double(expr.b.value);
		int intSize = int(size);
		vvpows(pointer, &b, a, &intSize);
	}
	template<class V, class B>
	NotConstant<B> fillExpr(RealPointer<float> pointer, expression::Sub<expression::ConstantExpr<V>, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *b = floats.real(expr.b, size, pointer);
		float a = float(expr.a.value), minusOne = -1;
		vDSP_vsmsa(b, 1, &minusOne, &a, pointer, 1, size);
	}
	template<class V, class B>
	NotConstant<B> fillExpr(RealPointer<double> pointer, expression::Sub<expression::ConstantExpr<V>, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *b = doubles.real(expr.b, size, pointer);
		double a = double(expr.a.value), minusOne = -1;
		vDSP_vsmsaD(b, 1, &minusOne, &a, pointer, 1, size);
	}
	template<class V, class B>
	NotConstant<B> fillExpr(RealPointer<float> pointer, expression::Div<expression::ConstantExpr<V>, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *b = floats.real(expr.b, size, pointer);
		float a = float(expr.a.value);
		vDSP_svdiv(&a, b, 1, pointer, 1, size);
	}
	template<class V, class B>
	NotConstant<B> fillExpr(RealPointer<double> pointer, expression::Div<expression::ConstantExpr<V>, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *b = doubles.real(expr.b, size, pointer);
		double a = double(expr.a.value);
		vDSP_svdivD(&a, b, 1, pointer, 1, size);
	}
	// Clamping (if the limits are the wrong way round, the result is constant)
	template<class A, class V1, class V2>
	void fillExpr(RealPointer<float> pointer, expression::Min<expression::Max<A, expression::ConstantExpr<V1>>, expression::ConstantExpr<V2>> expr, size_t size) {
		float low = float(expr.a.b.value), high = float(expr.b.value);
		if (low > high) return vDSP_vfill(&high, pointer, 1, size);
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a.a, size, pointer);
		vDSP_vclip(a, 1, &low, &high, pointer, 1, size);
	}
	template<class A, class V1, class V2>
	void fillExpr(RealPointer<double> pointer, expression::Min<expression::Max<A, expression::ConstantExpr<V1>>, expression::ConstantExpr<V2>> expr, size_t size) {
		double low = double(expr.a.b.value), high = double(expr.b.value);
		if (low > high) return vDSP_vfillD(&high, pointer, 1, size);
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a.a, size, pointer);
		vDSP_vclipD(a, 1, &low, &high, pointer, 1, size);
	}
	template<class A, class V1, class V2>
	void fillExpr(RealPointer<float> pointer, expression::Max<expression::Min<A, expression::ConstantExpr<V1>>, expression::ConstantExpr<V2>> expr, size_t size) {
		float high = float(expr.a.b.value), low = float(expr.b.value);
		if (low > high) return vDSP_vfill(&low, pointer, 1, size);
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a.a, size, pointer);
		vDSP_vclip(a, 1, &low, &high, pointer, 1, size);
	}
	template<class A, class V1, class V2>
	void fillExpr(RealPointer<double> pointer, expression::Max<expression::Min<A, expression::ConstantExpr<V1>>, expression::ConstantExpr<V2>> expr, size_t size) {
		double high = double(expr.a.b.value), low = double(expr.b.value);
		if (low > high) return vDSP_vfillD(&low, pointer, 1, size);
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a.a, size, pointer);
		vDSP_vclipD(a, 1, &low, &high, pointer, 1, size);
	}
	
	//---- Compound real expressions
	template<class A, class B, class C>
	void fillExpr(RealPointer<float> pointer, expression::Mul<expression::Add<A, B>, C> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a.a, size);
		auto *b = floats.real(expr.a.b, size);
		auto *c = floats.real(expr.b, size);
		vDSP_vam(a, 1, b, 1, c, 1, pointer, 1, size);
	}
	template<class A, class B, class C>
	void fillExpr(RealPointer<double> pointer, expression::Mul<expression::Add<A, B>, C> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a.a, size);
		auto *b = doubles.real(expr.a.b, size);
		auto *c = doubles.real(expr.b, size);
		vDSP_vamD(a, 1, b, 1, c, 1, pointer, 1, size);
	}
	template<class A, class B, class C>
	void fillExpr(RealPointer<float> pointer, expression::Mul<expression::Sub<A, B>, C> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a.a, size);
		auto *b = floats.real(expr.a.b, size);
		auto *c = floats.real(expr.b, size);
		vDSP_vsbm(a, 1, b, 1, c, 1, pointer, 1, size);
	}
	template<class A, class B, class C>
	void fillExpr(RealPointer<double> pointer, expression::Mul<expression::Sub<A, B>, C> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a.a, size);
		auto *b = doubles.real(expr.a.b, size);
		auto *c = doubles.real(expr.b, size);
		vDSP_vsbmD(a, 1, b, 1, c, 1, pointer, 1, size);
	}
	template<class A, class B, class V>
	void fillExpr(RealPointer<float> pointer, expression::Mul<expression::Add<A, B>, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a.a, size);
		auto *b = floats.real(expr.a.b, size);
		float c = float(expr.b.value);
		vDSP_vasm(a, 1, b, 1, &c, pointer, 1, size);
	}
	template<class A, class B, class V>
	void fillExpr(RealPointer<double> pointer, expression::Mul<expression::Add<A, B>, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a.a, size);
		auto *b = doubles.real(expr.a.b, size);
		double c = double(expr.b.value);
		vDSP_vasmD(a, 1, b, 1, &c, pointer, 1, size);
	}
	template<class A, class B, class V>
	void fillExpr(RealPointer<float> pointer, expression::Mul<expression::Sub<A, B>, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a.a, size);
		auto *b = floats.real(expr.a.b, size);
		float c = float(expr.b.value);
		vDSP_vsbsm(a, 1, b, 1, &c, pointer, 1, size);
	}
	template<class A, class B, class V>
	void fillExpr(RealPointer<double> pointer, expression::Mul<expression::Sub<A, B>, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a.a, size);
		auto *b = doubles.real(expr.a.b, size);
		double c = double(expr.b.value);
		vDSP_vsbsmD(a, 1, b, 1, &c, pointer, 1, size);
	}
	template<class A, class B, class C, class D>
	void fillExpr(RealPointer<float> pointer, expression::Mul<expression::Add<A, B>, expression::Add<C, D>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a.a, size);
		auto *b = floats.real(expr.a.b, size);
		auto *c = floats.real(expr.b.a, size);
		auto *d = floats.real(expr.b.b, size);
		vDSP_vaam(a, 1, b, 1, c, 1, d, 1, pointer, 1, size);
	}
	template<class A, class B, class C, class D>
	void fillExpr(RealPointer<double> pointer, expression::Mul<expression::Add<A, B>, expression::Add<C, D>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a.a, size);
		auto *b = doubles.real(expr.a.b, size);
		auto *c = doubles.real(expr.b.a, size);
		auto *d = doubles.real(expr.b.b, size);
		vDSP_vaamD(a, 1, b, 1, c, 1, d, 1, pointer, 1, size);
	}
	template<class A, class B, class C, class D>
	void fillExpr(RealPointer<float> pointer, expression::Mul<expression::Add<A, B>, expression::Sub<C, D>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a.a, size);
		auto *b = floats.real(expr.a.b, size);
		auto *c = floats.real(expr.b.a, size);
		auto *d = floats.real(expr.b.b, size);
		vDSP_vasbm(a, 1, b, 1, c, 1, d, 1, pointer, 1, size);
	}
	template<class A, class B, class C, class D>
	void fillExpr(RealPointer<double> pointer, expression::Mul<expression::Add<A, B>, expression::Sub<C, D>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a.a, size);
		auto *b = doubles.real(expr.a.b, size);
		auto *c = doubles.real(expr.b.a, size);
		auto *d = doubles.real(expr.b.b, size);
		vDSP_vasbmD(a, 1, b, 1, c, 1, d, 1, pointer, 1, size);
	}
	template<class A, class B, class C, class D>
	void fillExpr(RealPointer<float> pointer, expression::Mul<expression::Sub<A, B>, expression::Add<C, D>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.b.a, size);
		auto *b = floats.real(expr.b.b, size);
		auto *c = floats.real(expr.a.a, size);
		auto *d = floats.real(expr.a.b, size);
		vDSP_vasbm(a, 1, b, 1, c, 1, d, 1, pointer, 1, size);
	}
	template<class A, class B, class C, class D>
	void fillExpr(RealPointer<double> pointer, expression::Mul<expression::Sub<A, B>, expression::Add<C, D>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.b.a, size);
		auto *b = doubles.real(expr.b.b, size);
		auto *c = doubles.real(expr.a.a, size);
		auto *d = doubles.real(expr.a.b, size);
		vDSP_vasbmD(a, 1, b, 1, c, 1, d, 1, pointer, 1, size);
	}
	template<class A, class B, class C, class D>
	void fillExpr(RealPointer<float> pointer, expression::Mul<expression::Sub<A, B>, expression::Sub<C, D>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a.a, size);
		auto *b = floats.real(expr.a.b, size);
		auto *c = floats.real(expr.b.a, size);
		auto *d = floats.real(expr.b.b, size);
		vDSP_vsbsbm(a, 1, b, 1, c, 1, d, 1, pointer, 1, size);
	}
	template<class A, class B, class C, class D>
	void fillExpr(RealPointer<double> pointer, expression::Mul<expression::Sub<A, B>, expression::Sub<C, D>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a.a, size);
		auto *b = doubles.real(expr.a.b, size);
		auto *c = doubles.real(expr.b.a, size);
		auto *d = doubles.real(expr.b.b, size);
		vDSP_vsbsbmD(a, 1, b, 1, c, 1, d, 1, pointer, 1, size);
	}
	template<class A, class B, class C>
	void fillExpr(RealPointer<float> pointer, expression::Add<expression::Mul<A, B>, C> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a.a, size);
		auto *b = floats.real(expr.a.b, size);
		auto *c = floats.real(expr.b, size);
		vDSP_vma(a, 1, b, 1, c, 1, pointer, 1, size);
	}
	template<class A, class B, class C>
	void fillExpr(RealPointer<double> pointer, expression::Add<expression::Mul<A, B>, C> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a.a, size);
		auto *b = doubles.real(expr.a.b, size);
		auto *c = doubles.real(expr.b, size);
		vDSP_vmaD(a, 1, b, 1, c, 1, pointer, 1, size);
	}
	template<class A, class V, class C>
	void fillExpr(RealPointer<float> pointer, expression::Add<expression::Mul<A, expression::ConstantExpr<V>>, C> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a.a, size);
		float b = float(expr.a.b.value);
		auto *c = floats.real(expr.b, size);
		vDSP_vsma(a, 1, &b, c, 1, pointer, 1, size);
	}
	template<class A, class V, class C>
	void fillExpr(RealPointer<double> pointer, expression::Add<expression::Mul<A, expression::ConstantExpr<V>>, C> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a.a, size);
		double b = double(expr.a.b.value);
		auto *c = doubles.real(expr.b, size);
		vDSP_vsmaD(a, 1, &b, c, 1, pointer, 1, size);
	}
	template<class A, class B, class V>
	void fillExpr(RealPointer<float> pointer, expression::Add<expression::Mul<A, B>, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a.a, size);
		auto *b = floats.real(expr.a.b, size);
		float c = float(expr.b.value);
		vDSP_vmsa(a, 1, b, 1, &c, pointer, 1, size);
	}
	template<class A, class B, class V>
	void fillExpr(RealPointer<double> pointer, expression::Add<expression::Mul<A, B>, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a.a, size);
		auto *b = doubles.real(expr.a.b, size);
		double c = double(expr.b.value);
		vDSP_vmsaD(a, 1, b, 1, &c, pointer, 1, size);
	}
	template<class A, class V1, class V2>
	void fillExpr(RealPointer<float> pointer, expression::Add<expression::Mul<A, expression::ConstantExpr<V1>>, expression::ConstantExpr<V2>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a.a, size);
		float b = float(expr.a.b.value);
		float c = float(expr.b.value);
		vDSP_vsmsa(a, 1, &b, &c, pointer, 1, size);
	}
	template<class A, class V1, class V2>
	void fillExpr(RealPointer<double> pointer, expression::Add<expression::Mul<A, expression::ConstantExpr<V1>>, expression::ConstantExpr<V2>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a.a, size);
		double b = double(expr.a.b.value);
		double c = double(expr.b.value);
		vDSP_vsmsaD(a, 1, &b, &c, pointer, 1, size);
	}
	template<class A, class B, class C, class D>
	void fillExpr(RealPointer<float> pointer, expression::Add<expression::Mul<A, B>, expression::Mul<C, D>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a.a, size);
		auto *b = floats.real(expr.a.b, size);
		auto *c = floats.real(expr.b.a, size);
		auto *d = floats.real(expr.b.b, size);
		vDSP_vmma(a, 1, b, 1, c, 1, d, 1, pointer, 1, size);
	}
	template<class A, class B, class C, class D>
	void fillExpr(RealPointer<double> pointer, expression::Add<expression::Mul<A, B>, expression::Mul<C, D>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a.a, size);
		auto *b = doubles.real(expr.a.b, size);
		auto *c = doubles.real(expr.b.a, size);
		auto *d = doubles.real(expr.b.b, size);
		vDSP_vmmaD(a, 1, b, 1, c, 1, d, 1, pointer, 1, size);
	}
	template<class A, class V, class C, class D>
	void fillExpr(RealPointer<float> pointer, expression::Add<expression::Mul<A, expression::ConstantExpr<V>>, expression::Mul<C, D>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a.a, size);
		float b = float(expr.a.b.value);
		auto *c = floats.real(expr.b, size);
		vDSP_vsma(a, 1, &b, c, 1, pointer, 1, size);
	}
	template<class A, class V, class C, class D>
	void fillExpr(RealPointer<double> pointer, expression::Add<expression::Mul<A, expression::ConstantExpr<V>>, expression::Mul<C, D>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a.a, size);
		double b = double(expr.a.b.value);
		auto *c = doubles.real(expr.b, size);
		vDSP_vsmaD(a, 1, &b, c, 1, pointer, 1, size);
	}
	template<class A, class B, class C, class V>
	void fillExpr(RealPointer<float> pointer, expression::Add<expression::Mul<A, B>, expression::Mul<C, expression::ConstantExpr<V>>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *c = floats.real(expr.a, size);
		auto *a = floats.real(expr.b.a, size);
		float b = float(expr.b.b.value);
		vDSP_vsma(a, 1, &b, c, 1, pointer, 1, size);
	}
	template<class A, class B, class C, class V>
	void fillExpr(RealPointer<double> pointer, expression::Add<expression::Mul<A, B>, expression::Mul<C, expression::ConstantExpr<V>>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *c = doubles.real(expr.a, size);
		auto *a = doubles.real(expr.b.a, size);
		double b = double(expr.b.b.value);
		vDSP_vsmaD(a, 1, &b, c, 1, pointer, 1, size);
	}
	template<class A, class V1, class C, class V2>
	void fillExpr(RealPointer<float> pointer, expression::Add<expression::Mul<A, expression::ConstantExpr<V1>>, expression::Mul<C, expression::ConstantExpr<V2>>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a.a, size);
		float b = float(expr.a.b.value);
		auto *c = floats.real(expr.b.a, size);
		float d = float(expr.b.b.value);
		vDSP_vsmsma(a, 1, &b, c, 1, &d, pointer, 1, size);
	}
	template<class A, class V1, class C, class V2>
	void fillExpr(RealPointer<double> pointer, expression::Add<expression::Mul<A, expression::ConstantExpr<V1>>, expression::Mul<C, expression::ConstantExpr<V2>>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a.a, size);
		double b = double(expr.a.b.value);
		auto *c = doubles.real(expr.b.a, size);
		double d = double(expr.b.b.value);
		vDSP_vsmsmaD(a, 1, &b, c, 1, &d, pointer, 1, size);
	}
	template<class A, class B, class C>
	void fillExpr(RealPointer<float> pointer, expression::Sub<expression::Mul<A, B>, C> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a.a, size);
		auto *b = floats.real(expr.a.b, size);
		auto *c = floats.real(expr.b, size);
		vDSP_vmsb(a, 1, b, 1, c, 1, pointer, 1, size);
	}
	template<class A, class B, class C>
	void fillExpr(RealPointer<double> pointer, expression::Sub<expression::Mul<A, B>, C> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a.a, size);
		auto *b = doubles.real(expr.a.b, size);
		auto *c = doubles.real(expr.b, size);
		vDSP_vmsbD(a, 1, b, 1, c, 1, pointer, 1, size);
	}
	template<class A, class V, class C>
	void fillExpr(RealPointer<float> pointer, expression::Sub<expression::Mul<A, expression::ConstantExpr<V>>, C> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a.a, size);
		float b = float(expr.a.b.value);
		auto *c = floats.real(expr.b, size);
		vDSP_vsmsb(a, 1, &b, c, 1, pointer, 1, size);
	}
	template<class A, class V, class C>
	void fillExpr(RealPointer<double> pointer, expression::Sub<expression::Mul<A, expression::ConstantExpr<V>>, C> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a.a, size);
		double b = double(expr.a.b.value);
		auto *c = doubles.real(expr.b, size);
		vDSP_vsmsbD(a, 1, &b, c, 1, pointer, 1, size);
	}
	template<class A, class B, class V>
	void fillExpr(RealPointer<float> pointer, expression::Sub<expression::Mul<A, B>, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a.a, size);
		auto *b = floats.real(expr.a.b, size);
		float c = -float(expr.b.value);
		vDSP_vmsa(a, 1, b, 1, &c, pointer, 1, size);
	}
	template<class A, class B, class V>
	void fillExpr(RealPointer<double> pointer, expression::Sub<expression::Mul<A, B>, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a.a, size);
		auto *b = doubles.real(expr.a.b, size);
		double c = -double(expr.b.value);
		vDSP_vmsaD(a, 1, b, 1, &c, pointer, 1, size);
	}
	template<class A, class V1, class V2>
	void fillExpr(RealPointer<float> pointer, expression::Sub<expression::Mul<A, expression::ConstantExpr<V1>>, expression::ConstantExpr<V2>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a.a, size);
		float b = float(expr.a.b.value);
		float c = -float(expr.b.value);
		vDSP_vsmsa(a, 1, &b, &c, pointer, 1, size);
	}
	template<class A, class V1, class V2>
	void fillExpr(RealPointer<double> pointer, expression::Sub<expression::Mul<A, expression::ConstantExpr<V1>>, expression::ConstantExpr<V2>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a.a, size);
		double b = double(expr.a.b.value);
		double c = -double(expr.b.value);
		vDSP_vsmsaD(a, 1, &b, &c, pointer, 1, size);
	}
	template<class A, class B, class C, class D>
	void fillExpr(RealPointer<float> pointer, expression::Sub<expression::Mul<A, B>, expression::Mul<C, D>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a.a, size);
		auto *b = floats.real(expr.a.b, size);
		auto *c = floats.real(expr.b.a, size);
		auto *d = floats.real(expr.b.b, size);
		vDSP_vmmsb(a, 1, b, 1, c, 1, d, 1, pointer, 1, size);
	}
	template<class A, class B, class C, class D>
	void fillExpr(RealPointer<double> pointer, expression::Sub<expression::Mul<A, B>, expression::Mul<C, D>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a.a, size);
		auto *b = doubles.real(expr.a.b, size);
		auto *c = doubles.real(expr.b.a, size);
		auto *d = doubles.real(expr.b.b, size);
		vDSP_vmmsbD(a, 1, b, 1, c, 1, d, 1, pointer, 1, size);
	}
	template<class A, class V, class C, class D>
	void fillExpr(RealPointer<float> pointer, expression::Sub<expression::Mul<A, expression::ConstantExpr<V>>, expression::Mul<C, D>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a.a, size);
		float b = float(expr.a.b.value);
		auto *c = floats.real(expr.b, size);
		vDSP_vsmsb(a, 1, &b, c, 1, pointer, 1, size);
	}
	template<class A, class V, class C, class D>
	void fillExpr(RealPointer<double> pointer, expression::Sub<expression::Mul<A, expression::ConstantExpr<V>>, expression::Mul<C, D>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a.a, size);
		double b = double(expr.a.b.value);
		auto *c = doubles.real(expr.b, size);
		vDSP_vsmsbD(a, 1, &b, c, 1, pointer, 1, size);
	}
	template<class A, class V1, class C, class V2>
	void fillExpr(RealPointer<float> pointer, expression::Sub<expression::Mul<A, expression::ConstantExpr<V1>>, expression::Mul<C, expression::ConstantExpr<V2>>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a.a, size);
		float b = float(expr.a.b.value);
		auto *c = floats.real(expr.b.a, size);
		float d = -float(expr.b.b.value);
		vDSP_vsmsma(a, 1, &b, c, 1, &d, pointer, 1, size);
	}
	template<class A, class V1, class C, class V2>
	void fillExpr(RealPointer<double> pointer, expression::Sub<expression::Mul<A, expression::ConstantExpr<V1>>, expression::Mul<C, expression::ConstantExpr<V2>>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a.a, size);
		double b = double(expr.a.b.value);
		auto *c = doubles.real(expr.b.a, size);
		double d = -double(expr.b.b.value);
		vDSP_vsmsmaD(a, 1, &b, c, 1, &d, pointer, 1, size);
	}

	// Subtracting a scaled value (as a multiply-add with the scale negated)
	template<class V1, class A, class V2>
	void fillExpr(RealPointer<float> pointer, expression::Sub<expression::ConstantExpr<V1>, expression::Mul<A, expression::ConstantExpr<V2>>> expr, size_t size) {
		auto floats = cached.floatScope();
		float c = float(expr.a.value);
		auto *a = floats.real(expr.b.a, size);
		float b = -float(expr.b.b.value);
		vDSP_vsmsa(a, 1, &b, &c, pointer, 1, size);
	}
	template<class V1, class A, class V2>
	void fillExpr(RealPointer<double> pointer, expression::Sub<expression::ConstantExpr<V1>, expression::Mul<A, expression::ConstantExpr<V2>>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		double c = double(expr.a.value);
		auto *a = doubles.real(expr.b.a, size);
		double b = -double(expr.b.b.value);
		vDSP_vsmsaD(a, 1, &b, &c, pointer, 1, size);
	}
	template<class C, class A, class V>
	NotConstant<C> fillExpr(RealPointer<float> pointer, expression::Sub<C, expression::Mul<A, expression::ConstantExpr<V>>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *c = floats.real(expr.a, size);
		auto *a = floats.real(expr.b.a, size);
		float b = -float(expr.b.b.value);
		vDSP_vsma(a, 1, &b, c, 1, pointer, 1, size);
	}
	template<class C, class A, class V>
	NotConstant<C> fillExpr(RealPointer<double> pointer, expression::Sub<C, expression::Mul<A, expression::ConstantExpr<V>>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *c = doubles.real(expr.a, size);
		auto *a = doubles.real(expr.b.a, size);
		double b = -double(expr.b.b.value);
		vDSP_vsmaD(a, 1, &b, c, 1, pointer, 1, size);
	}
	template<class A, class B, class C, class V>
	void fillExpr(RealPointer<float> pointer, expression::Sub<expression::Mul<A, B>, expression::Mul<C, expression::ConstantExpr<V>>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *c = floats.real(expr.a, size);
		auto *a = floats.real(expr.b.a, size);
		float b = -float(expr.b.b.value);
		vDSP_vsma(a, 1, &b, c, 1, pointer, 1, size);
	}
	template<class A, class B, class C, class V>
	void fillExpr(RealPointer<double> pointer, expression::Sub<expression::Mul<A, B>, expression::Mul<C, expression::ConstantExpr<V>>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *c = doubles.real(expr.a, size);
		auto *a = doubles.real(expr.b.a, size);
		double b = -double(expr.b.b.value);
		vDSP_vsmaD(a, 1, &b, c, 1, pointer, 1, size);
	}


	//---- Complex/split copies and conversions
	void fillExpr(SplitPointer<float> pointer, expression::ReadableSplit<float> expr, size_t size) {
		std::memcpy(pointer.real, expr.pointer.real, size*sizeof(float));
		std::memcpy(pointer.imag, expr.pointer.imag, size*sizeof(float));
	}
	void fillExpr(SplitPointer<float> pointer, WritableSplit<float> expr, size_t size) {
		std::memcpy(pointer.real, expr.pointer.real, size*sizeof(float));
		std::memcpy(pointer.imag, expr.pointer.imag, size*sizeof(float));
	}
	void fillExpr(SplitPointer<double> pointer, expression::ReadableSplit<double> expr, size_t size) {
		std::memcpy(pointer.real, expr.pointer.real, size*sizeof(double));
		std::memcpy(pointer.imag, expr.pointer.imag, size*sizeof(double));
	}
	void fillExpr(SplitPointer<double> pointer, WritableSplit<double> expr, size_t size) {
		std::memcpy(pointer.real, expr.pointer.real, size*sizeof(double));
		std::memcpy(pointer.imag, expr.pointer.imag, size*sizeof(double));
	}
	void fillExpr(SplitPointer<float> pointer, expression::ReadableComplex<float> expr, size_t size) {
		auto p = dspSplit(pointer);
		vDSP_ctoz(dspComplex(expr.pointer), 2, &p, 1, size);
	}
	void fillExpr(SplitPointer<double> pointer, expression::ReadableComplex<double> expr, size_t size) {
		auto p = dspSplit(pointer);
		vDSP_ctozD(dspComplex(expr.pointer), 2, &p, 1, size);
	}
	void fillExpr(SplitPointer<float> pointer, WritableComplex<float> expr, size_t size) {
		auto p = dspSplit(pointer);
		vDSP_ctoz(dspComplex(expr.pointer), 2, &p, 1, size);
	}
	void fillExpr(SplitPointer<double> pointer, WritableComplex<double> expr, size_t size) {
		auto p = dspSplit(pointer);
		vDSP_ctozD(dspComplex(expr.pointer), 2, &p, 1, size);
	}
	void fillExpr(ComplexPointer<float> pointer, expression::ReadableSplit<float> expr, size_t size) {
		auto a = dspSplit(expr.pointer);
		vDSP_ztoc(&a, 1, dspComplex(pointer), 2, size);
	}
	void fillExpr(ComplexPointer<double> pointer, expression::ReadableSplit<double> expr, size_t size) {
		auto a = dspSplit(expr.pointer);
		vDSP_ztocD(&a, 1, dspComplex(pointer), 2, size);
	}
	void fillExpr(ComplexPointer<float> pointer, WritableSplit<float> expr, size_t size) {
		auto a = dspSplit(expr.pointer);
		vDSP_ztoc(&a, 1, dspComplex(pointer), 2, size);
	}
	void fillExpr(ComplexPointer<double> pointer, WritableSplit<double> expr, size_t size) {
		auto a = dspSplit(expr.pointer);
		vDSP_ztocD(&a, 1, dspComplex(pointer), 2, size);
	}

	//---- Complex unary ops
	template<class A>
	ItemTypeCF<A> fillExpr(ComplexPointer<float> pointer, expression::Conj<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.complex(expr.a, size, pointer));
		auto p = dspSplit(pointer);
		vDSP_zvconj(&a, 2, &p, 2, size);
	}
	template<class A>
	ItemTypeCD<A> fillExpr(ComplexPointer<double> pointer, expression::Conj<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.complex(expr.a, size, pointer));
		auto p = dspSplit(pointer);
		vDSP_zvconjD(&a, 2, &p, 2, size);
	}
	template<class A>
	ItemTypeCF<A> fillExpr(SplitPointer<float> pointer, expression::Conj<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a, size, pointer));
		auto p = dspSplit(pointer);
		vDSP_zvconj(&a, 1, &p, 1, size);
	}
	template<class A>
	ItemTypeCD<A> fillExpr(SplitPointer<double> pointer, expression::Conj<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a, size, pointer));
		auto p = dspSplit(pointer);
		vDSP_zvconjD(&a, 1, &p, 1, size);
	}
	// exp(x + iy) = e^x (cos(y) + i sin(y))
	template<class A>
	ItemTypeCF<A> fillExpr(ComplexPointer<float> pointer, expression::Exp<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = floats.split(expr.a, size);
		auto e = floats(size), sin = floats(size), cos = floats(size);
		int intSize = int(size);
		vvexpf(e, a.real, &intSize);
		vvsincosf(sin, cos, a.imag, &intSize);
		auto p = dspSplit(pointer);
		vDSP_vmul(e, 1, cos, 1, p.realp, 2, size);
		vDSP_vmul(e, 1, sin, 1, p.imagp, 2, size);
	}
	// log(z) = log(abs(z)) + i arg(z)
	template<class A>
	ItemTypeCF<A> fillExpr(ComplexPointer<float> pointer, expression::Log<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a, size));
		auto abs = floats(size), arg = floats(size);
		vDSP_zvabs(&a, 1, abs, 1, size);
		vDSP_zvphas(&a, 1, arg, 1, size);
		auto p = dspSplit(pointer);
		int intSize = int(size);
		vvlogf(abs, abs, &intSize);
		vDSP_mmov(abs, p.realp, 1, size, 1, 2);
		vDSP_mmov(arg, p.imagp, 1, size, 1, 2);
	}
	// log10(z) = log(z)/log(10)
	template<class A>
	ItemTypeCF<A> fillExpr(ComplexPointer<float> pointer, expression::Log10<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a, size));
		auto abs = floats(size), arg = floats(size);
		vDSP_zvabs(&a, 1, abs, 1, size);
		vDSP_zvphas(&a, 1, arg, 1, size);
		auto p = dspSplit(pointer);
		int intSize = int(size);
		vvlog10f(abs, abs, &intSize);
		vDSP_mmov(abs, p.realp, 1, size, 1, 2);
		float scale = float(1/std::log(10.0));
		vDSP_vsmul(arg, 1, &scale, p.imagp, 2, size);
	}
	// exp(x + iy) = e^x (cos(y) + i sin(y))
	template<class A>
	ItemTypeCD<A> fillExpr(ComplexPointer<double> pointer, expression::Exp<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = doubles.split(expr.a, size);
		auto e = doubles(size), sin = doubles(size), cos = doubles(size);
		int intSize = int(size);
		vvexp(e, a.real, &intSize);
		vvsincos(sin, cos, a.imag, &intSize);
		auto p = dspSplit(pointer);
		vDSP_vmulD(e, 1, cos, 1, p.realp, 2, size);
		vDSP_vmulD(e, 1, sin, 1, p.imagp, 2, size);
	}
	// log(z) = log(abs(z)) + i arg(z)
	template<class A>
	ItemTypeCD<A> fillExpr(ComplexPointer<double> pointer, expression::Log<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a, size));
		auto abs = doubles(size), arg = doubles(size);
		vDSP_zvabsD(&a, 1, abs, 1, size);
		vDSP_zvphasD(&a, 1, arg, 1, size);
		auto p = dspSplit(pointer);
		int intSize = int(size);
		vvlog(abs, abs, &intSize);
		vDSP_mmovD(abs, p.realp, 1, size, 1, 2);
		vDSP_mmovD(arg, p.imagp, 1, size, 1, 2);
	}
	// log10(z) = log(z)/log(10)
	template<class A>
	ItemTypeCD<A> fillExpr(ComplexPointer<double> pointer, expression::Log10<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a, size));
		auto abs = doubles(size), arg = doubles(size);
		vDSP_zvabsD(&a, 1, abs, 1, size);
		vDSP_zvphasD(&a, 1, arg, 1, size);
		auto p = dspSplit(pointer);
		int intSize = int(size);
		vvlog10(abs, abs, &intSize);
		vDSP_mmovD(abs, p.realp, 1, size, 1, 2);
		double scale = double(1/std::log(10.0));
		vDSP_vsmulD(arg, 1, &scale, p.imagp, 2, size);
	}
	// exp(x + iy) = e^x (cos(y) + i sin(y))
	template<class A>
	ItemTypeCF<A> fillExpr(SplitPointer<float> pointer, expression::Exp<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = floats.split(expr.a, size);
		auto e = floats(size), sin = floats(size), cos = floats(size);
		int intSize = int(size);
		vvexpf(e, a.real, &intSize);
		vvsincosf(sin, cos, a.imag, &intSize);
		auto p = dspSplit(pointer);
		vDSP_vmul(e, 1, cos, 1, p.realp, 1, size);
		vDSP_vmul(e, 1, sin, 1, p.imagp, 1, size);
	}
	// log(z) = log(abs(z)) + i arg(z)
	template<class A>
	ItemTypeCF<A> fillExpr(SplitPointer<float> pointer, expression::Log<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a, size));
		auto abs = floats(size), arg = floats(size);
		vDSP_zvabs(&a, 1, abs, 1, size);
		vDSP_zvphas(&a, 1, arg, 1, size);
		auto p = dspSplit(pointer);
		int intSize = int(size);
		vvlogf(abs, abs, &intSize);
		vDSP_mmov(abs, p.realp, 1, size, 1, 1);
		vDSP_mmov(arg, p.imagp, 1, size, 1, 1);
	}
	// log10(z) = log(z)/log(10)
	template<class A>
	ItemTypeCF<A> fillExpr(SplitPointer<float> pointer, expression::Log10<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a, size));
		auto abs = floats(size), arg = floats(size);
		vDSP_zvabs(&a, 1, abs, 1, size);
		vDSP_zvphas(&a, 1, arg, 1, size);
		auto p = dspSplit(pointer);
		int intSize = int(size);
		vvlog10f(abs, abs, &intSize);
		vDSP_mmov(abs, p.realp, 1, size, 1, 1);
		float scale = float(1/std::log(10.0));
		vDSP_vsmul(arg, 1, &scale, p.imagp, 1, size);
	}
	// exp(x + iy) = e^x (cos(y) + i sin(y))
	template<class A>
	ItemTypeCD<A> fillExpr(SplitPointer<double> pointer, expression::Exp<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = doubles.split(expr.a, size);
		auto e = doubles(size), sin = doubles(size), cos = doubles(size);
		int intSize = int(size);
		vvexp(e, a.real, &intSize);
		vvsincos(sin, cos, a.imag, &intSize);
		auto p = dspSplit(pointer);
		vDSP_vmulD(e, 1, cos, 1, p.realp, 1, size);
		vDSP_vmulD(e, 1, sin, 1, p.imagp, 1, size);
	}
	// log(z) = log(abs(z)) + i arg(z)
	template<class A>
	ItemTypeCD<A> fillExpr(SplitPointer<double> pointer, expression::Log<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a, size));
		auto abs = doubles(size), arg = doubles(size);
		vDSP_zvabsD(&a, 1, abs, 1, size);
		vDSP_zvphasD(&a, 1, arg, 1, size);
		auto p = dspSplit(pointer);
		int intSize = int(size);
		vvlog(abs, abs, &intSize);
		vDSP_mmovD(abs, p.realp, 1, size, 1, 1);
		vDSP_mmovD(arg, p.imagp, 1, size, 1, 1);
	}
	// log10(z) = log(z)/log(10)
	template<class A>
	ItemTypeCD<A> fillExpr(SplitPointer<double> pointer, expression::Log10<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a, size));
		auto abs = doubles(size), arg = doubles(size);
		vDSP_zvabsD(&a, 1, abs, 1, size);
		vDSP_zvphasD(&a, 1, arg, 1, size);
		auto p = dspSplit(pointer);
		int intSize = int(size);
		vvlog10(abs, abs, &intSize);
		vDSP_mmovD(abs, p.realp, 1, size, 1, 1);
		double scale = double(1/std::log(10.0));
		vDSP_vsmulD(arg, 1, &scale, p.imagp, 1, size);
	}
	// polar(abs, arg)
	template<class A, class B>
	void fillExpr(SplitPointer<float> pointer, expression::Polar<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		auto *b = floats.real(expr.b, size);
		auto sin = floats(size), cos = floats(size);
		int intSize = int(size);
		vvsincosf(sin, cos, b, &intSize);
		vDSP_vmul(a, 1, cos, 1, pointer.real, 1, size);
		vDSP_vmul(a, 1, sin, 1, pointer.imag, 1, size);
	}
	template<class A, class B>
	void fillExpr(ComplexPointer<float> pointer, expression::Polar<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		auto *b = floats.real(expr.b, size);
		auto sin = floats(size), cos = floats(size);
		int intSize = int(size);
		vvsincosf(sin, cos, b, &intSize);
		float *p = (float *)pointer;
		vDSP_vmul(a, 1, cos, 1, p, 2, size);
		vDSP_vmul(a, 1, sin, 1, p + 1, 2, size);
	}
	template<class A, class B>
	void fillExpr(SplitPointer<double> pointer, expression::Polar<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		auto *b = doubles.real(expr.b, size);
		auto sin = doubles(size), cos = doubles(size);
		int intSize = int(size);
		vvsincos(sin, cos, b, &intSize);
		vDSP_vmulD(a, 1, cos, 1, pointer.real, 1, size);
		vDSP_vmulD(a, 1, sin, 1, pointer.imag, 1, size);
	}
	template<class A, class B>
	void fillExpr(ComplexPointer<double> pointer, expression::Polar<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		auto *b = doubles.real(expr.b, size);
		auto sin = doubles(size), cos = doubles(size);
		int intSize = int(size);
		vvsincos(sin, cos, b, &intSize);
		double *p = (double *)pointer;
		vDSP_vmulD(a, 1, cos, 1, p, 2, size);
		vDSP_vmulD(a, 1, sin, 1, p + 1, 2, size);
	}

	//---- Complex binary ops
	template<class A, class B>
	ItemTypeCF<NotConstant<A, A>, ItemTypeCF<B>> fillExpr(ComplexPointer<float> pointer, expression::Add<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.complex(expr.a, size));
		auto b = dspSplit(floats.complex(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvadd(&a, 2, &b, 2, &p, 2, size);
	}
	template<class A, class B>
	ItemTypeCD<NotConstant<A, A>, ItemTypeCD<B>> fillExpr(ComplexPointer<double> pointer, expression::Add<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.complex(expr.a, size));
		auto b = dspSplit(doubles.complex(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvaddD(&a, 2, &b, 2, &p, 2, size);
	}
	template<class A, class B>
	ItemTypeCF<NotConstant<A, A>, ItemTypeCF<B>> fillExpr(SplitPointer<float> pointer, expression::Add<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a, size));
		auto b = dspSplit(floats.split(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvadd(&a, 1, &b, 1, &p, 1, size);
	}
	template<class A, class B>
	ItemTypeCD<NotConstant<A, A>, ItemTypeCD<B>> fillExpr(SplitPointer<double> pointer, expression::Add<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a, size));
		auto b = dspSplit(doubles.split(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvaddD(&a, 1, &b, 1, &p, 1, size);
	}
	template<class A, class B>
	ItemTypeCF<NotConstant<A, A>, ItemTypeF<B>> fillExpr(ComplexPointer<float> pointer, expression::Add<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.complex(expr.a, size));
		auto *b = floats.real(expr.b, size);
		auto p = dspSplit(pointer);
		vDSP_zrvadd(&a, 2, b, 1, &p, 2, size);
	}
	template<class A, class B>
	ItemTypeCD<NotConstant<A, A>, ItemTypeD<B>> fillExpr(ComplexPointer<double> pointer, expression::Add<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.complex(expr.a, size));
		auto *b = doubles.real(expr.b, size);
		auto p = dspSplit(pointer);
		vDSP_zrvaddD(&a, 2, b, 1, &p, 2, size);
	}
	template<class A, class B>
	ItemTypeCF<NotConstant<A, A>, ItemTypeF<B>> fillExpr(SplitPointer<float> pointer, expression::Add<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a, size));
		auto *b = floats.real(expr.b, size);
		auto p = dspSplit(pointer);
		vDSP_zrvadd(&a, 1, b, 1, &p, 1, size);
	}
	template<class A, class B>
	ItemTypeCD<NotConstant<A, A>, ItemTypeD<B>> fillExpr(SplitPointer<double> pointer, expression::Add<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a, size));
		auto *b = doubles.real(expr.b, size);
		auto p = dspSplit(pointer);
		vDSP_zrvaddD(&a, 1, b, 1, &p, 1, size);
	}
	template<class A, class B>
	ItemTypeF<NotConstant<A, A>, ItemTypeCF<B>> fillExpr(ComplexPointer<float> pointer, expression::Add<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		auto b = dspSplit(floats.complex(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zrvadd(&b, 2, a, 1, &p, 2, size);
	}
	template<class A, class B>
	ItemTypeD<NotConstant<A, A>, ItemTypeCD<B>> fillExpr(ComplexPointer<double> pointer, expression::Add<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		auto b = dspSplit(doubles.complex(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zrvaddD(&b, 2, a, 1, &p, 2, size);
	}
	template<class A, class B>
	ItemTypeF<NotConstant<A, A>, ItemTypeCF<B>> fillExpr(SplitPointer<float> pointer, expression::Add<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		auto b = dspSplit(floats.split(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zrvadd(&b, 1, a, 1, &p, 1, size);
	}
	template<class A, class B>
	ItemTypeD<NotConstant<A, A>, ItemTypeCD<B>> fillExpr(SplitPointer<double> pointer, expression::Add<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		auto b = dspSplit(doubles.split(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zrvaddD(&b, 1, a, 1, &p, 1, size);
	}
	template<class A, class B>
	ItemTypeCF<NotConstant<A, A>, ItemTypeCF<B>> fillExpr(ComplexPointer<float> pointer, expression::Sub<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.complex(expr.a, size));
		auto b = dspSplit(floats.complex(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvsub(&a, 2, &b, 2, &p, 2, size);
	}
	template<class A, class B>
	ItemTypeCD<A, ItemTypeCD<B>> fillExpr(ComplexPointer<double> pointer, expression::Sub<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.complex(expr.a, size));
		auto b = dspSplit(doubles.complex(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvsubD(&a, 2, &b, 2, &p, 2, size);
	}
	template<class A, class B>
	ItemTypeCF<A, ItemTypeCF<B>> fillExpr(SplitPointer<float> pointer, expression::Sub<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a, size));
		auto b = dspSplit(floats.split(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvsub(&a, 1, &b, 1, &p, 1, size);
	}
	template<class A, class B>
	ItemTypeCD<A, ItemTypeCD<B>> fillExpr(SplitPointer<double> pointer, expression::Sub<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a, size));
		auto b = dspSplit(doubles.split(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvsubD(&a, 1, &b, 1, &p, 1, size);
	}
	template<class A, class B>
	ItemTypeCF<A, ItemTypeF<B>> fillExpr(ComplexPointer<float> pointer, expression::Sub<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.complex(expr.a, size));
		auto *b = floats.real(expr.b, size);
		auto p = dspSplit(pointer);
		vDSP_zrvsub(&a, 2, b, 1, &p, 2, size);
	}
	template<class A, class B>
	ItemTypeCD<A, ItemTypeD<B>> fillExpr(ComplexPointer<double> pointer, expression::Sub<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.complex(expr.a, size));
		auto *b = doubles.real(expr.b, size);
		auto p = dspSplit(pointer);
		vDSP_zrvsubD(&a, 2, b, 1, &p, 2, size);
	}
	template<class A, class B>
	ItemTypeCF<A, ItemTypeF<B>> fillExpr(SplitPointer<float> pointer, expression::Sub<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a, size));
		auto *b = floats.real(expr.b, size);
		auto p = dspSplit(pointer);
		vDSP_zrvsub(&a, 1, b, 1, &p, 1, size);
	}
	template<class A, class B>
	ItemTypeCD<A, ItemTypeD<B>> fillExpr(SplitPointer<double> pointer, expression::Sub<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a, size));
		auto *b = doubles.real(expr.b, size);
		auto p = dspSplit(pointer);
		vDSP_zrvsubD(&a, 1, b, 1, &p, 1, size);
	}
	template<class A, class B>
	ItemTypeF<A, ItemTypeCF<B>> fillExpr(ComplexPointer<float> pointer, expression::Sub<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		auto b = dspSplit(floats.complex(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zrvsub(&b, 2, a, 1, &p, 2, size); // B - A
		vDSP_zvneg(&p, 2, &p, 2, size);
	}
	template<class A, class B>
	ItemTypeD<A, ItemTypeCD<B>> fillExpr(ComplexPointer<double> pointer, expression::Sub<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		auto b = dspSplit(doubles.complex(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zrvsubD(&b, 2, a, 1, &p, 2, size); // B - A
		vDSP_zvnegD(&p, 2, &p, 2, size);
	}
	template<class A, class B>
	ItemTypeF<A, ItemTypeCF<B>> fillExpr(SplitPointer<float> pointer, expression::Sub<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		auto b = dspSplit(floats.split(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zrvsub(&b, 1, a, 1, &p, 1, size); // B - A
		vDSP_zvneg(&p, 1, &p, 1, size);
	}
	template<class A, class B>
	ItemTypeD<A, ItemTypeCD<B>> fillExpr(SplitPointer<double> pointer, expression::Sub<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		auto b = dspSplit(doubles.split(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zrvsubD(&b, 1, a, 1, &p, 1, size); // B - A
		vDSP_zvnegD(&p, 1, &p, 1, size);
	}
	template<class A, class B>
	ItemTypeCF<A, ItemTypeCF<B>> fillExpr(ComplexPointer<float> pointer, expression::Mul<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.complex(expr.a, size));
		auto b = dspSplit(floats.complex(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvmul(&a, 2, &b, 2, &p, 2, size, 1);
	}
	template<class A, class B>
	ItemTypeCD<A, ItemTypeCD<B>> fillExpr(ComplexPointer<double> pointer, expression::Mul<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.complex(expr.a, size));
		auto b = dspSplit(doubles.complex(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvmulD(&a, 2, &b, 2, &p, 2, size, 1);
	}
	template<class A, class B>
	ItemTypeCF<A, ItemTypeCF<B>> fillExpr(SplitPointer<float> pointer, expression::Mul<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a, size));
		auto b = dspSplit(floats.split(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvmul(&a, 1, &b, 1, &p, 1, size, 1);
	}
	template<class A, class B>
	ItemTypeCD<A, ItemTypeCD<B>> fillExpr(SplitPointer<double> pointer, expression::Mul<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a, size));
		auto b = dspSplit(doubles.split(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvmulD(&a, 1, &b, 1, &p, 1, size, 1);
	}
	template<class A, class B>
	ItemTypeCF<A, ItemTypeF<B>> fillExpr(ComplexPointer<float> pointer, expression::Mul<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.complex(expr.a, size));
		auto *b = floats.real(expr.b, size);
		auto p = dspSplit(pointer);
		vDSP_zrvmul(&a, 2, b, 1, &p, 2, size);
	}
	template<class A, class B>
	ItemTypeCD<A, ItemTypeD<B>> fillExpr(ComplexPointer<double> pointer, expression::Mul<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.complex(expr.a, size));
		auto *b = doubles.real(expr.b, size);
		auto p = dspSplit(pointer);
		vDSP_zrvmulD(&a, 2, b, 1, &p, 2, size);
	}
	template<class A, class B>
	ItemTypeCF<A, ItemTypeF<B>> fillExpr(SplitPointer<float> pointer, expression::Mul<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a, size));
		auto *b = floats.real(expr.b, size);
		auto p = dspSplit(pointer);
		vDSP_zrvmul(&a, 1, b, 1, &p, 1, size);
	}
	template<class A, class B>
	ItemTypeCD<A, ItemTypeD<B>> fillExpr(SplitPointer<double> pointer, expression::Mul<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a, size));
		auto *b = doubles.real(expr.b, size);
		auto p = dspSplit(pointer);
		vDSP_zrvmulD(&a, 1, b, 1, &p, 1, size);
	}
	template<class A, class B>
	ItemTypeF<A, ItemTypeCF<B>> fillExpr(ComplexPointer<float> pointer, expression::Mul<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		auto b = dspSplit(floats.complex(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zrvmul(&b, 2, a, 1, &p, 2, size);
	}
	template<class A, class B>
	ItemTypeD<A, ItemTypeCD<B>> fillExpr(ComplexPointer<double> pointer, expression::Mul<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		auto b = dspSplit(doubles.complex(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zrvmulD(&b, 2, a, 1, &p, 2, size);
	}
	template<class A, class B>
	ItemTypeF<A, ItemTypeCF<B>> fillExpr(SplitPointer<float> pointer, expression::Mul<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		auto b = dspSplit(floats.split(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zrvmul(&b, 1, a, 1, &p, 1, size);
	}
	template<class A, class B>
	ItemTypeD<A, ItemTypeCD<B>> fillExpr(SplitPointer<double> pointer, expression::Mul<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		auto b = dspSplit(doubles.split(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zrvmulD(&b, 1, a, 1, &p, 1, size);
	}
	template<class A, class B>
	ItemTypeCF<A, ItemTypeCF<B>> fillExpr(ComplexPointer<float> pointer, expression::Div<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.complex(expr.a, size));
		auto b = dspSplit(floats.complex(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvdiv(&b, 2, &a, 2, &p, 2, size);
	}
	template<class A, class B>
	ItemTypeCD<A, ItemTypeCD<B>> fillExpr(ComplexPointer<double> pointer, expression::Div<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.complex(expr.a, size));
		auto b = dspSplit(doubles.complex(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvdivD(&b, 2, &a, 2, &p, 2, size);
	}
	template<class A, class B>
	ItemTypeCF<A, ItemTypeCF<B>> fillExpr(SplitPointer<float> pointer, expression::Div<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a, size));
		auto b = dspSplit(floats.split(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvdiv(&b, 1, &a, 1, &p, 1, size);
	}
	template<class A, class B>
	ItemTypeCD<A, ItemTypeCD<B>> fillExpr(SplitPointer<double> pointer, expression::Div<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a, size));
		auto b = dspSplit(doubles.split(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvdivD(&b, 1, &a, 1, &p, 1, size);
	}
	template<class A, class B>
	ItemTypeCF<A, ItemTypeF<B>> fillExpr(ComplexPointer<float> pointer, expression::Div<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.complex(expr.a, size));
		auto *b = floats.real(expr.b, size);
		auto p = dspSplit(pointer);
		vDSP_zrvdiv(&a, 2, b, 1, &p, 2, size);
	}
	template<class A, class B>
	ItemTypeCD<A, ItemTypeD<B>> fillExpr(ComplexPointer<double> pointer, expression::Div<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.complex(expr.a, size));
		auto *b = doubles.real(expr.b, size);
		auto p = dspSplit(pointer);
		vDSP_zrvdivD(&a, 2, b, 1, &p, 2, size);
	}
	template<class A, class B>
	ItemTypeCF<A, ItemTypeF<B>> fillExpr(SplitPointer<float> pointer, expression::Div<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a, size));
		auto *b = floats.real(expr.b, size);
		auto p = dspSplit(pointer);
		vDSP_zrvdiv(&a, 1, b, 1, &p, 1, size);
	}
	template<class A, class B>
	ItemTypeCD<A, ItemTypeD<B>> fillExpr(SplitPointer<double> pointer, expression::Div<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a, size));
		auto *b = doubles.real(expr.b, size);
		auto p = dspSplit(pointer);
		vDSP_zrvdivD(&a, 1, b, 1, &p, 1, size);
	}
	// vDSP_zvmul(..., -1) conjugates the first argument (a)
	template<class A, class B>
	ItemTypeCF<A, ItemTypeCF<B, NotConstant<B>>> fillExpr(ComplexPointer<float> pointer, expression::Mul<expression::Conj<A>, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.complex(expr.a.a, size));
		auto b = dspSplit(floats.complex(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvmul(&a, 2, &b, 2, &p, 2, size, -1);
	}
	// conj(A)*conj(B) = conj(A*B)
	template<class A, class B>
	ItemTypeCF<A, ItemTypeCF<B>> fillExpr(ComplexPointer<float> pointer, expression::Mul<expression::Conj<A>, expression::Conj<B>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.complex(expr.a.a, size));
		auto b = dspSplit(floats.complex(expr.b.a, size));
		auto p = dspSplit(pointer);
		vDSP_zvmul(&a, 2, &b, 2, &p, 2, size, 1);
		vDSP_zvconj(&p, 2, &p, 2, size);
	}
	template<class A, class B>
	ItemTypeCD<A, ItemTypeCD<B, NotConstant<B>>> fillExpr(ComplexPointer<double> pointer, expression::Mul<expression::Conj<A>, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.complex(expr.a.a, size));
		auto b = dspSplit(doubles.complex(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvmulD(&a, 2, &b, 2, &p, 2, size, -1);
	}
	// conj(A)*conj(B) = conj(A*B)
	template<class A, class B>
	ItemTypeCD<A, ItemTypeCD<B>> fillExpr(ComplexPointer<double> pointer, expression::Mul<expression::Conj<A>, expression::Conj<B>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.complex(expr.a.a, size));
		auto b = dspSplit(doubles.complex(expr.b.a, size));
		auto p = dspSplit(pointer);
		vDSP_zvmulD(&a, 2, &b, 2, &p, 2, size, 1);
		vDSP_zvconjD(&p, 2, &p, 2, size);
	}
	template<class A, class B>
	ItemTypeCF<A, ItemTypeCF<B, NotConstant<B>>> fillExpr(SplitPointer<float> pointer, expression::Mul<expression::Conj<A>, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a.a, size));
		auto b = dspSplit(floats.split(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvmul(&a, 1, &b, 1, &p, 1, size, -1);
	}
	// conj(A)*conj(B) = conj(A*B)
	template<class A, class B>
	ItemTypeCF<A, ItemTypeCF<B>> fillExpr(SplitPointer<float> pointer, expression::Mul<expression::Conj<A>, expression::Conj<B>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a.a, size));
		auto b = dspSplit(floats.split(expr.b.a, size));
		auto p = dspSplit(pointer);
		vDSP_zvmul(&a, 1, &b, 1, &p, 1, size, 1);
		vDSP_zvconj(&p, 1, &p, 1, size);
	}
	template<class A, class B>
	ItemTypeCD<A, ItemTypeCD<B, NotConstant<B>>> fillExpr(SplitPointer<double> pointer, expression::Mul<expression::Conj<A>, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a.a, size));
		auto b = dspSplit(doubles.split(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvmulD(&a, 1, &b, 1, &p, 1, size, -1);
	}
	// conj(A)*conj(B) = conj(A*B)
	template<class A, class B>
	ItemTypeCD<A, ItemTypeCD<B>> fillExpr(SplitPointer<double> pointer, expression::Mul<expression::Conj<A>, expression::Conj<B>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a.a, size));
		auto b = dspSplit(doubles.split(expr.b.a, size));
		auto p = dspSplit(pointer);
		vDSP_zvmulD(&a, 1, &b, 1, &p, 1, size, 1);
		vDSP_zvconjD(&p, 1, &p, 1, size);
	}

	//---- Complex binary ops with a constant
	template<class A, class V>
	ItemTypeCF<A> fillExpr(ComplexPointer<float> pointer, expression::Add<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.complex(expr.a, size, pointer));
		auto p = dspSplit(pointer);
		std::complex<float> b = expr.b.value;
		float br = b.real(), bi = b.imag();
		vDSP_vsadd(a.realp, 2, &br, p.realp, 2, size);
		vDSP_vsadd(a.imagp, 2, &bi, p.imagp, 2, size);
	}
	template<class A, class V>
	ItemTypeCF<A> fillExpr(ComplexPointer<float> pointer, expression::Sub<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.complex(expr.a, size, pointer));
		auto p = dspSplit(pointer);
		std::complex<float> b = expr.b.value;
		float br = -b.real(), bi = -b.imag();
		vDSP_vsadd(a.realp, 2, &br, p.realp, 2, size);
		vDSP_vsadd(a.imagp, 2, &bi, p.imagp, 2, size);
	}
	template<class V, class B>
	ItemTypeCF<B, NotConstant<B>> fillExpr(ComplexPointer<float> pointer, expression::Sub<expression::ConstantExpr<V>, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto b = dspSplit(floats.complex(expr.b, size, pointer));
		auto p = dspSplit(pointer);
		std::complex<float> a = expr.a.value;
		float ar = a.real(), ai = a.imag(), minusOne = -1;
		vDSP_vsmsa(b.realp, 2, &minusOne, &ar, p.realp, 2, size);
		vDSP_vsmsa(b.imagp, 2, &minusOne, &ai, p.imagp, 2, size);
	}
	template<class A, class V>
	ItemTypeCF<A> fillExpr(ComplexPointer<float> pointer, expression::Mul<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.complex(expr.a, size, pointer));
		auto p = dspSplit(pointer);
		std::complex<float> b = expr.b.value;
		float br = b.real(), bi = b.imag();
		if (bi == 0) {
			vDSP_vsmul(a.realp, 2, &br, p.realp, 2, size);
			vDSP_vsmul(a.imagp, 2, &br, p.imagp, 2, size);
		} else {
			auto bSplit = dspSplit({&br, &bi});
			vDSP_zvzsml(&a, 2, &bSplit, &p, 2, size);
		}
	}
	template<class A, class V>
	ItemTypeCF<A> fillExpr(ComplexPointer<float> pointer, expression::Div<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.complex(expr.a, size, pointer));
		auto p = dspSplit(pointer);
		std::complex<float> b = expr.b.value;
		float br = b.real(), bi = b.imag();
		if (bi == 0) {
			vDSP_vsdiv(a.realp, 2, &br, p.realp, 2, size);
			vDSP_vsdiv(a.imagp, 2, &br, p.imagp, 2, size);
		} else {
			b = float(1)/b;
			br = b.real();
			bi = b.imag();
			auto bSplit = dspSplit({&br, &bi});
			vDSP_zvzsml(&a, 2, &bSplit, &p, 2, size);
		}
	}
	template<class A, class V>
	ItemTypeCD<A> fillExpr(ComplexPointer<double> pointer, expression::Add<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.complex(expr.a, size, pointer));
		auto p = dspSplit(pointer);
		std::complex<double> b = expr.b.value;
		double br = b.real(), bi = b.imag();
		vDSP_vsaddD(a.realp, 2, &br, p.realp, 2, size);
		vDSP_vsaddD(a.imagp, 2, &bi, p.imagp, 2, size);
	}
	template<class A, class V>
	ItemTypeCD<A> fillExpr(ComplexPointer<double> pointer, expression::Sub<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.complex(expr.a, size, pointer));
		auto p = dspSplit(pointer);
		std::complex<double> b = expr.b.value;
		double br = -b.real(), bi = -b.imag();
		vDSP_vsaddD(a.realp, 2, &br, p.realp, 2, size);
		vDSP_vsaddD(a.imagp, 2, &bi, p.imagp, 2, size);
	}
	template<class V, class B>
	ItemTypeCD<B, NotConstant<B>> fillExpr(ComplexPointer<double> pointer, expression::Sub<expression::ConstantExpr<V>, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto b = dspSplit(doubles.complex(expr.b, size, pointer));
		auto p = dspSplit(pointer);
		std::complex<double> a = expr.a.value;
		double ar = a.real(), ai = a.imag(), minusOne = -1;
		vDSP_vsmsaD(b.realp, 2, &minusOne, &ar, p.realp, 2, size);
		vDSP_vsmsaD(b.imagp, 2, &minusOne, &ai, p.imagp, 2, size);
	}
	template<class A, class V>
	ItemTypeCD<A> fillExpr(ComplexPointer<double> pointer, expression::Mul<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.complex(expr.a, size, pointer));
		auto p = dspSplit(pointer);
		std::complex<double> b = expr.b.value;
		double br = b.real(), bi = b.imag();
		if (bi == 0) {
			vDSP_vsmulD(a.realp, 2, &br, p.realp, 2, size);
			vDSP_vsmulD(a.imagp, 2, &br, p.imagp, 2, size);
		} else {
			auto bSplit = dspSplit({&br, &bi});
			vDSP_zvzsmlD(&a, 2, &bSplit, &p, 2, size);
		}
	}
	template<class A, class V>
	ItemTypeCD<A> fillExpr(ComplexPointer<double> pointer, expression::Div<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.complex(expr.a, size, pointer));
		auto p = dspSplit(pointer);
		std::complex<double> b = expr.b.value;
		double br = b.real(), bi = b.imag();
		if (bi == 0) {
			vDSP_vsdivD(a.realp, 2, &br, p.realp, 2, size);
			vDSP_vsdivD(a.imagp, 2, &br, p.imagp, 2, size);
		} else {
			b = double(1)/b;
			br = b.real();
			bi = b.imag();
			auto bSplit = dspSplit({&br, &bi});
			vDSP_zvzsmlD(&a, 2, &bSplit, &p, 2, size);
		}
	}
	template<class A, class V>
	ItemTypeCF<A> fillExpr(SplitPointer<float> pointer, expression::Add<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a, size, pointer));
		auto p = dspSplit(pointer);
		std::complex<float> b = expr.b.value;
		float br = b.real(), bi = b.imag();
		vDSP_vsadd(a.realp, 1, &br, p.realp, 1, size);
		vDSP_vsadd(a.imagp, 1, &bi, p.imagp, 1, size);
	}
	template<class A, class V>
	ItemTypeCF<A> fillExpr(SplitPointer<float> pointer, expression::Sub<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a, size, pointer));
		auto p = dspSplit(pointer);
		std::complex<float> b = expr.b.value;
		float br = -b.real(), bi = -b.imag();
		vDSP_vsadd(a.realp, 1, &br, p.realp, 1, size);
		vDSP_vsadd(a.imagp, 1, &bi, p.imagp, 1, size);
	}
	template<class V, class B>
	ItemTypeCF<B, NotConstant<B>> fillExpr(SplitPointer<float> pointer, expression::Sub<expression::ConstantExpr<V>, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto b = dspSplit(floats.split(expr.b, size, pointer));
		auto p = dspSplit(pointer);
		std::complex<float> a = expr.a.value;
		float ar = a.real(), ai = a.imag(), minusOne = -1;
		vDSP_vsmsa(b.realp, 1, &minusOne, &ar, p.realp, 1, size);
		vDSP_vsmsa(b.imagp, 1, &minusOne, &ai, p.imagp, 1, size);
	}
	template<class A, class V>
	ItemTypeCF<A> fillExpr(SplitPointer<float> pointer, expression::Mul<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a, size, pointer));
		auto p = dspSplit(pointer);
		std::complex<float> b = expr.b.value;
		float br = b.real(), bi = b.imag();
		if (bi == 0) {
			vDSP_vsmul(a.realp, 1, &br, p.realp, 1, size);
			vDSP_vsmul(a.imagp, 1, &br, p.imagp, 1, size);
		} else {
			auto bSplit = dspSplit({&br, &bi});
			vDSP_zvzsml(&a, 1, &bSplit, &p, 1, size);
		}
	}
	template<class A, class V>
	ItemTypeCF<A> fillExpr(SplitPointer<float> pointer, expression::Div<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a, size, pointer));
		auto p = dspSplit(pointer);
		std::complex<float> b = expr.b.value;
		float br = b.real(), bi = b.imag();
		if (bi == 0) {
			vDSP_vsdiv(a.realp, 1, &br, p.realp, 1, size);
			vDSP_vsdiv(a.imagp, 1, &br, p.imagp, 1, size);
		} else {
			b = float(1)/b;
			br = b.real();
			bi = b.imag();
			auto bSplit = dspSplit({&br, &bi});
			vDSP_zvzsml(&a, 1, &bSplit, &p, 1, size);
		}
	}
	template<class A, class V>
	ItemTypeCD<A> fillExpr(SplitPointer<double> pointer, expression::Add<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a, size, pointer));
		auto p = dspSplit(pointer);
		std::complex<double> b = expr.b.value;
		double br = b.real(), bi = b.imag();
		vDSP_vsaddD(a.realp, 1, &br, p.realp, 1, size);
		vDSP_vsaddD(a.imagp, 1, &bi, p.imagp, 1, size);
	}
	template<class A, class V>
	ItemTypeCD<A> fillExpr(SplitPointer<double> pointer, expression::Sub<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a, size, pointer));
		auto p = dspSplit(pointer);
		std::complex<double> b = expr.b.value;
		double br = -b.real(), bi = -b.imag();
		vDSP_vsaddD(a.realp, 1, &br, p.realp, 1, size);
		vDSP_vsaddD(a.imagp, 1, &bi, p.imagp, 1, size);
	}
	template<class V, class B>
	ItemTypeCD<B, NotConstant<B>> fillExpr(SplitPointer<double> pointer, expression::Sub<expression::ConstantExpr<V>, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto b = dspSplit(doubles.split(expr.b, size, pointer));
		auto p = dspSplit(pointer);
		std::complex<double> a = expr.a.value;
		double ar = a.real(), ai = a.imag(), minusOne = -1;
		vDSP_vsmsaD(b.realp, 1, &minusOne, &ar, p.realp, 1, size);
		vDSP_vsmsaD(b.imagp, 1, &minusOne, &ai, p.imagp, 1, size);
	}
	template<class A, class V>
	ItemTypeCD<A> fillExpr(SplitPointer<double> pointer, expression::Mul<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a, size, pointer));
		auto p = dspSplit(pointer);
		std::complex<double> b = expr.b.value;
		double br = b.real(), bi = b.imag();
		if (bi == 0) {
			vDSP_vsmulD(a.realp, 1, &br, p.realp, 1, size);
			vDSP_vsmulD(a.imagp, 1, &br, p.imagp, 1, size);
		} else {
			auto bSplit = dspSplit({&br, &bi});
			vDSP_zvzsmlD(&a, 1, &bSplit, &p, 1, size);
		}
	}
	template<class A, class V>
	ItemTypeCD<A> fillExpr(SplitPointer<double> pointer, expression::Div<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a, size, pointer));
		auto p = dspSplit(pointer);
		std::complex<double> b = expr.b.value;
		double br = b.real(), bi = b.imag();
		if (bi == 0) {
			vDSP_vsdivD(a.realp, 1, &br, p.realp, 1, size);
			vDSP_vsdivD(a.imagp, 1, &br, p.imagp, 1, size);
		} else {
			b = double(1)/b;
			br = b.real();
			bi = b.imag();
			auto bSplit = dspSplit({&br, &bi});
			vDSP_zvzsmlD(&a, 1, &bSplit, &p, 1, size);
		}
	}


	// Real values combined with a (possibly complex) constant
	template<class A, class V>
	ItemTypeF<A> fillExpr(ComplexPointer<float> pointer, expression::Add<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		auto p = dspSplit(pointer);
		std::complex<float> b = expr.b.value;
		float br = b.real(), bi = b.imag();
		vDSP_vsadd(a, 1, &br, p.realp, 2, size);
		vDSP_vfill(&bi, p.imagp, 2, size);
	}
	template<class A, class V>
	ItemTypeF<A> fillExpr(ComplexPointer<float> pointer, expression::Mul<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		auto p = dspSplit(pointer);
		std::complex<float> b = expr.b.value;
		float br = b.real(), bi = b.imag();
		vDSP_vsmul(a, 1, &br, p.realp, 2, size);
		vDSP_vsmul(a, 1, &bi, p.imagp, 2, size);
	}
	template<class A, class V>
	ItemTypeD<A> fillExpr(ComplexPointer<double> pointer, expression::Add<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		auto p = dspSplit(pointer);
		std::complex<double> b = expr.b.value;
		double br = b.real(), bi = b.imag();
		vDSP_vsaddD(a, 1, &br, p.realp, 2, size);
		vDSP_vfillD(&bi, p.imagp, 2, size);
	}
	template<class A, class V>
	ItemTypeD<A> fillExpr(ComplexPointer<double> pointer, expression::Mul<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		auto p = dspSplit(pointer);
		std::complex<double> b = expr.b.value;
		double br = b.real(), bi = b.imag();
		vDSP_vsmulD(a, 1, &br, p.realp, 2, size);
		vDSP_vsmulD(a, 1, &bi, p.imagp, 2, size);
	}
	template<class A, class V>
	ItemTypeF<A> fillExpr(SplitPointer<float> pointer, expression::Add<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		auto p = dspSplit(pointer);
		std::complex<float> b = expr.b.value;
		float br = b.real(), bi = b.imag();
		vDSP_vsadd(a, 1, &br, p.realp, 1, size);
		vDSP_vfill(&bi, p.imagp, 1, size);
	}
	template<class A, class V>
	ItemTypeF<A> fillExpr(SplitPointer<float> pointer, expression::Mul<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		auto p = dspSplit(pointer);
		std::complex<float> b = expr.b.value;
		float br = b.real(), bi = b.imag();
		vDSP_vsmul(a, 1, &br, p.realp, 1, size);
		vDSP_vsmul(a, 1, &bi, p.imagp, 1, size);
	}
	template<class A, class V>
	ItemTypeD<A> fillExpr(SplitPointer<double> pointer, expression::Add<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		auto p = dspSplit(pointer);
		std::complex<double> b = expr.b.value;
		double br = b.real(), bi = b.imag();
		vDSP_vsaddD(a, 1, &br, p.realp, 1, size);
		vDSP_vfillD(&bi, p.imagp, 1, size);
	}
	template<class A, class V>
	ItemTypeD<A> fillExpr(SplitPointer<double> pointer, expression::Mul<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		auto p = dspSplit(pointer);
		std::complex<double> b = expr.b.value;
		double br = b.real(), bi = b.imag();
		vDSP_vsmulD(a, 1, &br, p.realp, 1, size);
		vDSP_vsmulD(a, 1, &bi, p.imagp, 1, size);
	}

	//---- Compound complex expressions
	template<class A, class B, class C>
	ItemTypeCF<A, ItemTypeCF<B, ItemTypeCF<C, NotConstant<C>>>> fillExpr(ComplexPointer<float> pointer, expression::Add<expression::Mul<A, B>, C> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.complex(expr.a.a, size));
		auto b = dspSplit(floats.complex(expr.a.b, size));
		auto c = dspSplit(floats.complex(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvma(&a, 2, &b, 2, &c, 2, &p, 2, size);
	}
	template<class A, class B, class C>
	ItemTypeCF<A, ItemTypeCF<B, ItemTypeCF<C, NotConstant<C>>>> fillExpr(ComplexPointer<float> pointer, expression::Add<expression::Mul<expression::Conj<A>, B>, C> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.complex(expr.a.a.a, size));
		auto b = dspSplit(floats.complex(expr.a.b, size));
		auto c = dspSplit(floats.complex(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvcma(&a, 2, &b, 2, &c, 2, &p, 2, size);
	}
	template<class A, class B, class C>
	ItemTypeCD<A, ItemTypeCD<B, ItemTypeCD<C, NotConstant<C>>>> fillExpr(ComplexPointer<double> pointer, expression::Add<expression::Mul<A, B>, C> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.complex(expr.a.a, size));
		auto b = dspSplit(doubles.complex(expr.a.b, size));
		auto c = dspSplit(doubles.complex(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvmaD(&a, 2, &b, 2, &c, 2, &p, 2, size);
	}
	template<class A, class B, class C>
	ItemTypeCD<A, ItemTypeCD<B, ItemTypeCD<C, NotConstant<C>>>> fillExpr(ComplexPointer<double> pointer, expression::Add<expression::Mul<expression::Conj<A>, B>, C> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.complex(expr.a.a.a, size));
		auto b = dspSplit(doubles.complex(expr.a.b, size));
		auto c = dspSplit(doubles.complex(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvcmaD(&a, 2, &b, 2, &c, 2, &p, 2, size);
	}
	template<class A, class B, class C>
	ItemTypeCF<A, ItemTypeCF<B, ItemTypeCF<C, NotConstant<C>>>> fillExpr(SplitPointer<float> pointer, expression::Add<expression::Mul<A, B>, C> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a.a, size));
		auto b = dspSplit(floats.split(expr.a.b, size));
		auto c = dspSplit(floats.split(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvma(&a, 1, &b, 1, &c, 1, &p, 1, size);
	}
	template<class A, class B, class C>
	ItemTypeCF<A, ItemTypeCF<B, ItemTypeCF<C, NotConstant<C>>>> fillExpr(SplitPointer<float> pointer, expression::Add<expression::Mul<expression::Conj<A>, B>, C> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a.a.a, size));
		auto b = dspSplit(floats.split(expr.a.b, size));
		auto c = dspSplit(floats.split(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvcma(&a, 1, &b, 1, &c, 1, &p, 1, size);
	}
	template<class A, class B, class C>
	ItemTypeCD<A, ItemTypeCD<B, ItemTypeCD<C, NotConstant<C>>>> fillExpr(SplitPointer<double> pointer, expression::Add<expression::Mul<A, B>, C> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a.a, size));
		auto b = dspSplit(doubles.split(expr.a.b, size));
		auto c = dspSplit(doubles.split(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvmaD(&a, 1, &b, 1, &c, 1, &p, 1, size);
	}
	template<class A, class B, class C>
	ItemTypeCD<A, ItemTypeCD<B, ItemTypeCD<C, NotConstant<C>>>> fillExpr(SplitPointer<double> pointer, expression::Add<expression::Mul<expression::Conj<A>, B>, C> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a.a.a, size));
		auto b = dspSplit(doubles.split(expr.a.b, size));
		auto c = dspSplit(doubles.split(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvcmaD(&a, 1, &b, 1, &c, 1, &p, 1, size);
	}
	// conj(A)*conj(B) + C: compute conj(B) first, then use zvcma
	template<class A, class B, class C>
	ItemTypeCF<A, ItemTypeCF<B, ItemTypeCF<C, NotConstant<C>>>> fillExpr(ComplexPointer<float> pointer, expression::Add<expression::Mul<expression::Conj<A>, expression::Conj<B>>, C> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.complex(expr.a.a.a, size));
		auto b = dspSplit(floats.complex(expr.a.b, size));
		auto c = dspSplit(floats.complex(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvcma(&a, 2, &b, 2, &c, 2, &p, 2, size);
	}
	// conj(A)*conj(B) + C: compute conj(B) first, then use zvcma
	template<class A, class B, class C>
	ItemTypeCD<A, ItemTypeCD<B, ItemTypeCD<C, NotConstant<C>>>> fillExpr(ComplexPointer<double> pointer, expression::Add<expression::Mul<expression::Conj<A>, expression::Conj<B>>, C> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.complex(expr.a.a.a, size));
		auto b = dspSplit(doubles.complex(expr.a.b, size));
		auto c = dspSplit(doubles.complex(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvcmaD(&a, 2, &b, 2, &c, 2, &p, 2, size);
	}
	// conj(A)*conj(B) + C: compute conj(B) first, then use zvcma
	template<class A, class B, class C>
	ItemTypeCF<A, ItemTypeCF<B, ItemTypeCF<C, NotConstant<C>>>> fillExpr(SplitPointer<float> pointer, expression::Add<expression::Mul<expression::Conj<A>, expression::Conj<B>>, C> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a.a.a, size));
		auto b = dspSplit(floats.split(expr.a.b, size));
		auto c = dspSplit(floats.split(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvcma(&a, 1, &b, 1, &c, 1, &p, 1, size);
	}
	// conj(A)*conj(B) + C: compute conj(B) first, then use zvcma
	template<class A, class B, class C>
	ItemTypeCD<A, ItemTypeCD<B, ItemTypeCD<C, NotConstant<C>>>> fillExpr(SplitPointer<double> pointer, expression::Add<expression::Mul<expression::Conj<A>, expression::Conj<B>>, C> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a.a.a, size));
		auto b = dspSplit(doubles.split(expr.a.b, size));
		auto c = dspSplit(doubles.split(expr.b, size));
		auto p = dspSplit(pointer);
		vDSP_zvcmaD(&a, 1, &b, 1, &c, 1, &p, 1, size);
	}

protected:
	CachedResults<LinearImpl> cached;

	static DSPSplitComplex dspSplit(ConstSplitPointer<float> x) {
		DSPSplitComplex dsp;
		dsp.realp = (float *)x.real;
		dsp.imagp = (float *)x.imag;
		return dsp;
	}
	static DSPSplitComplex dspSplit(ConstComplexPointer<float> x) {
		DSPSplitComplex dsp;
		dsp.realp = (float *)x;
		dsp.imagp = (float *)x + 1;
		return dsp;
	}
	static DSPDoubleSplitComplex dspSplit(ConstSplitPointer<double> x) {
		DSPDoubleSplitComplex dsp;
		dsp.realp = (double *)x.real;
		dsp.imagp = (double *)x.imag;
		return dsp;
	}
	static DSPDoubleSplitComplex dspSplit(ConstComplexPointer<double> x) {
		DSPDoubleSplitComplex dsp;
		dsp.realp = (double *)x;
		dsp.imagp = (double *)x + 1;
		return dsp;
	}
	static const DSPComplex * dspComplex(ConstComplexPointer<float> x) {
		return (const DSPComplex *)x;
	}
	static DSPComplex * dspComplex(ComplexPointer<float> x) {
		return (DSPComplex *)x;
	}
	static const DSPDoubleComplex * dspComplex(ConstComplexPointer<double> x) {
		return (const DSPDoubleComplex *)x;
	}
	static DSPDoubleComplex * dspComplex(ComplexPointer<double> x) {
		return (DSPDoubleComplex *)x;
	}
};

}}; // namespace
