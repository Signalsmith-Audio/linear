//#define ACCELERATE_NEW_LAPACK

// If possible, only include vecLib, since JUCE has conflicts with vImage
#if defined(__has_include) && __has_include(<vecLib/vecLib.h>)
#	include <vecLib/vecLib.h>
#else
#	include <Accelerate/Accelerate.h>
#endif

#include <cstring> // std::memcpy

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
	ItemType<Expr, float, void> fillBasic(SplitPointer<float> pointer, Expr expr, size_t size) {
		fillExpr(pointer.real, expr, size);
		clear(pointer.imag, size);
	}
	template<class Expr>
	ItemType<Expr, double, void> fillBasic(SplitPointer<double> pointer, Expr expr, size_t size) {
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

//	// Complex norm of a split pointer - rephrased as (r*r + i*i)
//	template<class V>
//	void fillExpr(RealPointer<V> pointer, expression::Norm<expression::ReadableSplit<V>> expr, size_t size) {
//		auto real = expression::ReadableReal<V>(expr.a.pointer.real);
//		auto imag = expression::ReadableReal<V>(expr.a.pointer.imag);
//		auto newExpr = expression::makeAdd(expression::makeMul(real, real), expression::makeMul(imag, imag));
//		return fillExpr(pointer, newExpr, size);
//	}
//	template<class V>
//	void fillExpr(RealPointer<V> pointer, expression::Norm<WritableSplit<V>> expr, size_t size) {
//		auto real = expression::ReadableReal<V>(expr.a.pointer.real);
//		auto imag = expression::ReadableReal<V>(expr.a.pointer.imag);
//		auto newExpr = expression::makeAdd(expression::makeMul(real, real), expression::makeMul(imag, imag));
//		return fillExpr(pointer, newExpr, size);
//	}
	
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
	ItemType<A, float, void> fillExpr(RealPointer<float> pointer, expression::Abs<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		vDSP_vabs(a, 1, pointer, 1, size);
	}
	template<class A>
	ItemType<A, double, void> fillExpr(RealPointer<double> pointer, expression::Abs<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		vDSP_vabsD(a, 1, pointer, 1, size);
	}
	template<class A>
	ItemType<A, std::complex<float>, void> fillExpr(RealPointer<float> pointer, expression::Abs<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a, size));
		vDSP_zvabs(&a, 1, pointer, 1, size);
	}
	template<class A>
	ItemType<A, std::complex<double>, void> fillExpr(RealPointer<double> pointer, expression::Abs<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a, size));
		vDSP_zvabsD(&a, 1, pointer, 1, size);
	}

	template<class A>
	ItemType<A, float, void> fillExpr(RealPointer<float> pointer, expression::Neg<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		vDSP_vneg(a, 1, pointer, 1, size);
	}
	template<class A>
	ItemType<A, double, void> fillExpr(RealPointer<double> pointer, expression::Neg<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		vDSP_vnegD(a, 1, pointer, 1, size);
	}
	template<class A>
	ItemType<A, std::complex<float>, void> fillExpr(ComplexPointer<float> pointer, expression::Neg<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.complex(expr.a, size));
		auto p = dspSplit(pointer);
		vDSP_zvneg(&a, 2, &p, 2, size);
	}
	template<class A>
	ItemType<A, std::complex<double>, void> fillExpr(ComplexPointer<double> pointer, expression::Neg<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.complex(expr.a, size));
		auto p = dspSplit(pointer);
		vDSP_zvnegD(&a, 2, &p, 2, size);
	}
	template<class A>
	ItemType<A, std::complex<float>, void> fillExpr(SplitPointer<float> pointer, expression::Neg<A> expr, size_t size) {
		auto floats = cached.floatScope();
		auto a = dspSplit(floats.split(expr.a, size));
		auto p = dspSplit(pointer);
		vDSP_zvneg(&a, 1, &p, 1, size);
	}
	template<class A>
	ItemType<A, std::complex<double>, void> fillExpr(SplitPointer<double> pointer, expression::Neg<A> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto a = dspSplit(doubles.split(expr.a, size));
		auto p = dspSplit(pointer);
		vDSP_zvnegD(&a, 1, &p, 1, size);
	}
	
	//---- Binary ops

	template<class A, class B>
	void fillExpr(RealPointer<float> pointer, expression::Div<A, B> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size);
		auto *b = floats.real(expr.b, size);
		int intSize = int(size);
		vvdivf(pointer, a, b, &intSize);
	}
	template<class A, class B>
	void fillExpr(RealPointer<double> pointer, expression::Div<A, B> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size);
		auto *b = doubles.real(expr.b, size);
		int intSize = int(size);
		vvdiv(pointer, a, b, &intSize);
	}

	//---- Ternary expressions
	
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

	//---- Expressions involving constants
	
	template<class A, class V>
	void fillExpr(RealPointer<float> pointer, expression::Add<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a, size, pointer);
		float b = expr.b.value;
		vDSP_vsadd(a, 1, &b, pointer, 1, size);
	}
	template<class A, class V>
	void fillExpr(RealPointer<double> pointer, expression::Add<A, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a, size, pointer);
		double b = expr.b.value;
		vDSP_vsaddD(a, 1, &b, pointer, 1, size);
	}

	template<class A, class B, class V>
	void fillMul(RealPointer<float> pointer, expression::Mul<expression::Add<A, B>, expression::ConstantExpr<V>> expr, size_t size) {
		auto floats = cached.floatScope();
		auto *a = floats.real(expr.a.a, size);
		auto *b = floats.real(expr.a.b, size);
		float c = expr.b.value;
		vDSP_vasm(a, 1, b, 1, &c, pointer, 1, size);
	}
	template<class A, class B, class V>
	void fillMul(RealPointer<double> pointer, expression::Mul<expression::Add<A, B>, expression::ConstantExpr<V>> expr, size_t size) {
		auto doubles = cached.doubleScope();
		auto *a = doubles.real(expr.a.a, size);
		auto *b = doubles.real(expr.a.b, size);
		double c = expr.b.value;
		vDSP_vasmD(a, 1, b, 1, &c, pointer, 1, size);
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
};

}}; // namespace
