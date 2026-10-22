#include <cstring> // std::memcpy
#include <limits>
#include <type_traits>

#include <immintrin.h>

#include "./basic-fill-warnings.h"

namespace signalsmith { namespace linear {

namespace sse2 {
	// Fallback: calculate as scalars and collect into SIMD register
	template<class Expr>
	bool isAligned(Expr expr, size_t i) {
		return false;
	}

	// Fallback: calculate as scalars and collect into SIMD register
	template<bool aligned, class Expr>
	__m128 getFloat(Expr expr, size_t i) {
		return _mm_setr_ps(expr.get(i), expr.get(i + 1), expr.get(i + 2), expr.get(i + 3));
	}
	template<bool aligned>
	__m128 getFloat(expression::ReadableReal<float> expr, size_t i) {
		if (aligned) {
			return _mm_loadu_ps(expr.pointer + i);
		} else {
			return _mm_loadu_ps(expr.pointer + i);
		}
	}
	template<bool aligned, class Expr>
	ItemTypeF<Expr, __m128> getFloat(expression::Abs<Expr> expr, size_t i) {
		__m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)); // clears the sign bit
		__m128 a = getFloat<aligned>(expr.a, i);
		return _mm_and_ps(a, mask);
	}
}

template<>
struct LinearImpl<true> : public LinearImplBase<true> {
	using Base = LinearImplBase<true>;

	LinearImpl() : Base(this) {
		basicFillWarningReset();
	}

	template<class V>
	void reserve(size_t) {/* doesn't need this */}

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

	template<class Expr>
	void fillBasic(RealPointer<float> pointer, Expr expr, size_t size) {
		// SSE2: width 4
		// sizes 0-3 handled explicitly (too short to be SIMD)
		switch(size) {
			case 3: pointer[2] = expr.get(2);
			case 2: pointer[1] = expr.get(1);
			case 1: pointer[0] = expr.get(0);
			case 0: return;
			default: break;
		}
		size_t i = 0;
		size_t alignmentResidual = (size_t(pointer)/4)&3;
		switch(alignmentResidual) {
			case 4:
				pointer[0] = expr.get(0);
				++i;
			case 8:
				pointer[i] = expr.get(i);
				++i;
			case 12:
				pointer[i] = expr.get(i);
				++i;
			default:
				break;
		}
		size_t simdEndI = i + (size - i)/4*4;
		if (sse2::isAligned(expr, i)) { // Don't explode the code-paths - have one for "everything is aligned", one otherwise
			while (i < simdEndI) {
				__m128 v = sse2::getFloat<true>(expr, i);
				_mm_store_ps(pointer + i, v);
				i += 4;
			}
		} else {
			while (i < simdEndI) {
				__m128 v = sse2::getFloat<false>(expr, i);
				_mm_store_ps(pointer + i, v);
				i += 4;
			}
		}
		while (i < size) {
			pointer[i] = expr.get(i);
			++i;
		}
	}

	template<class V, class Expr>
	void fillBasic(RealPointer<V> pointer, Expr expr, size_t size) {
		basicFillWarning<Expr>();
		for (size_t i = 0; i < size; ++i) {
			pointer[i] = expr.get(i);
		}
	}
	template<class V, class Expr>
	void fillBasic(ComplexPointer<V> pointer, Expr expr, size_t size) {
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
//	void clear(float *v, size_t size) {
//		vDSP_vclr(v, 1, size);
//	}
//	void clear(double *v, size_t size) {
//		vDSP_vclrD(v, 1, size);
//	}
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
	
//	// Filling with a constant
//	template<class V>
//	void fillExpr(RealPointer<float> pointer, expression::ConstantExpr<V> expr, size_t size) {
//		float v = expr.value;
//		vDSP_vfill(&v, pointer, 1, size);
//	}
//	template<class A>
//	ItemTypeF<A> fillExpr(RealPointer<float> pointer, expression::Abs<A> expr, size_t size) {
//		auto floats = cached.floatScope();
//		auto *a = floats.real(expr.a, size);
//		vDSP_vabs(a, 1, pointer, 1, size);
//	}
};

}}; // namespace
