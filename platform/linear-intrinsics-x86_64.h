#include <cstring> // std::memcpy
#include <limits>
#include <type_traits>

#include <immintrin.h>
#include "./linear-intrinsics/x86-sse2.h"
#include "./linear-intrinsics/x86-sse41.h"
#if defined(_MSC_VER) && !defined(__clang__)
#	include <intrin.h> // __cpuid
#endif

#include "./basic-fill-warnings.h"

namespace signalsmith { namespace linear {

template<>
struct LinearImpl<true> : public LinearImplBase<true> {
	using Base = LinearImplBase<true>;

	LinearImpl() : Base(this), simdArch(detectArch()) {
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
	enum class SimdArch{sse2, sse4_1};
	SimdArch simdArch = SimdArch::sse2;

	static SimdArch detectArch() {
#ifdef SIGNALSMITH_LINEAR_FORCE_X86
		return SimdArch::SIGNALSMITH_LINEAR_FORCE_X86;
#endif

#if defined(_MSC_VER) && !defined(__clang__)
		int info[4];
		__cpuid(info, 1);
		bool sse41 = info[2]&(1<<19);
#else
		bool sse41 = __builtin_cpu_supports("sse4.1");
#endif
		return sse41 ? SimdArch::sse4_1 : SimdArch::sse2;
	}

	using Sse2Fill = simd_sse2::SimdFill;
	using Sse41Fill = simd_sse41::SimdFill;

	template<class Pointer, class Expr>
	void fillExpr(Pointer pointer, Expr expr, size_t size) {
		fillBasic(pointer, expr, size);
	}

	// Most generic fill
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

	template<class Expr>
	void fillBasic(RealPointer<float> pointer, Expr expr, size_t size) {
		switch (simdArch) {
			case SimdArch::sse2: return Sse2Fill::fillReal(pointer, expr, size);
			case SimdArch::sse4_1: return Sse41Fill::fillReal(pointer, expr, size);
		}
	}
	template<class Expr>
	void fillBasic(RealPointer<double> pointer, Expr expr, size_t size) {
		switch (simdArch) {
			case SimdArch::sse2: return Sse2Fill::fillReal(pointer, expr, size);
			case SimdArch::sse4_1: return Sse41Fill::fillReal(pointer, expr, size);
		}
	}
	template<class Expr>
	ItemTypeCF<Expr> fillBasic(ComplexPointer<float> pointer, Expr expr, size_t size) {
		switch (simdArch) {
			case SimdArch::sse2: return Sse2Fill::fillComplex(pointer, expr, size);
			case SimdArch::sse4_1: return Sse41Fill::fillComplex(pointer, expr, size);
		}
	}
	template<class Expr>
	ItemTypeCD<Expr> fillBasic(ComplexPointer<double> pointer, Expr expr, size_t size) {
		switch (simdArch) {
			case SimdArch::sse2: return Sse2Fill::fillComplex(pointer, expr, size);
			case SimdArch::sse4_1: return Sse41Fill::fillComplex(pointer, expr, size);
		}
	}
	template<class Expr>
	ItemTypeCF<Expr> fillBasic(SplitPointer<float> pointer, Expr expr, size_t size) {
		switch (simdArch) {
			case SimdArch::sse2: return Sse2Fill::fillSplit(pointer.real, pointer.imag, expr, size);
			case SimdArch::sse4_1: return Sse41Fill::fillSplit(pointer.real, pointer.imag, expr, size);
		}
	}
	template<class Expr>
	ItemTypeCD<Expr> fillBasic(SplitPointer<double> pointer, Expr expr, size_t size) {
		switch (simdArch) {
			case SimdArch::sse2: return Sse2Fill::fillSplit(pointer.real, pointer.imag, expr, size);
			case SimdArch::sse4_1: return Sse41Fill::fillSplit(pointer.real, pointer.imag, expr, size);
		}
	}
	
	template<typename V>
	void clear(V *v, size_t size) {
		for (size_t i = 0; i < size; ++i) v[i] = 0;
	}
	void clear(float *v, size_t size) {
		std::memset(static_cast<void *>(v), 0, size*4);
	}
	void clear(double *v, size_t size) {
		std::memset(static_cast<void *>(v), 0, size*8);
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
};

}}; // namespace
