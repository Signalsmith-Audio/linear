#include "xsimd/xsimd.hpp"

#include <cstring> // std::memcpy
#include <cstdint> // uintptr_t
#include <type_traits>

#include "./basic-fill-warnings.h"

namespace signalsmith { namespace linear {

namespace _impl_fill {
	template<typename V, size_t size>
	XSIMD_INLINE V * getPrevAligned(V *array) {
		static_assert(sizeof(size_t) == sizeof(uintptr_t), "size_t != uintptr_t, which is valid but weird enough (on modern systems) to be suspicious");
		static constexpr uintptr_t alignBytes = sizeof(V)*size;
		static constexpr uintptr_t alignMask = ~(alignBytes - 1);
		uintptr_t asInt = uintptr_t(array);
		return reinterpret_cast<V *>(asInt&alignMask);
	}
	template<typename V, size_t size>
	XSIMD_INLINE V * getNextAligned(V *array) {
		return getPrevAligned<V, size>(array + (size - 1));
	}

#ifdef SIGNALSMITH_USE_XSIMD_DISPATCH
#	if defined(__i386__) || defined(__x86_64__)
#		define SIGNALSMITH_USE_XSIMD_DISPATCH_X86
#	elif defined(__ARM_NEON)
#		define SIGNALSMITH_USE_XSIMD_DISPATCH_ARM
#	endif

#	ifdef SIGNALSMITH_LINEAR_XSIMD_DISPATCH_ARCH
// Declare a concrete specialisation for one particular architecture
#		define SIGNALSMITH_LINEAR_ARCH_DISPATCH(fnName, ...) \
			template void fnName<SIGNALSMITH_LINEAR_XSIMD_DISPATCH_ARCH>(__VA_ARGS__, SIGNALSMITH_LINEAR_XSIMD_DISPATCH_ARCH);
#	elif defined(SIGNALSMITH_USE_XSIMD_DISPATCH_X86)
// List the arch-specific specialisations which will be defined in other units
#		define SIGNALSMITH_LINEAR_ARCH_DISPATCH(fnName, ...) \
			extern template void fnName<xsimd::sse2>(__VA_ARGS__, xsimd::sse2); \
			extern template void fnName<xsimd::sse3>(__VA_ARGS__, xsimd::sse3); \
			extern template void fnName<xsimd::sse4_2>(__VA_ARGS__, xsimd::sse4_2); \
			extern template void fnName<xsimd::avx>(__VA_ARGS__, xsimd::avx); \
			extern template void fnName<xsimd::avx2>(__VA_ARGS__, xsimd::avx2); \
			extern template void fnName<xsimd::avx512f>(__VA_ARGS__, xsimd::avx512f);
#	elif defined(SIGNALSMITH_USE_XSIMD_DISPATCH_ARM)
#		define SIGNALSMITH_LINEAR_ARCH_DISPATCH(fnName, ...) \
			extern template void fnName<xsimd::neon>(__VA_ARGS__, xsimd::neon); \
			extern template void fnName<xsimd::neon64>(__VA_ARGS__, xsimd::neon64);
#	endif
#else
#	define SIGNALSMITH_LINEAR_ARCH_DISPATCH(...)
#endif

	template<class Arch, class V>
	XSIMD_INLINE void fillConstantTyped(V *array, V constantValue, size_t size) {
		using Batch = xsimd::batch<V, Arch>;
		V *arrayEnd = array + size;
		V *alignedStart = getNextAligned<V, Batch::size>(array);
		V *alignedEnd = getPrevAligned<V, Batch::size>(arrayEnd);
		if (alignedEnd <= alignedStart) {
			// Too short to have an aligned section
			while (array != arrayEnd) {
				*array = constantValue;
				++array;
			}
			return;
		}
		while (array < alignedStart) {
			*array = constantValue;
			++array;
		}
		Batch constantBatch{constantValue};
		while (array < alignedStart) {
			constantBatch.store_aligned(array);
			array += Batch::size;
		}
		while (array < arrayEnd) {
			*array = constantValue;
			++array;
		}
	}
	template<class Arch>
	void fillConstant(float *array, float constantValue, size_t size, Arch) {
		if constexpr(Arch::supported()) fillConstantTyped<Arch, float>(array, constantValue, size);
	}
	template<class Arch>
	void fillConstant(double *array, double constantValue, size_t size, Arch) {
		if constexpr(Arch::supported()) fillConstantTyped<Arch, double>(array, constantValue, size);
	}
	template<class Arch>
	void fillConstant(std::complex<float> *array, std::complex<float> constantValue, size_t size, Arch) {
		if constexpr(Arch::supported()) fillConstantTyped<Arch, std::complex<float>>(array, constantValue, size);
	}
	template<class Arch>
	void fillConstant(std::complex<double> *array, std::complex<double> constantValue, size_t size, Arch) {
		if constexpr(Arch::supported()) fillConstantTyped<Arch, std::complex<double>>(array, constantValue, size);
	}
	SIGNALSMITH_LINEAR_ARCH_DISPATCH(fillConstant, float *, float, size_t)
	SIGNALSMITH_LINEAR_ARCH_DISPATCH(fillConstant, double *, double, size_t)
	SIGNALSMITH_LINEAR_ARCH_DISPATCH(fillConstant, std::complex<float> *, std::complex<float>, size_t)
	SIGNALSMITH_LINEAR_ARCH_DISPATCH(fillConstant, std::complex<double> *, std::complex<double>, size_t)
	
	template<class Arch, class Pointer, class Expr>
	XSIMD_INLINE void fillBasic(Pointer pointer, Expr expr, size_t size) {
		basicFillWarning<Expr>();
		for (size_t i = 0; i < size; ++i) {
			pointer[i] = expr.get(i);
		}
	}
	template<class Arch, class V, class Expr>
	XSIMD_INLINE void fillBasic(SplitPointer<V> pointer, Expr expr, size_t size) {
		basicFillWarning<Expr>();
		using Complex = typename SplitPointer<V>::Complex;
		for (size_t i = 0; i < size; ++i) {
			Complex c = expr.get(i);
			pointer.real[i] = c.real();
			pointer.imag[i] = c.imag();
		}
	}

	template<class Arch, class V, class Expr>
	void fill(RealPointer<V> pointer, Expr expr, size_t size, Arch) {
		fillBasic<Arch>(pointer, expr, size);
	}
	template<class Arch, class V, class Expr>
	void fill(ComplexPointer<V> pointer, Expr expr, size_t size, Arch) {
		fillBasic<Arch>(pointer, expr, size);
	}
	template<class Arch, class V, class Expr>
	void fill(SplitPointer<V> pointer, Expr expr, size_t size, Arch) {
		fillBasic<Arch>(pointer, expr, size);
	}

	template<class Arch>
	struct GetBatch {
		template<class V>
		XSIMD_INLINE static xsimd::batch<V, Arch> getBatch(const expression::ReadableReal<V> &expr, size_t index) {
			return xsimd::batch<V, Arch>::load_unaligned(expr.pointer + index);
		}
		template<class V>
		XSIMD_INLINE static xsimd::batch<std::complex<V>, Arch> getBatch(const expression::ReadableComplex<V> &expr, size_t index) {
			return xsimd::batch<std::complex<V>, Arch>::load_unaligned(expr.pointer + index);
		}
		template<class V>
		XSIMD_INLINE static xsimd::batch<std::complex<V>, Arch> getBatch(const expression::ReadableSplit<V> &expr, size_t index) {
			return xsimd::batch<std::complex<V>, Arch>::load_unaligned(expr.pointer.real + index, expr.pointer.imag + index);
		}

		template<class Expr>
		XSIMD_INLINE static auto getBatch(const expression::Abs<Expr> &expr, size_t index) -> xsimd::batch<std::remove_cv_t<decltype(expr.get(0))>, Arch> {
			return xsimd::abs(getBatch(expr.a, index));
		}
		template<class Expr>
		XSIMD_INLINE static auto getBatch(const expression::Norm<Expr> &expr, size_t index) -> xsimd::batch<std::remove_cv_t<decltype(expr.get(0))>, Arch> {
			return xsimd::norm(getBatch(expr.a, index));
		}
		template<class Expr>
		XSIMD_INLINE static auto getBatch(const expression::Exp<Expr> &expr, size_t index) -> xsimd::batch<std::remove_cv_t<decltype(expr.get(0))>, Arch> {
			return xsimd::exp(getBatch(expr.a, index));
		}
		template<class Expr>
		XSIMD_INLINE static auto getBatch(const expression::Exp2<Expr> &expr, size_t index) -> xsimd::batch<std::remove_cv_t<decltype(expr.get(0))>, Arch> {
			return xsimd::exp2(getBatch(expr.a, index));
		}
		template<class Expr>
		XSIMD_INLINE static auto getBatch(const expression::Log<Expr> &expr, size_t index) -> xsimd::batch<std::remove_cv_t<decltype(expr.get(0))>, Arch> {
			return xsimd::log(getBatch(expr.a, index));
		}
		template<class Expr>
		XSIMD_INLINE static auto getBatch(const expression::Log2<Expr> &expr, size_t index) -> xsimd::batch<std::remove_cv_t<decltype(expr.get(0))>, Arch> {
			return xsimd::log2(getBatch(expr.a, index));
		}
		template<class Expr>
		XSIMD_INLINE static auto getBatch(const expression::Log10<Expr> &expr, size_t index) -> xsimd::batch<std::remove_cv_t<decltype(expr.get(0))>, Arch> {
			return xsimd::log10(getBatch(expr.a, index));
		}
		template<class Expr>
		XSIMD_INLINE static auto getBatch(const expression::Neg<Expr> &expr, size_t index) -> xsimd::batch<std::remove_cv_t<decltype(expr.get(0))>, Arch> {
			return -getBatch(expr.a, index);
		}
		template<class Expr>
		XSIMD_INLINE static auto getBatch(const expression::Floor<Expr> &expr, size_t index) -> xsimd::batch<std::remove_cv_t<decltype(expr.get(0))>, Arch> {
			return xsimd::floor(getBatch(expr.a, index));
		}
		template<class Expr>
		XSIMD_INLINE static auto getBatch(const expression::Ceil<Expr> &expr, size_t index) -> xsimd::batch<std::remove_cv_t<decltype(expr.get(0))>, Arch> {
			return xsimd::ceil(getBatch(expr.a, index));
		}
		template<class Expr>
		XSIMD_INLINE static auto getBatch(const expression::Conj<Expr> &expr, size_t index) -> xsimd::batch<std::remove_cv_t<decltype(expr.get(0))>, Arch> {
			return xsimd::conj(getBatch(expr.a, index));
		}
		template<class Expr>
		XSIMD_INLINE static auto getBatch(const expression::Sqrt<Expr> &expr, size_t index) -> xsimd::batch<std::remove_cv_t<decltype(expr.get(0))>, Arch> {
			return xsimd::sqrt(getBatch(expr.a, index));
		}
		template<class Expr>
		XSIMD_INLINE static auto getBatch(const expression::Cbrt<Expr> &expr, size_t index) -> xsimd::batch<std::remove_cv_t<decltype(expr.get(0))>, Arch> {
			return xsimd::cbrt(getBatch(expr.a, index));
		}

		template<class ExprA, class ExprB>
		XSIMD_INLINE static auto getBatch(const expression::Add<ExprA, ExprB> &expr, size_t index) -> xsimd::batch<std::remove_cv_t<decltype(expr.get(0))>, Arch> {
			return getBatch(expr.a, index) + getBatch(expr.b, index);
		}
		template<class ExprA, class ExprB>
		XSIMD_INLINE static auto getBatch(const expression::Sub<ExprA, ExprB> &expr, size_t index) -> xsimd::batch<std::remove_cv_t<decltype(expr.get(0))>, Arch> {
			return getBatch(expr.a, index) - getBatch(expr.b, index);
		}
		template<class ExprA, class ExprB>
		XSIMD_INLINE static auto getBatch(const expression::Mul<ExprA, ExprB> &expr, size_t index) -> xsimd::batch<std::remove_cv_t<decltype(expr.get(0))>, Arch> {
			return getBatch(expr.a, index) * getBatch(expr.b, index);
		}
		template<class ExprA, class ExprB>
		XSIMD_INLINE static auto getBatch(const expression::Div<ExprA, ExprB> &expr, size_t index) -> xsimd::batch<std::remove_cv_t<decltype(expr.get(0))>, Arch> {
			return getBatch(expr.a, index) / getBatch(expr.b, index);
		}
		template<class ExprA, class ExprB>
		XSIMD_INLINE static auto getBatch(const expression::Max<ExprA, ExprB> &expr, size_t index) -> xsimd::batch<std::remove_cv_t<decltype(expr.get(0))>, Arch> {
			return xsimd::max(getBatch(expr.a, index), getBatch(expr.b, index));
		}
		template<class ExprA, class ExprB>
		XSIMD_INLINE static auto getBatch(const expression::Min<ExprA, ExprB> &expr, size_t index) -> xsimd::batch<std::remove_cv_t<decltype(expr.get(0))>, Arch> {
			return xsimd::min(getBatch(expr.a, index), getBatch(expr.b, index));
		}
	};

	template<class Arch, class V, class Expr>
	XSIMD_INLINE void fillSpecialisedReal(V *array, Expr expr, size_t size) {
		using Batch = xsimd::batch<V, Arch>;
		V *arrayEnd = array + size;
		V *alignedStart = getNextAligned<V, Batch::size>(array);
		V *alignedEnd = getPrevAligned<V, Batch::size>(arrayEnd);
		if (alignedEnd <= alignedStart) {
			// Too short to have an aligned section
			size_t i = 0;
			while (array + i != arrayEnd) {
				*(array + i) = V(expr.get(i));
				++i;
			}
			return;
		}
		size_t i = 0;
		while (array + i < alignedStart) {
			*(array + i) = V(expr.get(i));
			++i;
		}
		while (array + i < alignedEnd) {
			auto batch = GetBatch<Arch>::getBatch(expr, i);
			xsimd::batch_cast<V>(batch).store_aligned(array + i);
			i += Batch::size;
		}
		while (array + i < arrayEnd) {
			*(array + i) = V(expr.get(i));
			++i;
		}
	}

	template<class Arch, class V, class Expr>
	XSIMD_INLINE void fillSpecialisedComplex(std::complex<V> *array, Expr expr, size_t size) {
		using C = std::complex<V>;
		using Batch = xsimd::batch<C, Arch>;
		C *arrayEnd = array + size;
		C *alignedStart = getNextAligned<C, Batch::size>(array);
		C *alignedEnd = getPrevAligned<C, Batch::size>(arrayEnd);
		if (alignedEnd <= alignedStart) {
			// Too short to have an aligned section
			size_t i = 0;
			while (array + i != arrayEnd) {
				*(array + i) = expr.get(i);
				++i;
			}
			return;
		}
		size_t i = 0;
		while (array + i < alignedStart) {
			*(array + i) = C(expr.get(i));
			++i;
		}
		while (array + i < alignedEnd) {
			auto batch = GetBatch<Arch>::getBatch(expr, i);
			xsimd::batch_cast<C>(batch).store_aligned(array + i);
			i += Batch::size;
		}
		while (array + i < arrayEnd) {
			*(array + i) = C(expr.get(i));
			++i;
		}
	}
	template<class Arch, class V, class Expr>
	XSIMD_INLINE void fillSpecialisedSplit(SplitPointer<V> array, Expr expr, size_t size) {
		using C = std::complex<V>;
		using Batch = xsimd::batch<C, Arch>;
		V *real = array.real;
		V *realEnd = real + size;
		V *realAlignedStart = getNextAligned<V, Batch::size>(real);
		V *realAlignedEnd = getPrevAligned<V, Batch::size>(realEnd);
		
		bool alignmentReal = uintptr_t(array.real)&(Batch::size*sizeof(V) - 1);
		bool alignmentImag = uintptr_t(array.imag)&(Batch::size*sizeof(V) - 1);
		if (realAlignedEnd <= realAlignedStart || (alignmentReal != alignmentImag)) {
			// Either too short to have an aligned section, or the real/imaginary parts have unequal alignment
			size_t i = 0;
			while (real + i != realEnd) {
				array[i] = C(expr.get(i));
				++i;
			}
			return;
		}
		size_t i = 0;
		while (real + i < realAlignedStart) {
			array[i] = C(expr.get(i));
			++i;
		}
		while (real + i < realAlignedEnd) {
			auto batch = GetBatch<Arch>::getBatch(expr, i);
			xsimd::batch_cast<C>(batch).store_aligned(real + i, array.imag + i);
			i += Batch::size;
		}
		while (real + i < realEnd) {
			array[i] = C(expr.get(i));
			++i;
		}
	}
	
#define SIGNALSMITH_LINEAR_ARCH_SPECIALISE_REAL(V, ...) \
	template<class Arch> \
	void fill(RealPointer<V> pointer, __VA_ARGS__ expr, size_t size, Arch) { \
		if constexpr(Arch::supported()) fillSpecialisedReal<Arch>(pointer, expr, size); \
	} \
	template<class Arch> \
	void fill(ComplexPointer<V> pointer, __VA_ARGS__ expr, size_t size, Arch) { \
		if constexpr(Arch::supported()) fillSpecialisedComplex<Arch>(pointer, expr, size); \
	} \
	SIGNALSMITH_LINEAR_ARCH_DISPATCH(fill, RealPointer<V>, __VA_ARGS__, size_t) \
	SIGNALSMITH_LINEAR_ARCH_DISPATCH(fill, ComplexPointer<V>, __VA_ARGS__, size_t) \
	SIGNALSMITH_LINEAR_ARCH_DISPATCH(fill, SplitPointer<V>, __VA_ARGS__, size_t)
	
#define SIGNALSMITH_LINEAR_ARCH_SPECIALISE_COMPLEX(V, ...) \
	template<class Arch> \
	void fill(ComplexPointer<V> pointer, __VA_ARGS__ expr, size_t size, Arch) { \
		if constexpr(Arch::supported()) fillSpecialisedComplex<Arch>(pointer, expr, size); \
	} \
	template<class Arch> \
	void fill(SplitPointer<V> pointer, __VA_ARGS__ expr, size_t size, Arch) { \
		if constexpr(Arch::supported()) fillSpecialisedSplit<Arch>(pointer, expr, size); \
	} \
	SIGNALSMITH_LINEAR_ARCH_DISPATCH(fill, ComplexPointer<V>, __VA_ARGS__, size_t) \
	SIGNALSMITH_LINEAR_ARCH_DISPATCH(fill, SplitPointer<V>, __VA_ARGS__, size_t)

// Real unary operators
#define SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Name) \
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_REAL(float, expression::Name<expression::ReadableReal<float>>); \
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_REAL(double, expression::Name<expression::ReadableReal<double>>);
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Abs)
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Floor)
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Ceil)
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Neg)
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Exp)
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Exp2)
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Log)
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Log2)
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Log10)
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Sqrt)
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Cbrt)
#undef SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP

// Complex -> real unary operators
#define SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Name) \
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_REAL(float, expression::Name<expression::ReadableComplex<float>>); \
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_REAL(double, expression::Name<expression::ReadableComplex<double>>); \
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_REAL(float, expression::Name<expression::ReadableSplit<float>>); \
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_REAL(double, expression::Name<expression::ReadableSplit<double>>);
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Abs)
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Norm)
#undef SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP

// Complex unary operators
#define SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Name) \
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_COMPLEX(float, expression::Name<expression::ReadableComplex<float>>); \
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_COMPLEX(double, expression::Name<expression::ReadableComplex<double>>); \
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_COMPLEX(float, expression::Name<expression::ReadableSplit<float>>); \
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_COMPLEX(double, expression::Name<expression::ReadableSplit<double>>);
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Conj)
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Neg)
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Exp)
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Log)
#undef SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP

// Real binary operators
#define SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Name) \
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_REAL(float, expression::Name<expression::ReadableReal<float>, expression::ReadableReal<float>>); \
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_REAL(double, expression::Name<expression::ReadableReal<double>, expression::ReadableReal<double>>);
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Max)
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Min)
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Add)
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Sub)
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Mul)
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Div)
#undef SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP

// Complex binary operators
#define SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Name) \
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_COMPLEX(float, expression::Name<expression::ReadableComplex<float>, expression::ReadableComplex<float>>); \
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_COMPLEX(double, expression::Name<expression::ReadableComplex<double>, expression::ReadableComplex<double>>); \
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_COMPLEX(float, expression::Name<expression::ReadableSplit<float>, expression::ReadableSplit<float>>); \
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_COMPLEX(double, expression::Name<expression::ReadableSplit<double>, expression::ReadableSplit<double>>);
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Add)
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Sub)
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Mul)
	SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP(Div)
#undef SIGNALSMITH_LINEAR_ARCH_SPECIALISE_OP

}
#undef SIGNALSMITH_LINEAR_ARCH_DISPATCH

template<>
struct LinearImpl<true> : public LinearImplBase<true> {
	using Base = LinearImplBase<true>;

	LinearImpl() : Base(this) {
		basicFillWarningReset();
		chooseArchitecture();
	}

	template<class V>
	void reserve(size_t) {}

	template<class Pointer, class Expr>
	void fill(Pointer pointer, Expr expr, size_t size) {
		fillExprDispatch(pointer, expr, size);
	}

	template<class Pointer, class Expr>
	void fill(Pointer pointer, Expression<Expr> expr, size_t size) {
		return fillExprDispatch(pointer, (Expr &)expr, size);
	};
	template<class Pointer, class Expr>
	void fill(Pointer pointer, WritableExpression<Expr> expr, size_t size) {
		return fillExprDispatch(pointer, (Expr &)expr, size);
	};

private:

#ifdef SIGNALSMITH_USE_XSIMD_DISPATCH_X86
	enum class Arch{sse2, sse3, sse4_2, avx, avx2, avx512f, avx512er};
	Arch bestArch = Arch::sse2;

	void chooseArchitecture() {
		auto available = xsimd::available_architectures();
		if (available.has(xsimd::sse3{})) bestArch = Arch::sse3;
		if (available.has(xsimd::sse4_2{})) bestArch = Arch::sse4_2;
		if (available.has(xsimd::avx{})) bestArch = Arch::avx;
		if (available.has(xsimd::avx2{})) bestArch = Arch::avx2;
		if (available.has(xsimd::avx512f{})) bestArch = Arch::avx512f;
		if (available.has(xsimd::avx512er{})) bestArch = Arch::avx512er;
	}
	
	template<class Pointer, class Expr>
	void fillExprDispatch(Pointer pointer, Expr expr, size_t size) {
		switch(bestArch) {
			case Arch::sse2: return fillExpr<xsimd::sse2>(pointer, expr, size);
			case Arch::sse3: return fillExpr<xsimd::sse3>(pointer, expr, size);
			case Arch::sse4_2: return fillExpr<xsimd::sse4_2>(pointer, expr, size);
			case Arch::avx: return fillExpr<xsimd::avx>(pointer, expr, size);
			case Arch::avx2: return fillExpr<xsimd::avx2>(pointer, expr, size);
			case Arch::avx512f: return fillExpr<xsimd::avx512f>(pointer, expr, size);
			case Arch::avx512er: return fillExpr<xsimd::avx512er>(pointer, expr, size);
		}
	}
#elif defined(SIGNALSMITH_USE_XSIMD_DISPATCH_ARM)
	enum class Arch{neon, neon64};
	Arch bestArch = Arch::neon; // TODO: better fallback for non-NEON systems

	void chooseArchitecture() {
		auto available = xsimd::available_architectures();
		if (available.has(xsimd::neon64{})) bestArch = Arch::neon64;
	}
	
	template<class Pointer, class Expr>
	void fillExprDispatch(Pointer pointer, Expr expr, size_t size) {
		switch(bestArch) {
			case Arch::neon: return fillExpr<xsimd::neon>(pointer, expr, size);
			case Arch::neon64: return fillExpr<xsimd::neon64>(pointer, expr, size);
		}
	}
#else
	void chooseArchitecture() {/*nothing*/}

	template<class Pointer, class Expr>
	XSIMD_INLINE void fillExprDispatch(Pointer pointer, Expr expr, size_t size) {
		// Use best guaranteed arch
		fillExpr<xsimd::best_arch>(pointer, expr, size);
	}
#endif

	//---- Everything below this point only gets compiled for targets where Arch is fully supported ----//

	// Generic fill
	template<class Arch, class V, class Expr>
	XSIMD_INLINE void fillExpr(RealPointer<V> pointer, Expr expr, size_t size) {
		_impl_fill::fill<Arch>(pointer, expr, size, Arch{});
	}
	template<class Arch, class V, class Expr>
	XSIMD_INLINE void fillExpr(ComplexPointer<V> pointer, Expr expr, size_t size) {
		_impl_fill::fill<Arch>(pointer, expr, size, Arch{});
	}
	template<class Arch, class V, class Expr>
	XSIMD_INLINE void fillExpr(SplitPointer<V> pointer, Expr expr, size_t size) {
		_impl_fill::fill<Arch>(pointer, expr, size, Arch{});
	}
	// Filling a split-complex vector with real values won't hit the specialisations below, so we handle it here
	template<class Arch, class Expr>
	XSIMD_INLINE ItemType<Expr, float, void> fillExpr(SplitPointer<float> pointer, Expr expr, size_t size) {
		_impl_fill::fill<Arch>(pointer.real, expr, size, Arch{});
		_impl_fill::fillConstant(pointer.imag, 0.0f, size, Arch{});
	}
	template<class Arch, class Expr>
	XSIMD_INLINE ItemType<Expr, double, void> fillExpr(SplitPointer<double> pointer, Expr expr, size_t size) {
		_impl_fill::fill<Arch>(pointer.real, expr, size, Arch{});
		_impl_fill::fillConstant(pointer.imag, 0.0, size, Arch{});
	}
	
	// Copying from existing pointer
	template<class Arch>
	XSIMD_INLINE void fillExpr(RealPointer<float> pointer, expression::ReadableReal<float> expr, size_t size) {
		std::memcpy(pointer, expr.pointer, size*sizeof(float));
	}
	template<class Arch>
	XSIMD_INLINE void fillExpr(RealPointer<float> pointer, WritableReal<float> expr, size_t size) {
		std::memcpy(pointer, expr.pointer, size*sizeof(float));
	}
	template<class Arch>
	XSIMD_INLINE void fillExpr(RealPointer<double> pointer, expression::ReadableReal<double> expr, size_t size) {
		std::memcpy(pointer, expr.pointer, size*sizeof(double));
	}
	template<class Arch>
	XSIMD_INLINE void fillExpr(RealPointer<double> pointer, WritableReal<double> expr, size_t size) {
		std::memcpy(pointer, expr.pointer, size*sizeof(double));
	}
	template<class Arch>
	XSIMD_INLINE void fillExpr(ComplexPointer<float> pointer, expression::ReadableComplex<float> expr, size_t size) {
		std::memcpy(pointer, expr.pointer, size*sizeof(std::complex<float>));
	}
	template<class Arch>
	XSIMD_INLINE void fillExpr(ComplexPointer<float> pointer, WritableComplex<float> expr, size_t size) {
		std::memcpy(pointer, expr.pointer, size*sizeof(std::complex<float>));
	}
	template<class Arch>
	XSIMD_INLINE void fillExpr(ComplexPointer<double> pointer, expression::ReadableComplex<double> expr, size_t size) {
		std::memcpy(pointer, expr.pointer, size*sizeof(std::complex<double>));
	}
	template<class Arch>
	XSIMD_INLINE void fillExpr(ComplexPointer<double> pointer, WritableComplex<double> expr, size_t size) {
		std::memcpy(pointer, expr.pointer, size*sizeof(std::complex<double>));
	}
	
	// Filling with a constant
	template<class Arch, class V>
	XSIMD_INLINE void fillExpr(RealPointer<float> pointer, expression::ConstantExpr<V> expr, size_t size) {
		float constantValue = expr.value;
		_impl_fill::fillConstant(pointer, constantValue, size, Arch{});
	}
	template<class Arch, class V>
	XSIMD_INLINE void fillExpr(RealPointer<double> pointer, expression::ConstantExpr<V> expr, size_t size) {
		double constantValue = expr.value;
		_impl_fill::fillConstant(pointer, constantValue, size, Arch{});
	}
	template<class Arch, class V>
	XSIMD_INLINE void fillExpr(ComplexPointer<float> pointer, expression::ConstantExpr<V> expr, size_t size) {
		std::complex<float> constantValue = expr.value;
		_impl_fill::fillConstant(pointer, constantValue, size, Arch{});
	}
	template<class Arch, class V>
	XSIMD_INLINE void fillExpr(ComplexPointer<double> pointer, expression::ConstantExpr<V> expr, size_t size) {
		std::complex<double> constantValue = expr.value;
		_impl_fill::fillConstant(pointer, constantValue, size, Arch{});
	}
	template<class Arch, class V>
	XSIMD_INLINE void fillExpr(SplitPointer<float> pointer, expression::ConstantExpr<V> expr, size_t size) {
		std::complex<float> v = expr.value;
		float vr = v.real(), vi = v.imag();
		_impl_fill::fillConstant(pointer.real, vr, size, Arch{});
		_impl_fill::fillConstant(pointer.imag, vi, size, Arch{});
	}
	template<class Arch, class V>
	XSIMD_INLINE void fillExpr(SplitPointer<double> pointer, expression::ConstantExpr<V> expr, size_t size) {
		std::complex<double> v = expr.value;
		double vr = v.real(), vi = v.imag();
		_impl_fill::fillConstant(pointer.real, vr, size, Arch{});
		_impl_fill::fillConstant(pointer.imag, vi, size, Arch{});
	}
};

}}; // namespace

