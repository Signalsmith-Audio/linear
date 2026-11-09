// This meta-arch picks SSE2, SSE4.1, AVX or AVX2+FMA at runtime

#include <immintrin.h>
#if defined(_MSC_VER) && !defined(__clang__)
#	include <intrin.h> // __cpuid, __cpuidex
#endif

#include "./x86-sse2.h"
#include "./x86-sse41.h"
#include "./x86-avx1.h"
#include "./x86-avx2.h"

namespace signalsmith { namespace linear { namespace simd_x86_dispatch {

enum class SimdArch{sse2, sse4_1, avx, avx2_fma};

static inline SimdArch detectArch() {
#ifdef SIGNALSMITH_LINEAR_FORCE_X86
	return SimdArch::SIGNALSMITH_LINEAR_FORCE_X86;
#endif

#if defined(_MSC_VER) && !defined(__clang__)
	int info[4];
	__cpuid(info, 1);
	bool sse41 = info[2]&(1<<19);
	// AVX also needs the OS to save the 256-bit registers: OSXSAVE, then XCR0 has the XMM and YMM bits
	bool osYmm = (info[2]&(1<<27)) && (_xgetbv(0)&6) == 6;
	bool avx = osYmm && (info[2]&(1<<28));
	bool fma = osYmm && (info[2]&(1<<12));
	__cpuidex(info, 7, 0);
	bool avx2 = avx && (info[1]&(1<<5));
#else
	__builtin_cpu_init(); // in case we're called from a static initialiser, before it's run
	// these check OS support for AVX as well
	bool sse41 = __builtin_cpu_supports("sse4.1");
	bool avx = __builtin_cpu_supports("avx");
	bool avx2 = __builtin_cpu_supports("avx2");
	bool fma = __builtin_cpu_supports("fma");
#endif
	if (avx2 && fma) return SimdArch::avx2_fma;
	if (avx) return SimdArch::avx;
	return sse41 ? SimdArch::sse4_1 : SimdArch::sse2;
}

struct SimdFillDispatch {
	SimdFillDispatch() : simdArch(detectArch()) {}

	template<class S, class Expr>
	void fillReal(S *pointer, Expr expr, size_t size) const {
		switch (simdArch) {
			case SimdArch::sse2: return simd_sse2::SimdFill::fillReal(pointer, expr, size);
			case SimdArch::sse4_1: return simd_sse41::SimdFill::fillReal(pointer, expr, size);
			case SimdArch::avx: return simd_avx1::SimdFill::fillReal(pointer, expr, size);
			case SimdArch::avx2_fma: return simd_avx2::SimdFill::fillReal(pointer, expr, size);
		}
	}
	template<class S, class Expr>
	void fillComplex(std::complex<S> *pointer, Expr expr, size_t size) const {
		switch (simdArch) {
			case SimdArch::sse2: return simd_sse2::SimdFill::fillComplex(pointer, expr, size);
			case SimdArch::sse4_1: return simd_sse41::SimdFill::fillComplex(pointer, expr, size);
			case SimdArch::avx: return simd_avx1::SimdFill::fillComplex(pointer, expr, size);
			case SimdArch::avx2_fma: return simd_avx2::SimdFill::fillComplex(pointer, expr, size);
		}
	}
	template<class S, class Expr>
	void fillSplit(S *real, S *imag, Expr expr, size_t size) const {
		switch (simdArch) {
			case SimdArch::sse2: return simd_sse2::SimdFill::fillSplit(real, imag, expr, size);
			case SimdArch::sse4_1: return simd_sse41::SimdFill::fillSplit(real, imag, expr, size);
			case SimdArch::avx: return simd_avx1::SimdFill::fillSplit(real, imag, expr, size);
			case SimdArch::avx2_fma: return simd_avx2::SimdFill::fillSplit(real, imag, expr, size);
		}
	}

private:
	SimdArch simdArch;
};

}}}; // namespace
