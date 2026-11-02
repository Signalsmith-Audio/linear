#if defined(__clang__) || defined(__GNUC__)
#	pragma GCC push_options
// Here's where we would put `#pragma GCC target("foo,bar")`, but SSE2 is always supported
#endif

namespace signalsmith { namespace linear {

namespace sse2 {

#define SSMF_BITS 32
#define SSMF_VF __m128
#define SSMF_VF_0 _mm_setzero_ps()
#define SSMF_VF_F(v) _mm_set1_ps(float(v))
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
#define SSMF_VF_LSHIFT_K(a, bits) _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(a), bits))
// Round-trip through int32 (round-to-nearest with the default MXCSR), valid for |v| < 2^31.
#define SSMF_VF_ROUND(v) _mm_cvtepi32_ps(_mm_cvtps_epi32(v))
#if defined(__clang__) || defined(__GNUC__)
	inline __m128 asmBarrier(__m128 v) {
		__asm__("" : "+x"(v)); // empty ASM block which the compiler doesn't re-associate through
		return v;
	}
#	define SSMF_VF_BARRIER(v) asmBarrier(v)
#endif
#define SSMF_NAME(fn) ssmf_##fn##f
#include "../signalsmith-mathfun.h"

#define SSMF_BITS 64
#define SSMF_VF __m128d
#define SSMF_VF_0 _mm_setzero_pd()
#define SSMF_VF_F(v) _mm_set1_pd(double(v))
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
#define SSMF_VF_LSHIFT_K(a, b) _mm_castsi128_pd(_mm_slli_epi64(_mm_castpd_si128(a), b))
#if defined(__clang__) || defined(__GNUC__)
	inline __m128d asmBarrier(__m128d v) {
		__asm__("" : "+x"(v));
		return v;
	}
#	define SSMF_VF_BARRIER(v) asmBarrier(v)
#endif
#define SSMF_NAME(fn) ssmf_##fn##d
#include "../signalsmith-mathfun.h"

	struct Sse2 {
		struct ComplexFloat {
			__m128 real, imag;
		};
		struct ComplexDouble {
			__m128d real, imag;
		};

		static constexpr size_t alignmentSize = 16;
		template<class V>
		static bool pointerAligned(const V *ptr) {
			return (size_t(ptr)&(alignmentSize - 1)) == 0;
		}

		template<class Expr>
		static void fillFloat(float *pointer, Expr expr, size_t size) {
			static constexpr size_t simdStep = alignmentSize/4;
			size_t simdSize = size&~(simdStep - 1);
			// Don't explode the code-paths - have one for "everything is aligned", one otherwise
			if (pointerAligned(pointer) && isAligned(expr)) {
				for (size_t i = 0; i < simdSize; i += simdStep) {
					_mm_store_ps(pointer + i, getFloat<true>(expr, i));
				}
			} else {
				for (size_t i = 0; i < simdSize; i += simdStep) {
					_mm_storeu_ps(pointer + i, getFloat<false>(expr, i));
				}
			}
			for (size_t i = simdSize; i < size; ++i) {
				pointer[i] = expr.get(i);
			}
		}
		template<class Expr>
		static void fillDouble(double *pointer, Expr expr, size_t size) {
			static constexpr size_t simdStep = alignmentSize/8;
			size_t simdSize = size&~(simdStep - 1);
			// Don't explode the code-paths - have one for "everything is aligned", one otherwise
			if (pointerAligned(pointer) && isAligned(expr)) {
				for (size_t i = 0; i < simdSize; i += simdStep) {
					_mm_store_pd(pointer + i, getDouble<true>(expr, i));
				}
			} else {
				for (size_t i = 0; i < simdSize; i += simdStep) {
					_mm_storeu_pd(pointer + i, getDouble<false>(expr, i));
				}
			}
			for (size_t i = simdSize; i < size; ++i) {
				pointer[i] = expr.get(i);
			}
		}
		template<class Expr>
		static void fillCFloat(std::complex<float> *pointer, Expr expr, size_t size) {
			static constexpr size_t simdStep = alignmentSize/4;
			size_t simdSize = size&~(simdStep - 1);
			// Don't explode the code-paths - have one for "everything is aligned", one otherwise
			if (pointerAligned(pointer) && isAligned(expr)) {
				for (size_t i = 0; i < simdSize; i += simdStep) {
					ComplexFloat c = getCFloat<true>(expr, i);
					float *f = reinterpret_cast<float *>(pointer + i);
					_mm_store_ps(f, _mm_unpacklo_ps(c.real, c.imag));
					_mm_store_ps(f + simdStep, _mm_unpackhi_ps(c.real, c.imag));
				}
			} else {
				for (size_t i = 0; i < simdSize; i += simdStep) {
					ComplexFloat c = getCFloat<false>(expr, i);
					float *f = reinterpret_cast<float *>(pointer + i);
					_mm_storeu_ps(f, _mm_unpacklo_ps(c.real, c.imag));
					_mm_storeu_ps(f + simdStep, _mm_unpackhi_ps(c.real, c.imag));
				}
			}
			for (size_t i = simdSize; i < size; ++i) {
				pointer[i] = expr.get(i);
			}
		}
		template<class Expr>
		static void fillCDouble(std::complex<double> *pointer, Expr expr, size_t size) {
			static constexpr size_t simdStep = alignmentSize/8;
			size_t simdSize = size&~(simdStep - 1);
			if (pointerAligned(pointer) && isAligned(expr)) {
				for (size_t i = 0; i < simdSize; i += simdStep) {
					ComplexDouble c = getCDouble<true>(expr, i);
					double *f = reinterpret_cast<double *>(pointer + i);
					_mm_store_pd(f, _mm_unpacklo_pd(c.real, c.imag));
					_mm_store_pd(f + simdStep, _mm_unpackhi_pd(c.real, c.imag));
				}
			} else {
				for (size_t i = 0; i < simdSize; i += simdStep) {
					ComplexDouble c = getCDouble<false>(expr, i);
					double *f = reinterpret_cast<double *>(pointer + i);
					_mm_storeu_pd(f, _mm_unpacklo_pd(c.real, c.imag));
					_mm_storeu_pd(f + simdStep, _mm_unpackhi_pd(c.real, c.imag));
				}
			}
			for (size_t i = simdSize; i < size; ++i) {
				pointer[i] = expr.get(i);
			}
		}
		template<class Expr>
		static void fillCFloat(float *real, float *imag, Expr expr, size_t size) {
			static constexpr size_t simdStep = alignmentSize/4;
			size_t simdSize = size&~(simdStep - 1);
			// Don't explode the code-paths - have one for "everything is aligned", one otherwise
			if (pointerAligned(real) && pointerAligned(imag) && isAligned(expr)) {
				for (size_t i = 0; i < simdSize; i += simdStep) {
					ComplexFloat c = getCFloat<true>(expr, i);
					_mm_store_ps(real + i, c.real);
					_mm_store_ps(imag + i, c.imag);
				}
			} else {
				for (size_t i = 0; i < simdSize; i += simdStep) {
					ComplexFloat c = getCFloat<false>(expr, i);
					_mm_storeu_ps(real + i, c.real);
					_mm_storeu_ps(imag + i, c.imag);
				}
			}
			for (size_t i = simdSize; i < size; ++i) {
				std::complex<float> c = expr.get(i);
				real[i] = c.real();
				imag[i] = c.imag();
			}
		}
		template<class Expr>
		static void fillCDouble(double *real, double *imag, Expr expr, size_t size) {
			static constexpr size_t simdStep = alignmentSize/8;
			size_t simdSize = size&~(simdStep - 1);
			if (pointerAligned(real) && pointerAligned(imag) && isAligned(expr)) {
				for (size_t i = 0; i < simdSize; i += simdStep) {
					ComplexDouble c = getCDouble<true>(expr, i);
					_mm_store_pd(real + i, c.real);
					_mm_store_pd(imag + i, c.imag);
				}
			} else {
				for (size_t i = 0; i < simdSize; i += simdStep) {
					ComplexDouble c = getCDouble<false>(expr, i);
					_mm_storeu_pd(real + i, c.real);
					_mm_storeu_pd(imag + i, c.imag);
				}
			}
			for (size_t i = simdSize; i < size; ++i) {
				std::complex<double> c = expr.get(i);
				real[i] = c.real();
				imag[i] = c.imag();
			}
		}
//#ifndef DEV_NO_FALLBACK
		// Fallback
		template<class Expr>
		static bool isAligned(Expr) {
			return false;
		}
		template<bool aligned, class Expr>
		static __m128 getFloat(Expr expr, size_t i) {
			return _mm_setr_ps(expr.get(i), expr.get(i + 1), expr.get(i + 2), expr.get(i + 3));
		}
		template<bool aligned, class Expr>
		static __m128d getDouble(Expr expr, size_t i) {
			return _mm_setr_pd(expr.get(i), expr.get(i + 1));
		}
		template<bool aligned, class Expr>
		static ComplexFloat getCFloat(Expr expr, size_t i) {
			std::complex<float> c0 = expr.get(i), c1 = expr.get(i + 1), c2 = expr.get(i + 2), c3 = expr.get(i + 3);
			return {
				_mm_setr_ps(c0.real(), c1.real(), c2.real(), c3.real()),
				_mm_setr_ps(c0.imag(), c1.imag(), c2.imag(), c3.imag())
			};
		}
		template<bool aligned, class Expr>
		static ComplexDouble getCDouble(Expr expr, size_t i) {
			std::complex<double> c0 = expr.get(i), c1 = expr.get(i + 1);
			return {_mm_setr_pd(c0.real(), c1.real()), _mm_setr_pd(c0.imag(), c1.imag())};
		}
//#endif

		// ReadableReal
		template<class V>
		static bool isAligned(expression::ReadableReal<V> expr) {
			return (size_t(expr.pointer)&15) == 0;
		}
		template<bool aligned>
		static __m128 getFloat(expression::ReadableReal<float> expr, size_t i) {
			if (aligned) {
				return _mm_load_ps(expr.pointer + i);
			} else {
				return _mm_loadu_ps(expr.pointer + i);
			}
		}
		template<bool aligned>
		static __m128d getDouble(expression::ReadableReal<double> expr, size_t i) {
			if (aligned) {
				return _mm_load_pd(expr.pointer + i);
			} else {
				return _mm_loadu_pd(expr.pointer + i);
			}
		}
		
		// Constant
		template<class V>
		static bool isAligned(expression::ConstantExpr<V>) {
			return true;
		}
		template<bool aligned, class V>
		static __m128 getFloat(expression::ConstantExpr<V> expr, size_t) {
			return _mm_set1_ps(float(expr.value));
		}
		template<bool aligned, class V>
		static __m128d getDouble(expression::ConstantExpr<V> expr, size_t) {
			return _mm_set1_pd(double(expr.value));
		}

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
#undef SIGNALSMITH_EXPR_BINARY

		// Expressions which map neatly to a single intrinsic
#define SIGNALSMITH_EXPR_UNARY(Name, mmFloat, mmDouble) \
		template<class A> \
		static bool isAligned(expression::Name<A> expr) { \
			return isAligned(expr.a); \
		} \
		template<bool aligned, class A, class=ItemTypeF<A>> \
		static __m128 getFloat(expression::Name<A> expr, size_t i) { \
			return mmFloat(getFloat<aligned>(expr.a, i)); \
		} \
		template<bool aligned, class A, class=ItemTypeD<A>> \
		static __m128d getDouble(expression::Name<A> expr, size_t i) { \
			return mmDouble(getDouble<aligned>(expr.a, i)); \
		}
		SIGNALSMITH_EXPR_UNARY(Norm, ssmf_pow2f, ssmf_pow2d)
		SIGNALSMITH_EXPR_UNARY(Abs, ssmf_absf, ssmf_absd)
		SIGNALSMITH_EXPR_UNARY(Neg, ssmf_negf, ssmf_negd)
		SIGNALSMITH_EXPR_UNARY(Sqrt, _mm_sqrt_ps, _mm_sqrt_pd)
		SIGNALSMITH_EXPR_UNARY(Log, ssmf_logf, ssmf_logd)
#undef SIGNALSMITH_EXPR_UNARY
#define SIGNALSMITH_EXPR_BINARY(Name, mmFloat, mmDouble) \
		template<bool aligned, class A, class B> \
		static __m128 getFloat(expression::Name<A, B> expr, size_t i) { \
			return mmFloat(getFloat<aligned>(expr.a, i), getFloat<aligned>(expr.b, i)); \
		} \
		template<bool aligned, class A, class B> \
		static __m128d getDouble(expression::Name<A, B> expr, size_t i) { \
			return mmDouble(getDouble<aligned>(expr.a, i), getDouble<aligned>(expr.b, i)); \
		}
		SIGNALSMITH_EXPR_BINARY(Add, _mm_add_ps, _mm_add_pd)
		SIGNALSMITH_EXPR_BINARY(Sub, _mm_sub_ps, _mm_sub_pd)
		SIGNALSMITH_EXPR_BINARY(Mul, _mm_mul_ps, _mm_mul_pd)
		SIGNALSMITH_EXPR_BINARY(Div, _mm_div_ps, _mm_div_pd)
		SIGNALSMITH_EXPR_BINARY(Max, _mm_max_ps, _mm_max_pd)
		SIGNALSMITH_EXPR_BINARY(Min, _mm_min_ps, _mm_min_pd)
		SIGNALSMITH_EXPR_BINARY(Dim, ssmf_dimf, ssmf_dimd)
		SIGNALSMITH_EXPR_BINARY(Copysign, ssmf_copysignf, ssmf_copysignd)
#undef SIGNALSMITH_EXPR_BINARY
	};
}

}}; // namespace

#if defined(__clang__) || defined(__GNUC__)
#	pragma GCC pop_options
#endif
