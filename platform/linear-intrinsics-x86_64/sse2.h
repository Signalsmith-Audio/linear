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
#define SSMF_VF_NOT(a) _mm_xor_ps(a, _mm_castsi128_ps(_mm_set1_epi32(-1)))
#define SSMF_VF_ADD(a, b) _mm_add_ps(a, b)
#define SSMF_VF_SUB(a, b) _mm_sub_ps(a, b)
#define SSMF_VF_MUL(a, b) _mm_mul_ps(a, b)
#define SSMF_VF_DIV(a, b) _mm_div_ps(a, b)
#define SSMF_VF_MAX(a, b) _mm_max_ps(a, b)
#define SSMF_VF_MIN(a, b) _mm_min_ps(a, b)
#define SSMF_VF_SQRT(v) _mm_sqrt_ps(v)
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
#define SSMF_VF_NOT(a) _mm_xor_pd(a, _mm_castsi128_pd(_mm_set1_epi32(-1)))
#define SSMF_VF_ADD(a, b) _mm_add_pd(a, b)
#define SSMF_VF_SUB(a, b) _mm_sub_pd(a, b)
#define SSMF_VF_MUL(a, b) _mm_mul_pd(a, b)
#define SSMF_VF_DIV(a, b) _mm_div_pd(a, b)
#define SSMF_VF_MAX(a, b) _mm_max_pd(a, b)
#define SSMF_VF_MIN(a, b) _mm_min_pd(a, b)
#define SSMF_VF_SQRT(v) _mm_sqrt_pd(v)
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

// Overloads, so that the functions in Sse2 below can be written once for float and double
namespace ops {
	template<class V> V k(double v);
	template<> inline __m128 k<__m128>(double v) {return _mm_set1_ps(float(v));}
	template<> inline __m128d k<__m128d>(double v) {return _mm_set1_pd(v);}
#define SIGNALSMITH_OPS1(name, mmFloat, mmDouble) \
	inline __m128 name(__m128 a) {return mmFloat(a);} \
	inline __m128d name(__m128d a) {return mmDouble(a);}
#define SIGNALSMITH_OPS2(name, mmFloat, mmDouble) \
	inline __m128 name(__m128 a, __m128 b) {return mmFloat(a, b);} \
	inline __m128d name(__m128d a, __m128d b) {return mmDouble(a, b);}
	SIGNALSMITH_OPS2(add, _mm_add_ps, _mm_add_pd)
	SIGNALSMITH_OPS2(sub, _mm_sub_ps, _mm_sub_pd)
	SIGNALSMITH_OPS2(mul, _mm_mul_ps, _mm_mul_pd)
	SIGNALSMITH_OPS2(div, _mm_div_ps, _mm_div_pd)
	SIGNALSMITH_OPS2(copysign, ssmf_copysignf, ssmf_copysignd)
	SIGNALSMITH_OPS1(abs, ssmf_absf, ssmf_absd)
	SIGNALSMITH_OPS1(sqrt, _mm_sqrt_ps, _mm_sqrt_pd)
	SIGNALSMITH_OPS1(pow2, ssmf_pow2f, ssmf_pow2d)
	SIGNALSMITH_OPS1(log, ssmf_logf, ssmf_logd)
#undef SIGNALSMITH_OPS1
#undef SIGNALSMITH_OPS2
}

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
			return pointerAligned(expr.pointer);
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

		// ReadableComplex
		template<class V>
		static bool isAligned(expression::ReadableComplex<V> expr) {
			return pointerAligned(expr.pointer);
		}
		template<bool aligned>
		static ComplexFloat getCFloat(expression::ReadableComplex<float> expr, size_t i) {
			// Two interleaved sets
			const float *fPtr = reinterpret_cast<const float *>(expr.pointer + i);
			__m128 a = aligned ? _mm_load_ps(fPtr) : _mm_loadu_ps(fPtr);
			__m128 b = aligned ? _mm_load_ps(fPtr + 4) : _mm_loadu_ps(fPtr + 4);
			// de-interleave
			__m128 re = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0)); // r0 r1 r2 r3
			__m128 im = _mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1)); // i0 i1 i2 i3
			return {re, im};
		}
		template<bool aligned>
		static ComplexDouble getCDouble(expression::ReadableComplex<double> expr, size_t i) {
			// Two interleaved sets
			const double *dPtr = reinterpret_cast<const double *>(expr.pointer + i);
			__m128d a = aligned ? _mm_load_pd(dPtr) : _mm_loadu_pd(dPtr);
			__m128d b = aligned ? _mm_load_pd(dPtr + 2) : _mm_loadu_pd(dPtr + 2);
			// de-interleave
			__m128d re = _mm_unpacklo_pd(a, b); // r0 r1
			__m128d im = _mm_unpackhi_pd(a, b); // i0 i1
			return {re, im};
		}

		// ReadableSplit
		template<class V>
		static bool isAligned(expression::ReadableSplit<V> expr) {
			return pointerAligned(expr.pointer.real) && pointerAligned(expr.pointer.imag);
		}
		template<bool aligned>
		static ComplexFloat getCFloat(expression::ReadableSplit<float> expr, size_t i) {
			if (aligned) {
				return {_mm_load_ps(expr.pointer.real + i), _mm_load_ps(expr.pointer.imag + i)};
			} else {
				return {_mm_loadu_ps(expr.pointer.real + i), _mm_loadu_ps(expr.pointer.imag + i)};
			}
		}
		template<bool aligned>
		static ComplexDouble getCDouble(expression::ReadableSplit<double> expr, size_t i) {
			if (aligned) {
				return {_mm_load_pd(expr.pointer.real + i), _mm_load_pd(expr.pointer.imag + i)};
			} else {
				return {_mm_loadu_pd(expr.pointer.real + i), _mm_loadu_pd(expr.pointer.imag + i)};
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

		// Functions which mathfun doesn't define, composed from ones it does
		template<class V>
		static V asinh(V x) {
			// log(|x| + sqrt(x² + 1)): absolute (not relative) accuracy near 0, and x² overflows for |x| > ~1e19 (or ~1e154 for double)
			V ax = ops::abs(x);
			V r = ops::log(ops::add(ax, ops::sqrt(ops::add(ops::pow2(ax), ops::k<V>(1)))));
			return ops::copysign(r, x);
		}
		template<class V>
		static V acosh(V x) {
			// log(x + sqrt((x - 1)(x + 1))): absolute (not relative) accuracy near 1, and overflows for x > ~1e19 (or ~1e154 for double)
			V s = ops::sqrt(ops::mul(ops::sub(x, ops::k<V>(1)), ops::add(x, ops::k<V>(1))));
			return ops::log(ops::add(x, s));
		}
		template<class V>
		static V atanh(V x) {
			// log((1 + x)/(1 - x))/2: absolute (not relative) accuracy near 0
			V ratio = ops::div(ops::add(ops::k<V>(1), x), ops::sub(ops::k<V>(1), x));
			return ops::mul(ops::log(ratio), ops::k<V>(0.5));
		}

		// No vector approximation: call the `std::` function on each lane
#define SIGNALSMITH_SCALAR_FALLBACK(fn) \
		static __m128 fn(__m128 v) { \
			alignas(16) float array[4]; \
			_mm_store_ps(array, v); \
			for (auto &x : array) x = std::fn(x); \
			return _mm_load_ps(array); \
		} \
		static __m128d fn(__m128d v) { \
			alignas(16) double array[2]; \
			_mm_store_pd(array, v); \
			for (auto &x : array) x = std::fn(x); \
			return _mm_load_pd(array); \
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
#undef SIGNALSMITH_EXPR_BINARY

		// Expressions which map neatly to a single intrinsic or SSMF function
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
		SIGNALSMITH_EXPR_UNARY(Cbrt, ssmf_cbrtf, ssmf_cbrtd)
		SIGNALSMITH_EXPR_UNARY(Ceil, ssmf_ceilf, ssmf_ceild)
		SIGNALSMITH_EXPR_UNARY(Floor, ssmf_floorf, ssmf_floord)
		SIGNALSMITH_EXPR_UNARY(Trunc, ssmf_truncf, ssmf_truncd)
		SIGNALSMITH_EXPR_UNARY(Round, ssmf_roundf, ssmf_roundd)
		SIGNALSMITH_EXPR_UNARY(Log, ssmf_logf, ssmf_logd)
		SIGNALSMITH_EXPR_UNARY(Log2, ssmf_log2f, ssmf_log2d)
		SIGNALSMITH_EXPR_UNARY(Log10, ssmf_log10f, ssmf_log10d)
		SIGNALSMITH_EXPR_UNARY(Exp, ssmf_expf, ssmf_expd)
		SIGNALSMITH_EXPR_UNARY(Exp2, ssmf_exp2f, ssmf_exp2d)
		SIGNALSMITH_EXPR_UNARY(Sin, ssmf_sinf, ssmf_sind)
		SIGNALSMITH_EXPR_UNARY(Cos, ssmf_cosf, ssmf_cosd)
		SIGNALSMITH_EXPR_UNARY(Tan, ssmf_tanf, ssmf_tand)
		SIGNALSMITH_EXPR_UNARY(Asin, ssmf_asinf, ssmf_asind)
		SIGNALSMITH_EXPR_UNARY(Acos, ssmf_acosf, ssmf_acosd)
		SIGNALSMITH_EXPR_UNARY(Atan, ssmf_atanf, ssmf_atand)
		SIGNALSMITH_EXPR_UNARY(Sinh, ssmf_sinhf, ssmf_sinhd)
		SIGNALSMITH_EXPR_UNARY(Cosh, ssmf_coshf, ssmf_coshd)
		SIGNALSMITH_EXPR_UNARY(Tanh, ssmf_tanhf, ssmf_tanhd)
		SIGNALSMITH_EXPR_UNARY(Asinh, asinh, asinh)
		SIGNALSMITH_EXPR_UNARY(Acosh, acosh, acosh)
		SIGNALSMITH_EXPR_UNARY(Atanh, atanh, atanh)
		SIGNALSMITH_EXPR_UNARY(Erf, ssmf_erff, ssmf_erfd)
		SIGNALSMITH_EXPR_UNARY(Erfc, ssmf_erfcf, ssmf_erfcd)
		SIGNALSMITH_EXPR_UNARY(Tgamma, tgamma, tgamma)
		SIGNALSMITH_EXPR_UNARY(Lgamma, lgamma, lgamma)
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
		SIGNALSMITH_EXPR_BINARY(Pow, ssmf_powf, ssmf_powd)
		SIGNALSMITH_EXPR_BINARY(Atan2, ssmf_atan2f, ssmf_atan2d)
		SIGNALSMITH_EXPR_BINARY(Hypot, ssmf_hypotf, ssmf_hypotd)
#undef SIGNALSMITH_EXPR_BINARY
	};
}

}}; // namespace

#if defined(__clang__) || defined(__GNUC__)
#	pragma GCC pop_options
#endif
