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
	template<class V> V smallestNormal();
	template<> inline __m128 smallestNormal<__m128>() {return _mm_set1_ps(std::numeric_limits<float>::min());}
	template<> inline __m128d smallestNormal<__m128d>() {return _mm_set1_pd(std::numeric_limits<double>::min());}
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
	SIGNALSMITH_OPS2(max, _mm_max_ps, _mm_max_pd)
	SIGNALSMITH_OPS2(copysign, ssmf_copysignf, ssmf_copysignd)
	SIGNALSMITH_OPS2(hypot, ssmf_hypotf, ssmf_hypotd)
	SIGNALSMITH_OPS2(atan2, ssmf_atan2f, ssmf_atan2d)
	SIGNALSMITH_OPS1(abs, ssmf_absf, ssmf_absd)
	SIGNALSMITH_OPS1(neg, ssmf_negf, ssmf_negd)
	SIGNALSMITH_OPS1(sqrt, _mm_sqrt_ps, _mm_sqrt_pd)
	SIGNALSMITH_OPS1(pow2, ssmf_pow2f, ssmf_pow2d)
	SIGNALSMITH_OPS1(log, ssmf_logf, ssmf_logd)
	SIGNALSMITH_OPS1(atan, ssmf_atanf, ssmf_atand)
	SIGNALSMITH_OPS1(exp, ssmf_expf, ssmf_expd)
	SIGNALSMITH_OPS1(sinh, ssmf_sinhf, ssmf_sinhd)
	SIGNALSMITH_OPS1(cosh, ssmf_coshf, ssmf_coshd)
#undef SIGNALSMITH_OPS1
#undef SIGNALSMITH_OPS2
	inline void sincos(__m128 v, __m128 &sinOut, __m128 &cosOut) {ssmf_sincosf(v, &sinOut, &cosOut);}
	inline void sincos(__m128d v, __m128d &sinOut, __m128d &cosOut) {ssmf_sincosd(v, &sinOut, &cosOut);}
	// v < 0 ? a : b
	inline __m128 ifNegative(__m128 v, __m128 a, __m128 b) {
		__m128 mask = _mm_cmplt_ps(v, _mm_setzero_ps());
		return _mm_or_ps(_mm_and_ps(mask, a), _mm_andnot_ps(mask, b));
	}
	inline __m128d ifNegative(__m128d v, __m128d a, __m128d b) {
		__m128d mask = _mm_cmplt_pd(v, _mm_setzero_pd());
		return _mm_or_pd(_mm_and_pd(mask, a), _mm_andnot_pd(mask, b));
	}
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

		// Complex functions, from the real ones.  Unless noted, they have similar accuracy to the real ones (relative to |z| or |result|), and they work for any z which doesn't overflow or produce Inf/NaN.
		template<class C>
		static decltype(C::real) complexAbs(C z) {
			return ops::hypot(z.real, z.imag);
		}
		template<class C>
		static decltype(C::real) complexNorm(C z) {
			return ops::add(ops::pow2(z.real), ops::pow2(z.imag));
		}
		template<class C>
		static decltype(C::real) complexArg(C z) {
			return ops::atan2(z.imag, z.real); // nonsense for z == 0
		}
		template<class C>
		static C complexConj(C z) {
			return {z.real, ops::neg(z.imag)};
		}
		template<class C>
		static C complexMul(C a, C b) {
			return {ops::sub(ops::mul(a.real, b.real), ops::mul(a.imag, b.imag)), ops::add(ops::mul(a.real, b.imag), ops::mul(a.imag, b.real))};
		}
		template<class C>
		static C complexExp(C z) {
			using V = decltype(z.real);
			V e = ops::exp(z.real), s, c;
			ops::sincos(z.imag, s, c);
			return {ops::mul(e, c), ops::mul(e, s)};
		}
		template<class C>
		static C complexLog(C z) {
			// Absolute (not relative) accuracy for the real part when |z| is close to 1, and nonsense for z == 0
			return {ops::log(complexAbs(z)), complexArg(z)};
		}
		template<class C>
		static C complexLog10(C z) {
			using V = decltype(z.real);
			C l = complexLog(z);
			V scale = ops::k<V>(0.43429448190325182); // 1/ln(10)
			return {ops::mul(l.real, scale), ops::mul(l.imag, scale)};
		}
		template<class C>
		static C complexSqrt(C z) {
			using V = decltype(z.real);
			// t = sqrt((|z| + |re|)/2) and h = im/(2t) are both accurate, and are (re, im) or (|im|, ±re) depending on the sign of re
			V w = ops::mul(ops::add(complexAbs(z), ops::abs(z.real)), ops::k<V>(0.5));
			// Clamped so that t (and h) are exactly 0 when z == 0
			V invT = ops::div(ops::k<V>(1), ops::sqrt(ops::max(w, ops::smallestNormal<V>())));
			V t = ops::mul(w, invT), h = ops::mul(ops::mul(z.imag, ops::k<V>(0.5)), invT);
			return {ops::ifNegative(z.real, ops::abs(h), t), ops::ifNegative(z.real, ops::copysign(t, z.imag), h)};
		}
		template<class C>
		static C complexSin(C z) {
			using V = decltype(z.real);
			V s, c;
			ops::sincos(z.real, s, c);
			return {ops::mul(s, ops::cosh(z.imag)), ops::mul(c, ops::sinh(z.imag))};
		}
		template<class C>
		static C complexCos(C z) {
			using V = decltype(z.real);
			V s, c;
			ops::sincos(z.real, s, c);
			return {ops::mul(c, ops::cosh(z.imag)), ops::neg(ops::mul(s, ops::sinh(z.imag)))};
		}
		template<class C>
		static C complexSinh(C z) {
			using V = decltype(z.real);
			V s, c;
			ops::sincos(z.imag, s, c);
			return {ops::mul(ops::sinh(z.real), c), ops::mul(ops::cosh(z.real), s)};
		}
		template<class C>
		static C complexCosh(C z) {
			using V = decltype(z.real);
			V s, c;
			ops::sincos(z.imag, s, c);
			return {ops::mul(ops::cosh(z.real), c), ops::mul(ops::sinh(z.real), s)};
		}
		template<class C>
		static C complexTan(C z) {
			using V = decltype(z.real);
			// (sin(2a), sinh(2b))/(cos(2a) + cosh(2b)), with the denominator written as 2(cos²(a) + sinh²(b)) to avoid cancellation near the poles.  sinh²(b) overflows for |b| > ~44 (or ~355 for double)
			V s, c, sh = ops::sinh(z.imag), ch = ops::cosh(z.imag);
			ops::sincos(z.real, s, c);
			V invD = ops::div(ops::k<V>(1), ops::add(ops::pow2(c), ops::pow2(sh)));
			return {ops::mul(ops::mul(s, c), invD), ops::mul(ops::mul(sh, ch), invD)};
		}
		template<class C>
		static C complexTanh(C z) {
			using V = decltype(z.real);
			// (sinh(2a), sin(2b))/(cosh(2a) + cos(2b)), with the denominator written as 2(sinh²(a) + cos²(b)) to avoid cancellation near the poles.  sinh²(a) overflows for |a| > ~44 (or ~355 for double)
			V s, c, sh = ops::sinh(z.real), ch = ops::cosh(z.real);
			ops::sincos(z.imag, s, c);
			V invD = ops::div(ops::k<V>(1), ops::add(ops::pow2(sh), ops::pow2(c)));
			return {ops::mul(ops::mul(sh, ch), invD), ops::mul(ops::mul(s, c), invD)};
		}
		template<class C>
		static C complexAsinh(C z) {
			using V = decltype(z.real);
			// log(w + sqrt(w² + 1)) with w = ±z so that re(w) >= 0 (avoiding cancellation), then the sign put back.
			// Absolute (not relative) accuracy near 0 and near the branch points ±i, and w² overflows for |z| > ~1e19 (or ~1e154 for double)
			V sign = ops::copysign(ops::k<V>(1), z.real);
			C w{ops::abs(z.real), ops::mul(z.imag, sign)};
			C w2p1{ops::add(ops::mul(ops::sub(w.real, w.imag), ops::add(w.real, w.imag)), ops::k<V>(1)), ops::mul(ops::add(w.real, w.real), w.imag)};
			C q = complexSqrt(w2p1);
			C l = complexLog(C{ops::add(w.real, q.real), ops::add(w.imag, q.imag)});
			return {ops::mul(l.real, sign), ops::mul(l.imag, sign)};
		}
		template<class C>
		static C complexAsin(C z) {
			// -i*asinh(i*z)
			C a = complexAsinh(C{ops::neg(z.imag), z.real});
			return {a.imag, ops::neg(a.real)};
		}
		template<class C>
		static C complexAcos(C z) {
			using V = decltype(z.real);
			// pi/2 - asin(z)
			C a = complexAsin(z);
			return {ops::sub(ops::k<V>(1.5707963267948966), a.real), ops::neg(a.imag)};
		}
		template<class C>
		static C complexAcosh(C z) {
			using V = decltype(z.real);
			// log(z + sqrt(z + 1)*sqrt(z - 1)): absolute (not relative) accuracy near ±1, and z² overflows for |z| > ~1e19 (or ~1e154 for double)
			C s = complexMul(complexSqrt(C{ops::add(z.real, ops::k<V>(1)), z.imag}), complexSqrt(C{ops::sub(z.real, ops::k<V>(1)), z.imag}));
			return complexLog(C{ops::add(z.real, s.real), ops::add(z.imag, s.imag)});
		}
		template<class C>
		static C complexAtanh(C z) {
			using V = decltype(z.real);
			// (log(|1 + z|²/|1 - z|²)/4, atan2(2b, (1 - a)(1 + a) - b²)/2): absolute (not relative) accuracy near 0 and near the branch points ±1
			V b2 = ops::pow2(z.imag);
			V onePlus = ops::add(ops::k<V>(1), z.real), oneMinus = ops::sub(ops::k<V>(1), z.real);
			V ratio = ops::div(ops::add(ops::pow2(onePlus), b2), ops::add(ops::pow2(oneMinus), b2));
			V angle = ops::atan2(ops::add(z.imag, z.imag), ops::sub(ops::mul(oneMinus, onePlus), b2));
			return {ops::mul(ops::log(ratio), ops::k<V>(0.25)), ops::mul(angle, ops::k<V>(0.5))};
		}
		template<class C>
		static C complexAtan(C z) {
			// -i*atanh(i*z)
			C a = complexAtanh(C{ops::neg(z.imag), z.real});
			return {a.imag, ops::neg(a.real)};
		}
		static ComplexFloat complexPolar(__m128 r, __m128 theta) {
			__m128 s, c;
			ops::sincos(theta, s, c);
			return {ops::mul(r, c), ops::mul(r, s)};
		}
		static ComplexDouble complexPolar(__m128d r, __m128d theta) {
			__m128d s, c;
			ops::sincos(theta, s, c);
			return {ops::mul(r, c), ops::mul(r, s)};
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
		SIGNALSMITH_EXPR_BINARY(Polar)
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
		SIGNALSMITH_EXPR_COMPLEX_REAL(Arg, complexArg)
#undef SIGNALSMITH_EXPR_COMPLEX_REAL
		template<class A>
		static bool isAligned(expression::Arg<A> expr) {
			return isAligned(expr.a);
		}

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
}

}}; // namespace

#if defined(__clang__) || defined(__GNUC__)
#	pragma GCC pop_options
#endif
