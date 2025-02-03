#ifndef SIGNALSMITH_AUDIO_LINEAR_FFT_H
#define SIGNALSMITH_AUDIO_LINEAR_FFT_H

#include <complex>
#include <vector>
#include <cmath>

#if defined(__FAST_MATH__) && (__apple_build_version__ >= 16000000) && (__apple_build_version__ <= 16000099) && !defined(SIGNALSMITH_IGNORE_BROKEN_APPLECLANG)
#	error Apple Clang 16.0.0 generates incorrect SIMD for ARM. If you HAVE to use this version of Clang, turn off -ffast-math.
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace signalsmith { namespace linear {

namespace _impl {
	template<class V>
	void complexMul(std::complex<V> *a, const std::complex<V> *b, const std::complex<V> *c, size_t size) {
		for (size_t i = 0; i < size; ++i) {
			auto bi = b[i], ci = c[i];
			a[i] = {bi.real()*ci.real() - bi.imag()*ci.imag(), bi.imag()*ci.real() + bi.real()*ci.imag()};
		}
	}
	template<class V>
	void complexMulConj(std::complex<V> *a, const std::complex<V> *b, const std::complex<V> *c, size_t size) {
		for (size_t i = 0; i < size; ++i) {
			auto bi = b[i], ci = c[i];
			a[i] = {bi.real()*ci.real() + bi.imag()*ci.imag(), bi.imag()*ci.real() - bi.real()*ci.imag()};
		}
	}
	template<class V>
	void complexMul(V *ar, V *ai, const V *br, const V *bi, const V *cr, const V *ci, size_t size) {
		for (size_t i = 0; i < size; ++i) {
			V rr = br[i]*cr[i] - bi[i]*ci[i];
			V ri = br[i]*ci[i] + bi[i]*cr[i];
			ar[i] = rr;
			ai[i] = ri;
		}
	}
	template<class V>
	void complexMulConj(V *ar, V *ai, const V *br, const V *bi, const V *cr, const V *ci, size_t size) {
		for (size_t i = 0; i < size; ++i) {
			V rr = cr[i]*br[i] + ci[i]*bi[i];
			V ri = cr[i]*bi[i] - ci[i]*br[i];
			ar[i] = rr;
			ai[i] = ri;
		}
	}

	// Input: aStride elements next to each other -> output with bStride
	template<size_t aStride, class V>
	void interleaveCopy(const V *a, V *b, size_t bStride) {
		for (size_t bi = 0; bi < bStride; ++bi) {
			const V *offsetA = a + bi*aStride;
			V *offsetB = b + bi;
			for (size_t ai = 0; ai < aStride; ++ai) {
				offsetB[ai*bStride] = offsetA[ai];
			}
		}
	}
	template<class V>
	void interleaveCopy(const V *a, V *b, size_t aStride, size_t bStride) {
		for (size_t bi = 0; bi < bStride; ++bi) {
			const V *offsetA = a + bi*aStride;
			V *offsetB = b + bi;
			for (size_t ai = 0; ai < aStride; ++ai) {
				offsetB[ai*bStride] = offsetA[ai];
			}
		}
	}
	template<size_t aStride, class V>
	void interleaveCopy(const V *aReal, const V *aImag, V *bReal, V *bImag, size_t bStride) {
		for (size_t bi = 0; bi < bStride; ++bi) {
			const V *offsetAr = aReal + bi*aStride;
			const V *offsetAi = aImag + bi*aStride;
			V *offsetBr = bReal + bi;
			V *offsetBi = bImag + bi;
			for (size_t ai = 0; ai < aStride; ++ai) {
				offsetBr[ai*bStride] = offsetAr[ai];
				offsetBi[ai*bStride] = offsetAi[ai];
			}
		}
	}
	template<class V>
	void interleaveCopy(const V *aReal, const V *aImag, V *bReal, V *bImag, size_t aStride, size_t bStride) {
		for (size_t bi = 0; bi < bStride; ++bi) {
			const V *offsetAr = aReal + bi*aStride;
			const V *offsetAi = aImag + bi*aStride;
			V *offsetBr = bReal + bi;
			V *offsetBi = bImag + bi;
			for (size_t ai = 0; ai < aStride; ++ai) {
				offsetBr[ai*bStride] = offsetAr[ai];
				offsetBi[ai*bStride] = offsetAi[ai];
			}
		}
	}

	template<class V>
	void strideCopy(const std::complex<V> *a, size_t aStride, std::complex<V> *b, size_t size) {
		for (size_t i = 0; i < size; ++i) {
			b[i] = a[i*aStride];
		}
	}
	template<class V>
	void strideCopy(const V *ar, const V *ai, size_t aStride, V *br, V *bi, size_t size) {
		for (size_t i = 0; i < size; ++i) {
			br[i] = ar[i*aStride];
			bi[i] = ai[i*aStride];
		}
	}
}

/// Extremely simple and portable power-of-2 FFT
template<typename Sample>
struct SimpleFFT {
	using Complex = std::complex<Sample>;
	
	SimpleFFT(size_t maxSize=0) {
		resize(maxSize);
	}
	
	void resize(size_t maxSize) {
		twiddles.resize(maxSize/2);
		for (size_t i = 0; i < maxSize/2; ++i) {
			Sample twiddlePhase = -2*M_PI*i/maxSize;
			twiddles[i] = std::polar(Sample(1), twiddlePhase);
		}
		working.resize(maxSize);
	}
	
	void fft(size_t size, const Complex *time, Complex *freq) {
		if (size <= 1) {
			*freq = *time;
			return;
		}
		fftPass<false>(size, 1, time, freq, working.data());
	}

	void ifft(size_t size, const Complex *freq, Complex *time) {
		if (size <= 1) {
			*time = *freq;
			return;
		}
		fftPass<true>(size, 1, freq, time, working.data());
	}

	void fft(size_t size, const Sample *inR, const Sample *inI, Sample *outR, Sample *outI) {
		if (size <= 1) {
			*outR = *inR;
			*outI = *inI;
			return;
		}
		Sample *workingR = (Sample *)working.data(), *workingI = workingR + size;
		fftPass<false>(size, 1, inR, inI, outR, outI, workingR, workingI);
	}
	void ifft(size_t size, const Sample *inR, const Sample *inI, Sample *outR, Sample *outI) {
		if (size <= 1) {
			*outR = *inR;
			*outI = *inI;
			return;
		}
		Sample *workingR = (Sample *)working.data(), *workingI = workingR + size;
		fftPass<true>(size, 1, inR, inI, outR, outI, workingR, workingI);
	}
private:
	std::vector<Complex> twiddles;
	std::vector<Complex> working;

	// Calculate a [size]-point FFT, where each element is a block of [stride] values
	template<bool inverse>
	void fftPass(size_t size, size_t stride, const Complex *input, Complex *output, Complex *working) const {
		if (size > 2) {
			// Calculate the two half-size FFTs (odd and even) by doubling the stride
			fftPass<inverse>(size/2, stride*2, input, working, output);
			combine2<inverse>(size, stride, working, output);
		} else {
			// The input can already be considered a 1-point FFT
			combine2<inverse>(size, stride, input, output);
		}
	}

	// Combine interleaved even/odd results into a single spectrum
	template<bool inverse>
	void combine2(size_t size, size_t stride, const Complex *input, Complex *output) const {
		auto twiddleStep = twiddles.size()*2/size;
		for (size_t i = 0; i < size/2; ++i) {
			Complex twiddle = twiddles[i*twiddleStep];
			
			const Complex *inputA = input + 2*i*stride;
			const Complex *inputB = input + (2*i + 1)*stride;
			Complex *outputA = output + i*stride;
			Complex *outputB = output + (i + size/2)*stride;
			for (size_t s = 0; s < stride; ++s) {
				Complex a = inputA[s];
				Complex b = inputB[s];
				b = inverse ? Complex{b.real()*twiddle.real() + b.imag()*twiddle.imag(), b.imag()*twiddle.real() - b.real()*twiddle.imag()} : Complex{b.real()*twiddle.real() - b.imag()*twiddle.imag(), b.imag()*twiddle.real() + b.real()*twiddle.imag()};
				outputA[s] = a + b;
				outputB[s] = a - b;
			}
		}
	}

	// Calculate a [size]-point FFT, where each element is a block of [stride] values
	template<bool inverse>
	void fftPass(size_t size, size_t stride, const Sample *inputR, const Sample *inputI, Sample *outputR, Sample *outputI, Sample *workingR, Sample *workingI) const {
		if (size > 2) {
			// Calculate the two half-size FFTs (odd and even) by doubling the stride
			fftPass<inverse>(size/2, stride*2, inputR, inputI, workingR, workingI, outputR, outputI);
			combine2<inverse>(size, stride, workingR, workingI, outputR, outputI);
		} else {
			// The input can already be considered a 1-point FFT
			combine2<inverse>(size, stride, inputR, inputI, outputR, outputI);
		}
	}

	// Combine interleaved even/odd results into a single spectrum
	template<bool inverse>
	void combine2(size_t size, size_t stride, const Sample *inputR, const Sample *inputI, Sample *outputR, Sample *outputI) const {
		auto twiddleStep = twiddles.size()*2/size;
		for (size_t i = 0; i < size/2; ++i) {
			Complex twiddle = twiddles[i*twiddleStep];
			
			const Sample *inputAR = inputR + 2*i*stride;
			const Sample *inputAI = inputI + 2*i*stride;
			const Sample *inputBR = inputR + (2*i + 1)*stride;
			const Sample *inputBI = inputI + (2*i + 1)*stride;
			Sample *outputAR = outputR + i*stride;
			Sample *outputAI = outputI + i*stride;
			Sample *outputBR = outputR + (i + size/2)*stride;
			Sample *outputBI = outputI + (i + size/2)*stride;
			for (size_t s = 0; s < stride; ++s) {
				Complex a = {inputAR[s], inputAI[s]};
				Complex b = {inputBR[s], inputBI[s]};
				b = inverse ? Complex{b.real()*twiddle.real() + b.imag()*twiddle.imag(), b.imag()*twiddle.real() - b.real()*twiddle.imag()} : Complex{b.real()*twiddle.real() - b.imag()*twiddle.imag(), b.imag()*twiddle.real() + b.real()*twiddle.imag()};
				Complex sum = a + b, diff = a - b;
				outputAR[s] = sum.real();
				outputAI[s] = sum.imag();
				outputBR[s] = diff.real();
				outputBI[s] = diff.imag();
			}
		}
	}
};

/// A power-of-2 only FFT, specialised with platform-specific fast implementations where available
template<typename Sample>
struct Pow2FFT {
	static constexpr bool prefersSplit = true; // whether this FFT implementation is faster when given split-complex inputs
	using Complex = std::complex<Sample>;
	
	Pow2FFT(size_t size=0) {
		resize(size);
	}
	
	void resize(size_t size) {
		_size = size;
		simpleFFT.resize(size);
		tmp.resize(size);
	}
	
	void fft(const Complex *time, Complex *freq) {
		simpleFFT.fft(_size, time, freq);
	}
	void fft(const Sample *inR, const Sample *inI, Sample *outR, Sample *outI) {
		simpleFFT.fft(_size, inR, inI, outR, outI);
	}

	void ifft(const Complex *freq, Complex *time) {
		simpleFFT.ifft(_size, freq, time);
	}
	void ifft(const Sample *inR, const Sample *inI, Sample *outR, Sample *outI) {
		simpleFFT.ifft(_size, inR, inI, outR, outI);
	}

private:
	size_t _size;
	std::vector<Complex> tmp;
	SimpleFFT<Sample> simpleFFT;
};

/// An FFT which can be computed in chunks
template<typename Sample, bool splitComputation=false>
struct SplitFFT {
	using Complex = std::complex<Sample>;
	static constexpr size_t maxSplit = splitComputation ? 4 : 1;
	static constexpr size_t minInnerSize = 32;
	
	static size_t fastSizeAbove(size_t size) {
		size_t pow2 = 1;
		while (pow2 < 16 && pow2 < size) pow2 *= 2;
		while (pow2*8 < size) pow2 *= 2;
		size_t multiple = (size + pow2 - 1)/pow2; // will be 1-8
		if (multiple == 7) ++multiple;
		return multiple*pow2;
	}
	
	SplitFFT(size_t size=0) {
		resize(size);
	}
	
	void resize(size_t size) {
		innerSize = 1;
		outerSize = size;

		dftTmp.resize(0);
		dftTwists.resize(0);
		plan.resize(0);
		if (!size) return;
		
		// Inner size = largest power of 2 such that either the inner size >= minInnerSize, or we have the target number of splits
		while (!(outerSize&1) && (outerSize > maxSplit || innerSize < minInnerSize)) {
			innerSize *= 2;
			outerSize /= 2;
		}
		tmpFreq.resize(size);
		innerFFT.resize(innerSize);
		
		outerTwiddles.resize(innerSize*(outerSize - 1));
		outerTwiddlesR.resize(innerSize*(outerSize - 1));
		outerTwiddlesI.resize(innerSize*(outerSize - 1));
		for (size_t i = 0; i < innerSize; ++i) {
			for (size_t s = 1; s < outerSize; ++s) {
				Sample twiddlePhase = Sample(-2*M_PI*i/innerSize*s/outerSize);
				outerTwiddles[i + (s - 1)*innerSize] = std::polar(Sample(1), twiddlePhase);
			}
		}
		for (size_t i = 0; i < outerTwiddles.size(); ++i) {
			outerTwiddlesR[i] = outerTwiddles[i].real();
			outerTwiddlesI[i] = outerTwiddles[i].imag();
		}


		StepType interleaveStep = StepType::interleaveOrderN;
		StepType finalStep = StepType::finalOrderN;
		if (outerSize == 2) {
			interleaveStep = StepType::interleaveOrder2;
			finalStep = StepType::finalOrder2;
		}
		if (outerSize == 3) {
			interleaveStep = StepType::interleaveOrder3;
			finalStep = StepType::finalOrder3;
		}
		if (outerSize == 4) {
			interleaveStep = StepType::interleaveOrder4;
			finalStep = StepType::finalOrder4;
		}
		if (outerSize == 5) {
			interleaveStep = StepType::interleaveOrder5;
			finalStep = StepType::finalOrder5;
		}
		
		if (outerSize <= 1) {
			if (size > 0) plan.push_back(Step{StepType::passthrough, 0});
		} else {
			plan.push_back({interleaveStep, 0});
			plan.push_back({StepType::firstFFT, 0});
			for (size_t s = 1; s < outerSize; ++s) {
				plan.push_back({StepType::middleFFT, s*innerSize});
			}
			plan.push_back({StepType::twiddles, 0});
			plan.push_back({finalStep, 0});

			if (finalStep == StepType::finalOrderN) {
				dftTmp.resize(outerSize);
				dftTwists.resize(outerSize);
				for (size_t s = 0; s < outerSize; ++s) {
					Sample dftPhase = Sample(-2*M_PI*s/outerSize);
					dftTwists[s] = std::polar(Sample(1), dftPhase);
				}
			}
		}
	}
	
	size_t size() const {
		return innerSize*outerSize;
	}
	size_t steps() const {
		return plan.size();
	}
	
	void fft(const Complex *time, Complex *freq) {
		for (auto &step : plan) {
			fftStep<false>(step, time, freq);
		}
	}
	void fft(size_t step, const Complex *time, Complex *freq) {
		fftStep<false>(plan[step], time, freq);
	}
	void fft(const Sample *inR, const Sample *inI, Sample *outR, Sample *outI) {
		for (auto &step : plan) {
			fftStep<false>(step, inR, inI, outR, outI);
		}
	}
	void fft(size_t step, const Sample *inR, const Sample *inI, Sample *outR, Sample *outI) {
		fftStep<false>(plan[step], inR, inI, outR, outI);
	}
	
	void ifft(const Complex *freq, Complex *time) {
		for (auto &step : plan) {
			fftStep<true>(step, freq, time);
		}
	}
	void ifft(size_t step, const Complex *freq, Complex *time) {
		fftStep<true>(plan[step], freq, time);
	}
	void ifft(const Sample *inR, const Sample *inI, Sample *outR, Sample *outI) {
		for (auto &step : plan) {
			fftStep<true>(step, inR, inI, outR, outI);
		}
	}
	void ifft(size_t step, const Sample *inR, const Sample *inI, Sample *outR, Sample *outI) {
		fftStep<true>(plan[step], inR, inI, outR, outI);
	}
private:
	using InnerFFT = Pow2FFT<Sample>;
	InnerFFT innerFFT;

	size_t innerSize, outerSize;
	std::vector<Complex> tmpFreq;
	std::vector<Complex> outerTwiddles;
	std::vector<Sample> outerTwiddlesR, outerTwiddlesI;
	std::vector<Complex> dftTwists, dftTmp;

	enum class StepType {
		passthrough,
		interleaveOrder2, interleaveOrder3, interleaveOrder4, interleaveOrder5, interleaveOrderN,
		firstFFT, middleFFT,
		twiddles,
		finalOrder2, finalOrder3, finalOrder4, finalOrder5, finalOrderN
	};
	struct Step {
		StepType type;
		size_t offset;
	};
	std::vector<Step> plan;
	
	template<bool inverse>
	void fftStep(Step step, const Complex *time, Complex *freq) {
		switch (step.type) {
			case (StepType::passthrough): {
				if (inverse) {
					innerFFT.ifft(time, freq);
				} else {
					innerFFT.fft(time, freq);
				}
				break;
			}
			case (StepType::interleaveOrder2): {
				_impl::interleaveCopy<2>(time, tmpFreq.data(), innerSize);
				break;
			}
			case (StepType::interleaveOrder3): {
				_impl::interleaveCopy<3>(time, tmpFreq.data(), innerSize);
				break;
			}
			case (StepType::interleaveOrder4): {
				_impl::interleaveCopy<4>(time, tmpFreq.data(), innerSize);
				break;
			}
			case (StepType::interleaveOrder5): {
				_impl::interleaveCopy<5>(time, tmpFreq.data(), innerSize);
				break;
			}
			case (StepType::interleaveOrderN): {
				_impl::interleaveCopy(time, tmpFreq.data(), outerSize, innerSize);
				break;
			}
			case (StepType::firstFFT): {
				if (inverse) {
					innerFFT.ifft(tmpFreq.data(), freq);
				} else {
					innerFFT.fft(tmpFreq.data(), freq);
				}
				break;
			}
			case (StepType::middleFFT): {
				Complex *offsetOut = freq + step.offset;
				if (inverse) {
					innerFFT.ifft(tmpFreq.data() + step.offset, offsetOut);
				} else {
					innerFFT.fft(tmpFreq.data() + step.offset, offsetOut);
				}
				break;
			}
			case (StepType::twiddles): {
				if (inverse) {
					_impl::complexMulConj(freq + innerSize, freq + innerSize, outerTwiddles.data(), innerSize*(outerSize - 1));
				} else {
					_impl::complexMul(freq + innerSize, freq + innerSize, outerTwiddles.data(), innerSize*(outerSize - 1));
				}
				break;
			}
			case StepType::finalOrder2:
				finalPass2(freq);
				break;
			case StepType::finalOrder3:
				finalPass3<inverse>(freq);
				break;
			case StepType::finalOrder4:
				finalPass4<inverse>(freq);
				break;
			case StepType::finalOrder5:
				finalPass5<inverse>(freq);
				break;
			case StepType::finalOrderN:
				finalPassN<inverse>(freq);
				break;
		}
	}
	template<bool inverse>
	void fftStep(Step step, const Sample *inR, const Sample *inI, Sample *outR, Sample *outI) {
		Sample *tmpR = (Sample *)tmpFreq.data(), *tmpI = tmpR + tmpFreq.size();
		switch (step.type) {
			case (StepType::passthrough): {
				if (inverse) {
					innerFFT.ifft(inR, inI, outR, outI);
				} else {
					innerFFT.fft(inR, inI, outR, outI);
				}
				break;
			}
			case (StepType::interleaveOrder2): {
				_impl::interleaveCopy<2>(inR, tmpR, innerSize);
				_impl::interleaveCopy<2>(inI, tmpI, innerSize);
				break;
			}
			case (StepType::interleaveOrder3): {
				_impl::interleaveCopy<3>(inR, tmpR, innerSize);
				_impl::interleaveCopy<3>(inI, tmpI, innerSize);
				break;
			}
			case (StepType::interleaveOrder4): {
				_impl::interleaveCopy<4>(inR, tmpR, innerSize);
				_impl::interleaveCopy<4>(inI, tmpI, innerSize);
				break;
			}
			case (StepType::interleaveOrder5): {
				_impl::interleaveCopy<5>(inR, tmpR, innerSize);
				_impl::interleaveCopy<5>(inI, tmpI, innerSize);
				break;
			}
			case (StepType::interleaveOrderN): {
				_impl::interleaveCopy(inR, inI, tmpR, tmpI, outerSize, innerSize);
				break;
			}
			case (StepType::firstFFT): {
				if (inverse) {
					innerFFT.ifft(tmpR, tmpI, outR, outI);
				} else {
					innerFFT.fft(tmpR, tmpI, outR, outI);
				}
				break;
			}
			case (StepType::middleFFT): {
				size_t offset = step.offset;
				Sample *offsetOutR = outR + offset;
				Sample *offsetOutI = outI + offset;
				if (inverse) {
					innerFFT.ifft(tmpR + offset, tmpI + offset, offsetOutR, offsetOutI);
				} else {
					innerFFT.fft(tmpR + offset, tmpI + offset, offsetOutR, offsetOutI);
				}
				break;
			}
			case(StepType::twiddles): {
				auto *twiddlesR = outerTwiddlesR.data();
				auto *twiddlesI = outerTwiddlesI.data();
				if (inverse) {
					_impl::complexMulConj(outR + innerSize, outI + innerSize, outR + innerSize, outI + innerSize, twiddlesR, twiddlesI, innerSize*(outerSize - 1));
				} else {
					_impl::complexMul(outR + innerSize, outI + innerSize, outR + innerSize, outI + innerSize, twiddlesR, twiddlesI, innerSize*(outerSize - 1));
				}
				break;
			}
			case StepType::finalOrder2:
				finalPass2(outR, outI);
				break;
			case StepType::finalOrder3:
				finalPass3<inverse>(outR, outI);
				break;
			case StepType::finalOrder4:
				finalPass4<inverse>(outR, outI);
				break;
			case StepType::finalOrder5:
				finalPass5<inverse>(outR, outI);
				break;
			case StepType::finalOrderN:
				finalPassN<inverse>(outR, outI);
				break;
		}
	}
	
	void finalPass2(Complex *f0) {
		auto *f1 = f0 + innerSize;
		for (size_t i = 0; i < innerSize; ++i) {
			Complex a = f0[i], b = f1[i];
			f0[i] = a + b;
			f1[i] = a - b;
		}
	}
	void finalPass2(Sample *f0r, Sample *f0i) {
		auto *f1r = f0r + innerSize;
		auto *f1i = f0i + innerSize;
		for (size_t i = 0; i < innerSize; ++i) {
			Sample ar = f0r[i], ai = f0i[i];
			Sample br = f1r[i], bi = f1i[i];
			f0r[i] = ar + br;
			f0i[i] = ai + bi;
			f1r[i] = ar - br;
			f1i[i] = ai - bi;
		}
	}
	template<bool inverse>
	void finalPass3(Complex *f0) {
		auto *f1 = f0 + innerSize;
		auto *f2 = f0 + innerSize*2;
		const Complex tw1{Sample(-0.5), Sample(-std::sqrt(0.75)*(inverse ? -1 : 1))};
		for (size_t i = 0; i < innerSize; ++i) {
			Complex a = f0[i], b = f1[i], c = f2[i];
			f0[i] = a + b + c;
			f1[i] = a + b*tw1 + c*std::conj(tw1);
			f2[i] = a + b*std::conj(tw1) + c*tw1;
		}
	}
	template<bool inverse>
	void finalPass3(Sample *f0r, Sample *f0i) {
		auto *f1r = f0r + innerSize;
		auto *f1i = f0i + innerSize;
		auto *f2r = f0r + innerSize*2;
		auto *f2i = f0i + innerSize*2;
		const Sample tw1r = -0.5, tw1i = -std::sqrt(0.75)*(inverse ? -1 : 1);
		
		for (size_t i = 0; i < innerSize; ++i) {
			Sample ar = f0r[i], ai = f0i[i], br = f1r[i], bi = f1i[i], cr = f2r[i], ci = f2i[i];

			f0r[i] = ar + br + cr;
			f0i[i] = ai + bi + ci;
			f1r[i] = ar + br*tw1r - bi*tw1i + cr*tw1r + ci*tw1i;
			f1i[i] = ai + bi*tw1r + br*tw1i - cr*tw1i + ci*tw1r;
			f2r[i] = ar + br*tw1r + bi*tw1i + cr*tw1r - ci*tw1i;
			f2i[i] = ai + bi*tw1r - br*tw1i + cr*tw1i + ci*tw1r;
		}
	}
	template<bool inverse>
	void finalPass4(Complex *f0) {
		auto *f1 = f0 + innerSize;
		auto *f2 = f0 + innerSize*2;
		auto *f3 = f0 + innerSize*3;
		for (size_t i = 0; i < innerSize; ++i) {
			Complex a = f0[i], b = f1[i], c = f2[i], d = f3[i];
			
			Complex ac0 = a + c, ac1 = a - c;
			Complex bd0 = b + d, bd1 = inverse ? (b - d) : (d - b);
			Complex bd1i = {-bd1.imag(), bd1.real()};
			f0[i] = ac0 + bd0;
			f1[i] = ac1 + bd1i;
			f2[i] = ac0 - bd0;
			f3[i] = ac1 - bd1i;
		}
	}
	template<bool inverse>
	void finalPass4(Sample *f0r, Sample *f0i) {
		auto *f1r = f0r + innerSize;
		auto *f1i = f0i + innerSize;
		auto *f2r = f0r + innerSize*2;
		auto *f2i = f0i + innerSize*2;
		auto *f3r = f0r + innerSize*3;
		auto *f3i = f0i + innerSize*3;
		for (size_t i = 0; i < innerSize; ++i) {
			Sample ar = f0r[i], ai = f0i[i], br = f1r[i], bi = f1i[i], cr = f2r[i], ci = f2i[i], dr = f3r[i], di = f3i[i];
			
			Sample ac0r = ar + cr, ac0i = ai + ci;
			Sample ac1r = ar - cr, ac1i = ai - ci;
			Sample bd0r = br + dr, bd0i = bi + di;
			Sample bd1r = br - dr, bd1i = bi - di;
			
			f0r[i] = ac0r + bd0r;
			f0i[i] = ac0i + bd0i;
			f1r[i] = inverse ? (ac1r - bd1i) : (ac1r + bd1i);
			f1i[i] = inverse ? (ac1i + bd1r) : (ac1i - bd1r);
			f2r[i] = ac0r - bd0r;
			f2i[i] = ac0i - bd0i;
			f3r[i] = inverse ? (ac1r + bd1i) : (ac1r - bd1i);
			f3i[i] = inverse ? (ac1i - bd1r) : (ac1i + bd1r);
		}
	}
	template<bool inverse>
	void finalPass5(Complex *f0) {
		auto *f1 = f0 + innerSize;
		auto *f2 = f0 + innerSize*2;
		auto *f3 = f0 + innerSize*3;
		auto *f4 = f0 + innerSize*4;
		const Sample tw1r = 0.30901699437494745;
		const Sample tw1i = -0.9510565162951535*(inverse ? -1 : 1);
		const Sample tw2r = -0.8090169943749473;
		const Sample tw2i = -0.5877852522924732*(inverse ? -1 : 1);
		for (size_t i = 0; i < innerSize; ++i) {
			Complex a = f0[i], b = f1[i], c = f2[i], d = f3[i], e = f4[i];

			Complex be0 = b + e, be1 = {e.imag() - b.imag(), b.real() - e.real()}; // (b - e)*i
			Complex cd0 = c + d, cd1 = {d.imag() - c.imag(), c.real() - d.real()};
			
			Complex bcde01 = be0*tw1r + cd0*tw2r;
			Complex bcde02 = be0*tw2r + cd0*tw1r;
			Complex bcde11 = be1*tw1i + cd1*tw2i;
			Complex bcde12 = be1*tw2i - cd1*tw1i;

			f0[i] = a + be0 + cd0;
			f1[i] = a + bcde01 + bcde11;
			f2[i] = a + bcde02 + bcde12;
			f3[i] = a + bcde02 - bcde12;
			f4[i] = a + bcde01 - bcde11;
		}
	}
	template<bool inverse>
	void finalPass5(Sample *f0r, Sample *f0i) {
		auto *f1r = f0r + innerSize;
		auto *f1i = f0i + innerSize;
		auto *f2r = f0r + innerSize*2;
		auto *f2i = f0i + innerSize*2;
		auto *f3r = f0r + innerSize*3;
		auto *f3i = f0i + innerSize*3;
		auto *f4r = f0r + innerSize*4;
		auto *f4i = f0i + innerSize*4;
		
		const Sample tw1r = 0.30901699437494745;
		const Sample tw1i = -0.9510565162951535*(inverse ? -1 : 1);
		const Sample tw2r = -0.8090169943749473;
		const Sample tw2i = -0.5877852522924732*(inverse ? -1 : 1);
		for (size_t i = 0; i < innerSize; ++i) {
			Sample ar = f0r[i], ai = f0i[i], br = f1r[i], bi = f1i[i], cr = f2r[i], ci = f2i[i], dr = f3r[i], di = f3i[i], er = f4r[i], ei = f4i[i];

			Sample be0r = br + er, be0i = bi + ei;
			Sample be1r = ei - bi, be1i = br - er;
			Sample cd0r = cr + dr, cd0i = ci + di;
			Sample cd1r = di - ci, cd1i = cr - dr;

			Sample bcde01r = be0r*tw1r + cd0r*tw2r, bcde01i = be0i*tw1r + cd0i*tw2r;
			Sample bcde02r = be0r*tw2r + cd0r*tw1r, bcde02i = be0i*tw2r + cd0i*tw1r;
			Sample bcde11r = be1r*tw1i + cd1r*tw2i, bcde11i = be1i*tw1i + cd1i*tw2i;
			Sample bcde12r = be1r*tw2i - cd1r*tw1i, bcde12i = be1i*tw2i - cd1i*tw1i;

			f0r[i] = ar + be0r + cd0r;
			f0i[i] = ai + be0i + cd0i;
			f1r[i] = ar + bcde01r + bcde11r;
			f1i[i] = ai + bcde01i + bcde11i;
			f2r[i] = ar + bcde02r + bcde12r;
			f2i[i] = ai + bcde02i + bcde12i;
			f3r[i] = ar + bcde02r - bcde12r;
			f3i[i] = ai + bcde02i - bcde12i;
			f4r[i] = ar + bcde01r - bcde11r;
			f4i[i] = ai + bcde01i - bcde11i;
		}
	}

	template<bool inverse>
	void finalPassN(Complex *f0) {
		for (size_t i = 0; i < innerSize; ++i) {
			Complex *offsetFreq = f0 + i;
			Complex sum = 0;
			for (size_t i2 = 0; i2 < outerSize; ++i2) {
				sum += (dftTmp[i2] = offsetFreq[i2*innerSize]);
			}
			offsetFreq[0] = sum;
			
			for (size_t f = 1; f < outerSize; ++f) {
				Complex sum = dftTmp[0];

				for (size_t i2 = 1; i2 < outerSize; ++i2) {
					size_t twistIndex = (i2*f)%outerSize;
					Complex twist = inverse ? std::conj(dftTwists[twistIndex]) : dftTwists[twistIndex];
					sum += Complex{
						dftTmp[i2].real()*twist.real() - dftTmp[i2].imag()*twist.imag(),
						dftTmp[i2].imag()*twist.real() + dftTmp[i2].real()*twist.imag()
					};
				}

				offsetFreq[f*innerSize] = sum;
			}
		}
	}
	template<bool inverse>
	void finalPassN(Sample *f0r, Sample *f0i) {
		Sample *tmpR = (Sample *)dftTmp.data(), *tmpI = tmpR + outerSize;
		
		for (size_t i = 0; i < innerSize; ++i) {
			Sample *offsetR = f0r + i;
			Sample *offsetI = f0i + i;
			Sample sumR = 0, sumI = 0;
			for (size_t i2 = 0; i2 < outerSize; ++i2) {
				sumR += (tmpR[i2] = offsetR[i2*innerSize]);
				sumI += (tmpI[i2] = offsetI[i2*innerSize]);
			}
			offsetR[0] = sumR;
			offsetI[0] = sumI;
			
			for (size_t f = 1; f < outerSize; ++f) {
				Sample sumR = *tmpR, sumI = *tmpI;

				for (size_t i2 = 1; i2 < outerSize; ++i2) {
					size_t twistIndex = (i2*f)%outerSize;
					Complex twist = inverse ? std::conj(dftTwists[twistIndex]) : dftTwists[twistIndex];
					sumR += tmpR[i2]*twist.real() - tmpI[i2]*twist.imag();
					sumI += tmpI[i2]*twist.real() + tmpR[i2]*twist.imag();
				}

				offsetR[f*innerSize] = sumR;
				offsetI[f*innerSize] = sumI;
			}
		}
	}
};

template<typename Sample, bool splitComputation=false>
using FFT = SplitFFT<Sample, splitComputation>;

template<typename Sample, bool splitComputation=false, bool halfBinShift=false>
struct RealFFT {
	using Complex = std::complex<Sample>;

	static size_t fastSizeAbove(size_t size) {
		return ComplexFFT::fastSizeAbove((size + 1)/2)*2;
	}
	
	RealFFT(size_t size=0) {
		resize(size);
	}
	
	void resize(size_t size) {
		size_t hSize = size/2;
		complexFft.resize(hSize);
		tmpFreq.resize(hSize);
		tmpTime.resize(hSize);
		
		twiddles.resize(hSize/2 + 1);
		
		if (!halfBinShift) {
			for (size_t i = 0; i < twiddles.size(); ++i) {
				Sample rotPhase = i*(-2*M_PI/size) - M_PI/2; // bake rotation by (-i) into twiddles
				twiddles[i] = std::polar(Sample(1), rotPhase);
			}
		} else {
			for (size_t i = 0; i < twiddles.size(); ++i) {
				Sample rotPhase = (i + 0.5)*(-2*M_PI/size) - M_PI/2;
				twiddles[i] = std::polar(Sample(1), rotPhase);
			}
	
			halfBinTwists.resize(hSize);
			for (size_t i = 0; i < hSize; ++i) {
				Sample twistPhase = -2*M_PI*i/size;
				halfBinTwists[i] = std::polar(Sample(1), twistPhase);
			}
		}
	}
	
	size_t size() const {
		return complexFft.size()*2;
	}
	size_t steps() const {
		return complexFft.steps() + 2;
	}
	
	void fft(const Sample *time, Complex *freq) {
		for (size_t s = 0; s < steps(); ++s) {
			fft(s, time, freq);
		}
	}
	void fft(size_t step, const Sample *time, Complex *freq) {
		if (step-- == 0) {
			size_t hSize = complexFft.size();
			if (halfBinShift) {
				for (size_t i = 0; i < hSize; ++i) {
					Sample tr = time[2*i], ti = time[2*i + 1];
					Complex twist = halfBinTwists[i];
					tmpTime[i] = {
						tr*twist.real() - ti*twist.imag(),
						ti*twist.real() + tr*twist.imag()
					};
				}
			} else {
				for (size_t i = 0; i < hSize; ++i) {
					tmpTime[i] = {time[2*i], time[2*i + 1]};
				}
			}
		} else if (step < complexFft.steps()) {
			complexFft.fft(step, tmpTime.data(), tmpFreq.data());
		} else {
			size_t hSize = complexFft.size(), qSize = hSize/2;

			if (!halfBinShift) {
				Complex bin0 = tmpFreq[0];
				freq[0] = { // pack DC & Nyquist together
					bin0.real() + bin0.imag(),
					bin0.real() - bin0.imag()
				};
			}
			
			for (size_t i = halfBinShift ? 0 : 1; i <= qSize; ++i) {
				size_t conjI = halfBinShift ? (hSize - 1 - i) : (hSize - i);
				Complex twiddle = twiddles[i];

				Complex odd = (tmpFreq[i] + std::conj(tmpFreq[conjI]))*Sample(0.5);
				Complex evenI = (tmpFreq[i] - std::conj(tmpFreq[conjI]))*Sample(0.5);
				Complex evenRotMinusI = { // twiddle includes a factor of -i
					evenI.real()*twiddle.real() - evenI.imag()*twiddle.imag(),
					evenI.imag()*twiddle.real() + evenI.real()*twiddle.imag()
				};
				evenRotMinusI = evenI*twiddle;

				freq[i] = odd + evenRotMinusI;
				freq[conjI] = {odd.real() - evenRotMinusI.real(), evenRotMinusI.imag() - odd.imag()};
			}
		}
	}
	void fft(const Sample *inR, Sample *outR, Sample *outI) {
		for (size_t s = 0; s < steps(); ++s) {
			fft(s, inR, outR, outI);
		}
	}
	void fft(size_t step, const Sample *inR, Sample *outR, Sample *outI) {
		if (step-- == 0) {
			size_t hSize = complexFft.size();
			if (halfBinShift) {
				for (size_t i = 0; i < hSize; ++i) {
					Sample tr = inR[2*i], ti = inR[2*i + 1];
					Complex twist = halfBinTwists[i];
					tmpTime[i] = {
						tr*twist.real() - ti*twist.imag(),
						ti*twist.real() + tr*twist.imag()
					};
				}
			} else {
				for (size_t i = 0; i < hSize; ++i) {
					tmpTime[i] = {inR[2*i], inR[2*i + 1]};
				}
			}
		} else if (step < complexFft.steps()) {
			complexFft.fft(step, tmpTime.data(), tmpFreq.data());
		} else {
			size_t hSize = complexFft.size(), qSize = hSize/2;

			if (!halfBinShift) {
				Complex bin0 = tmpFreq[0];
				outR[0] = bin0.real() + bin0.imag();
				outI[0] = bin0.real() - bin0.imag();
			}

			for (size_t i = halfBinShift ? 0 : 1; i <= qSize; ++i) {
				size_t conjI = halfBinShift ? (hSize - 1 - i) : (hSize - i);
				Complex twiddle = twiddles[i];

				Complex odd = (tmpFreq[i] + std::conj(tmpFreq[conjI]))*Sample(0.5);
				Complex evenI = (tmpFreq[i] - std::conj(tmpFreq[conjI]))*Sample(0.5);
				Complex evenRotMinusI = { // twiddle includes a factor of -i
					evenI.real()*twiddle.real() - evenI.imag()*twiddle.imag(),
					evenI.imag()*twiddle.real() + evenI.real()*twiddle.imag()
				};
				evenRotMinusI = evenI*twiddle;

				outR[i] = odd.real() + evenRotMinusI.real();
				outI[i] = odd.imag() + evenRotMinusI.imag();
				outR[conjI] = odd.real() - evenRotMinusI.real();
				outI[conjI] = evenRotMinusI.imag() - odd.imag();
			}
		}
	}
	
	void ifft(const Complex *freq, Sample *time) {
		for (size_t s = 0; s < steps(); ++s) {
			ifft(s, freq, time);
		}
	}
	void ifft(size_t step, const Complex *freq, Sample *time) {
		if (step-- == 0) {
			size_t hSize = complexFft.size(), qSize = hSize/2;

			Complex bin0 = freq[0];
			if (!halfBinShift) {
				tmpFreq[0] = {
					bin0.real() + bin0.imag(),
					bin0.real() - bin0.imag()
				};
			}
			for (size_t i = halfBinShift ? 0 : 1; i <= qSize; ++i) {
				size_t conjI = halfBinShift ? (hSize - 1 - i) : (hSize - i);
				Complex twiddle = twiddles[i];

				Complex odd = freq[i] + std::conj(freq[conjI]);
				Complex evenRotMinusI = freq[i] - std::conj(freq[conjI]);
				Complex evenI = { // Conjugate
					evenRotMinusI.real()*twiddle.real() + evenRotMinusI.imag()*twiddle.imag(),
					evenRotMinusI.imag()*twiddle.real() - evenRotMinusI.real()*twiddle.imag()
				};

				tmpFreq[i] = odd + evenI;
				tmpFreq[conjI] = {odd.real() - evenI.real(), evenI.imag() - odd.imag()};
			}
		} else if (step < complexFft.steps()) {
			// Can't just use time as (Complex *), since it might not be aligned properly
			complexFft.ifft(step, tmpFreq.data(), tmpTime.data());
		} else {
			if (halfBinShift) {
				for (size_t i = 0; i < tmpTime.size(); ++i) {
					Complex t = tmpTime[i];
					Complex twist = halfBinTwists[i];
					time[2*i] = 	t.real()*twist.real() + t.imag()*twist.imag();
					time[2*i + 1] = t.imag()*twist.real() - t.real()*twist.imag();
				}
			} else {
				for (size_t i = 0; i < tmpTime.size(); ++i) {
					time[2*i] = tmpTime[i].real();
					time[2*i + 1] = tmpTime[i].imag();
				}
			}
		}
	}
	void ifft(const Sample *inR, const Sample *inI, Sample *outR) {
		for (size_t s = 0; s < steps(); ++s) {
			ifft(s, inR, inI, outR);
		}
	}
	void ifft(size_t step, const Sample *inR, const Sample *inI, Sample *outR) {
		if (step-- == 0) {
			size_t hSize = complexFft.size(), qSize = hSize/2;

			Sample bin0r = inR[0], bin0i = inI[0];
			if (!halfBinShift) {
				tmpFreq[0] = {
					bin0r + bin0i,
					bin0r - bin0i
				};
			}
			for (size_t i = halfBinShift ? 0 : 1; i <= qSize; ++i) {
				size_t conjI = halfBinShift ? (hSize - 1 - i) : (hSize - i);
				Complex twiddle = twiddles[i];
				Sample fir = inR[i], fii = inI[i];
				Sample fcir = inR[conjI], fcii = inI[conjI];

				Complex odd = {fir + fcir, fii - fcii};
				Complex evenRotMinusI = {fir - fcir, fii + fcii};
				Complex evenI = { // Conjugate
					evenRotMinusI.real()*twiddle.real() + evenRotMinusI.imag()*twiddle.imag(),
					evenRotMinusI.imag()*twiddle.real() - evenRotMinusI.real()*twiddle.imag()
				};

				tmpFreq[i] = odd + evenI;
				tmpFreq[conjI] = {odd.real() - evenI.real(), evenI.imag() - odd.imag()};
			}
		} else if (step < complexFft.steps()) {
			// Can't just use time as (Complex *), since it might not be aligned properly
			complexFft.ifft(step, tmpFreq.data(), tmpTime.data());
		} else {
			if (halfBinShift) {
				for (size_t i = 0; i < tmpTime.size(); ++i) {
					Complex t = tmpTime[i];
					Complex twist = halfBinTwists[i];
					outR[2*i] = 	t.real()*twist.real() + t.imag()*twist.imag();
					outR[2*i + 1] = t.imag()*twist.real() - t.real()*twist.imag();
				}
			} else {
				for (size_t i = 0; i < tmpTime.size(); ++i) {
					outR[2*i] = tmpTime[i].real();
					outR[2*i + 1] = tmpTime[i].imag();
				}
			}
		}
	}
private:
	std::vector<Complex> tmpFreq, tmpTime;
	std::vector<Complex> twiddles, halfBinTwists;

	using ComplexFFT = SplitFFT<Sample, splitComputation>;
	ComplexFFT complexFft;
};

template<typename Sample, bool splitComputation=false>
using ModifiedRealFFT = RealFFT<Sample, splitComputation, true>;

}} // namespace

// Platform-specific
#if defined(SIGNALSMITH_USE_ACCELERATE)
#	include "./platform/fft-accelerate.h"
#elif defined(SIGNALSMITH_USE_IPP)
#	include "./platform/fft-ipp.h"
#endif

#endif // include guard
