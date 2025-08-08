#ifndef SIGNALSMITH_LINEAR_PLATFORM_FFT_IPP_H
#define SIGNALSMITH_LINEAR_PLATFORM_FFT_IPP_H

#include "pffft/pffft.h"

#include <memory>
#include <cmath>
#include <complex>
#include <cassert>

namespace signalsmith { namespace linear {

template<>
struct Pow2FFT<float> {
	static constexpr bool prefersSplit = false;

	using Complex = std::complex<float>;

	Pow2FFT(size_t size=0) {
		resize(size);
	}
	~Pow2FFT() {
		resize(0); // frees everything
	}

	void resize(size_t size) {
		_size = size;
		if (hasSetup) pffft_destroy_setup(fftSetup);
		if (work) pffft_aligned_free(work);
		if (tmpInterleaved) pffft_aligned_free(tmpInterleaved);

		// We use this for split-real, even if there's no PFFFT setup
		tmpInterleaved = (float *)pffft_aligned_malloc(sizeof(float)*size*2);

		if (size < 32) {
			// PFFFT doesn't support smaller sizes
			hasSetup = false;
			return;
		}
		
		work = (float *)pffft_aligned_malloc(sizeof(float)*size*2);
		fftSetup = pffft_new_setup(int(size), PFFFT_COMPLEX);
		hasSetup = fftSetup;
	}

	void fft(const Complex* input, Complex* output) {
		fftInner(input, output, PFFFT_FORWARD);
	}
	void fft(const float *inR, const float *inI, float *outR, float *outI) {
		fftInner(inR, inI, outR, outI, PFFFT_FORWARD);
	}

	void ifft(const Complex* input, Complex* output) {
		fftInner(input, output, PFFFT_BACKWARD);
	}
	void ifft(const float *inR, const float *inI, float *outR, float *outI) {
		fftInner(inR, inI, outR, outI, PFFFT_BACKWARD);
	}

private:
	void fftInner(const Complex *input, Complex *output, pffft_direction_t direction) {
		// For small FFT sizes (no setup), use the fallback
		if (!hasSetup) {
			if (output != input) {
				std::memcpy(output, input, sizeof(Complex)*_size);
			}
			if (direction == PFFFT_FORWARD) {
				_impl::fallbackInPlaceFFT(_size, output);
			} else {
				_impl::fallbackInPlaceIFFT(_size, output);
			}
			return;
		}
		// 16-byte alignment
		if (size_t(input)&0x0F) {
			// `tmpInterleaved` is always aligned, so copy into that
			std::memcpy(tmpInterleaved, input, sizeof(Complex)*_size);
			input = (const Complex *)tmpInterleaved;
		}
		if (size_t(output)&0x0F) {
			// Output to `tmpInterleaved` - might be in-place if input is unaligned, but that's fine
			pffft_transform_ordered(fftSetup, (const float *)input, tmpInterleaved, work, direction);
			std::memcpy(output, tmpInterleaved, sizeof(Complex)*_size);
		} else {
			pffft_transform_ordered(fftSetup, (const float *)input, (float *)output, work, direction);
		}
	}
	void fftInner(const float *inR, const float *inI, float *outR, float *outI, pffft_direction_t direction) {
		for (size_t i = 0; i < _size; ++i) {
			tmpInterleaved[2*i] = inR[i];
			tmpInterleaved[2*i + 1] = inI[i];
		}
		// PFFFT supports in-place transforms
		fftInner((const Complex *)tmpInterleaved, (Complex *)tmpInterleaved, direction);
		// Un-interleave
		for (size_t i = 0; i < _size; ++i) {
			outR[i] = tmpInterleaved[2*i];
			outI[i] = tmpInterleaved[2*i + 1];
		}
	}

	size_t _size = 0;
	bool hasSetup = false;
	PFFFT_Setup *fftSetup;
	float *work = nullptr, *tmpInterleaved = nullptr;
};

/*
template<>
struct Pow2RealFFT<float> {
	static constexpr bool prefersSplit = true;

	using Complex = std::complex<float>;

	Pow2RealFFT(size_t size = 0) {
		resize(size);
	}
	~Pow2RealFFT() {
		if (hasSetup) vDSP_destroy_fftsetup(fftSetup);
	}

	void resize(size_t size) {
		_size = size;
		if (hasSetup) vDSP_destroy_fftsetup(fftSetup);
		if (!size) {
			hasSetup = false;
			return;
		}

		splitReal.resize(size);
		splitImag.resize(size);
		log2 = std::log2(size);
		fftSetup = vDSP_create_fftsetup(log2, FFT_RADIX2);
		hasSetup = fftSetup;
	}

	void fft(const float* input, Complex* output) {
		float mul = 0.5f;
		vDSP_vsmul(input, 2, &mul, splitReal.data(), 1, _size/2);
		vDSP_vsmul(input + 1, 2, &mul, splitImag.data(), 1, _size/2);
		DSPSplitComplex tmpSplit{splitReal.data(), splitImag.data()};
		vDSP_fft_zrip(fftSetup, &tmpSplit, 1, log2, kFFTDirection_Forward);
		vDSP_ztoc(&tmpSplit, 1, (DSPComplex *)output, 2, _size/2);
	}
	void fft(const float *inR, float *outR, float *outI) {
		DSPSplitComplex outputSplit{outR, outI};
		float mul = 0.5f;
		vDSP_vsmul(inR, 2, &mul, outR, 1, _size/2);
		vDSP_vsmul(inR + 1, 2, &mul, outI, 1, _size/2);
		vDSP_fft_zrip(fftSetup, &outputSplit, 1, log2, kFFTDirection_Forward);
	}

	void ifft(const Complex * input, float * output) {
		DSPSplitComplex tmpSplit{splitReal.data(), splitImag.data()};
		vDSP_ctoz((DSPComplex*)input, 2, &tmpSplit, 1, _size/2);
		vDSP_fft_zrip(fftSetup, &tmpSplit, 1, log2, kFFTDirection_Inverse);
		DSPSplitComplex outputSplit{output, output + 1};
		vDSP_zvmov(&tmpSplit, 1, &outputSplit, 2, _size/2);
	}
	void ifft(const float *inR, const float *inI, float *outR) {
		DSPSplitComplex inputSplit{(float *)inR, (float *)inI};
		DSPSplitComplex tmpSplit{splitReal.data(), splitImag.data()};
		vDSP_fft_zrop(fftSetup, &inputSplit, 1, &tmpSplit, 1, log2, kFFTDirection_Inverse);
		DSPSplitComplex outputSplit{outR, outR + 1};
		// We can't use vDSP_ztoc without knowing the alignment
		vDSP_zvmov(&tmpSplit, 1, &outputSplit, 2, _size/2);
	}

private:
	size_t _size = 0;
	bool hasSetup = false;
	FFTSetup fftSetup;
	int log2 = 0;
	std::vector<float> splitReal, splitImag;
};
*/

}} // namespace
#endif // include guard
