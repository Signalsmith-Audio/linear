#ifndef FFT2_FFT2_ACCELERATE_H
#define FFT2_FFT2_ACCELERATE_H

#include <Accelerate/Accelerate.h>

namespace signalsmith { namespace fft2 {

template<>
struct Pow2FFT<float> {
	using Complex = std::complex<float>;

	Pow2FFT(size_t size = 0) {
		if (size > 0) resize(size);
	}
	~Pow2FFT() {
		if (hasSetup) vDSP_destroy_fftsetup(fftSetup);
	}

	void resize(int size) {
		_size = size;
		if (hasSetup) vDSP_destroy_fftsetup(fftSetup);
		log2 = std::round(std::log2(size));
		fftSetup = vDSP_create_fftsetup(log2, FFT_RADIX2);
		hasSetup = true;

		splitReal.resize(size);
		splitImag.resize(size);
	}

	void fft(const Complex* input, Complex* output) {
		DSPSplitComplex splitComplex{ splitReal.data(), splitImag.data() };
		vDSP_ctoz((DSPComplex*)input, 2, &splitComplex, 1, _size);
		vDSP_fft_zip(fftSetup, &splitComplex, 1, log2, kFFTDirection_Forward);
		vDSP_ztoc(&splitComplex, 1, (DSPComplex*)output, 2, _size);
	}

	void fftStrideTime(size_t stride, const Complex* input, Complex* output) {
		DSPSplitComplex splitComplex{ splitReal.data(), splitImag.data() };
		vDSP_ctoz((DSPComplex*)input, 2 * stride, &splitComplex, 1, _size);
		vDSP_fft_zip(fftSetup, &splitComplex, 1, log2, kFFTDirection_Forward);
		vDSP_ztoc(&splitComplex, 1, (DSPComplex*)output, 2, _size);
	}
private:
	size_t _size = 0;
	bool hasSetup = false;
	FFTSetup fftSetup;
	int log2 = 0;
	std::vector<float> splitReal, splitImag;
};
template<>
struct Pow2FFT<double> {
	using Complex = std::complex<double>;

	Pow2FFT(size_t size = 0) {
		if (size > 0) resize(size);
	}
	~Pow2FFT() {
		if (hasSetup) vDSP_destroy_fftsetupD(fftSetup);
	}

	void resize(int size) {
		_size = size;
		if (hasSetup) vDSP_destroy_fftsetupD(fftSetup);
		log2 = std::round(std::log2(size));
		fftSetup = vDSP_create_fftsetupD(log2, FFT_RADIX2);
		hasSetup = true;

		splitReal.resize(size);
		splitImag.resize(size);
	}

	void fft(const Complex* input, Complex* output) {
		DSPDoubleSplitComplex splitComplex{ splitReal.data(), splitImag.data() };
		vDSP_ctozD((DSPDoubleComplex*)input, 2, &splitComplex, 1, _size);
		vDSP_fft_zipD(fftSetup, &splitComplex, 1, log2, kFFTDirection_Forward);
		vDSP_ztocD(&splitComplex, 1, (DSPDoubleComplex*)output, 2, _size);
	}

	void fftStrideTime(size_t stride, const Complex* input, Complex* output) {
		DSPDoubleSplitComplex splitComplex{ splitReal.data(), splitImag.data() };
		vDSP_ctozD((DSPDoubleComplex*)input, 2 * stride, &splitComplex, 1, _size);
		vDSP_fft_zipD(fftSetup, &splitComplex, 1, log2, kFFTDirection_Forward);
		vDSP_ztocD(&splitComplex, 1, (DSPDoubleComplex*)output, 2, _size);
	}
private:
	size_t _size = 0;
	bool hasSetup = false;
	FFTSetupD fftSetup;
	int log2 = 0;
	std::vector<double> splitReal, splitImag;
};

}} // namespace

#endif // include guard
