#include "./stopwatch.h"
#if defined(__has_include) && __has_include("plot/signalsmith.h")
#	include "plot/signalsmith.h"
#else
#	include "plot/plot.h"
#endif

#include <iostream>
#include <complex>
#include <vector>
#include <random>

#define LOG_EXPR(expr) std::cout << #expr << " = " << (expr) << std::endl;

template<typename Sample>
struct RunData {
	using Complex = std::complex<Sample>;

	const int size;
	const int maxSize;
	std::vector<Complex> input, output;
	
	RunData(int size, int maxSize, int seed=0) : size(size), maxSize(maxSize), input(size), output(size), randomEngine(seed) {
		randomise();
	}
	
	void randomise() {
		std::uniform_real_distribution<Sample> dist{-1, 1};
		for (auto &v : input) {
			v = {dist(randomEngine), dist(randomEngine)};
		}
	}
	
	double errorCheck(int count=-1) {
		double error2 = 0;
		
		if (count < 0) count = size;
		std::uniform_int_distribution<int> dist{0, size - 1};
		
		for (int r = 0; r < count; ++r) {
			int f = (count == size) ? r : dist(randomEngine);
			Complex actual = output[f];
			Complex sum = 0;
			for (int i = 0; i < size; ++i) {
				Sample phase = Sample(-2*M_PI*f*i/size);
				sum += input[i]*std::polar(Sample(1), phase);
			}
			Complex error = sum - actual;
			error2 += std::norm(error);
		}
		return std::sqrt(error2/size);
	}
	
private:
	std::default_random_engine randomEngine;
};

template<class FftWrapper>
struct Runner {
	static constexpr double testSeconds = 0.05;//0.5;
	static constexpr double testChunk = 0.01;//0.1;

	const char *name;
	signalsmith::plot::Line2D &line;
	Stopwatch stopwatch{false};
	FftWrapper fft;
	
	Runner(const char *name, signalsmith::plot::Line2D &line, signalsmith::plot::Legend &legend) : name(name), line(line) {
		legend.add(line, name);
	}
	
	template<class Data>
	void run(double x, Data &data) {
		fft.prepare(data.size, data.maxSize);
		size_t rounds = 0, roundStep = 1;
		
		double dummySum = 1;
		double seconds = 0;
		while (seconds < testSeconds) {
			stopwatch.start();
			
			for (size_t r = 0; r < roundStep; ++r) {
				dummySum += fft.run(data);
			}
			
			double lap = stopwatch.seconds(stopwatch.lap());
			if (lap < testChunk) {
				roundStep *= 2;
			} else {
				seconds += lap;
				rounds += roundStep;
			}
		}
		double rps = rounds/seconds;
		double ref = 1e-8*data.size*(std::log2(data.size) + 1);
		double scaledRps = rps*ref;
		line.add(x, scaledRps);
		
		std::cout << data.size << "\t" << name;
		for (size_t c = std::strlen(name); c < 20; ++c) std::cout << " ";
		std::cout << "\tspeed: " << scaledRps;
		if (data.size <= 256) {
			std::cout << "\terror: " << data.errorCheck() << "\n";
		} else {
			std::cout << "\tdummy: " << dummySum << "\n";
		}
	}
};

// ---------- wrappers

#include "../simple-fft.h"
template<class Sample>
struct SimpleWrapper {
	signalsmith::fft2::SimpleFFT<Sample> fft;

	void prepare(int size, int) {
		fft.resize(size);
	}
	
	template<class Data>
	double run(Data &data) {
		fft.fft(data.size, data.input.data(), data.output.data());
		return data.output[0].real();
	}
};

#include "../split-fft.h"
template<class Sample>
struct SplitWrapper {
	signalsmith::fft2::SplitFFT<Sample> fft;

	void prepare(int size, int) {
		fft.resize(size);
	}
	
	template<class Data>
	double run(Data &data) {
		fft.fft(data.input.data(), data.output.data());
		return data.output[0].real();
	}
};

#include "./others/signalsmith-fft.h"
template<class Sample>
struct SignalsmithFFTWrapper {
	signalsmith::FFT<Sample> fft{1};

	void prepare(int size, int) {
		fft.setSize(size);
	}
	
	template<class Data>
	double run(Data &data) {
		fft.fft(data.input.data(), data.output.data());
		return data.output[0].real();
	}
};

#include "dsp/fft.h"
template<class Sample>
struct SignalsmithDSPWrapper {
	signalsmith::fft::FFT<Sample> fft{1};

	void prepare(int size, int) {
		fft.setSize(size);
	}
	
	template<class Data>
	double run(Data &data) {
		fft.fft(data.input.data(), data.output.data());
		return data.output[0].real();
	}
};

#ifdef INCLUDE_KISS
#define KISSFFT_DATATYPE float
#include "others/kissfft/kiss_fft.h"
struct KissFloatWrapper {
	kiss_fft_cfg cfg;
	bool hasConfig = false;
	
	~KissFloatWrapper() {
		if (hasConfig) kiss_fft_free(cfg);
	}

	void prepare(int size, int) {
		if (hasConfig) kiss_fft_free(cfg);
		cfg = kiss_fft_alloc(size, false, 0, 0);
		hasConfig = true;
	}
	
	template<class Data>
	double run(Data &data) {
		kiss_fft(cfg, (kiss_fft_cpx *)data.input.data(), (kiss_fft_cpx *)data.output.data());
		return data.output[0].real();
	}
};
#endif

#include <Accelerate/Accelerate.h>
struct AccelerateFloatWrapper {
	bool hasSetup = false;
	FFTSetup fftSetup;
	int log2 = 0;
	
	std::vector<float> splitReal, splitImag;

	AccelerateFloatWrapper() {}
	~AccelerateFloatWrapper() {
		if (hasSetup) vDSP_destroy_fftsetup(fftSetup);
	}
	
	void prepare(int size, int) {
		if (hasSetup) vDSP_destroy_fftsetup(fftSetup);
		log2 = std::round(std::log2(size));
		fftSetup =  vDSP_create_fftsetup(log2, FFT_RADIX2);
		hasSetup = true;
		
		splitReal.resize(size);
		splitImag.resize(size);
	}
	
	template<class Data>
	double run(Data &data) {
		DSPSplitComplex splitComplex{splitReal.data(), splitImag.data()};
		vDSP_ctoz((DSPComplex *)data.input.data(), 2, &splitComplex, 1, data.size);
		
		vDSP_fft_zip(fftSetup, &splitComplex, 1, log2, kFFTDirection_Forward);

		vDSP_ztoc(&splitComplex, 1, (DSPComplex *)data.output.data(), 2, data.size);
		return data.output[0].real();
	}
};
struct AccelerateDoubleWrapper {
	bool hasSetup = false;
	FFTSetupD fftSetup;
	int log2 = 0;
	
	std::vector<double> splitReal, splitImag;

	AccelerateDoubleWrapper() {}
	~AccelerateDoubleWrapper() {
		if (hasSetup) vDSP_destroy_fftsetupD(fftSetup);
	}
	
	void prepare(int size, int) {
		if (hasSetup) vDSP_destroy_fftsetupD(fftSetup);
		log2 = std::round(std::log2(size));
		fftSetup =  vDSP_create_fftsetupD(log2, FFT_RADIX2);
		hasSetup = true;
		
		splitReal.resize(size);
		splitImag.resize(size);
	}
	
	template<class Data>
	double run(Data &data) {
		DSPDoubleSplitComplex splitComplex{splitReal.data(), splitImag.data()};
		vDSP_ctozD((DSPDoubleComplex *)data.input.data(), 2, &splitComplex, 1, data.size);
		
		vDSP_fft_zipD(fftSetup, &splitComplex, 1, log2, kFFTDirection_Forward);

		vDSP_ztocD(&splitComplex, 1, (DSPDoubleComplex *)data.output.data(), 2, data.size);
		return data.output[0].real();
	}
};

// ---------- main code

int main() {
	signalsmith::plot::Plot2D fastSizePlot(200, 200);
	auto &fastSizeLine = fastSizePlot.line();
	for (int n = 1; n < 65536; ++n) {
		int fastN = signalsmith::fft2::SplitFFT<double>::fastSizeAbove(n);
		fastSizeLine.add(std::log2(n), std::log2(fastN));
	}
	fastSizePlot.line().add(0, 0).add(16, 16);
	fastSizePlot.line().add(0, std::log2(1.25)).add(16 - std::log2(1.25), 16);
	fastSizePlot.write("fast-sizes.svg");

	signalsmith::plot::Figure figure;
	auto &plot = figure.plot(800, 250);
	plot.x.label("FFT size");
	
	auto &legend = plot.legend(0, 1);
	Runner<SimpleWrapper<double>> simpleDouble("simple (double)", plot.line(), legend);
	Runner<SimpleWrapper<float>> simpleFloat("simple (float)", plot.line(), legend);
	Runner<SplitWrapper<double>> splitDouble("split (double)", plot.line(), legend);
	Runner<SplitWrapper<float>> splitFloat("split (float)", plot.line(), legend);
	Runner<SignalsmithDSPWrapper<double>> dspDouble("DSP library (double)", plot.line(), legend);
	Runner<SignalsmithDSPWrapper<float>> dspFloat("DSP library (float)", plot.line(), legend);
	Runner<AccelerateDoubleWrapper> accelerateDouble("Accelerate (double)", plot.line(), legend);
	Runner<AccelerateFloatWrapper> accelerateFloat("Accelerate (float)", plot.line(), legend);
#ifdef INCLUDE_KISS
	Runner<KissFloatWrapper> kissFloat("KISS (float)", plot.line(), legend);
#endif

	int maxSize = 65536*8;
	bool first = true;
	auto runSize = [&](int n, int pow3=0, int pow5=0, int pow7=0){
		double x = std::log2(n);
		RunData<double> dataDouble(n, maxSize, n);
		RunData<float> dataFloat(n, maxSize, n);

		if (pow3 + pow5 + pow7 == 0) {
			simpleDouble.run(x, dataDouble);
			simpleFloat.run(x, dataFloat);
			accelerateDouble.run(x, dataDouble);
			accelerateFloat.run(x, dataFloat);
		}
		if (pow5 + pow7 == 0) {
			dspDouble.run(x, dataDouble);
			dspFloat.run(x, dataFloat);
		}
#ifdef INCLUDE_KISS
		kissFloat.run(x, dataFloat);
#endif
		if (signalsmith::fft2::SplitFFT<double>::fastSizeAbove(n) == size_t(n)) {
			splitDouble.run(x, dataDouble);
			splitFloat.run(x, dataFloat);
		}

		if (first) {
			first = false;
			plot.x.major(x, std::to_string(n));
		} else if (pow3 + pow5 + pow7 == 0) {
			plot.x.tick(x, std::to_string(n));
		}
	};
	for (int n = 1; n <= maxSize; n *= 2) {
		if (n/16) runSize(n*9/16, 2);
		if (n/8) runSize(n*5/8, 0, 1);
		if (n/4) runSize(n*3/4, 1);
		if (n/8) runSize(n*7/8, 1, 1);
		//if (n/16) runSize(n*15/16, 1, 1);
		runSize(n);
	}
	plot.y.major(0); // auto-scaled range includes 0
	plot.y.blankLabels().label("speed"); // values don't matter, only the comparison
	figure.write("comparison.svg");
}
