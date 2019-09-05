//# define NDEBUG // switch off all the assert calls. 
//#undef NDEBUG


#include<WN.hpp>
#include<upsample.hpp>
#include<hparams.hpp>
#include<logger.hpp>

#include<conv.hpp>
#include<data_types.hpp>
#include<cnpy.hpp>
#include<string>
#include<vector>
#include <chrono>
#include <unistd.h>

using namespace livai::tts;
using namespace livai::tts::common;

void testWaveglow(cudnnHandle_t& cudnn)
	{
		using namespace livai::tts::waveglow;
		using namespace std;
		using namespace livai::tts::common;

		//make and set objects of WN ans upsample//
		WN waveglow;
		upsample upsample;
		size_t max_length = hparams::max_length;
		waveglow.set(cudnn, max_length);
		upsample.set(cudnn, max_length);

		/* input-mel generated from either text-2-mel models such as tacotron, deepvoice or 
		        from ground truth fft in format [channels, length]*/
		auto input_m = cnpy::npy_load("/shared1/saurabh.m/waveglow/input_mel.npy");
		
		// initialize float_arrays of required dimension
		gpu_float_array input_mel, audio, d_workspace, upsampled_mel;
		d_workspace.init(100000000,1);
		input_mel.init(input_m.shape);
		audio.init(input_m.shape[2]*256,1);
		upsampled_mel.init(640, input_m.shape[2]*32);

		
		cudaMemcpy(input_mel.ptr, input_m.data<float_t>(), input_mel.size()*sizeof(float_t), cudaMemcpyHostToDevice);
		
		cudaDeviceSynchronize();
		auto start = chrono::steady_clock::now(), upsampler_end = start;

		int test_count=1;
		while(test_count>0)
		{
			upsample(cudnn, input_mel, upsampled_mel, d_workspace);
			cudaDeviceSynchronize();
			upsampler_end = chrono::steady_clock::now();

			waveglow(cudnn, upsampled_mel, audio, d_workspace);
			test_count--;
		}

		cudaDeviceSynchronize();
		auto end = chrono::steady_clock::now();

		log_d("Gen_out audio in waveglow", audio.log("gen_out.npy"));

		std::cout << "Elapsed time in milliseconds : " 
			<< chrono::duration_cast<chrono::milliseconds>(end - start).count()
			<< " ms" << std::endl;

		std::cout << "Time elapsed time in upsampler in milliseconds : " 
			<< chrono::duration_cast<chrono::milliseconds>(upsampler_end - start).count()
			<< " ms" << std::endl;

		}


int main()
{
	
	// create a cuda handle
	size_t device_id = 3;
	cudnnHandle_t cudnn;

	cudaSetDevice(device_id);

	checkCUDNN(cudnnCreate(&cudnn));
	testWaveglow(cudnn);
	cudnnDestroy(cudnn);
}


