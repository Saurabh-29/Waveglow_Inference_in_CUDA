//# define NDEBUG // switch off all the assert calls. 
//#undef NDEBUG


#include<WN.hpp>
#include<upsample.hpp>
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

void testupsampler(cudnnHandle_t& cudnn)
	{
		using namespace livai::tts::waveglow;
		using namespace std;
		using namespace livai::tts::common ;

		upsample upsample;

		std::cout<<"test upsample code is running"<<"\n";
		auto input_m = cnpy::npy_load("/shared1/saurabh.m/waveglow/input_spect.npy");

		gpu_float_array input_mel;

		input_mel.init(input_m.shape);

		cudaMemcpy(input_mel.ptr, input_m.data<float_t>(), input_mel.size()*sizeof(float_t), cudaMemcpyHostToDevice);
		upsample.set(cudnn, input_mel.shape[2]);

		auto start = chrono::steady_clock::now();

		upsample(cudnn, input_mel);
		
		cudaDeviceSynchronize();
		auto end = chrono::steady_clock::now();

		// log_d("Gen_out audio in wavegen", final_out.log("gen_out.npy"));

		std::cout << "Elapsed time in milliseconds : " 
			<< chrono::duration_cast<chrono::milliseconds>(end - start).count()
			<< " ms" << std::endl;
		}

void testWN(cudnnHandle_t& cudnn)
	{
		using namespace livai::tts::waveglow;
		using namespace std;
		using namespace livai::tts::common ;

		WN wavenet;

		std::cout<<"test waveglow code is running"<<"\n";
		// auto input_t = cnpy::npy_load("/shared1/saurabh.m/waveglow/input_audio.npy");
		auto input_t = cnpy::npy_load("/shared1/saurabh.m/waveglow/audio_11.npy");
		auto input_m = cnpy::npy_load("/shared1/saurabh.m/waveglow/input_spect.npy");

		auto z_4 = cnpy::npy_load("/shared1/saurabh.m/waveglow/z_4.npy");
		auto z_8 = cnpy::npy_load("/shared1/saurabh.m/waveglow/z_8.npy");

		gpu_float_array input_tensor, input_mel, final_out, d_workspace, z4, z8;

		input_tensor.init(1, 8, input_t.shape[2]);
		input_mel.init(input_m.shape);
		z4.init(z_4.shape);
		z8.init(z_8.shape);
		input_tensor.reshape(1,4,input_t.shape[2]);

		cudaMemcpy(input_tensor.ptr, input_t.data<float_t>(), input_tensor.size()*sizeof(float_t), cudaMemcpyHostToDevice);
		cudaMemcpy(input_mel.ptr, input_m.data<float_t>(), input_mel.size()*sizeof(float_t), cudaMemcpyHostToDevice);
		cudaMemcpy(z4.ptr, z_4.data<float_t>(), z4.size()*sizeof(float_t), cudaMemcpyHostToDevice);
		cudaMemcpy(z8.ptr, z_8.data<float_t>(), z8.size()*sizeof(float_t), cudaMemcpyHostToDevice);


		// final_out.init(1,totalNum*256);

		wavenet.set(cudnn, 11, input_tensor.shape[2]);
		d_workspace.init(2170112/2,1);

		auto start = chrono::steady_clock::now();

		wavenet(cudnn, input_tensor, input_mel, z4, z8);
		
		cudaDeviceSynchronize();
		auto end = chrono::steady_clock::now();

		// log_d("Gen_out audio in wavegen", final_out.log("gen_out.npy"));

		std::cout << "Elapsed time in milliseconds : " 
			<< chrono::duration_cast<chrono::milliseconds>(end - start).count()
			<< " ms" << std::endl;
		}


int main()
{
// create a cuda handle
	cudnnHandle_t cudnn;
	(cudaSetDevice(1));
	checkCUDNN(cudnnCreate(&cudnn));
	testupsampler(cudnn);
	// testWN(cudnn);
	cudnnDestroy(cudnn);
}


