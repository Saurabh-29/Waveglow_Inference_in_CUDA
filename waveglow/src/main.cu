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
		auto input_m = cnpy::npy_load("/shared1/saurabh.m/waveglow/input_mel.npy");

		gpu_float_array input_mel, upsampled_mel, d_workspace;

		input_mel.init(input_m.shape);
		upsampled_mel.init(640, input_m.shape[2]*32);
		d_workspace.init(1000000,1);

		cudaMemcpy(input_mel.ptr, input_m.data<float_t>(), input_mel.size()*sizeof(float_t), cudaMemcpyHostToDevice);
		upsample.set(cudnn, input_mel.shape[2]);

		auto start = chrono::steady_clock::now();

		// upsample(cudnn, input_mel, upsampled_mel);
		
		cudaDeviceSynchronize();
		auto end = chrono::steady_clock::now();

		log_d("final mel", upsampled_mel.log("gen_upsamplee_mel.npy"));

		std::cout << "Elapsed time in milliseconds : " 
			<< chrono::duration_cast<chrono::milliseconds>(end - start).count()
			<< " ms" << std::endl;
		}

void testWN(cudnnHandle_t& cudnn)
	{
		using namespace livai::tts::waveglow;
		using namespace std;
		using namespace livai::tts::common;

		WN wavenet;
		upsample upsample;
		size_t max_length = 1500*32;
		wavenet.set(cudnn, max_length);
		upsample.set(cudnn, max_length);

		std::cout<<"test waveglow code is running"<<"\n";
		auto input_m = cnpy::npy_load("/shared1/saurabh.m/waveglow/input_mel.npy");

		gpu_float_array input_mel, audio, d_workspace, upsampled_mel;
		d_workspace.init(1000000,1);
		input_mel.init(input_m.shape);
		audio.init(input_m.shape[2]*256,1);
		
		cudaMemcpy(input_mel.ptr, input_m.data<float_t>(), input_mel.size()*sizeof(float_t), cudaMemcpyHostToDevice);
	
		upsampled_mel.init(640, input_m.shape[2]*32);


		auto start = chrono::steady_clock::now();
		int test_count=1;
		while(test_count>0)
		{
			upsample(cudnn, input_mel, upsampled_mel, d_workspace);
			wavenet(cudnn, upsampled_mel, audio, d_workspace);
			test_count--;
		}

		cudaDeviceSynchronize();
		auto end = chrono::steady_clock::now();

		log_d("Gen_out audio in waveglow", audio.log("gen_out.npy"));

		std::cout << "Elapsed time in milliseconds : " 
			<< chrono::duration_cast<chrono::milliseconds>(end - start).count()
			<< " ms" << std::endl;

		// std::cout << "Elapsed time in milliseconds : " 
		// 	<< chrono::duration_cast<chrono::milliseconds>(end2 - start).count()
		// 	<< " ms" << std::endl;
		}


int main()
{
// create a cuda handle
	cudnnHandle_t cudnn;
	(cudaSetDevice(1));
	checkCUDNN(cudnnCreate(&cudnn));
	// testupsampler(cudnn);
	testWN(cudnn);
	cudnnDestroy(cudnn);
}


