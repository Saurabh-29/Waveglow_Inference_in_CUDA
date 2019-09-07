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
#include <fstream>
#include <string>

using namespace livai::tts;
using namespace livai::tts::common;

void testWaveglow(cudnnHandle_t& cudnn, std::string filename, std::string out_folder)
	{
		using namespace livai::tts::waveglow;
		using namespace std;
		using namespace livai::tts::common;

		std::cout<<"the folder_name and out_folder is"<<filename<<"\t"<<out_folder<<"\n";

		std::string line;
		std::ifstream infile(filename);

		//make and set objects of WN ans upsample//
		WN waveglow;
		upsample upsample;
		size_t max_length = hparams::max_length;
		waveglow.set(cudnn, max_length);
		upsample.set(cudnn, max_length);


		// initialize float_arrays of required max dimension
		gpu_float_array input_mel, audio, d_workspace, upsampled_mel;
		d_workspace.init(100000000,1);
		input_mel.init(hparams::mel_dim, max_length);
		audio.init(max_length*hparams::stride,1);
		upsampled_mel.init(640, max_length*32);


		while (std::getline(infile, line))
		{
			std::string outputFilename = out_folder+line.substr(line.find_last_of("/")+1);
			std::cout<<"\nsynthesizing for " << line<<"\t and saving at  "<<outputFilename<<"\n";
		
			auto input_m = cnpy::npy_load(line);
			
			input_mel.reshape(input_m.shape[0], input_m.shape[1]);
			audio.reshape(input_m.shape[1]*256,1);
			upsampled_mel.reshape(640, input_m.shape[1]*32);

			
			cudaMemcpy(input_mel.ptr, input_m.data<float_t>(), input_mel.size()*sizeof(float_t), cudaMemcpyHostToDevice);
			
			cudaDeviceSynchronize();
			auto start = chrono::steady_clock::now();

			
			upsample(cudnn, input_mel, upsampled_mel, d_workspace);
			cudaDeviceSynchronize();

			waveglow(cudnn, upsampled_mel, audio, d_workspace);
				
			cudaDeviceSynchronize();
			auto end = chrono::steady_clock::now();

			log_d("audio is generated", audio.log(outputFilename));

			std::cout << "Elapsed time in milliseconds : " 
				<< chrono::duration_cast<chrono::milliseconds>(end - start).count()
				<< " ms" << "\n";
		}

	}


int main(int argc, char *argv[])
{
	
	std::string filename(argv[1]);
	std::string out_folder(argv[2]);

	std::cout<<argc<<"\t"<<filename<<"\t"<<out_folder<<"\n";

	// create a cuda handle
	size_t device_id = 3;
	cudnnHandle_t cudnn;
	cudaSetDevice(device_id);
	checkCUDNN(cudnnCreate(&cudnn));
	testWaveglow(cudnn, filename, out_folder);
	cudnnDestroy(cudnn);
}


