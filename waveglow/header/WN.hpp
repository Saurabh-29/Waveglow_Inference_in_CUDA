#ifndef __TACOTRON_WN_HPP__
#define __TACOTRON_WN_HPP__

#pragma once 

#include<memory> 
#include<conv.hpp>
#include<hparams.hpp>
#include<dense.hpp>
#include <curand.h>


namespace livai
{
	namespace tts
	{
		namespace waveglow
		{
			class WN
			{
			private:
				sys::conv start_conv[12], end_conv[12], inv_conv[12]; 
				sys::conv in_conv[12][8]; 
				sys::conv cond_conv[12][8]; 
				sys::conv res_skip_conv[12][8]; 
				
				curandGenerator_t rng;

				cudnnTensorDescriptor_t input_desc, out_desc;
				
				size_t input_len, dil_t, num_values, n_threads, num_blocks;
				size_t n_channels, n_flows, n_layers, n_groups, n_rem_channels;
				gpu_float_array f1,f2,f3,f4,temp, temp_input, f6, d_workspace, z, input_t;

			public:
				noCopy(WN);
				WN(){}
				void operator () (cudnnHandle_t& cudnn, gpu_float_array& mel_input, gpu_float_array& d_output);
			 	void set(cudnnHandle_t& cudnn, size_t totalNum);
			 	// void dynamic_set(cudnnHandle_t& cudnn, size_t totalNum);
				~WN(); 
			};
		}
	}
}


#endif
