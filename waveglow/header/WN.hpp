#ifndef __TACOTRON_WN_HPP__
#define __TACOTRON_WN_HPP__

#pragma once 

#include<memory> 
#include<conv.hpp>
#include<hparams.hpp>
#include<dense.hpp>


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
				
				cudnnTensorDescriptor_t input_desc, out_desc, out_desc2;
				
				size_t input_len, dil_t, num_values, threads, num_blocks;
				gpu_float_array f1,f2,f3,f4,temp, temp_input, f6, d_workspace;

			public:
				noCopy(WN);
				WN(){}
				void operator () (cudnnHandle_t& cudnn, gpu_float_array& input_t, gpu_float_array& mel_input, 
								gpu_float_array& z4, gpu_float_array& z8, gpu_float_array& d_output);
			 	void set(cudnnHandle_t& cudnn, size_t k, size_t totalNum);
			 	// void dynamic_set(cudnnHandle_t& cudnn, size_t totalNum);
				~WN(); 
			};
		}
	}
}


#endif
