#ifndef __TACOTRON_UPSAMPLE_HPP__
#define __TACOTRON_UPSAMPLE_HPP__

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
			class upsample
			{
			private:
				sys::conv up_conv;
						
				cudnnTensorDescriptor_t input_desc, out_desc;
				
				gpu_float_array f1,f2, d_workspace;

			public:
				noCopy(upsample);
				upsample(){}
				void operator () (cudnnHandle_t& cudnn, gpu_float_array& input_t);
			 	void set(cudnnHandle_t& cudnn,  size_t totalNum);
				~upsample(); 
			};
		}
	}
}


#endif
