#ifndef __ACTIVATION_HPP__
#define __ACTIVATION_HPP__

#include <data_types.hpp>
#include <cudnn.h>
#include<logger.hpp>


namespace livai {
	namespace tts {

		using namespace common;

		namespace sys {

			class activation
			{
			private:
				cudnnTensorDescriptor_t in_out_descriptor;
				cudnnActivationDescriptor_t activation_descriptor;

			public:
				noCopy(activation);
				activation(){}
				/* 
				Alloc memory at device and memcopy the parameters ( shared memory )
				* */
				void init(size_t in_rows, size_t in_cols, size_t in_channels,
					size_t batch_size = 1, cudnnActivationMode_t mode = CUDNN_ACTIVATION_RELU)
				{
					checkCUDNN(cudnnCreateTensorDescriptor(&in_out_descriptor));
					checkCUDNN(cudnnSetTensor4dDescriptor(in_out_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/batch_size,
                                      /*channels=*/in_channels,
                                      /*image_height=*/in_rows,
                                      /*image_width=*/in_cols));

						checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
						checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,
                                        /*mode=*/mode,
                                        /*reluNanOpt=*/CUDNN_PROPAGATE_NAN,
                                        /*relu_coef=*/0));

					}

					void operator () (cudnnHandle_t& cudnn, gpu_float_array& d_input)
					{
						const float alpha = 1, beta = 0;
					// Perform the forward pass of the activation
						checkCUDNN(cudnnActivationForward(cudnn,
							activation_descriptor,
							&alpha,
							in_out_descriptor,
							d_input.ptr,
							&beta,
							in_out_descriptor,
							d_input.ptr));


						d_input.reset(d_input.size());
						
					}

				// free host & device memory
					~activation()
					{
						cudnnDestroyTensorDescriptor(in_out_descriptor);
						cudnnDestroyActivationDescriptor(activation_descriptor);
					}
				};
			}
		}
	}
#endif