#ifndef __CONV_HPP__
#define __CONV_HPP__

#include <data_types.hpp>
#include <cudnn.h>
#include<logger.hpp>


namespace livai {
	namespace tts {

		using namespace common;

		namespace sys {

			class conv
			{
			private:
				cudnnTensorFormat_t default_tensor_format;
				cudnnDataType_t default_data_type;

				gpu_float_array d_workspace; 
				gpu_float_array d_kernel;
				gpu_float_array d_bias;
				gpu_float_array d_z;
				
				size_t workspace_bytes;

				cudnnHandle_t cudnn;
				cudnnTensorDescriptor_t input_descriptor;
				cudnnTensorDescriptor_t z_descriptor;

				cudnnTensorDescriptor_t output_descriptor;
				cudnnFilterDescriptor_t kernel_descriptor;
				cudnnTensorDescriptor_t bias_descriptor;
				cudnnConvolutionDescriptor_t convolution_descriptor;
				cudnnConvolutionFwdAlgo_t convolution_algorithm;
				cudnnActivationDescriptor_t activation_descriptor;

				
				void initModelWeight(const cnpy::NpyArray& h_kernel, const cnpy::NpyArray& h_bias)
				{
					// load kernel
					d_kernel.init(h_kernel.shape);
					cudaMemcpy(d_kernel.ptr, h_kernel.data<float_t>(), d_kernel.size()*sizeof(float_t), cudaMemcpyHostToDevice);

					// load bias
					d_bias.init(h_bias.shape);
					cudaMemcpy(d_bias.ptr, h_bias.data<float_t>(), d_bias.size()*sizeof(float_t), cudaMemcpyHostToDevice);
				}

				void initConvTensors(size_t in_rows, size_t in_cols, size_t in_channels, 
					size_t out_rows, size_t out_cols, size_t out_channels, 
					size_t kernel_height, size_t kernel_width, 
					size_t batch_size)
				{
					checkCUDNN(cudnnCreate(&cudnn));
				
					checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
					checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                      /*format=*/default_tensor_format,
                                      /*dataType=*/default_data_type,
                                      /*batch_size=*/batch_size,
                                      /*channels=*/in_channels,
                                      /*image_height=*/in_rows,
                                      /*image_width=*/in_cols));

					checkCUDNN(cudnnCreateTensorDescriptor(&z_descriptor));
					checkCUDNN(cudnnSetTensor4dDescriptor(z_descriptor,
                                      /*format=*/default_tensor_format,
                                      /*dataType=*/default_data_type,
                                      /*batch_size=*/batch_size,
                                      /*channels=*/out_channels,
                                      /*image_height=*/out_rows,
                                      /*image_width=*/out_cols));

					checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
					checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                      /*format=*/default_tensor_format,
                                      /*dataType=*/default_data_type,
                                      /*batch_size=*/batch_size,
                                      /*channels=*/out_channels,
                                      /*image_height=*/out_rows,
                                      /*image_width=*/out_cols));

					checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
					checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                      /*dataType=*/default_data_type,
                                      /*format=*/default_tensor_format,
                                      /*out_channels=*/out_channels,
                                      /*in_channels=*/in_channels,
                                      /*kernel_height=*/kernel_height,
                                      /*kernel_width=*/kernel_width));

					checkCUDNN(cudnnCreateTensorDescriptor(&bias_descriptor));
					checkCUDNN(cudnnSetTensor4dDescriptor(bias_descriptor,
                                      /*format=*/default_tensor_format,
                                      /*dataType=*/default_data_type,
                                      /*batch_size=*/1,
                                      /*channels=*/out_channels,
                                      /*image_height=*/1,
                                      /*image_width=*/1));

					checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
					checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,
                                        /*mode=*/CUDNN_ACTIVATION_RELU,
                                        /*reluNanOpt=*/CUDNN_NOT_PROPAGATE_NAN,
                                        /*relu_coef=*/0));

					// compute pad
					size_t pad_height = (out_rows - 1 + kernel_height - in_rows) / 2 ; 
					size_t pad_width = (out_cols - 1 + kernel_width - in_cols) / 2 ; 

					checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
					checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                           /*pad_height=*/pad_height,
                                           /*pad_width=*/pad_width,
                                           /*vertical_stride=*/1,
                                           /*horizontal_stride=*/1,
                                           /*dilation_height=*/1,
                                           /*dilation_width=*/1,
                                           /*mode=*/CUDNN_CROSS_CORRELATION,
                                           /*computeType=*/CUDNN_DATA_FLOAT));

					//convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_​PRECOMP_GEMM;

					
					checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
                                        input_descriptor,
                                        kernel_descriptor,
                                        convolution_descriptor,
                                        output_descriptor,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                        0,
                                        &convolution_algorithm));
					

					workspace_bytes = 0;
					checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                   input_descriptor,
                                                   kernel_descriptor,
                                                   convolution_descriptor,
                                                   output_descriptor,
                                                   convolution_algorithm,
                                                   &workspace_bytes));

					log_d("Workspace size",workspace_bytes);
					size_t workspace_size = (workspace_bytes / sizeof(float_t)) + 1;
					d_workspace.init(workspace_size);
				}

			public:
				noCopy(conv);
				conv(){}
				/* 
				Alloc memory at device and memcopy the parameters ( shared memory )
				* */
				void init(const cnpy::NpyArray& h_kernel, const cnpy::NpyArray& h_bias, 
					size_t in_rows, size_t in_cols, size_t in_channels, 
					size_t out_rows, size_t out_cols, size_t out_channels, 
					size_t kernel_height, size_t kernel_width, 
					size_t batch_size = 1)
				{
					default_tensor_format = CUDNN_TENSOR_NHWC;
					default_data_type = CUDNN_DATA_FLOAT;

					initConvTensors(in_rows, in_cols, in_channels, out_rows, out_cols, out_channels, kernel_height, kernel_width, batch_size);
					initModelWeight(h_kernel, h_bias);

					// initialise d_z with zeros..
					d_z.init(out_cols,out_channels);
					d_z.reset();
				}

				void operator () (const gpu_float_array& d_input, gpu_float_array& d_output)
				{
					const float alpha = 1, beta = 0;
					/*
					//cudnnConvolutionBiasActivationForward
					//CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_​PRECOMP_GEMM
					//CUDNN_ACTIVATION_IDENTITY
					checkCUDNN(cudnnConvolutionForward(cudnn,
                                   &alpha,
                                   input_descriptor,
                                   d_input.ptr,
                                   kernel_descriptor,
                                   d_kernel.ptr,
                                   convolution_descriptor,
                                   convolution_algorithm, 
                                   (void*)d_workspace.ptr,
                                   workspace_bytes,
                                   &beta,
                                   output_descriptor,
                                   d_output.ptr));
                                   */

					checkCUDNN(cudnnConvolutionBiasActivationForward(cudnn,
												    &alpha,
												    input_descriptor,
												    d_input.ptr,
												    kernel_descriptor,
												    d_kernel.ptr,
												    convolution_descriptor,
												    convolution_algorithm,
												    (void*)d_workspace.ptr,
												    workspace_bytes,,
												    &beta,
												    z_descriptor,
												    d_z.ptr,
												    bias_descriptor,
												    d_bias.ptr,
												    activation_descriptor,
												    output_descriptor,
												    d_output.ptr));
				}

				// free host & device memory
				~conv()
				{
					cudnnDestroyTensorDescriptor(input_descriptor);
					cudnnDestroyTensorDescriptor(z_descriptor);
					cudnnDestroyTensorDescriptor(output_descriptor);
					cudnnDestroyFilterDescriptor(kernel_descriptor);
					cudnnDestroyConvolutionDescriptor(convolution_descriptor);
					cudnnDestroy(cudnn);

				}
			};
		}
	}
}
#endif