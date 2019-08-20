#ifndef __CONV_HPP__
#define __CONV_HPP__

#include <data_types.hpp>
#include <cudnn.h>
#include <logger.hpp>


namespace livai 
{
    
    namespace tts 
    {

          using namespace common;

          namespace sys 
          {

                  class conv
                  {
                          private:
                                  cudnnTensorFormat_t default_tensor_format;
                                  cudnnDataType_t default_data_type;
                                  gpu_float_array d_workspace; 
                                  gpu_float_array d_kernel;
                                  gpu_float_array d_bias;

                                  size_t workspace_bytes;

                                  // cudnnTensorDescriptor_t input_descriptor;
                                  // cudnnTensorDescriptor_t output_descriptor;
                                  cudnnFilterDescriptor_t kernel_descriptor;
                                  cudnnTensorDescriptor_t bias_descriptor;

                                  cudnnConvolutionDescriptor_t convolution_descriptor;
                                  cudnnConvolutionFwdAlgo_t convolution_algorithm;

                                  


                          public:
                                  noCopy(conv);
                                  conv(){};
                                  void initModelWeight(const cnpy::NpyArray& h_kernel, const cnpy::NpyArray& h_bias);
                                  void initConvTensors(cudnnHandle_t& cudnn, size_t in_rows, size_t in_cols, size_t in_channels, 
                                                        size_t out_rows, size_t out_cols, size_t out_channels, 
                                                        size_t kernel_height, size_t kernel_width, 
                                                        size_t dilation_height, size_t dilation_width,
                                                        size_t batch_size);

                                  void init(cudnnHandle_t& cudnn, const cnpy::NpyArray& h_kernel, const cnpy::NpyArray& h_bias, 
                                              size_t in_rows, size_t in_cols, size_t in_channels, 
                                              size_t out_rows, size_t out_cols, size_t out_channels, 
                                              size_t kernel_height, size_t kernel_width,
                                              size_t dilation_height = 1, size_t dilation_width =1, 
                                              size_t batch_size = 1);
                                  void operator () (cudnnHandle_t& cudnn, const gpu_float_array& d_input, gpu_float_array& d_output, 
                                                      cudnnTensorDescriptor_t input_desc, cudnnTensorDescriptor_t output_desc, 
                                                        gpu_float_array& d_workspace, size_t has_bias=1);
                                  ~conv();

                  };

          }
    }
}

#endif