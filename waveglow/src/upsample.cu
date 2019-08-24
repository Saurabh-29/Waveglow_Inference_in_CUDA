#include <upsample.hpp>
#include <hparams.hpp>
#include <data_types.hpp>

#include<cublas_v2.h>
#include<iostream>
#include<vector>
#include<logger.hpp>
#include<utils.hpp>

using namespace livai::tts::waveglow;
using namespace livai::tts::common;





__global__ void fractional_stride_nchw(size_t num_values, size_t stride, float_t* src, float_t* dest, size_t ld_src, size_t ld_dest)
{
    size_t index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index < num_values)
    {
        size_t frame_id = (index/ ld_src)*ld_dest + (stride)*(index%ld_src) ;
        dest[frame_id] = src[index];
    }
}

void upsample::set(cudnnHandle_t& cudnn, size_t audio_len)
{
    size_t total_input_size = audio_len;
    size_t input_len = audio_len; 

    std::string kernel_fname = hparams::up_conv_weight;
    std::string bias_fname = hparams::up_conv_bias;   
    auto kernel_weight = cnpy::npy_load(kernel_fname); 
    auto bias_weight = cnpy::npy_load(bias_fname);

    size_t kernel_width = kernel_weight.shape[2];
    size_t in_channel_size = kernel_weight.shape[1];
    size_t out_channel_size = kernel_weight.shape[0];

    std::cout<<kernel_width<<"\t"<<in_channel_size<<"\t"<<out_channel_size<<"\n";
    size_t input_rows = audio_len+(audio_len-1)*255;
    size_t output_rows = audio_len*256+768;
    up_conv.init(cudnn, kernel_weight, bias_weight, 1, input_rows, in_channel_size,
        1, output_rows, out_channel_size, 1, kernel_width);
    

    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&out_desc);

    f1.init(80, 256*input_len+1024);
    f2.init(80, 256*input_len+1024);
    d_workspace.init(100000,1);

}


void upsample::operator() (cudnnHandle_t& cudnn, gpu_float_array& input_mel)
{   

    size_t input_len = input_mel.shape[2];
    std::cout<<"the value is"<<input_len<<"\t"<<input_mel.size()<<"\n";

    size_t input_rows = input_len+(input_len-1)*255;
    size_t output_rows = input_len*256+768;
    f1.reset();
    f1.reshape(80, input_rows);
    f2.reshape(80, output_rows);
    
    cudnnSetTensor4dDescriptor(input_desc,
                                      /*format=*/cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                                      /*dataType=*/cudnnDataType_t::CUDNN_DATA_FLOAT,
                                      /*batch_size=*/1,
                                      /*channels=*/80,
                                      /*image_height=*/1,
                                      /*image_width=*/input_rows);
    
    cudnnSetTensor4dDescriptor(out_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, 80, 1, output_rows);
    
    size_t num_values = input_mel.size();
    fractional_stride_nchw<<<(num_values+1023)/1024, 1024>>>(num_values, 256, input_mel.ptr, f1.ptr, input_len, input_rows);

    up_conv(cudnn, f1, f2, input_desc, out_desc, d_workspace);
    log_d("added upsampling", f2.log("upsampled_mel.npy"));

}


upsample::~upsample()
{

}
