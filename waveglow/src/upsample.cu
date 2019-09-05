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




/* kernel to add zero padding in input to treat transposedConv1d as Conv1d, not used anymore */
__global__ void fractional_stride_nchw(size_t num_values, size_t stride, float_t* src, float_t* dest, size_t ld_src, size_t ld_dest)
{
    size_t index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index < num_values)
    {
        size_t frame_id = (index/ ld_src)*ld_dest + (stride)*(index%ld_src) ;
        dest[frame_id] = src[index];
    }
}

/* kernel to reshape srs with leading dimension (ld_src) to dest with leading dimension (ld_dest)*/
__global__ void reshape(size_t num_values, float_t* src, float_t* dest, size_t ld_src, size_t ld_dest)
{
    size_t index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index < num_values)
    {
        size_t src_index = (index/ld_dest)*ld_src+ index%ld_dest;
        dest[index] = src[src_index];
    }
}

/*kernel to apply the given series of transformation on spect
        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)
*/
__global__ void transformation(size_t num_values, float_t* src, float_t* dest, size_t ld_src, size_t ld_dest)
{
    size_t index = blockIdx.x*blockDim.x + threadIdx.x;

    if(index < num_values)
    {
        size_t dest_index = (index/ld_src)*ld_src + ((index%ld_src)%8)*ld_dest+ (index%ld_src)/8;
        dest[dest_index] = src[index];
    }
}

void upsample::set(cudnnHandle_t& cudnn, size_t max_mel_length)
/* initialize the weights and biases of the upsampling Convolution layers 
Arguments:
    --------------
    cudnn: A cudnnHandle 
        A cudnn handle used by various cudnn layers

    max_mel_length: integer to denote size of mel i.e. (len of audio)/256
        maximum mel_length that is in the dataset, so that cuda malloc can be avoided
*/

{
    size_t input_len = max_mel_length; 
    mel_dim = hparams::mel_dim;
    stride = hparams::stride; 
    n_threads = 1024;

    /* When posing conv as transpose Conv2d, we can use this conv layer
    {
        std::string kernel_fname = hparams::up_conv_weight;
        std::string bias_fname = hparams::up_conv_bias;   
        auto kernel_weight = cnpy::npy_load(kernel_fname); 
        auto bias_weight = cnpy::npy_load(bias_fname);

        size_t kernel_width = kernel_weight.shape[2];
        kernel_len = kernel_width;
        size_t in_channel_size = kernel_weight.shape[1];
        size_t out_channel_size = kernel_weight.shape[0];

        size_t input_rows = max_mel_length+(max_mel_length-1)*(stride-1);
        size_t output_rows = max_mel_length*stride+kernel_len-stride;
        up_conv.init(cudnn, kernel_weight, bias_weight, 1, input_rows, in_channel_size,
            1, output_rows, out_channel_size, 1, kernel_width);

    }
    (/)
    /* transpose conv1d initialization*/
    {
        std::string kernel_fname = hparams::up_conv_weight_orig;
        std::string bias_fname = hparams::up_conv_bias;   
        auto kernel_weight = cnpy::npy_load(kernel_fname); 
        auto bias_weight = cnpy::npy_load(bias_fname);

        size_t kernel_width = kernel_weight.shape[2];
        kernel_len = kernel_width;
        size_t in_channel_size = kernel_weight.shape[1];
        size_t out_channel_size = kernel_weight.shape[0];

        size_t input_rows = max_mel_length;
        size_t output_rows = max_mel_length*stride+kernel_len-stride;
        trans_conv.init(cudnn, kernel_weight, bias_weight, 1, input_rows, in_channel_size,
            1, output_rows, out_channel_size, 1, kernel_width);
    }
    

    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&out_desc);

    f1.init(mel_dim, stride*input_len+1024);
    f2.init(mel_dim, stride*input_len+1024);

}


void upsample::operator() (cudnnHandle_t& cudnn, gpu_float_array& input_mel, gpu_float_array& d_output, gpu_float_array& d_workspace)
/*
    Function to upsample the input mel
    
    Arguments:
    --------------
    cudnn: A cudnnHandle 
        A cudnn handle used by various cudnn layers

    input_mel: a float array of size [80,x]
        input-mel generated from either text-2-mel models such as tacotron, deepvoice or 
        from ground truth fft.

    d_output: A float array of size [640,x*32]
        Pointer to store values of transformed upsampled mel

    d_workspace: A float array of large size ( greater than required by any convolution)
        A chunk of memory to be used by convolution workspace, alternatively we can set size to a
         given maximum by selecting such algorithms in conv
*/

{   

    size_t input_len = input_mel.shape[2];
    size_t input_rows = input_len+(input_len-1)*(stride-1);
    size_t output_rows = input_len*stride+kernel_len-stride;

    f1.reset();
    f1.reshape(mel_dim, input_rows);
    f2.reshape(mel_dim, output_rows);
    
    cudnnSetTensor4dDescriptor(input_desc,
                                      /*format=*/cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                                      /*dataType=*/cudnnDataType_t::CUDNN_DATA_FLOAT,
                                      /*batch_size=*/1,
                                      /*channels=*/mel_dim,
                                      /*image_height=*/1,
                                      /*image_width=*/input_rows);
    
    cudnnSetTensor4dDescriptor(out_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, mel_dim, 1, output_rows);
    
    size_t num_values = input_mel.size();
    // fractional_stride_nchw<<<(num_values+n_threads-1)/n_threads, n_threads>>>(num_values, stride, input_mel.ptr, f1.ptr, input_len, input_rows);
    // up_conv(cudnn, f1, f2, input_desc, out_desc, d_workspace, 0);

    cudnnSetTensor4dDescriptor(input_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, mel_dim, 1, input_len);
    cudnnSetTensor4dDescriptor(out_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, mel_dim, 1, output_rows);
    trans_conv(cudnn, input_mel, f2, input_desc, out_desc, d_workspace);

    size_t upsampled_dim = input_len*stride;
    f1.reshape(mel_dim, upsampled_dim);
    num_values = f1.size();
    reshape<<<(num_values+n_threads-1)/n_threads, n_threads>>>(num_values, f2.ptr, f1.ptr, output_rows, upsampled_dim);

    f2.reshape(640, upsampled_dim/8);
    transformation<<<(num_values+n_threads-1)/n_threads, n_threads>>>(num_values, f1.ptr, d_output.ptr, upsampled_dim, upsampled_dim/8);
}


upsample::~upsample()
{

}
