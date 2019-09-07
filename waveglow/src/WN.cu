#include <WN.hpp>
#include <hparams.hpp>
#include <data_types.hpp>

#include<cublas_v2.h>
#include<iostream>
#include<vector>
#include<logger.hpp>
#include<utils.hpp>

using namespace livai::tts::waveglow;
using namespace livai::tts::common;

__forceinline__ __device__ float sigmoidf(float in) {
   return 1.f / (1.f + expf(-in));  
}

/* kernel to apply gated activation function on input */

__global__ void fused_add_tanh_sigm_mul(size_t sz, float_t* in_conv_out, float* f3, float_t* dest)
{
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    
        if(index < sz)
        {
            dest[index] = tanhf(in_conv_out[index]+f3[index])* sigmoidf(in_conv_out[index+sz] + f3[index+sz]);
        }
}

/* kernel to apply affine transformation on one half of audio */
__global__ void affine_transform(size_t sz, float_t* audio, float_t* end_out, size_t stride)
{
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    
        if(index < sz)
        {
            audio[index+stride] = (audio[index+stride]-end_out[index])/expf(end_out[index+stride]);
        }
}

/* kernel to add skip and res results to global skip and res */
__global__ void skip_res_add(size_t sz, float_t* f5, float* f1, float_t* skip_out_sum, size_t stride)
{
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    
        if(index < sz)
        {
            skip_out_sum[index] += f5[index+stride];
            f1[index] += f5[index]; 
        }
}

/* kernel to add skip value to global skip in last layer */
__global__ void skip_add(size_t sz, float_t* f1, float* skip_out_sum)
{
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    
        if(index < sz)
        {
            skip_out_sum[index] += f1[index];
        }
}

/* kernel to copy src value to dest */
__global__ void copy_kernel(size_t sz, float_t* src, float_t* dest)
{
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(index < sz)
    {
        dest[index]=src[index];
    }
}

/* kernel to compute transpose of src and store in dest */
__global__ void transpose_kernel(size_t sz, float_t* src, float_t* dest, size_t ld_src, size_t ld_dest)
{
    size_t index = blockIdx.x*blockDim.x + threadIdx.x;
    size_t i = index/ld_src, j= index%ld_src;
    size_t dest_index = j*ld_dest + i;

    if(index < sz)
    {
        dest[dest_index] = src[index];
    }
}

/* kernel to concatenate z in audio after every 4 rounds of flow */

__global__ void concat_z(size_t sz, float_t* src, float_t* dest, float_t* z, size_t stride)
{
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(index < sz)
    {
        if(index>=stride)
        {
            dest[index]=src[index-stride];
        }
        else
        {
            dest[index]=z[index];
        }
    }
}


void WN::set(cudnnHandle_t& cudnn, size_t max_audio_len)
/* initialize the weights and biases of the Convolution layers */
{
    input_len = max_audio_len; 
    
    n_channels = hparams::n_channels;
    n_flows = hparams::n_flows;
    n_layers = hparams::n_layers;
    n_groups = hparams::n_groups;
    n_rem_channels = hparams::n_rem_channels;

    n_threads = 512;

    for (int k=0; k<n_flows; k++)
    {   
        std::string kernel_fname = get_param_name(hparams::start_conv_weight, k);
        std::string bias_fname = get_param_name(hparams::start_conv_bias, k);   
        auto kernel_weight = cnpy::npy_load(kernel_fname); 
        auto bias_weight = cnpy::npy_load(bias_fname);

        size_t kernel_width = kernel_weight.shape[2];
        size_t in_channel_size = kernel_weight.shape[1];
        size_t out_channel_size = kernel_weight.shape[0];

        start_conv[k].init(cudnn, kernel_weight, bias_weight, 1, input_len, in_channel_size,
            1, input_len, out_channel_size, 1, kernel_width);
    }
    
    for (int k=0; k<n_flows; k++)
    {   
        size_t dilation = 1;

            for(int i=0; i<n_layers; i++)
            {
                std::string kernel_fname = get_res_name(hparams::in_conv_weight, k, i);
                std::string bias_fname = get_res_name(hparams::in_conv_bias, k, i); 
                auto kernel_weight = cnpy::npy_load(kernel_fname); 
                auto bias_weight = cnpy::npy_load(bias_fname);

                size_t kernel_width = kernel_weight.shape[2];
                size_t in_channel_size = kernel_weight.shape[1];
                size_t out_channel_size = kernel_weight.shape[0];

                in_conv[k][i].init(cudnn, kernel_weight, bias_weight, 1, input_len, in_channel_size,
                    1, input_len, out_channel_size, 1, kernel_width, 1, dilation);

                
                kernel_fname = get_res_name(hparams::cond_conv_weight, k, i);
                bias_fname = get_res_name(hparams::cond_conv_bias, k, i);   
                kernel_weight = cnpy::npy_load(kernel_fname); 
                bias_weight = cnpy::npy_load(bias_fname);

                kernel_width = kernel_weight.shape[2];
                in_channel_size = kernel_weight.shape[1];
                out_channel_size = kernel_weight.shape[0];

                cond_conv[k][i].init(cudnn, kernel_weight, bias_weight, 1, input_len, in_channel_size,
                    1, input_len, out_channel_size, 1, kernel_width);

                kernel_fname = get_res_name(hparams::res_skip_conv_weight, k, i);
                bias_fname = get_res_name(hparams::res_skip_conv_bias, k, i);   
                kernel_weight = cnpy::npy_load(kernel_fname); 
                bias_weight = cnpy::npy_load(bias_fname);

                kernel_width = kernel_weight.shape[2];
                in_channel_size = kernel_weight.shape[1];
                out_channel_size = kernel_weight.shape[0];

                res_skip_conv[k][i].init(cudnn, kernel_weight, bias_weight, 1, input_len, in_channel_size,
                    1, input_len, out_channel_size, 1, kernel_width);

                dilation*=2;

            }
    }
    for (int k=0; k<n_flows; k++)
    {   

        std::string kernel_fname = get_param_name(hparams::end_conv_weight, k);
        std::string bias_fname = get_param_name(hparams::end_conv_bias, k); 
        auto kernel_weight = cnpy::npy_load(kernel_fname); 
        auto bias_weight = cnpy::npy_load(bias_fname);

        size_t kernel_width = kernel_weight.shape[2];
        size_t in_channel_size = kernel_weight.shape[1];
        size_t out_channel_size = kernel_weight.shape[0];

        end_conv[k].init(cudnn, kernel_weight, bias_weight, 1, input_len, in_channel_size,
            1, input_len, out_channel_size, 1, kernel_width);
    
        kernel_fname = get_param_name(hparams::inv_conv_weight, k);
        bias_fname = get_param_name(hparams::end_conv_bias, k); 
        kernel_weight = cnpy::npy_load(kernel_fname); 
        bias_weight = cnpy::npy_load(bias_fname);

        kernel_width = kernel_weight.shape[2];
        in_channel_size = kernel_weight.shape[1];
        out_channel_size = kernel_weight.shape[0];

        inv_conv[k].init(cudnn, kernel_weight, bias_weight, 1, input_len, in_channel_size,
            1, input_len, out_channel_size, 1, kernel_width);
    }


    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&out_desc);

    // std::cout<<"input length is "<<input_len<<"\n";
    {
        audio_0.init(n_groups/2, input_len);
        f1.init(n_channels, input_len);
        in_conv_out.init(2*n_channels, input_len);
        f3.init(2*n_channels, input_len);
        gated_activation_output.init(n_channels, input_len);
        skip_out_sum.init(n_channels, input_len);
        audio.init(n_groups, input_len);
        z.init(2, 2*input_len);
        input_t.init(n_groups,input_len);
    }

    {
        checkCURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
        checkCURAND(curandSetPseudoRandomGeneratorSeed(rng, 1337ull));
    }

}


void WN::operator() (cudnnHandle_t& cudnn,  gpu_float_array& mel_input, gpu_float_array& d_output, gpu_float_array& d_workspace)
/*
    This function transforms noise to audio through series of normalising flows 
    
    Arguments:
    --------------
    cudnn: A cudnnHandle 
        A cudnn handle used by various cudnn layers

    mel_input: a float array of size [640,x]
        Upsampled mel generated from upsamper layer, used for conditioning waveglow

    d_output: A float array of size [8*x,1]
        Pointer to store values of audiov (output)

    d_workspace: A float array of large size ( greater than required by any convolution)
        A chunk of memory to be used by convolution workspace, alternatively we can set size to a
         given maximum by selecting such algorithms in conv
*/
{   

    size_t input_len = mel_input.shape[1];
    size_t aud_channels = n_rem_channels;
    // std::cout<<"the value is"<<input_len<<"\t"<<input_t.shape[2]<<"\t"<<mel_input.shape[1]<<"\n";

    input_t.reshape(aud_channels, input_len);
    curandGenerateNormal(rng, input_t.ptr, input_t.size(), 0.0f, 0.6);

    f1.reshape(n_channels, input_len);
    in_conv_out.reshape(2*n_channels, input_len);
    f3.reshape(2*n_channels, input_len);
    gated_activation_output.reshape(n_channels, input_len);
    skip_out_sum.reshape(n_channels, input_len);
    audio_0.reshape(aud_channels/2, input_len);
    audio.reshape(aud_channels, input_len);


    for(int k=n_flows-1; k>-1; k--)
    {
        copy_kernel <<< (audio_0.size()+n_threads-1)/n_threads, n_threads >>>(audio_0.size(), input_t.ptr, audio_0.ptr);

        cudnnSetTensor4dDescriptor(input_desc,
                                          /*format=*/cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                                          /*dataType=*/cudnnDataType_t::CUDNN_DATA_FLOAT,
                                          /*batch_size=*/1,
                                          /*channels=*/aud_channels/2,
                                          /*image_height=*/1,
                                          /*image_width=*/input_len);
        
        cudnnSetTensor4dDescriptor(out_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, n_channels, 1, input_len);
        start_conv[k](cudnn, audio_0, f1, input_desc, out_desc, d_workspace);

        skip_out_sum.reset();
        for(int j=0; j<n_layers; j++)
        {
                // log_d("input", f1.log("inp_in" + std::to_string(j)+ ".npy"));

                cudnnSetTensor4dDescriptor(input_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, n_channels, 1, input_len);
                cudnnSetTensor4dDescriptor(out_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, 2*n_channels, 1, input_len);
                in_conv[k][j](cudnn, f1, in_conv_out, input_desc, out_desc, d_workspace);
                // log_d("in_out", in_conv_out.log("in_out" + std::to_string(j)+ ".npy"));

                cudnnSetTensor4dDescriptor(input_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, 640, 1, input_len);
                cudnnSetTensor4dDescriptor(out_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, 2*n_channels, 1, input_len);
                cond_conv[k][j](cudnn, mel_input, f3, input_desc, out_desc, d_workspace);
                // log_d("cond_out", f3.log("cond_out" + std::to_string(j)+ ".npy"));

                fused_add_tanh_sigm_mul <<< (gated_activation_output.size()+n_threads-1)/n_threads, n_threads >>>(gated_activation_output.size(), in_conv_out.ptr, f3.ptr, gated_activation_output.ptr);
                // log_d("acts ", gated_activation_output.log("acts_out" + std::to_string(j)+ ".npy"));

                
                if(j<7)
                {
                    cudnnSetTensor4dDescriptor(input_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, n_channels, 1, input_len);
                    cudnnSetTensor4dDescriptor(out_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, 2*n_channels, 1, input_len);
                    res_skip_conv[k][j](cudnn, gated_activation_output, f3, input_desc, out_desc, d_workspace);
                    skip_res_add <<< (f1.size()+n_threads-1)/n_threads, n_threads >>>(f1.size(), f3.ptr, f1.ptr, skip_out_sum.ptr, 256*input_len);
                }
                else
                {
                    cudnnSetTensor4dDescriptor(input_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, n_channels, 1, input_len);
                    cudnnSetTensor4dDescriptor(out_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, n_channels, 1, input_len);
                    res_skip_conv[k][j](cudnn, gated_activation_output, f1, input_desc, out_desc, d_workspace);
                    skip_add <<< (f1.size()+n_threads-1)/n_threads, n_threads >>>(f1.size(), f1.ptr, skip_out_sum.ptr);
                }

        }

        cudnnSetTensor4dDescriptor(input_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, n_channels, 1, input_len);
        cudnnSetTensor4dDescriptor(out_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, aud_channels, 1, input_len);
        end_conv[k](cudnn, skip_out_sum, audio, input_desc, out_desc, d_workspace);

        affine_transform <<< (audio.size()/2+n_threads-1)/n_threads, n_threads >>>(audio.size()/2, input_t.ptr, audio.ptr, audio.size()/2);

        cudnnSetTensor4dDescriptor(input_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, aud_channels, 1, input_len);
        cudnnSetTensor4dDescriptor(out_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, aud_channels, 1, input_len);
        inv_conv[k](cudnn, input_t, audio, input_desc, out_desc, d_workspace, 0);

        copy_kernel<<<(input_t.size()+n_threads-1)/n_threads, n_threads>>>(input_t.size(), audio.ptr, input_t.ptr);


        if((k%4==0) && (k>0))
            {
                aud_channels +=2;

                input_t.reshape(aud_channels, input_len);
                z.reshape(2, input_len);
                curandGenerateNormal(rng, z.ptr, z.size(), 0.0f, 0.6);
                concat_z<<<(input_t.size()+n_threads-1)/n_threads, n_threads>>>(input_t.size(), audio.ptr, input_t.ptr, z.ptr, 2*input_len);
                
                audio_0.reshape(aud_channels/2, input_len);
                audio.reshape(aud_channels, input_len);
            }
    }

    transpose_kernel<<<(d_output.size()+n_threads-1)/n_threads, n_threads>>>(d_output.size(), input_t.ptr, d_output.ptr, input_t.shape[1], input_t.shape[0]);

    // std::cout<<input_t.shape[1]<<"\t"<<input_t.shape[0]<<"\n";

}


WN::~WN()
{

}
