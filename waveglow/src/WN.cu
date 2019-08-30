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


__global__ void fused_add_tanh_sigm_mul(size_t sz, float_t* f2, float* f3, float_t* dest)
{
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    
        if(index < sz)
        {
            dest[index] = tanhf(f2[index]+f3[index])* sigmoidf(f2[index+sz] + f3[index+sz]);
        }
}

__global__ void affine_transform(size_t sz, float_t* audio, float_t* end_out, size_t stride)
{
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    
        if(index < sz)
        {
            audio[index+stride] = (audio[index+stride]-end_out[index])/expf(end_out[index+stride]);
        }
}


__global__ void skip_res_add(size_t sz, float_t* f5, float* f1, float_t* f6, size_t stride)
{
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    
        if(index < sz)
        {
            f6[index] += f5[index+stride];
            f1[index] += f5[index]; 
        }
}

__global__ void skip_add(size_t sz, float_t* f1, float* f6)
{
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    
        if(index < sz)
        {
            f6[index] += f1[index];
        }
}


__global__ void copy_kernel(size_t sz, float_t* src, float_t* dest)
{
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(index < sz)
    {
        dest[index]=src[index];
    }
}


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
{
    input_len = max_audio_len; 


    for (int k=0;k<12;k++)
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
    
    for (int k=0;k<12;k++)
    {   
        size_t dilation = 1;

            for(int i=0;i<8;i++)
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
    for (int k=0;k<12;k++)
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

    std::cout<<"input length is "<<input_len<<"\n";
    temp_input.init(4, input_len);
    f1.init(256, input_len);
    f2.init(512, input_len);
    f3.init(512, input_len);
    f4.init(256, input_len);
    f6.init(256, input_len);
    temp.init(8, input_len);
    z.init(2, 2*input_len);
    input_t.init(1,8,input_len);
    d_workspace.init(1000000,1);

    {
        checkCURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
        checkCURAND(curandSetPseudoRandomGeneratorSeed(rng, 1337ull));
    }

}


void WN::operator() (cudnnHandle_t& cudnn, gpu_float_array& mel_input, gpu_float_array& d_output)
{   

    size_t input_len = mel_input.shape[1], n_channels=4;
    std::cout<<"the value is"<<input_len<<"\t"<<input_t.shape[2]<<"\t"<<mel_input.shape[1]<<"\n";

    input_t.reshape(1, n_channels, input_len);
    curandGenerateNormal(rng, input_t.ptr, input_t.size(), 0.0f, 0.6);

    f1.reshape(256, input_len);
    f2.reshape(512, input_len);
    f3.reshape(512, input_len);
    f4.reshape(256, input_len);
    f6.reshape(256, input_len);
    temp_input.reshape(n_channels/2, input_len);
    temp.reshape(n_channels, input_len);


    for(int k=11;k>-1;k--)
    {
        copy_kernel <<< (temp_input.size()+511)/512, 512 >>>(temp_input.size(), input_t.ptr, temp_input.ptr);

        cudnnSetTensor4dDescriptor(input_desc,
                                          /*format=*/cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                                          /*dataType=*/cudnnDataType_t::CUDNN_DATA_FLOAT,
                                          /*batch_size=*/1,
                                          /*channels=*/n_channels/2,
                                          /*image_height=*/1,
                                          /*image_width=*/input_len);
        
        cudnnSetTensor4dDescriptor(out_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, 256, 1, input_len);
        start_conv[k](cudnn, temp_input, f1, input_desc, out_desc, d_workspace);

        f6.reset();
        for(int j=0; j<8;j++)
        {
                // log_d("input", f1.log("inp_in" + std::to_string(j)+ ".npy"));

                cudnnSetTensor4dDescriptor(input_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, 256, 1, input_len);
                cudnnSetTensor4dDescriptor(out_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, 512, 1, input_len);
                in_conv[k][j](cudnn, f1, f2, input_desc, out_desc, d_workspace);
                // log_d("in_out", f2.log("in_out" + std::to_string(j)+ ".npy"));

                cudnnSetTensor4dDescriptor(input_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, 640, 1, input_len);
                cudnnSetTensor4dDescriptor(out_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, 512, 1, input_len);
                cond_conv[k][j](cudnn, mel_input, f3, input_desc, out_desc, d_workspace);
                // log_d("cond_out", f3.log("cond_out" + std::to_string(j)+ ".npy"));

                fused_add_tanh_sigm_mul <<< (f4.size()+511)/512, 512 >>>(f4.size(), f2.ptr, f3.ptr, f4.ptr);
                // log_d("acts ", f4.log("acts_out" + std::to_string(j)+ ".npy"));

                
                if(j<7)
                {
                    cudnnSetTensor4dDescriptor(input_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, 256, 1, input_len);
                    cudnnSetTensor4dDescriptor(out_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, 512, 1, input_len);
                    res_skip_conv[k][j](cudnn, f4, f3, input_desc, out_desc, d_workspace);
                    // log_d("res_skip_acts ", f3.log("res_skip_acts" + std::to_string(j)+ ".npy"));

                    skip_res_add <<< (f1.size()+511)/512, 512 >>>(f1.size(), f3.ptr, f1.ptr, f6.ptr, 256*input_len);
                    // log_d("outputs ", f6.log("outputs" + std::to_string(j)+ ".npy"));
                }
                else
                {
                    cudnnSetTensor4dDescriptor(input_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, 256, 1, input_len);
                    cudnnSetTensor4dDescriptor(out_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, 256, 1, input_len);
                    res_skip_conv[k][j](cudnn, f4, f1, input_desc, out_desc, d_workspace);
                    // log_d("res_skip_acts ", f1.log("res_skip_acts" + std::to_string(j)+ ".npy"));

                    skip_add <<< (f1.size()+511)/512, 512 >>>(f1.size(), f1.ptr, f6.ptr);
                    // log_d("outputs ", f6.log("outputs" + std::to_string(j)+ ".npy"));
                }

        }

        cudnnSetTensor4dDescriptor(input_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, 256, 1, input_len);
        cudnnSetTensor4dDescriptor(out_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, n_channels, 1, input_len);
        end_conv[k](cudnn, f6, temp, input_desc, out_desc, d_workspace);
        // log_d(" end conv outputs ", temp.log("end_out.npy"));

        affine_transform <<< (temp.size()/2+511)/512, 512 >>>(temp.size()/2, input_t.ptr, temp.ptr, temp.size()/2);
        // log_d("audio transformed", input_t.log("audio_tr.npy"));

        cudnnSetTensor4dDescriptor(input_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, n_channels, 1, input_len);
        cudnnSetTensor4dDescriptor(out_desc, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, n_channels, 1, input_len);
        inv_conv[k](cudnn, input_t, temp, input_desc, out_desc, d_workspace, 0);

        copy_kernel<<<(input_t.size()+511)/512, 512>>>(input_t.size(), temp.ptr, input_t.ptr);


        if((k%4==0) && (k>0))
            {
                n_channels +=2;

                input_t.reshape(1, n_channels, input_len);
                z.reshape(2, input_len);
                curandGenerateNormal(rng, z.ptr, z.size(), 0.0f, 0.6);
                concat_z<<<(input_t.size()+511)/512, 512>>>(input_t.size(), temp.ptr, input_t.ptr, z.ptr, 2*input_len);
                // log_d("vaue of z", z8.log("z8.npy"));
                temp_input.reshape(n_channels/2, input_len);
                temp.reshape(n_channels, input_len);
            }
     
        log_d("audio transformed inv", input_t.log("audio_after_step" + std::to_string(k)+ ".npy"));

    }

    transpose_kernel<<<(d_output.size()+511)/512, 512>>>(d_output.size(), input_t.ptr, d_output.ptr, input_t.shape[2], input_t.shape[1]);

    std::cout<<input_t.shape[1]<<"\t"<<input_t.shape[0]<<"\n";

}


WN::~WN()
{

}
