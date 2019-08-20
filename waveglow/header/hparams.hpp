#ifndef _HPARAMS_HPP__
#define _HPARAMS_HPP__

#pragma once 

#include<string>

namespace hparams
{
	using namespace std;

    static const string base_folder_save = "/shared1/saurabh.m/test_cuda/weights_new_2/";
    static const size_t mel_dim = 80;

	//Res_block
	static const size_t res_block_layers=10;
	static const size_t flow_layers = 6;	


	static const size_t upsampler_conv_layers = 2;
	static const vector<size_t> upsampling_scales{16, 16};
	static const float lrelu_negative_slope = 0.4;

	static const string upsampler_conv_layer_kernel = base_folder_save + "upsample_conv_{}_weight.npy";
	static const string upsampler_conv_layer_bias = base_folder_save + "upsample_conv_{}_bias.npy";

	static const string front_conv_bias = base_folder_save + "{}_front_conv_0_conv_bias.npy";
	static const string front_conv_weight = base_folder_save + "{}_front_conv_0_conv_weight.npy";

	static const string filter_conv_bias = base_folder_save + "{}_res_blocks_{}_filter_conv_conv_bias.npy";
	static const string filter_conv_weight = base_folder_save + "{}_res_blocks_{}_filter_conv_conv_weight.npy";

	static const string	gate_conv_bias = base_folder_save + "{}_res_blocks_{}_gate_conv_conv_bias.npy";
	static const string	gate_conv_weight = base_folder_save + "{}_res_blocks_{}_gate_conv_conv_weight.npy";

	static const string	res_conv_bias = base_folder_save +	"{}_res_blocks_{}_res_conv_bias.npy";
	static const string	res_conv_weight = base_folder_save + "{}_res_blocks_{}_res_conv_weight.npy";

	static const string	skip_conv_bias = base_folder_save + "{}_res_blocks_{}_skip_conv_bias.npy";
	static const string	skip_conv_weight = base_folder_save + "{}_res_blocks_{}_skip_conv_weight.npy";

	static const string	filter_conv_c_bias = base_folder_save + "{}_res_blocks_{}_filter_conv_c_bias.npy";
	static const string	filter_conv_c_weight = base_folder_save + "{}_res_blocks_{}_filter_conv_c_weight.npy";

	static const string	gate_conv_c_bias = base_folder_save + "{}_res_blocks_{}_gate_conv_c_bias.npy";
	static const string	gate_conv_c_weight = base_folder_save + "{}_res_blocks_{}_gate_conv_c_weight.npy";

	static const string final_conv_bias = base_folder_save + "{}_final_conv_{}_conv_bias.npy";
	static const string final_conv_weight = base_folder_save + "{}_final_conv_{}_conv_weight.npy";

	static const string input_mel = base_folder_save + "input_mel_T.npy";
	static const string input_tensor = base_folder_save + "out_front_T.npy";
	static const string input_front = base_folder_save + "input_front_T.npy";
	static const string conv_test = base_folder_save +"conv_front_T.npy";

	static const string base_folder = "/shared1/saurabh.m/waveglow/weights_new/";

	static const string start_conv_weight = base_folder + "{}_start_weight.npy";
	static const string start_conv_bias = base_folder + "{}_start_bias.npy";

	static const string in_conv_weight = base_folder + "{}_in_layers_{}_weight.npy";
	static const string in_conv_bias = base_folder + "{}_in_layers_{}_bias.npy";

	static const string cond_conv_weight = base_folder + "{}_cond_layers_{}_weight.npy";
	static const string cond_conv_bias = base_folder + "{}_cond_layers_{}_bias.npy";

	static const string res_skip_conv_weight = base_folder + "{}_res_skip_layers_{}_weight.npy";
	static const string res_skip_conv_bias = base_folder + "{}_res_skip_layers_{}_bias.npy";

	static const string end_conv_weight = base_folder + "{}_end_weight.npy";
	static const string end_conv_bias = base_folder + "{}_end_bias.npy";

	static const string inv_conv_weight = base_folder + "{}_conv_weight_inv.npy";

};

#endif
