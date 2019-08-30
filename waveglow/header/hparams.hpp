#ifndef _HPARAMS_HPP__
#define _HPARAMS_HPP__

#pragma once 

#include<string>

namespace hparams
{
	using namespace std;

    static const string base_folder = "/shared1/saurabh.m/waveglow/weights_new/";

    static const size_t max_length = 1500*32;

	static const size_t mel_dim = 80;
	static const size_t stride = 256;

	static const size_t n_channels = 256;
	static const size_t n_flows = 12;
	static const size_t n_layers = 8;
	static const size_t n_groups = 8;
	static const size_t n_rem_channels = 4;

    static const string up_conv_weight = base_folder + "weight_flip.npy";
	static const string up_conv_bias = base_folder + "bias.npy";

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
