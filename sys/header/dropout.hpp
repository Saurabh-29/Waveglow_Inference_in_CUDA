#ifndef __DROPOUT_HPP__
#define __DROPOUT_HPP__

#include <data_types.hpp>
#include <cudnn.h>
#include <curand.h>


namespace livai {
	namespace tts {

		using namespace common;

		namespace sys {

			class dropout
			{
			private:
				curandGenerator_t rng;
				gpu_float_array d_data;  // this is to keep the randomly generated floats

				dim3 blockDim;
				dim3 gridDim;

			public:
				noCopy(dropout);
				dropout(){}
				/* 
				Alloc memory at device
				* */
				void init(size_t num);
				/*
				  in place droppout operation ( TBD:: add seed(integer) as param ? )
				*/
				void operator () (gpu_float_array& input, float_t drop_rate);

				// free host & device memory
				~dropout();
			};
		}
	}
}
#endif