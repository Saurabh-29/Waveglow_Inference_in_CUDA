#ifndef __DENSE_MV_HPP__
#define __DENSE_MV_HPP__

#include <data_types.hpp>
#include <cudnn.h>
#include<logger.hpp>
#include <cublas_v2.h>


namespace livai {
	namespace tts {

		using namespace common;

		namespace sys {

			class dense_mv
			{
			private:
				cublasHandle_t handle ;
				gpu_float_array d_kernel;  // this is Rows*Cols [ column major memory layoput ]
				gpu_float_array d_bias;  // Bias vecotr

				bool hasbias;

			public:
				noCopy(dense_mv);
				dense_mv();
				/* 
				Alloc memory at device and memcopy the parameters ( shared memory )
				* */
				void init(const cnpy::NpyArray& h_kernel, const cnpy::NpyArray& h_bias);
				void init(const cnpy::NpyArray& h_kernel);
				
				void operator () (cudnnHandle_t& cudnn, const gpu_float_array& d_input, gpu_float_array& d_output);

				// free host & device memory
				~dense_mv();
			};
		}
	}
}
#endif