#ifndef __data_types_hpp__
#define __data_types_hpp__

#include<vector>
#include<cassert>
#include<memory>
#include<macros.hpp>
#include<cnpy.hpp>

namespace livai
{
	namespace tts
	{
		namespace common
		{
			//typedef boost:numpy parameter;
			typedef float float_t;
		
			template<typename T> 
			struct gup_data_type
			{
				T* ptr;
				std::vector<size_t> shape;
				size_t capacity;

				// not copyable; pass using reference only
				noCopy(gup_data_type);

				gup_data_type()
				: ptr(NULL)
				,capacity(0)
				{ }

				// This sets the max memory of array
				// you can change the shape but not the total size of memory
				void init(const std::vector<size_t>& dims)
				{
					assert(dims.size() <= 4);
					shape = dims;

					// make the shape of lenght 4 for consistency
					for(size_t i=shape.size(); i<4;++i)
					{
						shape.push_back(1);
					}


					capacity = size();
					cudaMalloc(&ptr, capacity*sizeof(T));
				}

				void init(size_t dim1, size_t dim2=1, size_t dim3=1, size_t dim4 = 1)
				{
					std::vector<size_t> shape_;
					shape_.push_back(dim1);
					shape_.push_back(dim2);
					shape_.push_back(dim3);
					shape_.push_back(dim4);

					init(shape_);
				}

				inline void reset(size_t beginIndex = 0)
				{
					cudaMemset(ptr+beginIndex, 0, (capacity-beginIndex)*sizeof(T));
				}

				inline void copy(const T* src, size_t size)
				{
					cudaMemcpy(ptr, src, size*sizeof(T), cudaMemcpyHostToDevice);
				}

				inline size_t dim(size_t index) const
				{
					assert(index < shape.size());
					return shape[index];
				}

				size_t size() const
				{
					size_t numVals = 1;
					for(size_t i = 0;i < shape.size();i++) 
					{
						numVals *= shape[i];
					}

					return numVals;
				}

				// sets non-zero dimensions
				void reshape(size_t dim1, size_t dim2=1, size_t dim3=1, size_t dim4 = 1)
				{
					shape[0] =  dim1; 
					shape[1] =  dim2;
					shape[2] =  dim3; 
					shape[3] =  dim4;

					assert(size() <= capacity);
				}

				std::string log(const std::string& fname) const
				{
					std::string logStr = "shape [" + std::to_string(shape[0]) + ", " 
					+ std::to_string(shape[1]) + ", " 
					+ std::to_string(shape[2]) + ", "
					+ std::to_string(shape[3]) + "]\n";

					size_t N = shape[0] * shape[1] * shape[2] * shape[3];
					T* h_f = new T[N];
					cudaMemcpy(h_f, ptr, N*sizeof(T), cudaMemcpyDeviceToHost);  // what if this fails...
					std::vector<size_t> tempShape;
					for(int i=0;i<shape.size();++i)
					{
						if(shape[i] > 1)
						{
							tempShape.push_back(shape[i]);
						}
					}

					cnpy::npy_save(fname, h_f, tempShape);
					delete h_f;

					return logStr;
				}

				std::string log(size_t rowFrom, size_t rowTo =0) const
				{
					std::string logStr = "shape [" + std::to_string(shape[0]) + ", " 
					+ std::to_string(shape[1]) + ", " 
					+ std::to_string(shape[2]) + ", "
					+ std::to_string(shape[3]) + "]\n";

					if(rowTo == 0)
					{
						// go till the end..
						rowTo = rowFrom;
					}

					size_t totalRows = rowTo - rowFrom + 1;

					if(totalRows > 0)
					{
						size_t dataPerRow = shape[1]*shape[2]*shape[3];
						size_t N = totalRows * dataPerRow;
						T* h_f = new T[N];
						T* ptrBegin = ptr + rowFrom*dataPerRow;
						cudaMemcpy(h_f, ptrBegin, N*sizeof(T), cudaMemcpyDeviceToHost);  // what if this fails...
						for(size_t i=0;i<N;++i)
						{
							if(i % dataPerRow == 0)
							{
								if(i != 0)
								{
									logStr += "]\n\n";
								}

								logStr += "row:" + std::to_string((i/dataPerRow)+rowFrom) + " [";
							}

							logStr += std::to_string(h_f[i]) + ",";
						}

						logStr += "]";

						delete h_f;
					}

					return logStr;
				}

				~gup_data_type()
				{
					cudaFree(ptr);
				}
			};



			typedef gup_data_type<float_t> gpu_float_array;
			typedef gup_data_type<__constant__ float_t> const_gpu_float_array;
			typedef gup_data_type<bool> gpu_bool_array;
			
		}
	}
}

#endif

