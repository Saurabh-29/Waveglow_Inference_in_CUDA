#ifndef __MACROS_HPP__
#define __MACROS_HPP__

#include <string>

#pragma once

#define noCopy(class_name) class_name(const class_name&) = delete;\
                           class_name& operator=(const class_name&) = delete


#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on file : line "                   \
                << __FILE__ << ": "                          \
                << __LINE__ << ": "                          \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

  #define checkCUDAERROR(expression)                               \
  {                                                          \
    cudaError_t status = (expression);                     \
    if (status != cudaSuccess) {                    \
      std::cerr << "Error on file : line "                   \
                << __FILE__ << ": "                          \
                << __LINE__ << ": "                       \
                << cudaGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

  #define checkCUBLAS(expression)                               \
  {                                                          \
    cublasStatus_t status = (expression);                     \
    if (status != CUBLAS_STATUS_SUCCESS) {                    \
      std::cerr << "Error on file : line "                   \
                << __FILE__ << ": "                          \
                << __LINE__ << ": "         \
                << status << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

  #define checkCURAND(expression)                               \
  {                                                          \
    curandStatus_t status = (expression);                     \
    if (status != CURAND_STATUS_SUCCESS) {                    \
      std::cerr << "Error on file : line "                   \
                << __FILE__ << ": "                          \
                << __LINE__ << ": "         \
                << status << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }


#endif
