

# Waveglow_Inference_in_CUDA

C++ Code to run  **optimized inference  in CUDA** of [Waveglow](https://arxiv.org/pdf/1811.00002.pdf), this implementation gives **25% speedup** over [Nvidia's Pytorch implementation](https://github.com/NVIDIA/waveglow) in full precision and **2.5-3x speedup** when using TensorCore
>By default, this code will use GPU's TensorCore when running on NVIDIA's Volta GPU



# Waveglow 
Cuda C++ implementation of NVIDIA's Waveglow. 

The model architecure based on flows is described in this paper. [WaveGlow: a Flow-based Generative Network for Speech Synthesis](https://arxiv.org/pdf/1811.00002.pdf). 

**_Waveglow, a flow-based network is capable of generating high quality speech from mel-spectograms_**. It combines insights from [Glow](https://arxiv.org/pdf/1807.03039.pdf) and  [Wavenet](https://arxiv.org/pdf/1609.03499.pdf)  in order to provide fast, efficient and high-quality audio synthesis, without the need for auto-regression. 

WaveGlow is implemented using only a single network, trained using only a single cost function: maximizing the likelihood of the training data, which makes the training procedure simple and stable. 

Paper claims that  *in full-precision* (32 bit float) waveglow produces speech at the 500kHz on V100 but typically it is about **300-325kHz** with pytorch's implementation and **400-420kHz** using our implementation in full precision and around **1000kHz** using TensorCore in full precision.


# Repository Structure
	cpp
	├── common			(All common files; logger, utils, numpy reader)
	│   └── header
	│   ├── src
	│        
	├── sys		        (ML units i.e conv, dense, activation)
	│   └── header
	│   ├── src      	
	│   
	├── Waveglow		(WN, upsample, main)
	│   └── header
	│   ├── src  
	├── tools
		└── get_waveglow_weights.py
		└── npy_2_aud.py	


# Getting Started
1.  Git clone the repository
2. Download [waveglow_weights](https://drive.google.com/file/d/170W_2vua0xAOZ5YpmwMufrUg9HYbpe5E/view?usp=sharing)
3.  Download [mel_spectrograms](https://drive.google.com/open?id=1VD1OTQ5yBWUTGVrAdMzmz25As2XMGLRx)
4.  Update waveglow_weights path in waveglow/header/hparams.hpp file 
5.  Run this 
```
	make
	ls -d path_2_mel_folder  >  filename.txt
	./waveglow_tts filename.txt OutputDir
	python tools/npy_2_aud.py OutputDir 
  ```
6.  Audio will be stored in OutputDir in .wav format
# Traning
You can also train your model using [this](https://github.com/NVIDIA/waveglow) and then use copy tools/get_waveglow_weights.py file in waveglow folder and run
```
 python get_waveglow_weights.py <checkpoint path>
 ```

# Inference and Results
> Currently the code takes around 250ms to generate 10secs of speech

# Resources and refrences

 - [Waveglow paper](https://arxiv.org/pdf/1811.00002.pdf)
 - [Waveglow open-source code](https://github.com/NVIDIA/waveglow)
 - [Blog on Normalising flows by Eric jang](https://blog.evjang.com/2018/01/nf1.html)