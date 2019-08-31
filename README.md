# waveglow_cuda_inference
> C++ Code to run  **optimized inference  in CUDA** of Waveglow, this implementation gives **25% speedup** over [Nvidia's Pytorch implementation](https://github.com/NVIDIA/waveglow)

# Waveglow 
Cuda C++ implementation of NVIDIA's Waveglow. 

The model architecure based on flows is described in this paper. [WaveGlow: a Flow-based Generative Network for Speech Synthesis](https://arxiv.org/pdf/1811.00002.pdf). 

**_Waveglow, a flow-based network is capable of generating high quality speech from mel-spectograms_**. It combines insights from [Glow](https://arxiv.org/pdf/1807.03039.pdf) and  [Wavenet](https://arxiv.org/pdf/1609.03499.pdf)  in order to provide fast, efficient and high-quality audio synthesis, without the need for auto-regression. 

WaveGlow is implemented using only a single network, trained using only a single cost function: maximizing the likelihood of the training data, which makes the training procedure simple and stable. 

Paper claims that  *in full-precision* (32 bit float) waveglow produces speech at the 500kHz on V100 but typically it is about **300-325kHz** with pytorch's implementation and **400-420kHz** using our implementation.


# Repository Structure

# Setup and Installation
1. Set up this original [Waveglow repository](https://github.com/NVIDIA/waveglow) first
2. Copy and run the *get_weights.py* file using [pretrained weights](https://drive.google.com/file/d/1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx/view)
3. Git clone this repository
4. Update the weights folder path in hparams.hpp (in waveglow/header/)
5. Run the makefile 
6. Run ./waveglow_tts object file

# Inference and Results
> Currently the code takes around 500ms to generate 10secs of speech

# Resources and refrences

 - [Waveglow paper](https://arxiv.org/pdf/1811.00002.pdf)
 - [Waveglow open-source code](https://github.com/NVIDIA/waveglow)
 - [Blog on Normalising flows by Eric jang](https://blog.evjang.com/2018/01/nf1.html)