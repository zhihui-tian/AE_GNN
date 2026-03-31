# NPS
Neural Phase Simulation

Implements neural networks for prediction of microstructure evolution in 2D and 3D

Networks:
  * Convolutional LSTM (PredRNN, PredRNN++, E3dLSTM)
  * Convolutional n-gram models
    * 1-gram ResNet
  * VAE with RNN on latent bottleneck (TBD)
  * Attention based, transformer like (TBD)
  
Loss functions:
  * L1, L2 loss of pixels/voxels
  * GAN loss (TBD)
  * Perceptual loss (TBD)
  
Special ops
  * Point group symmetry through data augmentation
  * Periodic convolution
  * Attention (TBD)
