# ml_in_chem_big_homework
This repository is our group assignment of class Machine Learning in Chemistry in 2021.
## Let's Begin
We reproduct the result in xxx.

wgan-gp is from previous work of Ishaan Gulrajani, et al.. Hans modified it to compatible with torh 1.10 and python3

structure and theory of wgan-div is from https://dx.doi.org/10.1007/978-3-030-01228-1_40
## model.py
Pytorch model of wgan-div, not including residual yet

DO NOT use batchnorm in Discriminator.
## eval.py
Evaluate the output of our model and render it into a gif of 10 frames.
## train_cifar10.py
Our trainer for the cifar10 dataset.
## tran_lsun.ipynb
Our trainer for the lsun church dataset.
## cifar10
result in in ani.gif
## lsun church
eval result with fid: https://github.com/mseitzer/pytorch-fid. randomly pick 10000 real images and 10000 fake images. 
lsun church 90000 cht FID:  16.719440690521083.
Note: according to pytorch-fid, this fid is slightly different from that calculated by tensorflow.

120000 FID:  27.405582796819232
150000 FID:  30.643652300209567
