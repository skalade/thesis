# Thesis code

## Intro

This repository contains sets of notebooks and scripts used to reproduce the results of my thesis "Deep Learning for Wireless Communications: Flexible Architectures and Multitask Learning". 

Proper software people, please avert your eyes, I realize much of this codebase is riddled with bad practices, copies of functions and global variables. I did try to make this as reproducible as possible, locked down random seeds, etc. I hope at least some of this will be useful for fellow future researchers going down the path I had decided to embark upon so long ago.

## Prerequisites

Software versions used to run all the notebooks:
* Python 3.8
* NumPy 1.20.1
* PyTorch 1.10
* CUDA 10.1

I might update these to more up-to-date versions at some point, but this is what the original results were obtained with. That said, I'm pretty sure most of the functions I've used should be backwards compatible with latest PyTorch releases.

**I highly recommend running the training notebooks with a GPU.** Even on my 1080ti the Seq2Seq models take days to train. Don't attempt on CPU.

## Contents

* **Background** - more educational than anything, you can use the code in these to reproduce the wireless communications background chapter figures. It covers baseband modulation, channel impairments and some high order stats (moments).
* **Seq2Seq** - Sequence-to-Sequence models based on RNNs. I used these models for simultaneous AMC and demodulation. Proof of concept model can do 2 classes - BPSK and QPSK. Doesn't have the bells and whistles of attention, or bidirectionality. This is a good example if you want to do comms Seq2Seq, however for general Seq2Seq education I highly recommend this repo: https://github.com/bentrevett/pytorch-seq2seq.
* **FCN** - Fully Convolutional Neural Network work on frame synchronization. There's some material in here that explains frame sync from scratch, which might be useful if you're new to the field. Primarily focuses on training FCNs for short preamble lengths you can find in Internet of Things (IoT) or sensor networks applications (e.g. 4 bytes).
* **MTL** - Multitask learning. This one's split into two:
    * **amc** - contains a set of notebooks for deep learning-based SNR estimation, then combines a regular VGG-like AMC network with an SNR estimator to create an SNR-aware AMC CNN.
    * **fcn** - expands upon the FCN work on frame sync and adds CFO estimation as an additional task to improve performance at high carrier offsets. Also shows how to train a fully convolutional double-headed frame synchronizer + SNR estimator.