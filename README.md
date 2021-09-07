Autos and AI for Low Code
==============================


<img src="https://user-images.githubusercontent.com/9223799/132366378-f7721343-a55d-4d91-b47d-7c733923df17.png" width="25%">
<br />
<br />

## Overview
A sample machine vision project that accepts an image of a car (presumably with forms of damage), and returns a picture annotated with boxes around likely areas of damage.  
<br />
## Motivation
We built this project to show how custom trained AI models are incorporated into low code platforms such as Mendix, referencing great work already done in the references section.  The aim of this project is to get readers excited about working with AI models, with code examples showing training, inference, containerization and prototype serving (not production) of a simple consumer facing use case. This is by no means a study into each area, but more an attempt at an end to end example of training to serving.   
<br />

## Quick start (in two sections):
* Git clone this project, making sure weight file (final.pth) downloads into models/weights/
* Make sure prereqs installed, docker, Nvidia drivers (if you are training).  If you are running inference or annotating pictures only without a GPU, it will take 5-10 seconds for a response.  
* For training, I would advise against training without a GPU unless you're running only a few epochs (or iterations).  We trained this model for 100 cycles, with around 80+ showing convergence (for prototype accuracy).

### Inference
* Use the makefile and run make build-prereqs, make build-image, make run-image (or run-image-gpu if you have Nvidia GPU access)
* Post  http://yourservername:5000/annotate with form data of type file, referencing an image from the references/test-images folder
* Expect 5-10 seconds inference time for CPU, or ~500 ms for GPU

### Training (code published in next version)
* Sign up for a free Weights and Biases account  https://wandb.ai, we use them for hyperparameter tracking.  Weight files are configured to be stored locally (so as to not exceed cloud storage space)
* Download dataset from Kaggle to ./data/external, and unzip (this is also in the main notebook)
* Assuming you're running a GPU locally or on a cloud instance, at the command line typing nvidia-smi should show you meaningful info.
* Additional steps here... (TBD)

<br />

## Tech used
* Python v3.9 
* Nvidia (Cuda 11 +) for GPU acceleration
* Docker, Pytorch, Faster R-CNN, Resnet50, COCO
* Kaggle, for data set hosting
* Weights and Biases for hyperparameter and results training
* Inference served with AWS g4dn.xlarge (Nvidia T4), AMI of Deep Learning Ubuntu 18
* Training locally with Ubuntu 20.04 (passthrough Nvidia 1080Ti) VM, Proxmox VE 7

<br />

## References (Huge thanks for their groundwork!):
* [Car Damage Classification](https://medium.com/analytics-vidhya/car-damage-classification-using-deep-learning-d29fa1e9a520_)
* [CNN Car Exterior Damage](https://medium.com/@sourish.syntel/cnn-application-detecting-car-exterior-damage-full-implementable-code-1b205e3cb48c)
* [Pytorch Object Detection](https://www.pyimagesearch.com/2021/08/02/pytorch-object-detection-with-pre-trained-networks/)
* [How to train Object Detector with COCO dataset in Pytorch](https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5) 

<br />
## Upcoming enhancements
* Publish of training components
* Conversion to ONNX and running direct within Mendix runtime

<br />
### Project Organization (not all parts are implemented)

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
