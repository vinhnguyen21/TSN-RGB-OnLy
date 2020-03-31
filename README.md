# TSN-RGB-OnLy

## Description
Our code is reconstructed code from origin github: https://github.com/yjxiong/tsn-pytorch for only RGB stream.

## Setup environment

Run the scripts to create a virtual environment and install dependency libraries

```
$conda create -n tsn python=3.6
$conda activate tsn
$pip install -r requirements.txt

```

## Dataset

- extract RGB frames from each video in UCF101 dataset with sampling rate: 10 fps

- Download the preprocessed data directly from : https://github.com/feichtenhofer/twostreamfusion

```
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.001
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.002
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.003

cat ucf101_jpegs_256.zip* > ucf101_jpegs_256.zip
unzip ucf101_jpegs_256.zip

```

## Training 
To train this project, we just run the command

```
$python train.py

```

where `train_config.json` contains parameters for training:

There is the list of parameters in the config file:
- *dataset* : string, *choices=['ucf101', 'hmdb51', 'kinetics']*
- *modality*: string, because of building only for RGB stream so in here we just have 1 option: *RGB*
- *model_type*: string, *"resnet18", "resnet34", "resnet50", "resnet101", "vgg16", "BNInception"*
- *data_path*: string, *"path to folder which contains folder videos"*
- *class_label*: string, *"path to class index, txt file"*
- *weight_folder*: string, *"path to folder which contains training'weights"*
- *model_path*: string, *"path to folder which contains weight for testing"*
- *num_segments*: int, *"number of segments that we split the video into"*
- *batch_size*: int, *"number of videos we will load while training"*
- *size*: int, *"size of each frame"*
- *consensus_type*: string, *choices = ['avg', 'attention']*. With 'avg', Average all output vector of N segments. With 'attention', find attention-score of each N segments and compute attention-vector.
- *dropout*: float, *"dropout ratio"*
- *loss_type*: string, *"nll"* cross-entropy
- *epochs*: integer, number of epochs for training
- *valid_size*: float, size of validation for splitting train-test-split
- *lr*: float, learning rate
- *lr_steps*: int, learning rate step
- *momentum*: float, momentum for optimization
- *weight_decay*: float, weight decay for optimization
- *clip_grad*: int, threshhold of gradient clipping
- *partialbn*: boolean, *"true": freeze batch-norm2d except last one, "false": unfreeze all batch-norm*
- *print_freq*: int, *"print loss top1 and top5 accuracy after print_freq iterations"*
- *eval_freq*: int, *"Saving weight of model after number of epochs"*
- *num_worker*: int, *"number of worker"*
- *gpus*: string, *"multi": using multiple of gpus, "cuda:0": specific gpu, ...*

## Testing
Adjust "model_path" in config file for loading training weight to model

Then run:

```
python test.py

```
