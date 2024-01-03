# TeSTra: Real-time Online Video Detection with Temporal Smoothing Transformers

## Introduction

This is a PyTorch implementation for our ECCV 2022 paper "[`Real-time Online Video Detection with Temporal Smoothing Transformers`](https://arxiv.org/pdf/2209.09236.pdf)".

![teaser](assets/testra_teaser.png?raw=true)

## Environment
```shell
conda create -n testra python=3.7.7
conda activate testra
pip install -r requirements.txt
```

## Data Preparation

### Feature extractor
We selected "ResNet-50 pretrained on Kinetics-400" (tsn_r50_320p_1x1x8_100e_kinetics400_rgb checkpoint from [MMAction2](https://github.com/open-mmlab/mmaction2)) for RGB inputs.\
For optical flow inputs, we selected "NVIDIA Optical Flow SDK" or [DenseFlow](https://github.com/open-mmlab/denseflow) and BN-Inception.

You can directly download the pre-extracted feature (.zip) from the UTBox links below.


### (Alternative) Prepare dataset from scratch

You can also try to prepare the datasets from scratch by yourself. 

#### THUMOS14

For TH14, please refer to [LSTR](https://github.com/amazon-research/long-short-term-transformer#data-preparation).

#### EK100

For EK100, please find more details at [RULSTM](https://github.com/fpv-iplab/rulstm).



### Data Structure
1. If you want to use our [dataloaders](src/rekognition_online_action_detection/datasets), please make sure to put the files as the following structure:

    * THUMOS'14 dataset:
        ```
        $YOUR_PATH_TO_THUMOS_DATASET
        ├── rgb_kinetics_resnet50/
        |   ├── video_validation_0000051.npy (of size L x 2048)
        │   ├── ...
        ├── flow_kinetics_bninception/
        |   ├── video_validation_0000051.npy (of size L x 1024)
        |   ├── ...
        ├── target_perframe/
        |   ├── video_validation_0000051.npy (of size L x 22)
        |   ├── ...
        ```
    
    * EK100 dataset:
        ```
        $YOUR_PATH_TO_EK_DATASET
        ├── rgb_kinetics_bninception/
        |   ├── P01_01.npy (of size L x 2048)
        │   ├── ...
        ├── flow_kinetics_bninception/
        |   ├── P01_01.npy (of size L x 2048)
        |   ├── ...
        ├── target_perframe/
        |   ├── P01_01.npy (of size L x 3807)
        |   ├── ...
        ├── noun_perframe/
        |   ├── P01_01.npy (of size L x 301)
        |   ├── ...
        ├── verb_perframe/
        |   ├── P01_01.npy (of size L x 98)
        |   ├── ...
        ```

2. Create softlinks of datasets:
    ```
    cd TeSTra
    ln -s $YOUR_PATH_TO_THUMOS_DATASET data/THUMOS
    ln -s $YOUR_PATH_TO_EK_DATASET data/EK100
    ```

## Training
We select "testra_lite_long_512_work_8_kinetics_1x_box", which utilizes NVIDIA Optical Flow.\
\
The commands for training are as follows.\
From scratch
```shell
CUDA_VISIBLE_DEVICES=4 python tools/train_net.py --config_file configs/THUMOS/TESTRA/testra_lite_long_512_work_8_kinetics_1x_box.yaml --gpu 4
```

Finetune from a pre-trained model

```shell
CUDA_VISIBLE_DEVICES=4 python tools/train_net.py --config_file configs/THUMOS/TESTRA/testra_lite_long_512_work_8_kinetics_1x_box.yaml --gpu 4 MODEL.CHECKPOINT pretrained_weights/testra_lite_th14_long_512_work_8_nv_box.epoch-25.pth
```

## Online Inference

For existing checkpoints, please refer to the next [section](#main-results-and-checkpoints).

### Stream mode

```shell
CUDA_VISIBLE_DEVICES=4 python tools/test_net.py --config_file configs/THUMOS/TESTRA/testra_lite_long_512_work_8_kinetics_1x_box.yaml --gpu 4 MODEL.CHECKPOINT pretrained_weights/testra_lite_th14_long_512_work_8_nv_box.epoch-25.pth MODEL.LSTR.INFERENCE_MODE stream
```


Run the online inference in `stream mode` to calculate runtime in the streaming setting. 

    ```
    cd TeSTra/
    # Online inference in stream mode
    python tools/test_net.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES \
        MODEL.CHECKPOINT $PATH_TO_CHECKPOINT MODEL.LSTR.INFERENCE_MODE stream
    # The above one will take quite long over the entire dataset,
    # If you only want to look at a particular video, attach an additional argument:
    python tools/test_net.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES \
        MODEL.CHECKPOINT $PATH_TO_CHECKPOINT MODEL.LSTR.INFERENCE_MODE stream \
        DATA.TEST_SESSION_SET "['$VIDEO_NAME']"
    ```

For more details on the difference between `batch mode` and `stream mode`, please check out [LSTR](https://github.com/amazon-research/long-short-term-transformer#online-inference).