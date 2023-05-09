<!-- <p align="center"> -->
<!-- </p> -->
## DiaASQ
<a href="https://github.com/unikcc/DiaASQ">
  <img src="https://img.shields.io/badge/DiaASQ-0.1-blue" alt="pytorch 1.8.1">
</a>
<a href="https://github.com/unikcc/DiaASQ" rel="nofollow">
  <img src="https://img.shields.io/badge/pytorch-1.8.1-green" alt="pytorch 1.8.1">
</a>
<a href="https://huggingface.co/docs/transformers/index" rel="nofollow">
  <img src="https://img.shields.io/badge/transformers-4.24.0-orange" alt="Build Status">
</a>

This repository contains data and code for our paper "DiaASQ: A Benchmark of Conversational Aspect-based Sentiment Quadruple Analysis](https://arxiv.org/abs/2211.05705)"

To clone the repository, please run the following command:

```bash
git clone https://github.com/unikcc/DiaASQ
```

## News
:sparkles: `2023-05-10`: Released training code.  
:loudspeaker: `2023-05-10`: Released the train and valid dataset.  
:zap: `2022-12-10`: Created repository.  


## Quick Links
- [Overview](#overview)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Model Usage](#model-usage)
- [Citation](#citation)


## Overview
In this work, we propose a new task named DiaASQ, which aims to extract Target-Aspect-Opinion-Sentiment quadruples from the given dialogue.
More details about the task can be found in our [paper](https://arxiv.org/abs/2211.05705).

<center>
<img src="./data/fig_sample.png" width="50%" />
</center>


## Requirements

The model is implemented using PyTorch. The versions of the main packages:

+ python>=3.7
+ torch>=1.8.1

Install the other required packages:
``` bash
pip install -r requirements.txt
```

## Data Preparation

<!-- You can either choose to use the parsed data or build the data manually. -->

##### Parsed data 
You can download the parsed data with json format from [Google Drive Link](https://drive.google.com/file/d/1MsY8LqbnQ40te-i_OmL5wOT6vQr6PuQi/view?usp=share_link).
Unzip the files and place them under the data directory like the following:

```bash
data/dataset/jsons_zh
data/dataset/jsons_en
```

The dataset currently only includes the train and valid sets. The test set will be released at a later date, refer to [this issue](https://github.com/unikcc/DiaASQ/issues/5#issuecomment-1495612887) for more information.

<!-- 
##### Build data manually
You can also manually run the scripts to transform the ann and txt format to json format.
1. Download the source data (ann and txt) from [Google Drive Link]
2. Then, unzip the files and place them under the data directory like the following:
```
./data/dataset/annotation_zh
./data/dataset/annotation_en

```  
3. Run the following commands, then you will obtained the parsed file with json format.
```bash
python src/prepare_data.py
python src/prepare_data.py --lang en
``` -->

## Model Usage 

+ Train && Evaluate for Chinese dataset
  ```bash 
  bash scripts/train_zh.sh
  ```

+ Train && Evaluate for English dataset
  ```bash 
  bash scripts/train_en.sh
  ```

+ If you do not have a `test` set yet, you can run the following command to train and evaluate the model on the `valid` set.
  ```bash 
  bash scripts/train_zh_notest.sh
  bash scripts/train_en_notest.sh
  ```

+ Cuda memory requirements 

| Dataset | Batch size | GPU Memory |
| --- | --- | --- |
| Chinese | 2 |  8GB. |
| English | 2 | 16GB. |

+ Customized hyperparameters:  
You can set hyperparameters in `main.py` or `src/config.yaml`, and the former has a higher priority.


## Citation
If you use our dataset, please cite the following paper:
```
@article{lietal2022arxiv,
  title={DiaASQ: A Benchmark of Conversational Aspect-based Sentiment Quadruple Analysis},
  author={Bobo Li, Hao Fei, Fei Li, Yuhan Wu, Jinsong Zhang, Shengqiong Wu, Jingye Li, Yijiang Liu, Lizi Liao, Tat-Seng Chua, Donghong Ji}
  journal={arXiv preprint arXiv:2211.05705},
  year={2022}
}
```
