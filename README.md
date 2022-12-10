<!-- <p align="center"> -->
# DiaASQ
<a href="https://github.com/unikcc/DiaASQ">
  <img src="https://img.shields.io/badge/DiaASQ-0.1-blue" alt="DiaASQ">
</a>
<a href="https://github.com/unikcc/DiaASQ" rel="nofollow">
  <img src="https://img.shields.io/badge/pytorch-1.8.1-green" alt="pytorch 1.8.1">
</a>
<a href="https://huggingface.co/docs/transformers/index" rel="nofollow">
  <img src="https://img.shields.io/badge/transformers-4.24.0-orange" alt="Transformers">
</a>
<a href="https://github.com/unikcc/DiaASQ/blob/master/LICENSE" rel="nofollow">
  <img src="https://img.shields.io/badge/LICENSE-MIT-cyan" alt="LICENSE">
</a>

<!-- </p> -->
This repository contains code(to be done) and data for our paper "DiaASQ: A Benchmark of Conversational Aspect-based Sentiment Quadruple Analysis](https://arxiv.org/abs/2211.05705)"

To clone the repository, please run the following command:

```bash
git clone https://github.com/unikcc/DiaASQ
```


# Quick Links
+ [Overview](#overview)

+ [Requirements](#requirements)

+ [Data Preparation](#data-preparation)

+ [Citation](#citation)


# Overview
In this work, we propose a new task named DiaASQ, which aims to extract Target-Aspect-Opinion-Sentiment quadruples from the given dialogue.
You can find more details in our [paper](https://arxiv.org/abs/2211.05705).
<center>
<img src="./figures/fig_sample.png" width="50%" />
</center>


# Requirements

The model is implemented using PyTorch. The versions of the main packages used are shown below.

+ python>=3.8
+ attrdict>=2.0.1
+ jieba>=0.42.1
+ PyYAML>=6.0
+ spacy>=3.4.2
+ torch>=1.8.1

To set up the dependencies, you can run the following command:
``` bash
pip install -r requirements.txt
```

# Data Preparation

You can download the source data from [Google Drive Link](https://drive.google.com/file/d/1DTRq8lVAzJev75rdFP0y7GaQnzYFwORC/view?usp=sharing)

Then, unzip the files and place them under the data directory like the following:
```
./data/dataset/annotation_zh
./data/dataset/annotation_en
```

Generate JSON format files for Chinese data and English data(You should download ``)
```bash
python prepare_data.py --lang zh
python prepare_data.py --lang en
```

For example, the Chinese version train dataset with JSON format should locate at:
```
./data/dataset/json_zh/train.json
```

## Citation
If you want to use our dataset, please cite the following paper:
```
@article{lietal2022arxiv,
  title={DiaASQ: A Benchmark of Conversational Aspect-based Sentiment Quadruple Analysis},
  author={Bobo Li, Hao Fei, Fei Li, Yuhan Wu, Jinsong Zhang, Shengqiong Wu, Jingye Li, Yijiang Liu, Lizi Liao, Tat-Seng Chua, Donghong Ji}
  journal={arXiv preprint arXiv:2211.05705},
  year={2022}
}
```