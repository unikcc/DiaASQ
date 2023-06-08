<!-- <p align="center"> -->
<!-- </p> -->
# DiaASQ
<a href="https://github.com/unikcc/DiaASQ">
  <img src="https://img.shields.io/badge/DiaASQ-0.1-blue" alt="pytorch 1.8.1">
</a>
<a href="https://github.com/unikcc/DiaASQ" rel="nofollow">
  <img src="https://img.shields.io/badge/pytorch-1.8.1-green" alt="pytorch 1.8.1">
</a>
<a href="https://huggingface.co/docs/transformers/index" rel="nofollow">
  <img src="https://img.shields.io/badge/transformers-4.24.0-orange" alt="Build Status">
</a>

This repository contains data and code for the ACL23 (findings) paper: [DiaASQ: A Benchmark of Conversational Aspect-based Sentiment Quadruple Analysis](https://arxiv.org/abs/2211.05705)

Also see the [project page](https://conasq.pages.dev/) for more details.

------

To clone the repository, please run the following command:

```bash
git clone https://github.com/unikcc/DiaASQ
```

## News 🎉
<!-- :sparkles: `2023-05-10`: Released code and dataset.   -->
:loudspeaker: `2023-05-10`: Released code and dataset.  
:zap: `2022-12-10`: Created repository.  


## Quick Links
- [Overview](#overview)
- [Requirements](#requirements)
- [Code Usage](#code-usage)
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

## Code Usage 

+ Dataset: the dataset can be found at:
  ```bash
  data/dataset
  ```

+ Train && Evaluate on the Chinese dataset
  ```bash 
  bash scripts/train_zh.sh
  ```

+ Train && Evaluate on the English dataset
  ```bash 
  bash scripts/train_en.sh
  ```

+ GPU memory requirements 

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
