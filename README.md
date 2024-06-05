# Timer (Large Time Series Model)

This repo provides official code and checkpoints for [Timer: Generative Pre-trained Transformers Are Large Time Series Models](https://arxiv.org/abs/2402.02368).

# Updates

:triangular_flag_on_post: **News** (2024.6) Pre-training dataset (UTSD) is available in [HuggingFace](https://huggingface.co/datasets/thuml/UTSD)!

:triangular_flag_on_post: **News** (2024.5) Accepted by ICML 2024, a [camera-ready version](https://arxiv.org/abs/2402.02368) of **31 pages**.

:triangular_flag_on_post: **News** (2024.4) The pre-training scale has been extended, enabling zero-shot forecasting.

:triangular_flag_on_post: **News** (2024.2) Releasing model checkpoints and code for adaptation.

## Introduction

**Tim**e Series Transfor**mer** (Timer) is a Generative Pre-trained Transformer for general time series analysis. You can visit our [Homepage](https://thuml.github.io/timer/) for a more detailed introduction.

<p align="center">
<img src="./figures/abilities.png" alt="" align=center />
</p>

## Datasets

We curate [Unified Time Series Datasets (UTSD)]((https://huggingface.co/datasets/thuml/UTSD)) comprised of **1B time points** and **4 volumes** to facilitate the research on large time series models and pre-training.

<p align="center">
<img src="./figures/utsd.png" alt="" align=center />
</p>

## Tasks

> **[Forecasting](./scripts/forecast/README.md)**: We provide all scripts as well as datasets for few-shot forecasting in this repo.

> **[Imputation](./scripts/imputation/README.md)**:  We propose segment-level imputation, which is more challenging than point-level imputation.

> **[Anomaly Detection](scripts/anomaly_detection/README.md)**: We provide new benchmarks of predictive anomaly detection on [UCR Anomaly Archive](https://arxiv.org/pdf/2009.13807).

See each detailed README file in the folder ```./scripts/```.

## Code for Fine-tuning 

1. Install Pytorch and necessary dependencies.

```
pip install -r requirements.txt
```

2. Put the datasets from [Google Drive](https://drive.google.com/file/d/1yffcQBcMLasQcT7cdotjOVcg-2UKRarw/view?usp=drive_link) and [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/6bc31f9a003b4d75a10b/) under the folder ```./dataset/```.

3. Put the checkpoint from [Google Drive](https://drive.google.com/file/d/1vDy-nAwYwrppl61nvGpzordLlJAYLWf7/view?usp=drive_link) and [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/fd356758318847218605/) under the folder ```./checkpoints/```.

4. Train and evaluate the model. We provide the above tasks under the folder ```./scripts/```.

```bash
# forecasting
bash ./scripts/forecast/ECL.sh

# segement-level imputation
bash ./scripts/imputation/ECL.sh

# anomaly detection
bash ./scripts/anomaly_detection/UCR.sh
```

We provide the option for Direct Multi-Step (DMS) and Iterated Multi-Step (IMS) generation in each task through the argument `--use_ims`.

## Approach

### Pre-training and Adaptation

To facilitate pre-training on heterogeneous time series, we propose **single-series sequence (S3)**, reserving series variations with the unified context length. For diverse tasks, we convert forecasting, imputation, and anomaly detection into a **unified generative task**.

<p align="center">
<img src="./figures/pretrain_adaptation.png" align=center />
</p>

### Model Architecture

Given the limited exploration of the backbone for large time series models, we extensively evaluate candidate backbones and adopt the decoder-only Transformer with autoregressive generation towards LTSMs.

<p align="center">
<img src="./figures/architecture.png" align=center />
</p>


## Performance

We compare Timer with state-of-the-art approaches and present the pre-training benefit on few-shot scenarios.

<p align="center">
<img src="./figures/performance.png" align=center />
</p>

## Scalability

By increasing the parameters and pre-training scale, Timer achieves notable performance improvement: 0.231 $\to$ 0.138 (−40.3%), surpassing the previous state-of-the-art deep forecasters.

<p align="center">
<img src="./figures/scale.png" alt="300" align=center />
</p>

## Flexible Sequence Length

The decoder-only architecture provides the flexibility to accommodate time series of different lookback and forecast lengths.

<p align="center">
<img src="./figures/length.png" alt="300" align=center />
</p>

## Showcases

> **Forecasting under data scarcity**

<p align="center">
<img src="./figures/showcases_forecast.png" alt="" align=center />
</p>

> **Imputation with few-shot samples**

<p align="center">
<img src="./figures/showcases_imputation.png" alt="" align=center />
</p>

> **Anomaly detection on UCR Anomaly Archive**

<p align="center">
<img src="./figures/showcases_detection.png" alt="" align=center />
</p>

## Future Work

We are preparing to provide the online service for zero-shot forecasting ([demo](https://thuml.github.io/timer/img/demo.mp4)). Please stay tuned for the update!
  

## Citation

If you find this repo helpful, please cite our paper. 

```
@article{liu2024timer,
 title={Timer: Transformers for Time Series Analysis at Scale},
 author={Liu, Yong and Zhang, Haoran and Li, Chenyu and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng},
 journal={arXiv preprint arXiv:2402.02368},
 year={2024} 
}
```

## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- 

## Contact

If you have any questions or want to use the code, feel free to contact:
* Yong Liu (liuyong21@mails.tsinghua.edu.cn)
* Haoran Zhang (z-hr20@mails.tsinghua.edu.cn)
