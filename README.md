# Spatial feature fusion in 3D convolutional architectures for lung tumor segmentation from 3D CT images

## Publication

Najeeb, S., & Bhuiyan, M. I. H. (2022). **Spatial feature fusion in 3D convolutional autoencoders for lung tumor segmentation from 3D CT images.** *Biomedical Signal Processing and Control, 78*, 103996.

[Read Online](https://www.sciencedirect.com/science/article/pii/S174680942200444X?casa_token=X-Q6xxJMUTgAAAAA:obFFHXJATpV7j-oAybaqx8dZYB2xGfeTh6KQ_C2WxXFruL3etqAsU_nsqy4Nr93Chfv7Inxubg) | [PDF](https://pdf.sciencedirectassets.com/273545/1-s2.0-S1746809422X00050/1-s2.0-S174680942200444X/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIF8SKzZyo4bOWvXNrUf2UpXg%2BmF8tjsW%2FD1hLqIhISh9AiB8L33g%2F7RdiBT%2FijLG2JzF5oda5DjmrsJSQU4GSc1mnCrVBAi7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAUaDDA1OTAwMzU0Njg2NSIMooMWYelr8nWcl9HzKqkEsHgqp2eDLIj4xBTGnYd5qyY56FbK5kgWd7LJ%2FQwXv%2BPB6llILKMOZxYgMtFElnAoKJ4NevitLMvaxxYMTGj9p2dD5nALRK0sKY8a854P%2BnkoiYIP8frdCgsJOHCJ3ESdbxt%2BWnjU%2FvTAxyiYME6Ohl%2FhTYFkv4mxX%2FG3OtT0F0gDt8dOyCl5dPooetKHET2BSzqI94jwMtS9pcdc47tj0Vy9PwmyiuHmbQyaY%2FiYeXglikuq4eC63N7snayHOGDtNnr14axq6LX1NCNO27jzDy5iBZNHVAaGdGukDAVLlUsIzua8rZ%2Fu%2FGzfl60bIU8UogylHqALX7x2uZsq1rtkeRYuSjteQVTVj4Gjtp4FuiDFOcBr94HgbZ5mLuacrdvkBhNj4%2FfqMKogsD9it%2F5BXhRmKf5y64R%2BYKI52GpqQjjGSQWdR9mNkEqaPYGKE4AZULWVV7PvD2p70dNupMXqrKD0yZZpvHUb11hKuvjDtQ1YDQ2UnlBpTtU%2FvjAb6J8unQeG3s%2F0fIadiEgzeWac4dagZ%2FyIDVZGXDA%2FXDI5Tnw6jDnVwPK0wEFKX4Tp64myKrSOO50VRd8N0ENruoPwcA0Cp%2FeF303%2BqU9GQfnajblpmML0izJ%2B5%2F2B8h5uIK06EGNNv9hqHkmTPkoOMrP3l6pu%2BLY8IHGMefQuX2MK0%2BaeGb22stSUbfBlONF7XalwAwMrVvOpGs8z7g85aRwWsOT41nvv8%2FY8DTCBhMObBjqqAcUeoCD6cNn8HnLT3CW7kGOlU%2BmHZwM%2FQUgm6EYoJyfj%2F%2BbOUCjojQsuKc6MImwza9bkgozUraMIfE4ZG7Gtzu3fg3uHfvt8GFnyeLXHXONtb2PkWgN5mupfGGOWRvlsxtt1vsboUCN4Coj4qndV9Zz0hV1osRyo1FEfuyBzNeZrZ6x3c2htw6HEFZ%2BisMry7xmhLfznGye0gh1gLHVYe8tIMiX%2Ff6vykNO%2B&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20221113T105339Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYWP6BF2GI%2F20221113%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=b9699deac0db01a4347287b6bc57a9a375783b0f3866c1d8cdc3fce2dfa08727&hash=63ba0220e66eec7e4895ab97aaea9773aff0b432415d6b2dbc1c1c7c228322e1&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S174680942200444X&tid=spdf-efa1a12f-e548-4099-a25f-f479fb5ebdf4&sid=6b46bf4f74a3574a0e7a59e8af08c0fd2294gxrqa&type=client&ua=4d505c5101575350560756&rr=7696f7024a9fdf1c)

## Abstract

Accurate detection and segmentation of lung tumors from volumetric CT scans is a critical area of research for the development of computer aided diagnosis systems for lung cancer. Several existing methods of 2D biomedical image segmentation based on convolutional autoencoders show decent performance for the task. However, it is imperative to make use of volumetric data for 3D segmentation tasks. Existing 3D segmentation networks are computationally expensive and have several limitations. In this paper, we introduce a novel approach which makes use of the spatial features learned at different levels of a 2D convolutional autoencoder to create a 3D segmentation network capable of more efficiently utilizing spatial and volumetric information. Our studies show that without any major changes to the underlying architecture and minimum computational overhead, our proposed approach can improve lung tumor segmentation performance by 1.61%, 2.25%, and 2.42% respectively for the 3D-UNet, 3D-MultiResUNet, and Recurrent-3D-DenseUNet networks on the LOTUS dataset in terms of mean 2D dice coefficient. Our proposed models also respectively report 7.58%, 2.32%, and 4.28% improvement in terms of 3D dice coefficient. The proposed modified version of the 3D-MultiResUNet network outperforms existing segmentation architectures on the dataset with a mean 2D dice coefficient of 0.8669. A key feature of our proposed method is that it can be applied to different convolutional autoencoder based segmentation networks to improve segmentation performance.

## Citation

```bib

@article{NAJEEB2022103996,
title = {Spatial feature fusion in 3D convolutional autoencoders for lung tumor segmentation from 3D CT images},
journal = {Biomedical Signal Processing and Control},
volume = {78},
pages = {103996},
year = {2022},
issn = {1746-8094},
doi = {https://doi.org/10.1016/j.bspc.2022.103996},
url = {https://www.sciencedirect.com/science/article/pii/S174680942200444X},
author = {Suhail Najeeb and Mohammed Imamul Hassan Bhuiyan},
keywords = {Segmentation, CT scan, Lung tumor, Convolutional autoencoders, Deep learning},
abstract = {Accurate detection and segmentation of lung tumors from volumetric CT scans is a critical area of research for the development of computer aided diagnosis systems for lung cancer. Several existing methods of 2D biomedical image segmentation based on convolutional autoencoders show decent performance for the task. However, it is imperative to make use of volumetric data for 3D segmentation tasks. Existing 3D segmentation networks are computationally expensive and have several limitations. In this paper, we introduce a novel approach which makes use of the spatial features learned at different levels of a 2D convolutional autoencoder to create a 3D segmentation network capable of more efficiently utilizing spatial and volumetric information. Our studies show that without any major changes to the underlying architecture and minimum computational overhead, our proposed approach can improve lung tumor segmentation performance by 1.61%, 2.25%, and 2.42% respectively for the 3D-UNet, 3D-MultiResUNet, and Recurrent-3D-DenseUNet networks on the LOTUS dataset in terms of mean 2D dice coefficient. Our proposed models also respectively report 7.58%, 2.32%, and 4.28% improvement in terms of 3D dice coefficient. The proposed modified version of the 3D-MultiResUNet network outperforms existing segmentation architectures on the dataset with a mean 2D dice coefficient of 0.8669. A key feature of our proposed method is that it can be applied to different convolutional autoencoder based segmentation networks to improve segmentation performance.}
}

```


## Proposed Methodology

![proposed_method](figures/proposed_method.jpg)

## Dataset

[LOTUS Benchmark](http://i-sip.encs.concordia.ca/datasets.html#Radiomics) (Lung-Originated Tumor Region Segmentation)
- Prepared as part of the IEEE VIP Cup 2018 Challenge
- Modified version of NSCLC-Radiomics Dataset
- Contains Computed Tomography (CT) scans of 300 lung cancer patients with NSCLC.
- CT Scan Resolution: 512x512

Segmentation Task: 
- GTV (Gross-Tumor Volume)

Dataset Statistics:

|||CT Scanner||Number of Slices||
|--|--|--|--|--|--|
|Dataset|Patients|CMS|Siemens|Tumor|Non-Tumor|
|Train|260|60|200|4296(13.7%)|26951(86.3%)|
|Test|40|34|6|848(18.9%)|3610(81.1%)|

## Data Processing

- Read scans: PyDicom Library
- Normalize HU Values between 0~1
- Slices resized to resolution 256x256

Sample Scans:

![samples](figures/samples.jpg)

## Data Augmentation

Data augmemntation is performed on-the-fly. One or more of the following data augmantations are performed on each training samples - 

(a) Random Rotation
(b) Horizontal Flip
(c) Random Elastic Deformation
(d) Random Contrast Normalization
(e) Random Noise
(f) Blurring

Illustration: 

![augs](figures/augs.jpg)

## Baseline Architectures

1. UNet (3D)
2. MultiResUNet (3D)
3. Recurrent-3D-DenseUNet

## Multi-Scale Spatial Feature Extractor:

The encoders of the 2D segmentation networks are utilized to extract spatial (2D) features from 2D CT slices. 

![feature_extract](figures/feature_extract.gif)


## Proposed Pipeline

![method](figures/method.gif)

Architecutre: 

![architecture](figures/architecture.png)

## Mask Generation

- 3D Segmentation networks: produce results on 8 consecutive slices
- To produce final segmentation masks, overlapping stacks of CT slices are processed by the segmentation networks.
- Overlapping masks are averaged which serves as a post-processing step to remove noise.
- Segmentation mask values are within 0~1 where 0 signifies no tumor and 1 signifies tumor
- Two-step thresholding approach is applied to generate final segmentation mask
    - Step 1: Apply a threshold of 0.7 to filter out false-positive slices
    - Step 2: Apply a threshold of 0.5 to generate the final tumor volume

![mask_gen](figures/mask_gen.jpg)

## Results

![results](figures/results_3d.jpg)

In terms of 2D dice coefficient, the proposed models with SFF achieve performance improvements of - 

|2D Dice Score|Without SFF|With SFF|Improvement|
|--|--|--|--|
|3D-UNet|0.8144|0.8275|1.61%|
|3D-MultiResUNet|0.8478|0.8669|2.25%|
|Recurrent-3D-DenseUNet|0.8080|0.8276|2.42%|

In terms of 3D dice coefficient, the proposed models with SFF achieve performance improvements of -

|3D Dice Score|Without SFF|With SFF|Improvement|
|--|--|--|--|
|3D-UNet|0.5440|0.5853|7.58%|
|3D-MultiResUNet|0.5803|0.5938|2.32%|
|Recurrent-3D-DenseUNet|0.5634|0.5874|4.28%|

## Visual Analysis

Sample 1

![sample_1](figures/sample_1.png)

Sample 2

![sample_2](figures/sample_2.png)

Sample 3
![sample_3](figures/sample_3.png)


## Edge Cases

![edge_cases](figures/edge_cases.png)

## Summary

- We have proposed three novel architecutres which incorporate multi-scale spatial feature fusion and improve lung tumor segmentation performance with minimal computational overhead. 
- Our proposed architectures achieved performance improvements of 1.61%, 2.25%, and 2.42% respectively in terms of 2D dice coefficient. 
- Our proposed architectures also achieved performance improvements of 7.58%, 2.32%, and 4.28% in terms of 3D dice coefficient. 

## Source Code

To obtain the baseline 2D models train using the script `code/train_2d.py`. The 3D data generation/pipeline can be found in here - https://github.com/muntakimrafi/TIA2020-Recurrent-3D-DenseUNet , https://github.com/udaykamal20/Team_Spectrum which is the implementation of the Recurrent-3D-DenseUNet. The models used in this publication can be found in `code/model_lib.py` folder. Example usage of the models with multi-scale-spatial-feature-fusion: 

```python
def get_HybridUNet001(model_fe, input_shape = (256, 256, 8, 1)):
    '''
    Usage: 
    base_model = load_model('preferred type: UNet2Dx3 (functional)') 
    model_fe = get_model_fe(base_model)
    model = get_HybridUNet001(model_fe)
    '''
    pass
```