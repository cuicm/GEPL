# “Pre-train, prompt" Framework to Boost Graph Neural Networks Performance in EEG Analysis

![image](https://github.com/user-attachments/assets/464f92d7-7e0c-49b7-931c-cd05a344c4f3)


Overview of workflow and GEPL. (a) EEG signals are resampled, cropped with a fixed window, and transformed using the Fourier transform to construct a graph structure with electrodes as nodes and correlation coefficients as the adjacency matrix. 
(b) In the pre-training phase, contrastive learning generates graph augmentations, and contrastive loss optimizes the model's generalization to EEG data. 
(c) In the target dataset, graph prompt tuning adjusts node features and graph connections with learnable prompts to enhance task-specific performance.

Preprocessed data can be downloaded from https://zenodo.org/records/13219018

Pre-training:
```
python Pretrain.py
```
Prompt tuning:
```
python tuning_.py --dataset PROCESSED/SHL.pth --tuning eeg-pro --model_path saved_models/encoder.pth --lr_decay True --batch_size 64 --num_epochs 500 
```
