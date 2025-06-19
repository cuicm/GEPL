# Code for “Pre-train, prompt” Framework to Boost Graph Neural Networks Performance in EEG Analysis

![p1 (2)_00](https://github.com/user-attachments/assets/94d8537f-b144-446d-bed3-61f1d9ffd8df)
Overview of workflow and GEPL. (a) EEG signals are resampled and cropped using a fixed-length window, then transformed using the Fourier transform to create a graph structure where electrodes serve as nodes and correlation coefficients form the adjacency matrix. In the pre-training dataset, the EEG signals from the subjects are segmented as extensively as possible, whereas in the downstream task dataset, each subject's EEG signal is segmented into a single segment. 
(b) During the pre-training phase, contrastive learning is used to generate graph augmentations, while contrastive loss optimizes the model’s ability to generalize to EEG data. All parameters of the model are updated throughout this process. 
(c) In the target dataset, graph prompt tuning modifies node features and graph connections using learnable prompts to enhance task-specific performance. During this phase, only the parameters of the learnable graph prompt and the linear classification layer are updated, while the parameters of the pre-trained model remain unchanged.


