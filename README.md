# Memory-Net-Inverse

**_Comprehensive Examination of Unrolled Networks for Solving Linear Inverse Problems_**

This repository includes the accompanying code for the paper "Comprehensive Examination of Unrolled Networks for Solving Linear Inverse Problems," including a novel generalized deep architecture for solving linear inverse problems called Long Memory Unrolled Networks, as well as a comprehensive ablation study on the impact of various hyperparameter choices and on robustness under various sampling conditions. Below are the details regarding the dataset, model architecture, results, and additional resources used.

---

## 📁 Repository Structure

- **`Denoising_Algorithms/Memory_Network/`**: The model architecture for the Long Memory Unrolled Network is located here.
  
- **`Results/`**: Contains all results, including:
  - Model weights
  - Training histories
  - Evaluation losses

- **`Auxiliary_Functions/` & `Denoising_Algorithms/DL_Training/`**: Houses auxiliary functions for image processing, training, and other utility scripts.

---

## 📊 Dataset

We utilize a subset of the 2012 ImageNet Object Large Scale Visual Recognition Challenge (ILSVRC 2012) - Object Localization Challenge validation dataset, which contains 50,000 images.

🔗 **Dataset link**: [Kaggle - ILSVRC 2012 Validation Data](https://www.kaggle.com/c/imagenet-object-localization-challenge/data)