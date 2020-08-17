# A Weakly Supervised Adaptive Densenet for Classifying Thoracic Diseases and Identifying Abnormalities

Bo Zhou, Yuemeng Li, Jiangcong Wang

[[Paper](https://arxiv.org/pdf/1807.01257.pdf)]

This repository contains the PyTorch implementation of adaptive densenet for chest x-ray's weakly supervised learning.

### Citation
If you use this code for your research or project, please cite:

    @article{zhou2018weakly,
      title={A weakly supervised adaptive densenet for classifying thoracic diseases and identifying abnormalities},
      author={Zhou, Bo and Li, Yuemeng and Wang, Jiangcong},
      journal={arXiv preprint arXiv:1807.01257},
      year={2018}
    }


### Environment and Dependencies
Requirements:
* Python 3.7
* Pytorch 0.4.1
* scipy
* scikit-image
* opencv-python
* tqdm

Our code has been tested with Python 3.7, Pytorch 0.4.1, CUDA 10.0 on Ubuntu 18.04.


### Dataset Setup
    ../
    Data/
    ChestXray14
    ├── images                   # contain all the 1024x1024 imaging data in .png format
    │   ├── 00000001_000.png         
    │   ├── 00000001_001.png 
    │   ├── ...         
    │   └── 00030805_000.png 
    │
    ├── labels                   # contain train / val / test .txt label splitted files
    │   ├── train_list.txt         
    │   ├── val_list.txt      
    │   └── test_list.txt 
    │            
    └── ...

Each .png is an image data with intensity ranged between 0~255. 

Please download the ChestXray14 dataset from [LINK](https://academictorrents.com/details/557481faacd824c83fbf57dcf7b6da9383b3235a). 

### To Run Our Code
- Train the model
```bash
python train.py --experiment_name 'train_bone_msunet' --model_type 'model_bone' --dataset 'DE' --data_root './example_data/' --net_G 'msunet' --net_D 'patchGAN' --wr_recon 50 --batch_size 2 --lr 1e-4 --AUG
```
where \
`--experiment_name` provides the experiment name for the current run, and save all the corresponding results under the experiment_name's folder. \
`--data_root`  provides the data folder directory (with structure illustrated above). \
`--AUG` adds for using data augmentation option (rotation, random cropping, scaling). \
Other hyperparameters can be adjusted in the code as well.

- Test the model
```bash
python test.py --resume './output/train_bone_msunet/checkpoints/model_best.pt' --experiment_name 'test_bone_msunet' --model_type 'model_bone' --dataset 'DE' --data_root './example_data/' --net_G 'msunet' --net_D 'patchGAN'
```
where \
`--resume` defines which checkpoint for testing and evaluation. The 'model_best.pt' is available upon request.  \
The test will output an eval.mat containing model's input and prediction for evaluation in the '--experiment_name' folder.

Sample training/test scripts are provided under './scripts/' and can be directly executed.

### Contact 
If you have any question, please file an issue or contact the author:
```
Bo Zhou: bo.zhou@yale.edu
```