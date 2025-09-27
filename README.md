# WGambaNet
This is the official code repository for "WGambaNet: Multi-scale Wavelet Boundary Enhancement and Dynamic Gating Fusion Network for Medical Image Segmentation".
## 0. Environments
- To set up the environment, you can use the following command to create a conda environment:
```bash
conda env create -f environment.yml
```
- Alternatively, install the required dependencies with pip:
```bash
pip install -r requirements.txt
```
## 1. Prepare the dataset
- The ISIC17 and ISIC18 datasets, which are divided into a 7:3 ratio, can be downloaded by following the instructions in [VM-UNet](https://github.com/JCruan519/VM-UNet).

- After downloading the datasets, you are supposed to put them into './data/isic17/' and './data/isic18/', and the file format reference is as follows. (take the ISIC17 dataset as an example.)

- './data/isic17/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png
    
### ACDC datasets
- For the ACDC dataset, you could follow [CSwin-UNet](https://github.com/eatbeanss/CSWin-UNet) to download the dataset, or you could download them from {[Baidu](https://pan.baidu.com/s/1CYogWveMjiqnjEekW_huWA?pwd=trv9 )}.

- After downloading the datasets, you are supposed to put them into `./data/acdc/`, then run the `split_acdc.py` file to split the dataset. The file format reference is as follows.

- './data/acdc/'
  - train
    - patientxxx
      - patientxxx_framexx_gt.nii
      - patientxxx_framexx.nii
  - test
    - patientxxx
      - patientxxx_framexx_gt.nii
      - patientxxx_framexx.nii
  - val
    - patientxxx
      - patientxxx_framexx_gt.nii
      - patientxxx_framexx.nii
      
### Synapse datasets
- For the Synapse dataset, you could follow [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet) to download the dataset, or you could download them from {[Baidu](https://pan.baidu.com/s/1JCXBfRL9y1cjfJUKtbEhiQ?pwd=9jti)}.

- After downloading the datasets, you are supposed to put them into './data/Synapse/', and the file format reference is as follows.

- './data/Synapse/'
  - lists
    - list_Synapse
      - all.lst
      - test_vol.txt
      - train.txt
  - test_vol_h5
    - casexxxx.npy.h5
  - train_npz
    - casexxxx_slicexxx.npz

## 2. Prepare the pre_trained weights

- The weights of the pre-trained VMamba could be downloaded from [Baidu](https://pan.baidu.com/s/1ci_YvPPEiUT2bIIK5x8Igw?pwd=wnyy) or [GoogleDrive](https://drive.google.com/drive/folders/1ZJjc7sdyd-6KfI7c8R6rDN8bcTz3QkCx?usp=sharing). After that, the pre-trained weights should be stored in './pretrained_weights/'.

## 3. Train the WGambaNet
```bash
cd WGambaNet
python train.py  # Train and test WGambaNet on the ISIC17 or ISIC18 dataset.
python train_acdc.py  # Train and test WGambaNet on the ACDC dataset.
python train_synapse.py  # Train and test WGambaNet on the Synapse dataset.
```

## 4. Obtain the outputs
- After trianing, you could obtain the results in './results/'

## 5. Acknowledgments

- We thank the authors of [VMamba](https://github.com/MzeroMiko/VMamba) and [VM-UNet](https://github.com/JCruan519/VM-UNet) for their open-source codes.