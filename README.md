
# BORT2
Pytorch implementation of BORT<sup>2</sup> for multi-source domain adaptation.

Pattern Recognition 2026 version: [Zhongying Deng, Da Li, Xiaojiang Peng, Yi-Zhe Song, Tao Xiang. "BORT<sup>2</sup>: Bi-level optimization for robust target training in multi-source domain adaptation." Pattern Recognition (2026)](https://doi.org/10.1016/j.patcog.2025.112367)

BMVC 2022 version: [Zhongying Deng, Da Li, Yi-Zhe Song, Tao Xiang. "Robust Target Training for Multi-Source Domain Adaptation." BMVC (2022)](https://bmvc2022.mpi-inf.mpg.de/0778.pdf)


## Installation

- Please first install the [Dassl codebase](https://github.com/KaiyangZhou/Dassl.pytorch#get-started) as follows (or refer to [Dassl codebase](https://github.com/KaiyangZhou/Dassl.pytorch#get-started) for more details):


Make sure [conda](https://www.anaconda.com/distribution/) is installed properly. The following installation commands are adapted from [Dassl codebase](https://github.com/KaiyangZhou/Dassl.pytorch#get-started).

```bash
# Clone this repo
git clone https://github.com/Zhongying-Deng/BORT2.git
cd BORT2

# Create a conda environment
conda create -n dassl python=3.7

# Activate the environment
conda activate dassl

# Install dependencies
pip install -r requirements.txt

# Install torch and torchvision (select a version that suits your machine)
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```

After that, `pytorch 1.7.1 + cuda 10.1, python 3.7` should be installed. **Note that installing Dassl is a must.**

- Follow the instructions in [DATASETS.md](./DATASETS.md) to install the datasets. All the data should be stored under the `data` folder.

- You may also be interested in my other implementations of unsupervised domain adaptation work: [DAC-Net (BMVC 2021)](https://github.com/Zhongying-Deng/DAC-Net), [DIDA-Net (IEEE T-IP 2022)](https://github.com/Zhongying-Deng/DIDA), [GeNRT (Pattern Recognition 2026)](https://github.com/Zhongying-Deng/GeNRT). They are also implemented using [Dassl codebase](https://github.com/KaiyangZhou/Dassl.pytorch#get-started).

### Training

- The training scripts for the first-step training are provided in the bash files, such as `train_digit5_ms_cutmix.sh` for training on Digit-Five using FixMatch-CM. The backbone models for the FixMatch-CM can be found at `dassl/modeling/backbone`, such as `resnet_mixstyle.py` or `cnn_digit5_m3sda_mixstyle.py`, all with the suffix ‘_mixstyle.py’. You can also implement other existing MSDA methods as the first-step training strategies.

- After the training with FixMatch-CM, you will obtain the first-step trained model, such as `model.pth`. 

- The second step BORT2 training uses the script `train_digit5_mscm_dist_net_meta_train_retrain.sh` (similar names apply to PACS). Note that in the bash script, you will need to specify the checkpoint from the first-step training by `MODEL.INIT_WEIGHTS model.pth`. The config files and the trainer (like `FixMatchMSCMDistNetMetaLearnRetrain`) are specified in the script. And the trainer for BORT2 is named  `FixMatchMSCMDistNetMetaLearnRetrain`, of which the py file can be found at `dassl/engine/da/fixmatch_mscm_dist_net_meta_learn_retrain.py`. 

=========================

The zip file `Dassl_BORT2.zip` contains an initial version of the code for BORT<sup>2</sup>, but it has been updated and replaced by this new version. The instructions for using the code in the zip file can be found at [this issue](https://github.com/Zhongying-Deng/BORT2/issues/1#issuecomment-1655890786).


## Citation
Please cite the following paper if you find Dassl useful to your research.

```
@article{DENG2026112367,
title = {BORT2: Bi-level optimization for robust target training in multi-source domain adaptation},
journal = {Pattern Recognition},
volume = {172},
pages = {112367},
year = {2026},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2025.112367},
url = {https://www.sciencedirect.com/science/article/pii/S0031320325010283},
author = {Zhongying Deng and Da Li and Xiaojiang Peng and Yi-Zhe Song and Tao Xiang},
keywords = {Multi-source domain adaptation, BORT, Bi-level optimization, Stochastic CNN, Implicit differentiation}
}
```

and

```
@inproceedings{Deng_2022_BMVC,
author    = {Zhongying Deng and Da Li and Yi-Zhe Song and Tao Xiang},
title     = {Robust Target Training for Multi-Source Domain Adaptation},
booktitle = {33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
publisher = {{BMVA} Press},
year      = {2022},
url       = {https://bmvc2022.mpi-inf.mpg.de/0778.pdf}
}
```
