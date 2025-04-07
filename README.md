<h2 align="center">🩸BP-nnUNet: Blood Pressure Assisted Cerebral Microbleed Segmentation via Meta-matching</h2>

<p align="center">
  <b>Junmo Kwon<sup>1</sup>, Jonghun Kim<sup>1,2</sup>, Taehyeon Kim<sup>3</sup>, Sang Won Seo<sup>4</sup>, Hwan-ho Cho<sup>5</sup>, Hyunjin Park<sup>1,2</sup></b>
</p>

<p align="center">
  <sup>1</sup>Department of Electrical and Computer Engineering, Sungkyunkwan University, Suwon 16419, South Korea<br>
  <sup>2</sup>Center for Neuroscience Imaging Research, Institute for Basic Science, Suwon 16419, South Korea<br>
  <sup>3</sup>Department of Computer and Information Technology, Purdue University, West Lafayette, IN 47907, USA<br>
  <sup>4</sup>Department of Neurology, Samsung Medical Center, Sungkyunkwan University School of Medicine, Seoul 06351, South Korea<br>
  <sup>5</sup>Department of Electronics Engineering, Incheon National University, Incheon 22012, South Korea<br>
</p>

<p align="center">
  🎉 Our work has been provisionally accepted for MICCAI 2025! (top 9% among submissions) 🎉<br>
  🎉 Our work has been selected for oral presentation at MICCAI 2025! (top 2% among submissions) 🎉<br>
</p>

## Notes on nnU-Net Version
We provide compatible source codes for both [**nnUNet-v1**](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) and [**nnUNet-v2**](https://github.com/MIC-DKFZ/nnUNet).

Please see the [**Installation**](#installation) section if you want to integrate **BP-nnUNet** into an existing **nnU-Net** framework.

## Prerequisites
**Meta-matching** and **BiomedCLIP** embeddings should be provided before running **BP-nnUNet**.

### Running Pre-trained Meta-matching v1.0
To feed the blood pressure data as a joint deep prompt, run [**Meta-matching v1.0**](https://github.com/ThomasYeoLab/Meta_matching_models/tree/main/T1/v1.0) using the T1 MRI scan.

To run the pre-trained Meta-matching model called **simple fully convolutional network (SFCN; [Peng et al., MedIA 2021](https://doi.org/10.1016/j.media.2020.101871))**:
1. Download the [**pre-trained weights**](https://github.com/ThomasYeoLab/Meta_matching_models/blob/main/T1/v1.0/model/CBIG_ukbb_dnn_run_0_epoch_98.pkl_torch) ([**Wulan et al., Imaging Neuroscience 2024**](https://doi.org/10.1162/imag_a_00251))
2. Pre-process T1 MRI scans using the [**PreFreesurfer Pipeline**](https://github.com/Washington-University/HCPpipelines/blob/v4.3.0-rc.3/Examples/Scripts/PreFreeSurferPipelineBatch.sh). We highly recommend using [**docker container of HCP Pipelines**](https://hub.docker.com/r/bids/hcppipelines).
3. Resample ``./MNINonLinear/xfms/T1w_acpc_dc_restore_brain_to_MNILinear.nii.gz`` from 0.7mm isotropic to 1mm isotropic resolution.
4. Extract ICV using [**CBIG_mics.py**](https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/predict_phenotypes/Naren2024_MMT1/cbig/CBIG_mics.py#L401) and ``./MNINonLinear/xfms/acpc2MNILinear.mat``.
5. Follow the [**notebook example**](https://github.com/ThomasYeoLab/Meta_matching_models/blob/main/T1/v1.0/meta_matching_v1.0.ipynb) to get Meta-matching phentoypes.
```
import numpy as np
from CBIG_util import metamatching_infer

y_dummy = np.zeros(T1_MRI.shape[0], 1)  # irrelevant but required field to infer meta-matching
phenotypes = metamatching_infer(resampled_T1_MRI, ICV_from_acpc2MNILinear, y_dummy, model_path)
```

### Running Pre-trained BiomedCLIP
To generate text embeddings using [**BiomedCLIP**](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224), Please check [``initialize_biomedclip.py``](./initialize_biomedclip.py). Before running the script, please modify ``OUTPUT_DIR`` to a proper output directory. We also provide text embeddings in [``embeddings``](./embeddings) directory.

## Installation

Please follow the official nnU-Net installation guide. Visit [**nnUNet-v1**](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) and [**nnUNet-v2**](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md) for details.

### Integration to Existing nnU-Net Framework

For **nnUNet-v1**, replace [``inference/predict.py``](./nnUNetv1/inference/predict.py) and copy the following files:
* [``utilities/bp_nnunet.py``](./nnUNetv1/utilities/bp_nnunet.py)
* [``network_architecture/bp_nnUNet.py``](./nnUNetv1/network_architecture/bp_nnUNet.py)
* [``network_training/BP_Trainer.py``](./nnUNetv1/training/network_training/BP_Trainer.py)

For **nnUNet-v2**, replace [``inference/predict_from_raw_data.py``](./nnUNetv2/inference/predict_from_raw_data.py) and copy the following files:
* [``utilities/bp_nnunet.py``](./nnUNetv2/utilities/bp_nnunet.py)
* [``training/nnUNetTrainer/BP_Trainer.py``](./nnUNetv2/training/nnUNetTrainer/BP_Trainer.py)

## Major Changes in Existing nnU-Net Framework

Following changes should be made to run **BP-nnUNet**.
1. Import blood pressure data into **BP-nnUNet**
2. Set an additional environment variables for additional input data
3. Modify ``Tensor`` class to feed additional input data
4. Implement a proposed ``BP_nnUNet`` class

### Import Blood Pressure Data

Write a JSON file containing 67 phenotypes for each subject. The JSON file should be readable through ``load_json`` in [**batchgenerators**](https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/utilities/file_and_folder_operations.py#L102).

The JSON file should follow the format ``"subject_name": {"phenotypes": [val1, val2, val3, ...]}``, where ``phenotypes`` list must contain exactly 67 floating-point numbers.

An example form of JSON file should be like this. See [``BP_example.json``](./BP_example.json) for details.
```
{
    "example1": {
        "phenotypes": [0.1234, 0.5678, 0.9012, ...]
    },
    "example2": {
        "phenotypes": [0.1234, 0.5678, 0.9012, ...]
    }
}
```

### Set Environment Variables

Specify a path of blood pressure data in JSON and text embeddings from **BiomedCLIP** in ``.pth``.

An example of environment variables should be like this.
```
os.environ["BP_nnUNet_BP_JSON"] = "/path/to/bp.json"
os.environ["BP_nnUNet_text_embed"] = "/path/to/text/embeddings"
```
The ``BP_nnUNet_text_embed`` directory should contain the following text embeddings with matching filenames, which are provided in [``embeddings``](./embeddings).
```
cerebral_microbleed.pth
deep_microbleed.pth
lobar_microbleed.pth
```
If you wish to manually generate text embeddings, please refer to [``initialize_biomedclip.py``](./initialize_biomedclip.py).

### Modify ``Tensor`` Class

To minimize changing source codes, we define ``MetaTensor`` to feed additional input data. See ``bp_nnunet.py`` in ``utilities`` for [**nnUNet-v1**](./nnUNetv1/utilities/bp_nnunet.py) and [**nnUNet-v2**](./nnUNetv2/utilities/bp_nnunet.py).

``MetaTensor`` only works when both environment variables ``BP_nnUNet_BP_JSON`` and ``BP_nnUNet_text_embed`` are set.

It is imperative to modify the inference process, which is [``predict.py``](./nnUNetv1/inference/predict.py) in **nnUNet-v1** and [``predict_from_raw_data.py``](./nnUNetv2/inference/predict_from_raw_data.py) in **nnUNet-v2**.

To see modification details, search for the keyword ``BP-nnUNet`` in ``predict.py`` and ``predict_from_raw_data.py``.

### Implement ``BP_nnUNet``

See [``bp_nnUNet.py``](./nnUNetv1/network_architecture/bp_nnUNet.py) for **nnUNet-v1** and [``BP_Trainer.py``](./nnUNetv2/training/nnUNetTrainer/BP_Trainer.py) for **nnUNet-v2**.

## nnUNet-v1 Training Example (Jupyter Notebook)

```
import os

# Make sure to configure accordingly.
working_dir = "/path/to/working/directory"
task_id = 401
bp_json = "blood_pressure.json"
text_embed = "text_embed"

os.environ['nnUNet_raw_data_base'] = os.path.join(working_dir, 'nnUNet_raw_data_base')
os.environ['nnUNet_preprocessed'] = os.path.join(working_dir, 'preprocessed')
os.environ['RESULTS_FOLDER'] = os.path.join(working_dir, 'nnUNet_trained_models')
os.environ['BP_nnUNet_BP_JSON'] = os.path.join(working_dir, bp_json)
os.environ['BP_nnUNet_text_embed'] = os.path.join(working_dir, text_embed)

os.chdir(working_dir)
!nnUNet_plan_and_preprocess -t "{task_id}"
for fold in [0, 1, 2, 3, 4]:
    !nnUNet_train 3d_fullres BP_Trainer "{task_id}" "{fold}"
```

## nnUNet-v2 Training Example (Jupyter Notebook)

```
import os

# Make sure to configure accordingly.
working_dir = "/path/to/working/directory"
dataset_id = 1
bp_json = "blood_pressure.json"
text_embed = "text_embed"

os.environ['nnUNet_raw'] = os.path.join(working_dir, 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = os.path.join(working_dir, 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = os.path.join(working_dir, 'nnUNet_results')
os.environ['BP_nnUNet_BP_JSON'] = os.path.join(working_dir, bp_json)
os.environ['BP_nnUNet_text_embed'] = os.path.join(working_dir, text_embed)

os.chdir(working_dir)
!nnUNetv2_plan_and_preprocess -d "{dataset_id}" -pl nnUNetPlannerResEncL
for fold in [0, 1, 2, 3, 4]:
    !nnUNetv2_train "{dataset_id}" 3d_fullres "{fold}" -p nnUNetResEncUNetLPlans -tr BP_Trainer
```

## Acknowledgement

Part of the codes are referred from the following open-source projects:

* https://github.com/MIC-DKFZ/nnUNet
* https://github.com/ljwztc/CLIP-Driven-Universal-Model
* https://github.com/yeerwen/UniSeg

## Citation

If you find this code useful in your research, please consider citing:

```
@inproceedings{kwon2025blood,
    title={Blood Pressure Assisted Cerebral Microbleed Segmentation via Meta-matching},
    author={Kwon, Junmo and Kim, Jonghun and Kim, Taehyeon and Seo, Sang Won and Cho, Hwan-ho and Park, Hyunjin},
    booktitle={28th International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
    pages={77--86},
    doi={10.1007/978-3-032-04927-8_8},
    year={2025}
}

@inproceedings{kwon2024anatomically,
    title={Anatomically-Guided Segmentation of Cerebral Microbleeds in T1-weighted and T2*-weighted MRI},
    author={Kwon, Junmo and Seo, Sang Won and Park, Hyunjin},
    booktitle={27th International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
    pages={24--33},
    doi={10.1007/978-3-031-72069-7_3},
    year={2024}
}
```
