# CycleSTTN: Specular Augmentation in Endoscopy
![teaser](./docs/ImageCompare2023v3.png?raw=true)
***Fig. 1.** Sample consecutive video frames from the model output using pseudo ground truth VR as input in columns 1-4 and real videos VA as input in columns 5-8.*

### [Paper](????) | [BibTex](#citation)

CycleSTTN: A Learning-Based Temporal Model for Specular Augmentation in Endoscopy<br>

Rema Daher, O. León Barbed, Ana C. Murillo, Francisco Vasconcelos, and Danail Stoyanov<br> 
_The 26th International Conference on Medical Image Computing and Computer Assisted Intervention, MICCAI 2023_.

## Citation
If any part of our paper and repository is helpful to your work, please generously cite with:
```
@article{daher2023cyclesttn,
  title={CycleSTTN: A Learning-Based Temporal Model for Specular Augmentation in Endoscopy},
  author={Daher, Rema and Barbed, O. León and Murillo, Ana C. and Vasconcelos, Francisco and Stoyanov, Danail},
  journal={International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2023}
}
```

Since this code is based on [STTN](https://github.com/researchmm/STTN) and [Endo-STTN](https://github.com/endomapper/Endo-STTN), please also cite: 
```
@inproceedings{yan2020sttn,
  author = {Zeng, Yanhong and Fu, Jianlong and Chao, Hongyang},
  title = {Learning Joint Spatial-Temporal Transformations for Video Inpainting},
  booktitle = {The Proceedings of the European Conference on Computer Vision (ECCV)},
  year = {2020}
}

@article{daher2023temporal,
  title={A Temporal Learning Approach to Inpainting Endoscopic Specularities and Its Effect on Image Correspondence},
  author={Daher, Rema and Vasconcelos, Francisco and Stoyanov, Danail},
  journal={Medical Image Analysis},
  year={2023}
}
```

## Paper Contributions 
* We propose the CycleSTTN training pipeline as an extension of STTN to a cyclic structure.
* We use CycleSTTN to train a model for synthetic generation of temporally consistent and realistic specularities in endoscopy videos. We compare results of our method against CycleGAN.
* We demonstrate CycleSTTN as a data augmentation technique that improves the performance of SuperPoint feature detector in endoscopy videos.

![Flowchart](./docs/CycleSTTN-2023-1-2.png?raw=true)
***Fig. 2.** CycleSTTN training pipeline with 3 main steps: $\fbox{1}$ Paired Dataset Generation, $\fbox{2}$ $STTN_A$ Pre-training, and $\fbox{3}$  $STTN_R, STTN_A$ Joint Training.*


## Installation  
```
git clone https://github.com/RemaDaher/CycleSTTN.git
cd CycleSTTN/
conda create --name sttn python=3.8.5
pip install -r requirements.txt
```

To install Pytorch, please refer to [Pytorch](https://pytorch.org/).
In our experiments we use the following installation for cuda 11.1: 
```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
``` 

## Dataset Preparation

Navigate to [./dataset_prep/README.md](./dataset_prep/README.md) for more details.

## Pretrained Models
For your reference, we provide our pretrained models:
  - [$STTN_{R0}, D_{R0}$](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucabrd0_ucl_ac_uk/Ess2Ea9oHLlFq_v0abebqv8BspwLmRKodazZpzBPSMZd9g?e=8wfEGb) named as **STTN_removal_model**. 
    - From [Endo-STTN](https://github.com/endomapper/Endo-STTN).
  - [$STTN_{A0}, D_{A0}$](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucabrd0_ucl_ac_uk/EoAGnYt_DoxFuLDMRKNpHdgB1DjAL-ddAiOWLTUDTrFddw?e=3ToxAv) named as **STTN_addition_model**. 
    - Tested with [test.py](./test.py)
  - [$STTN_{A1}, D_{A1}$; $STTN_{R1}, D_{R1}$](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucabrd0_ucl_ac_uk/Evo8wo8qjCxNk0KLboW7GNsBoFBYEUI7qZNwUgm89vj9bw?e=Obudws) named as **CycleSTTN_model**. 
    - Tested with [test-cycle.py](./test-cycle.py)

Download and unzip them in [./release_model/](./release_model/)


## Add and Remove specularities Using Pretrained Models
### Testing script:
1. Arguments that can be set with [test.py](./test.py) and [test-cycle.py](./test-cycle.py):
    - --overlaid: used to overlay the original frame pixels outside the mask region on your output. 
    - --shifted: used to inpaint using a shifted mask.
    - --framelimit: used to set the maximum number of frames per video (Default = 927).
    - --Dil: used to set the size of the structuring element used for dilation (Default = 8). If set to 0, no dilation will be made.
    - --nomask: used to only take image without mask as input.

<br />

2. To test using [test.py](./test.py) or [test-cycle.py](./test-cycle.py) on all the test videos in your dataset, listed in your test.json:
    ```
    python <<test.py or test-cycle.py>> --gpu <<INSERT GPU INDEX>> --nomask \
    --output <<INSERT OUTPUT DIR>> \
    --frame <<INSERT FRAMES DIR>> \
    --mask <<INSERT ANNOTATIONS DIR>> \
    -c <<INSERT PRETRAINED PARENT DIR>> \
    -cn <<INSERT PRETRAINED MODEL NUMBER>> \
    --zip
    ``` 

    - For example, using our pretrained models: 
      ```
      python test.py --gpu 1 --nomask \
      --output results/STTN_addition_model/ \
      --frame datasets/EndoSTTN_dataset/JPEGImages/ \
      --mask datasets/EndoSTTN_dataset/Annotations/ \
      -c release_model/STTN_addition_model/ \
      -cn 3 \
      --zip
      ```
      ```
      python test-cycle.py --gpu 1 --nomask \
      --output results/CycleSTTN_model/ \
      --frame datasets/EndoSTTN_dataset/JPEGImages/ \
      --mask datasets/EndoSTTN_dataset/Annotations/ \
      -c release_model/CycleSTTN_model/ \
      -cn 2 \
      --zip
      ```
    >>**_NOTE_**: When running this script the loaded frames and masks are saved as npy files in datasets/EndoSTTN_dataset/files/so that loading them would be easier if you want to rerun this script. To load these npy files use the --readfiles argument. This is useful when experimenting with a large dataset.

3. To test on 1 video: 
    ```
    python <<test.py or test-cycle.py>> --gpu <<INSERT GPU INDEX>> --nomask \
    --output <<INSERT VIDEO OUTPUT DIR>> \
    --frame <<INSERT VIDEO FRAMES DIR>> \
    --mask <<INSERT VIDEO ANNOTATIONS DIR>> \
    -c <<INSERT PRETRAINED PARENT DIR>> \
    -cn <<INSERT PRETRAINED MODEL NUMBER>>
    ``` 

    - For example, for a folder "ExampleVideo1_Frames" containing the video frames, using our pretrained models: 

      ``` 
      python test.py  --gpu 1 --nomask \
      --output results/STTN_addition_model/ \
      --frame datasets/ExampleVideo1_Frames/ \
      --mask datasets/ExampleVideo1_Annotations/ \
      -c release_model/STTN_addition_model/ \
      -cn 3
      ```
      ```
      python test-cycle.py  --gpu 1 --nomask \
      --output results/CycleSTTN_model/ \
      --frame datasets/ExampleVideo1_Frames/ \
      --mask datasets/ExampleVideo1_Annotations/ \
      -c release_model/CycleSTTN_model/ \
      -cn 2
      ``

4. Single frame testing:

    To test a single frame at a time and thus removing the temporal component, follow the same steps above but use [test-singleframe.py](./test-singleframe.py) instead of [test.py](./test.py) and [test-cycle-singleframe.py](./test-cycle-singleframe.py) instead of [test-cycle.py](./test-cycle.py).


## Training New Models
Once the dataset is ready, new models can be trained:
- Prepare the configuration file (ex: [STTN_addition_model.json](./configs/STTN_addition_model.json), [CycleSTTN_model.json](./configs/CycleSTTN_model.json)):
  - "gpu": \<INSERT GPU INDICES EX: "1,2"\>
  - "data_root": \<INSERT DATASET ROOT\>
  - "name": \<INSERT NAME OF DATASET FOLDER\>
  - "frame_limit": used to set the maximum number of frames per video (Default = 927)
  - "Dil": used to set the size of the structuring element used for dilation (Default = 8). If set to 0, no dilation will be made.

### Training Script
```
python <<train.py or train-cycle.py>> --model sttn \
--config <<INSERT CONFIG FILE DIR>> \
-c <<INSERT INITIALIZATION MODEL PARENT DIR>> \
-cn <<INSERT INITIALIZATION MODEL NUMBER>>
```
>For train-cycle.py, in addition to the arguments *(-c, -cn)*, we added *(-cRem, cnRem)* for the removal model and *(-cAdd, -cnAdd)* for the addition model. This was done for the case of separate initialization models for removal and addition.

- For example: 
  ```
  python train.py --model sttn \
  --config configs/STTN_addition_model.json
  ```
  ```
  python train-cycle.py --model sttn \
  --config configs/CycleSTTN_model.json \
  -cRem release_model/STTN_removal_model/ \
  -cnRem 9 \
  -cAdd release_model/STTN_addition_model/ \
  -cnAdd 3
  ```

# TODO: 
- MAKE SURE evaluation works and write section
## Evaluation
To quantitatively evaluate results using the pseudo-ground truth:
1. Test all videos using [Testing Script (2.)](#testing-script) with removed specularity frames as input (JPEGImagesNS folder instead of JPEGImages)
2. Use [quantifyResultsAddingSpecs.ipynb](./quantifyResultsAddingSpecs.ipynb) to generate csv files containing the quantitative results.

## Training Monitoring  

We provide training monitoring on losses by running: 
```
tensorboard --logdir release_model                                                    
```

<!-- ---------------------------------------------- -->
## Contact
If you have any questions or suggestions about this paper, feel free to contact me (remadaher711@gmail.com).

