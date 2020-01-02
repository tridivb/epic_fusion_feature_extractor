# Epic Fusion Feature Extractor

Extract features from videos with a pre-trained Epic Fusion (Temporal Binding) Model. The original project page can be found here, [EPIC-Fusion](https://ekazakos.github.io/TBN/) along with the paper. The code was built on top of the one provided by the authors. The github repo is, [EPIC-Fusion Code](https://github.com/ekazakos/temporal-binding-network)

## Getting Started

Clone the repo and set it up in your local drive.

```
git clone https://github.com/tridivb/epic_fusion_feature_extractor.git
```

### Prerequisites

Python >= 3.6\
[Pytorch](https://pytorch.org/)  >= 1.1\
[OmegaConf](https://github.com/omry/omegaconf) \
[Numpy](https://numpy.org/) \
[Tqdm](https://github.com/tqdm/tqdm) \
[PIL](https://pillow.readthedocs.io/en/stable/) \
[Parse](https://pypi.org/project/parse/) \

\
We use the same pre-processing steps as the original authors and create the following hierarchy for the dataset files,

```
|---<path to dataset>
|   | rgb_prefix
|   |   |--- video_1
|   |   |   |--- img_0000000000
|   |   |   |--- x_0000000000
|   |   |   |--- y_0000000000
|   |   |   |--- .
|   |   |   |--- .
|   |   |   |--- .
|   |   |--- .
|   |   |--- .
|   |   |--- .
|   |   |--- video_100
|   |   |   |--- img_0000000000
|   |   |   |--- x_0000000000
|   |   |   |--- y_0000000000
|   |   |   |--- .
|   |   |   |--- .
|   |   |   |--- .
|   |   |--- .
|   |   |--- .
|   |   |--- .
|   audio_prefix
|   |   |--- video_1.wav
|   |   |--- .
|   |   |--- .
|   |   |--- video_100.wav
|   |   |--- .
|   |   |--- .
|   |   |--- .
|   |---vid_list.txt
```

A script to create the pre-processing symlinks and extract audio for epic-kitchens can be found at the original [repo](https://github.com/ekazakos/temporal-binding-network/tree/master/preprocessing_epic). However you can also separate the flow files into a separate sub directory. Only the naming conventions have to be the same as shown above.
\
The vid_list.txt should have the names of all the video files or subdirectories for extracted frames, which are to be processed.
Based on the hierarchy above, it should be like:
```
video_1
video_2
.
.
video_100
.
.
```

### Installing
\
Navigate to the ```epic_fusion_feature_extractor``` directory and create a directory for the pretrained weights.

```
git clone https://github.com/tridivb/epic_fusion_feature_extractor.git
cd epic_fusion_feature_extractor
mkdir weights
```
\
Download the pre-trained [weights](https://drive.google.com/uc?export=download&id=1c2z0xrshfpLvhcbkIpNJVcdyPe5rEO-g). \
Rename the file as "model_best.pth" and put it in the ```weights``` directory.

```
mv epic_tbn_rgbflowaudio.pth.tar <path/to/repo>/epic_fusion_feature_extractor/weights/
```

### Configure the paramters

Copy the existing config file ```./configs/config.yaml``` and rename it as ```<config_file>.local.yaml```
\
Set the paths in the ```./configs/<config_file>.local.yaml``` file:

```
# Dataset Parameters
DATA:
  # Root directory of dataset
  ROOT_DIR: "<absolute/path/to/dataset/root>"
  RGB_DIR_PREFIX: "<prefix/to/rgb/frames>"
  FLOW_DIR_PREFIX: "<prefix/to/flow/frames>"
  AUDIO_DIR_PREFIX: "<prefix/to/audio/files>"
  # Currently only supports multimodal feature extraction (Don't change)
  MODALITY: ["RGB", "Flow", "Spec"]
  # Input video/rgb fps
  VID_FPS: 60
  # Fps of extracted flow frames
  FLOW_FPS: 30
  # Desired output feature fps 
  OUT_FPS: 10
  RGB_FILE_FMT: "img_{:010d}.jpg"
  FLOW_FILE_FMT: "{}_{:010d}.jpg"
  AUDIO_FILE_FMT: "{}.wav"
  # Number of consecutive flow files to interleave
  FLOW_WIN_LENGTH: 5
  # Audio sampling rate
  SAMPLING_RATE: 24000
  # Length of audio sample in seconds
  AUDIO_LENGTH: 1.279
  # List of videos to process
  VID_LIST: "<absolute/path/to/video/list/file.txt>"
  # Output directory
  OUT_DIR: "<absolute/path/to/output/dir>"
# Model Parameters
MODEL:
  # Only supports BNInception for now
  ARCH: "BNInception"
  # List of classes. Default one is for epic fusion dataset
  NUM_CLASSES: [125, 352]
  # Pretrained model weights
  CHECKPOINT: "<path/to/repo>/epic_fusion_feature_extractor/weights/model_best.pth"
# Misc
NUM_WORKERS: 8
```

### Extracting the features and detections

To extract features, execute the extract_features.py as follows:

```
python extract_features.py ./configs/<config_file>.local.yaml
```

We use the same pre-trained weights and parameters provided by the authors.

### Saved Featues

The features are saved as dictionaries to a pickle file using torch.save. The pickle file format is,

```
{
    "<rgb_frame_idx_0>": {
        "RGB": torch.tensor(...),
        "Flow": torch.tensor(...),
        "Spec": torch.tensor(...)
    }
    "<rgb_frame_idx_1>": {
        "RGB": torch.tensor(...),
        "Flow": torch.tensor(...),
        "Spec": torch.tensor(...)
    .
    .
    .
}
```

The file can be loaded as follows:

```
torch.load("<path/to/features/file>/<video_id>_<out_fps>.pkl", map_location="cpu")
```

Since the features are saved as torch tensors, they are by default gpu tensors. If you want to keep them on the gpu, just remove the ```map_location="cpu"``` part from the above code snippet.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.\
\
Please note, the original Epic-Fusion framework is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. Please respect the original licenses as well.

## Acknowledgments

1. EPIC-Fusion: Audio-Visual Temporal Binding for Egocentric Action Recognition
    ```
    @InProceedings{kazakos2019TBN,
    author    = {Kazakos, Evangelos and Nagrani, Arsha and Zisserman, Andrew and Damen, Dima},
    title     = {EPIC-Fusion: Audio-Visual Temporal Binding for Egocentric Action Recognition},
    booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2019}
    }
    ```

2. Readme Template -> https://gist.github.com/PurpleBooth/109311bb0361f32d87a2