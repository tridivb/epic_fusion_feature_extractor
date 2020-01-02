# Source: https://github.com/ekazakos/temporal-binding-network
# Modified for feature extraction

from torch.utils.data import Dataset

import librosa as lr
from PIL import Image
import os
import numpy as np
from parse import parse


class VideoDataSet(Dataset):
    """
    Video Dataset class

    Args
    ----------
    cfg: OmegaConf dict
        Dictonary of config parameters
    vid_id: str
        Untrimmed Video Id
    modality: list
        List of modalities
    transform: list, default = None
        List of transforms to apply
    use_audio_pickle: boolean, default = False
        Flag to read audio from wav or pickle files
    
    """
    def __init__(
        self, cfg, vid_id, modality, transform=None, use_audio_pickle=False,
    ):

        self.cfg = cfg

        self.rgb_path = os.path.join(cfg.DATA.ROOT_DIR, cfg.DATA.RGB_DIR_PREFIX, vid_id)
        self.flow_path = os.path.join(
            cfg.DATA.ROOT_DIR, cfg.DATA.FLOW_DIR_PREFIX, vid_id
        )
        self.aud_path = os.path.join(cfg.DATA.ROOT_DIR, cfg.DATA.AUDIO_DIR_PREFIX)

        self.vid_id = vid_id
        self.flow_win_length = cfg.DATA.FLOW_WIN_LENGTH
        self.modality = modality
        self.rgb_fmt = cfg.DATA.RGB_FILE_FMT
        self.flow_fmt = cfg.DATA.FLOW_FILE_FMT
        self.audio_fmt = cfg.DATA.AUDIO_FILE_FMT
        self.transform = transform
        self.resampling_rate = cfg.DATA.SAMPLING_RATE
        self.vid_fps = self.cfg.DATA.VID_FPS
        self.flow_fps = self.cfg.DATA.FLOW_FPS
        self.flow_drop_rate = self.vid_fps / self.flow_fps
        self.out_fps = self.cfg.DATA.OUT_FPS
        self.step_size = self.vid_fps // self.out_fps
        self.use_audio_pickle = use_audio_pickle

        # TODO Make this robust
        all_rgb_files = sorted(
            filter(
                lambda x: x.startswith("img") and x.endswith("jpg"),
                os.listdir(self.rgb_path),
            ),
            key=lambda x: parse(self.rgb_fmt, x)[0],
        )

        self.first_index = parse(self.rgb_fmt, all_rgb_files[0])[0]
        self.last_index = parse(self.rgb_fmt, all_rgb_files[-1])[0]

        self.frame_indices = list(
            range(self.first_index, self.last_index + 1, self.step_size)
        )

        self.aud_sample = self._read_audio_sample()

    def __len__(self):
        """
        Get length of dataset
        """

        return len(self.frame_indices)

    def _log_specgram(self, audio, window_size=10, step_size=5, eps=1e-6):
        """
        Helper function to create 2D audio spectogram from 1D audio sample

        Args
        ----------
        audio: np.ndarray
            1D array of untrimmed audio sample
        window_size: int, default = 10
            Temporal size in millisecond of window function for STFT
        step_size: int, default = 5
            Hop size in millisecond between each sample for STFT
        eps: double, default = 1e-6
            Correction term to prevent divide by zero
        
        Returns
        ----------
        spec: np.ndarray
            Array of audio spectrogram

        """

        nperseg = int(round(window_size * self.resampling_rate / 1e3))
        noverlap = int(round(step_size * self.resampling_rate / 1e3))

        spec = lr.stft(
            audio,
            n_fft=511,
            window="hann",
            hop_length=noverlap,
            win_length=nperseg,
            pad_mode="constant",
        )

        spec = np.log(np.real(spec * np.conj(spec)) + eps)
        return spec

    def _extract_sound_feature(self, idx):
        """
        Helper function to create 2D audio spectogram from 1D audio sample

        Args
        ----------
        idx: int
            Center index of sound sample
        
        Returns
        ----------
        spec: np.ndarray
            Array of audio spectrogram

        """

        centre_sec = idx / self.vid_fps
        left_sec = centre_sec - 0.639
        right_sec = centre_sec + 0.639

        duration = self.aud_sample.shape[0] / float(self.resampling_rate)

        left_sample = int(round(left_sec * self.resampling_rate))
        right_sample = int(round(right_sec * self.resampling_rate))

        if left_sec < 0:
            samples = self.aud_sample[: int(round(self.resampling_rate * 1.279))]

        elif right_sec > duration:
            samples = self.aud_sample[-int(round(self.resampling_rate * 1.279)) :]
        else:
            samples = self.aud_sample[left_sample:right_sample]

        return self._log_specgram(samples)

    def _read_audio_sample(self):
        """
        Helper function to read audio sample from raw wav or pickle file
        
        Returns
        ----------
        sample: np.ndarray
            Array of 1D audio sample

        """

        if self.use_audio_pickle:
            # Read from numpy file
            npy_file = os.path.join(self.aud_path, "{}.npy".format(self.vid_id))
            try:
                sample = np.load(npy_file)
            except Exception as e:
                raise Exception(
                    "Failed to read audio sample {} with error {}".format(npy_file, e)
                )
        else:
            # Read from raw file
            aud_file = os.path.join(self.aud_path, self.audio_fmt.format(self.vid_id))
            try:
                sample, _ = lr.core.load(aud_file, sr=self.resampling_rate, mono=True,)
            except Exception as e:
                raise Exception(
                    "Failed to read audio sample {} with error {}".format(aud_file, e)
                )

        return sample

    def _load_data(self, modality, idx):
        """
        Helper function to load images or get spectrogram for an index

        Args
        ----------
        modality: str
            Input modality to process
        idx: int
            Index of the frame to be read
        
        Returns
        ----------
        img/spec: list
            List of array(s) containing the image or spectrogram

        """

        if modality == "RGB":
            img = Image.open(
                os.path.join(self.rgb_path, self.rgb_fmt.format(idx))
            ).convert("RGB")
            return [img]
        elif modality == "Flow":
            x_img = Image.open(
                os.path.join(self.flow_path, self.flow_fmt.format("x", idx))
            ).convert("L")
            y_img = Image.open(
                os.path.join(self.flow_path, self.flow_fmt.format("y", idx))
            ).convert("L")
            return [x_img, y_img]
        elif modality == "Spec":
            spec = self._extract_sound_feature(idx)
            return [Image.fromarray(spec)]

    def __getitem__(self, index):
        """
        Retrieve input data for a specified index

        Args
        ----------
        index: int
            Index of the dataset to be processed

        Returns
        ----------
        input: dict
            Dictionary of input data containing rgb frame index and images/spectogram

        """

        input = {}
        frame_idx = self.frame_indices[index]
        input["frame_idx"] = frame_idx

        for m in self.modality:

            img = self.get(m, frame_idx)
            input[m] = img

        return input

    def get(self, modality, frame_idx):
        """
        Helper function to get the transformed input data

        Args
        ----------
        modality: str
            Input modality to process
        frame_idx: int
            Index of the frame to be read
        
        Returns
        ----------
        imgages: list
            List of array(s) containing the image or spectrogram

        """

        images = list()

        if modality == "Flow":
            if self.flow_drop_rate == 1:
                last_flow_index = self.last_index
            else:
                last_flow_index = int(self.last_index / self.flow_drop_rate)
                # In case of mismatch in frame numbers because of approximation
                while True:
                    if os.path.exists(
                        os.path.join(
                            self.flow_path, self.flow_fmt.format("x", last_flow_index)
                        )
                    ):
                        break
                    else:
                        last_flow_index -= 1

            # handle boundary conditions
            start = min(
                max((frame_idx - self.flow_win_length) // 2, 0),
                (last_flow_index - self.flow_win_length + 1),
            )
            end = start + self.flow_win_length
        else:
            start = frame_idx
            end = frame_idx + 1
        for idx in range(start, end):
            imgs = self._load_data(modality, idx)
            images.extend(imgs)

        if self.transform:
            images = self.transform[modality](images)
        return images
