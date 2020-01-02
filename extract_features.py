import os
import argparse
import time
import torch
import torchvision
from omegaconf import OmegaConf
from tqdm import tqdm

from dataset.dataset import VideoDataSet
from models.models import TBN
from dataset.transforms import (
    GroupScale,
    GroupCenterCrop,
    Stack,
    ToTorchFormatTensor,
    GroupNormalize,
)
import pickle


def parse_args():
    parser = argparse.ArgumentParser(
        description="Video feature extractor using the BNInceptopn Epic Fusion model"
    )
    parser.add_argument("cfg", type=str, help="config file")

    return parser.parse_args()


def get_time_diff(start_time, end_time):
    """
    Helper function to calculate time difference

    Args
    ----------
    start_time: float
        Start time in seconds since January 1, 1970, 00:00:00 (UTC)
    end_time: float
        End time in seconds since January 1, 1970, 00:00:00 (UTC)

    Returns
    ----------
    hours: int
        Difference of hours between start and end time
    minutes: int
        Difference of minutes between start and end time
    seconds: int
        Difference of seconds between start and end time
    """

    hours = int((end_time - start_time) / 3600)
    minutes = int((end_time - start_time) / 60) - (hours * 60)
    seconds = round((end_time - start_time) % 60)
    return (hours, minutes, seconds)


def extract_features(cfg):
    """
    Helper function to extract and save features from epic fusion tbn model

    Args
    ----------
    cfg: OmegaConf 
        Config dictionary of parameters
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modality = cfg.DATA.MODALITY

    out_dir = cfg.DATA.OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    num_classes = tuple(cfg.MODEL.NUM_CLASSES)

    net = TBN(
        num_classes,
        1,
        cfg.DATA.MODALITY,
        base_model=cfg.MODEL.ARCH,
        consensus_type="avg",
        dropout=0.5,
        midfusion="concat",
    )

    checkpoint = torch.load(cfg.MODEL.CHECKPOINT)
    base_dict = {
        ".".join(k.split(".")[1:]): v for k, v in list(checkpoint["state_dict"].items())
    }
    net.load_state_dict(base_dict)
    print("Pretrained weights loaded from {}".format(cfg.MODEL.CHECKPOINT))
    print("----------------------------------------------------------")

    # TODO
    # if device.type == "cuda" and torch.cuda.device_count() > 1:
    #     net = torch.nn.DataParallel(net)
    # else:
    #     net = net.to(device)

    net = net.to(device)
    print("Model loaded to {}".format(device))
    print("----------------------------------------------------------")

    with open(cfg.DATA.VID_LIST) as f:
        vid_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
    print("Video List loaded.")
    print("----------------------------------------------------------")

    transform = {}
    for m in modality:
        if m != "Spec":
            cropping = torchvision.transforms.Compose(
                [GroupScale(net.scale_size[m]), GroupCenterCrop(net.input_size[m]),]
            )
            transform[m] = torchvision.transforms.Compose(
                [
                    cropping,
                    Stack(roll=cfg.MODEL.ARCH == "BNInception"),
                    ToTorchFormatTensor(div=cfg.MODEL.ARCH != "BNInception"),
                    GroupNormalize(net.input_mean[m], net.input_std[m]),
                ]
            )
        else:
            transform[m] = torchvision.transforms.Compose(
                [
                    Stack(roll=cfg.MODEL.ARCH == "BNInception"),
                    ToTorchFormatTensor(div=False),
                ]
            )

    start = time.time()
    for vid_id in vid_list:
        print("Processing {}...".format(vid_id))
        dataset = VideoDataSet(cfg, vid_id, modality, transform=transform,)

        # TODO Implement multi batch processing
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=cfg.NUM_WORKERS
        )

        with torch.no_grad():
            net.eval()
            feat_dict = {}

            for data in tqdm(dataloader):
                for m in modality:
                    data[m] = data[m].to(device)

                feat = net(data)
                feat_dict[str(data["frame_idx"].item())] = feat

        torch.save(
            feat_dict,
            os.path.join(out_dir, "{}_{}.pkl".format(vid_id, cfg.DATA.OUT_FPS)),
        )

    print(
        "Done. Total time taken (HH:MM:SS): {}".format(
            get_time_diff(start, time.time())
        )
    )


def main():

    args = parse_args()

    cfg = OmegaConf.load(args.cfg)
    print(cfg.pretty())
    print("----------------------------------------------------------")

    extract_features(cfg)


if __name__ == "__main__":
    main()
