import os
import argparse
import time

import torch.nn.parallel
import pandas as pd

from dataset.dataset import VideoDataSet
from models.models import TBN
from dataset.transforms import *
import pickle


def evaluate_model(num_class):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = TBN(num_class, 1, args.modality,
              base_model=args.arch,
              consensus_type=args.crop_fusion_type,
              dropout=args.dropout,
              midfusion=args.midfusion)

    weights = '{weights_dir}/model_best.pth.tar'.format(
        weights_dir=args.weights_dir)
    checkpoint = torch.load(weights)
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    net.load_state_dict(base_dict)

    # test_transform = {}
    # image_tmpl = {}
    # for m in args.modality:
    #     if m != 'Spec':
    #         if args.test_crops == 1:
    #             cropping = torchvision.transforms.Compose([
    #                 GroupScale(net.scale_size[m]),
    #                 GroupCenterCrop(net.input_size[m]),
    #             ])
    #         elif args.test_crops == 10:
    #             cropping = torchvision.transforms.Compose([
    #                 GroupOverSample(net.input_size[m], net.scale_size[m])
    #             ])
    #         else:
    #             raise ValueError("Only 1 and 10 crops are supported" +
    #                              " while we got {}".format(args.test_crops))


    #         test_transform[m] = torchvision.transforms.Compose([
    #             cropping, Stack(roll=args.arch == 'BNInception'),
    #             ToTorchFormatTensor(div=args.arch != 'BNInception'),
    #             GroupNormalize(net.input_mean[m], net.input_std[m]), ])

    #         # Prepare dictionaries containing image name templates
    #         # for each modality
    #         if m in ['RGB', 'RGBDiff']:
    #             image_tmpl[m] = "img_{:010d}.jpg"
    #         elif m == 'Flow':
    #             image_tmpl[m] = args.flow_prefix + "{}_{:010d}.jpg"
    #     else:

    #         test_transform[m] = torchvision.transforms.Compose([
    #             Stack(roll=args.arch == 'BNInception'),
    #             ToTorchFormatTensor(div=False), ])


    # data_length = net.new_length

    # if args.dataset != 'epic' or args.test_list is not None:
    #     # For other datasets, and EPIC when using EPIC_val_action_labels.pkl
    #     test_loader = torch.utils.data.DataLoader(
    #         TBNDataSet(args.dataset,
    #                    pd.read_pickle(args.test_list),
    #                    data_length,
    #                    args.modality,
    #                    image_tmpl,
    #                    visual_path=args.visual_path,
    #                    audio_path=args.audio_path,
    #                    num_segments=args.test_segments,
    #                    mode='test',
    #                    transform=test_transform,
    #                    fps=args.fps,
    #                    resampling_rate=args.resampling_rate),
    #         batch_size=1, shuffle=False,
    #         num_workers=args.workers * 2)
    # else:
    #     # When test_list is not provided,
    #     # Seen and Unseen timestamps will be automatically loaded
    #     # just to extract scores on EPIC
    #     test_seen_loader = torch.utils.data.DataLoader(
    #         TBNDataSet(args.dataset,
    #                    test_timestamps('seen'),
    #                    data_length,
    #                    args.modality,
    #                    image_tmpl,
    #                    visual_path=args.visual_path,
    #                    audio_path=args.audio_path,
    #                    num_segments=args.test_segments,
    #                    mode='test',
    #                    transform=test_transform,
    #                    fps=args.fps,
    #                    resampling_rate=args.resampling_rate),
    #         batch_size=1, shuffle=False,
    #         num_workers=args.workers * 2)

    #     test_unseen_loader = torch.utils.data.DataLoader(
    #         TBNDataSet(args.dataset,
    #                    test_timestamps('unseen'),
    #                    data_length,
    #                    args.modality,
    #                    image_tmpl,
    #                    visual_path=args.visual_path,
    #                    audio_path=args.audio_path,
    #                    num_segments=args.test_segments,
    #                    mode='test',
    #                    transform=test_transform,
    #                    fps=args.fps,
    #                    resampling_rate=args.resampling_rate),
    #         batch_size=1, shuffle=False,
    #         num_workers=args.workers * 2)

    # net = torch.nn.DataParallel(net, device_ids=args.gpus).to(device)
    # with torch.no_grad():
    #     net.eval()
    #     data_gen_dict = {}
    #     results_dict = {}
    #     if args.dataset != 'epic' or args.test_list is not None:
    #         data_gen_dict['test'] = test_loader
    #     else:
    #         data_gen_dict['test_seen'] = test_seen_loader
    #         data_gen_dict['test_unseen'] = test_unseen_loader

    #     for split, data_gen in data_gen_dict.items():
    #         results = []
    #         total_num = len(data_gen.dataset)

    #         proc_start_time = time.time()
    #         max_num = args.max_num if args.max_num > 0 else total_num
    #         for i, (data, label) in enumerate(data_gen):
    #             if i >= max_num:
    #                 break
    #             rst = eval_video(data, net, num_class, device)
    #             if label != -10000:  # label exists
    #                 if args.dataset != 'epic':
    #                     label_ = label.item()
    #                 else:
    #                     label_ = {k: v.item() for k, v in label.items()}
    #                 results.append((rst, label_))
    #             else:  # Test set (S1/S2)
    #                 results.append((rst,))
    #             cnt_time = time.time() - proc_start_time
    #             print('video {} done, total {}/{}, average {} sec/video'.format(
    #                 i, i + 1, total_num, float(cnt_time) / (i + 1)))

    #         results_dict[split] = results
    #     return results_dict

def parse_args():
    parser = argparse.ArgumentParser(description="Standard video-level" +
                                     " testing")
    parser.add_argument('dataset', type=str,
                        choices=['ucf101', 'hmdb51', 'kinetics', 'epic'])
    parser.add_argument('modality', type=str,
                        choices=['RGB', 'Flow', 'RGBDiff', 'Spec'],
                        nargs='+', default=['RGB', 'Flow', 'Spec'])
    parser.add_argument('weights_dir', type=str)
    parser.add_argument('--test_list')
    parser.add_argument('--visual_path')
    parser.add_argument('--audio_path')
    parser.add_argument('--arch', type=str, default="resnet101")
    parser.add_argument('--scores_root', type=str, default='scores')
    parser.add_argument('--test_segments', type=int, default=25)
    parser.add_argument('--max_num', type=int, default=-1)
    parser.add_argument('--test_crops', type=int, default=10)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--crop_fusion_type', type=str, default='avg',
                        choices=['avg', 'max', 'topk'])
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    parser.add_argument('--flow_prefix', type=str, default='')
    parser.add_argument('--fps', type=float, default=60)
    parser.add_argument('--resampling_rate', type=int, default=24000)
    parser.add_argument('--midfusion', choices=['concat', 'gating_concat', 'multimodal_gating'],
                    default='concat')
    parser.add_argument('--exp_suffix', default='')



def main():

    global args
    args = parse_args()

    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400
    elif args.dataset == 'beoid':
        num_class = 34
    elif args.dataset == 'epic':
        num_class = (125, 352)
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    evaluate_model(num_class)


if __name__ == '__main__':
    main()
