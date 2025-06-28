"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import cv2
import argparse
import util.io
import numpy as np
from tqdm import tqdm

from torchvision.transforms import Compose
from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from dpt.models import DPTDepthModel
from src.weatherkitti_dataset import WeatherKITTIDepthMixedDataset

DATASET_DIR = 'data/kitti'
BASE_SPLIT_PATH = 'data_split/kitti_depth/'
TRAIN_FILE = 'eigen_train_files_with_gt.txt'
TEST_FILE = 'eigen_test_files_with_gt.txt'


def run(args):
    """Run MonoDepthNN to compute depth maps.
    """
    print("initialize")
    input_path = args.input_path
    output_path = args.output_path
    model_path = args.model_weights
    model_type = args.model_type
    optimize = args.optimize
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load network
    if model_type == "dpt_large":  # DPT-Large
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
    elif model_type == "dpt_hybrid_kitti":
        net_w = 1216
        net_h = 352

        model = DPTDepthModel(
            path=model_path,
            scale=0.00006016,
            shift=0.00579,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
    elif model_type == "dpt_hybrid_nyu":
        net_w = 640
        net_h = 480
        model = DPTDepthModel(
            path=model_path,
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
    else:
        assert (
            False
        ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid|dpt_hybrid_kitti|dpt_hybrid_nyu|midas_v21]"

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)

    model.to(device)

    # get input
    dataset = WeatherKITTIDepthMixedDataset(
        mode='EVAL',
        train_crop=None,
        squeeze_channel=True,
        filename_ls_path=os.path.join(BASE_SPLIT_PATH, TEST_FILE),
        dataset_dir=DATASET_DIR,
        disp_name='Weather KITTI DPT',
    )

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")
    
    for ind, sample_data in enumerate(tqdm(dataset, desc="Processing images")):
        img_input = sample_data['rgb_norm']
        rel_path = sample_data['rgb_relative_path']

        with torch.no_grad():
            sample = img_input.to(device).unsqueeze(0)
            if optimize and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample

            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=sample.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().cpu().numpy())
            
            # Save prediction as npy file
            npy_filename = rel_path.replace('.png', '.npy')
            npy_path = os.path.join(output_path, npy_filename)
            
            # Create all necessary subdirectories
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            
            np.save(npy_path, prediction.astype(np.float32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_path", default="data/kitti/rgb", help="folder with input images"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="output_monodepth",
        help="folder for output images",
    )
    parser.add_argument(
        "-m", "--model_weights", 
        default='weights/dpt_hybrid_kitti-cb926ef4.pt', 
        help="path to model weights"
    )
    parser.add_argument(
        "-t",
        "--model_type",
        default="dpt_hybrid_kitti",
        help="model type [dpt_large|dpt_hybrid|dpt_hybrid_kitti]",
    )
    parser.add_argument("--kitti_crop", dest="kitti_crop", action="store_true")
    parser.add_argument("--absolute_depth", dest="absolute_depth", action="store_true")

    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")

    parser.set_defaults(optimize=True)
    parser.set_defaults(kitti_crop=False)
    parser.set_defaults(absolute_depth=False)

    args = parser.parse_args()
    default_models = {
        "midas_v21": "weights/midas_v21-f6b98070.pt",
        "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
        "dpt_hybrid_kitti": "weights/dpt_hybrid_kitti-cb926ef4.pt",
        "dpt_hybrid_nyu": "weights/dpt_hybrid_nyu-2ce69ec7.pt",
    }
    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]
    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # compute depth maps
    run(args)
