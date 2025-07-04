import os
import glob
import torch
import cv2
import logging
import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import util.io
from dpt.models import DPTDepthModel
from src.util.loss import ScaleAndShiftInvariantLoss
from src.dataset.weatherkitti_dataset import WeatherKITTIDepthMixedDataset

DATASET_DIR = 'data/kitti'
BASE_SPLIT_PATH = 'data_split/kitti_depth/'
TRAIN_FILE = 'eigen_train_files_with_gt.txt'
TEST_FILE = 'eigen_test_files_with_gt.txt'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_path", default="input", help="folder with input images"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="output_monodepth",
        help="folder for output images",
    )
    parser.add_argument(
        "-m", "--model_weights", default=None, help="path to model weights"
    )
    parser.add_argument(
        "-t",
        "--model_type",
        default="dpt_hybrid",
        choices=["dpt_large", "dpt_hybrid"],
        help="model type [dpt_large|dpt_hybrid]",
    )
    parser.add_argument(
        "--crop_size",
        default=384,
        type=int,
        help="crop size for training"
    )
    parser.add_argument(
        "--iterations",
        default=160000,
        type=int,
        help="iterations for training"
    )
    parser.add_argument(
        "--lr",
        default=1e-4,
        type=float,
        help="learning rate"
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="batch size for training"
    )
    parser.add_argument(
        "--logging_steps",
        default=200,
        type=int,
        help="logging steps for training"
    )
    parser.add_argument(
        "--output_dir",
        default='output',
        type=str,
        help="saving directory"
    )
    parser.add_argument(
        "--pretrained",
        default='weights/dpt_hybrid-midas-501f0c75.pt',
        type=str,
        help="pretrained weights directory"
    )

    parser.add_argument("--kitti_crop", dest="kitti_crop", action="store_true")
    parser.add_argument("--absolute_depth", dest="absolute_depth", action="store_true")

    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")

    parser.set_defaults(optimize=True)
    parser.set_defaults(kitti_crop=False)
    parser.set_defaults(absolute_depth=False)

    return parser.parse_args()
    

def build_model(args):
    model_type = args.model_type
    # load network
    if model_type == "dpt_large":  # DPT-Large
        model = DPTDepthModel(
            path=args.pretrained,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
       
    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        model = DPTDepthModel(
            path=args.pretrained,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
    else:
        raise NotImplementedError
    return model


def train_model(args):
    logging.info("Initialize for training.")

    model = build_model(args)
    model.train()

    # Set up device: automatically choose CUDA or CPU based on availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(memory_format=torch.channels_last).to(device)

    dataset = WeatherKITTIDepthMixedDataset(
        mode='train',
        train_crop=args.crop_size,
        squeeze_channel=True,
        filename_ls_path=os.path.join(BASE_SPLIT_PATH, TRAIN_FILE),
        dataset_dir=DATASET_DIR,
        disp_name='Weather KITTI DPT',
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iterations, eta_min=1e-6)

    criterion = ScaleAndShiftInvariantLoss()

    # Resume logic
    start_iter = 0
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "latest_checkpoint.pth")

    if os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['iter'] + 1
        logging.info(f"Resuming from iteration {start_iter}")

    for _ in range(start_iter):
        scheduler.step()

    current_iter = start_iter

    # Training loop
    for epoch in range(0, 10000):
        model.train()
        running_loss = 0.0
        
        for i, batch in enumerate(dataloader):
            # Move input data to the same device as the model
            inputs = batch['rgb_norm'].to(device)  # Move input to the device
            targets = batch['depth_filled_linear'].to(device)  # Move target to the device
            masks = torch.ones_like(targets).to(device)  # Move mask to the device
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, masks)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            current_iter += 1
            
            if current_iter % args.logging_steps == 0:
                logging.info(f"Iter: [{current_iter:6d}/{args.iterations:6d}], Loss: {loss.item():.4f}")

        # Checkpoint saving logic
        is_last = (current_iter == args.iterations)
        if (current_iter % 5000 == 0) or is_last:
            save_path = os.path.join(args.output_dir, f"checkpoint_iter_{current_iter}.pth")
            torch.save({
                'iter': current_iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
            logging.info(f"Saved checkpoint: {save_path}")
        
        # Optionally, update "checkpoint.pth" for resume
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)


if __name__ == '__main__':
    args = get_args()
    log_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir,f'{log_time}.log'), 
                encoding='utf-8'),
            logging.StreamHandler(),
        ]
    )
    train_model(args)
