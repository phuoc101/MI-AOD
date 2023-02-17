import os
import argparse
import time
import torch
from mmdet.apis.inference import init_detector, extract_features


def parse_args():
    parser = argparse.ArgumentParser(description="Inference on single images")
    parser.add_argument("--config_file", help="train config file path")
    parser.add_argument("--ckpt_file", help="model checkpoint file path")
    parser.add_argument("--img_file1", help="image file path")
    parser.add_argument("--img_file2", help="image file path")
    parser.add_argument("--out_file", help="output image file path")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    # cfg = Config.fromfile(args.config_file)
    model = init_detector(args.config_file, args.ckpt_file, device="cuda:0")
    start = time.perf_counter()
    result1 = extract_features(model, args.img_file1)
    result2 = extract_features(model, args.img_file2)
    feat1 = torch.flatten(result1[4])
    feat2 = torch.flatten(result2[4])
    print(f"MSE: {(torch.pow(feat1 - feat2, 2)).mean()}")
    end = time.perf_counter()
    elapsed = end - start
    print(f"inference time: {elapsed*1000} ms")
    print(f"fps: {1/elapsed}")


if __name__ == "__main__":
    main()
