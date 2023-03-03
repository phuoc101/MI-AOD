import os
import argparse
import time
import torch
from mmdet.apis.inference import init_detector, extract_features
import numpy as np
from tqdm import trange

from kmeans_pytorch import kmeans

DATA_ROOT = os.environ["DATASETS_CV_DIR"]
PICAM_ROOT = os.path.join(DATA_ROOT, "picam_data/HevoNaTiera")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference on single images")
    parser.add_argument("--config_file", help="train config file path")
    parser.add_argument("--ckpt_file", default=None, help="model checkpoint file path")
    parser.add_argument("--img_file", help="image files path")
    parser.add_argument("--out_file", help="output image file path")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--num_to_cluster", type=int, default=None)
    parser.add_argument("--num_clusters", type=int, default=100)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main():

    args = parse_args()
    # cfg = Config.fromfile(args.config_file)
    model = init_detector(args.config_file, args.ckpt_file, device="cuda:0")
    start = time.perf_counter()
    imgs = []
    with open(args.img_file, "r") as f:
        lines_from_file = f.readlines()
        if args.num_to_cluster is None:
            args.num_to_cluster = len(lines_from_file)
        lines = [line.replace("\n", "") for line in lines_from_file[-args.num_to_cluster :]]
        for im in lines:
            imgs.append(os.path.join(PICAM_ROOT, "obj_train_data", im + ".png"))
    imgs = np.array(imgs)
    features = None
    for im in trange(len(imgs), desc="Extracting features"):
        # extracted = torch.flatten(extract_features(model, imgs[im])[4])
        extracted_raw = extract_features(model, imgs[im])
        extracted = torch.flatten(
            torch.cat(
                (
                    torch.mean(extracted_raw[0], axis=[2, 3]),
                    torch.mean(extracted_raw[1], axis=[2, 3]),
                    torch.mean(extracted_raw[2], axis=[2, 3]),
                    torch.mean(extracted_raw[3], axis=[2, 3]),
                    torch.mean(extracted_raw[4], axis=[2, 3]),
                ),
                dim=1,
            )
        )
        if features is None:
            features = torch.zeros([len(imgs), extracted.shape[0]])
            features[0, :] = extracted
        else:
            features[im, :] = extracted
    end = time.perf_counter()
    elapsed = end - start
    print(f"inference time: {elapsed*1000} ms")
    cluster_idx, cluster_means = kmeans(
        X=features, num_clusters=args.num_clusters, distance="cosine", device=torch.device("cuda:0")
    )
    selected = dict()
    for i in trange(args.num_to_cluster, desc="Selecting images"):
        cluster = cluster_idx[i].item()
        dist2mean = np.linalg.norm(cluster_means[cluster].cpu().numpy() - features[i].cpu().numpy())
        if cluster not in selected.keys():
            selected[cluster] = {"img": os.path.basename(imgs[i]).replace(".png", ""), "dist2mean": dist2mean}
        else:
            if dist2mean > selected[cluster]["dist2mean"]:
                selected[cluster] = {
                    "img": os.path.basename(imgs[i]).replace(".png", ""),
                    "dist2mean": dist2mean,
                }
    print(selected)
    with open("to_label.txt", "w+") as f:
        for im in selected.values():
            f.write(im["img"] + "\n")


if __name__ == "__main__":
    main()
