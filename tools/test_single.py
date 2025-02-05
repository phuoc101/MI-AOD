from mmdet.apis.inference import init_detector, inference_detector, show_result_pyplot
from mmcv import Config
import os
import argparse
import numpy as np
import time
import torch
import random
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Inference on single images")
    parser.add_argument("--config_file", help="train config file path")
    parser.add_argument("--ckpt_file", help="model checkpoint file path")
    parser.add_argument("--img_file", help="image file path")
    parser.add_argument("--out_im", help="output image file path")
    parser.add_argument("--out_txt", help="output label txt file path")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--conf_thres", type=float, default=0.25)
    parser.add_argument("--iomin_thres", type=float, default=0.5)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def plot_one_box(x, img, color=None, label=None, line_thickness=1):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def nms(bbox_results, im_size, conf_thres=0.3, iomin_thres=0.5):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    if not bbox_results:
        return []
    bboxes = []
    for i, bb in enumerate(bbox_results):
        bboxes.append(np.hstack([bb, (np.ones((bb.shape[0], 1)) * i)]))
    boxes = np.vstack(bboxes)
    # Sort by confidence in descending order
    boxes = boxes[boxes[:, 4].argsort()[::-1]]
    boxes = boxes[boxes[..., 4] >= conf_thres]  # Candidates
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # Compute the area of the bounding boxes
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # add 1, because the pixel at the start as well as at the end counts
    indices = np.arange(boxes.shape[0])  # indices of all boxes
    for i, box in enumerate(boxes):
        # indices of other boxes
        tmp_ids = indices[indices != i]
        # Find out the coordinates of the intersection box
        xx1 = np.maximum(box[0], boxes[tmp_ids, 0])
        yy1 = np.maximum(box[1], boxes[tmp_ids, 1])
        xx2 = np.minimum(box[2], boxes[tmp_ids, 2])
        yy2 = np.minimum(box[3], boxes[tmp_ids, 3])
        # Find out the width and the height of the intersection box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # Overlapping ratio
        overlap = (w * h) / areas[tmp_ids] > iomin_thres
        overlapping_boxes = tmp_ids[overlap]
        for ob in overlapping_boxes:
            if boxes[ob, 4] < box[4]:
                indices = indices[indices != ob]
    boxes = boxes[indices]
    return boxes


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config_file)
    im_size = cfg.image_scaling
    model = init_detector(args.config_file, args.ckpt_file, device="cuda:0")
    start = time.perf_counter()
    bbox_results, uncertainty = inference_detector(model, args.img_file)
    bbox_results = nms(
        bbox_results=bbox_results, conf_thres=args.conf_thres, iomin_thres=args.iomin_thres, im_size=im_size
    )
    end = time.perf_counter()
    elapsed = end - start
    gn = torch.tensor(im_size)[[1, 0, 1, 0]]
    im0 = cv2.imread(args.img_file)
    classes = model.CLASSES
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in classes]
    # Write results
    for *xyxy, conf, cls in reversed(bbox_results):
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf)
        print(line)
        with open(args.out_txt, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")
        label = f"{classes[int(cls)]} {conf:.2f}"
        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
    print(f"inference time: {elapsed*1000} ms")
    print(f"fps: {1/elapsed}")
    print("Image uncertainty is: " + str(uncertainty.cpu().numpy()))
    cv2.imwrite(args.out_im, im0)


if __name__ == "__main__":
    main()
