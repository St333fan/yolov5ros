#!/usr/bin/env python
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.
Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

# rospy for the subscriber
import rospy
# ROS Image message
from std_msgs.msg import String
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError

from utils.augmentations import letterbox #fÃ¼r im0 to im in dataset
import time
import numpy as np

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

bridge = CvBridge()

@torch.no_grad()
class subscriber:

    def __init__(self):
        weights = ROOT / 'yolov5s.pt'  # model.pt path(s)
        source = ROOT / 'data/images'  # file/dir/URL/glob, 0 for webcam
        self.imgsz = (640, 640)  # inference size (height, width)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img = False  # show results
        self.save_txt = False  # save results to *.txt
        self.save_conf = False  # save confidences in --save-txt labels
        self.save_crop = False  # save cropped prediction boxes
        nosave = False  # do not save images/videos
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.visualize = False  # visualize features
        update = False  # update all models
        project = ROOT / 'runs/detect'  # save results to project/name
        name = 'exp'  # save results to project/name
        exist_ok = False  # existing project/name ok, do not increment
        self.line_thickness = 3  # bounding box thickness (pixels)
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
        self.half = False  # use FP16 half-precision inference
        dnn = False  # use OpenCV DNN for ONNX inference
        source = str(source)
        self.save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        self.save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn)
        self.stride, self.names, pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

        # Half
        self.half &= (pt or jit or engine) and self.device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt or jit:
            self.model.model.half() if self.half else self.model.model.float()

        # Dataloader
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=pt)
        bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        self.model.warmup(imgsz=(1, 3, *self.imgsz), half=self.half)  # warmup
        self.dt, self.seen = [0.0, 0.0, 0.0], 0

        self.sub = rospy.Subscriber("/video_frames", Image, self.callback)

    def callback(self, data):

        t1 = time_sync()
        img = bridge.imgmsg_to_cv2(data, "8UC3") #bgr8
        # Letterbox
        im0s = img.copy()
        im = letterbox(im0s, self.imgsz, stride=self.stride, auto=True)[0]
        
        img = img[np.newaxis, :, :, :]

        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        self.dt[0] += t2 - t1

        # Inference
        self.visualize = False
        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        t3 = time_sync()
        self.dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        self.dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            self.seen += 1
            im0 = im0s.copy()
            s=''
            p = Path(p)  # to Path
            save_path = str(self.save_dir / p.name)  # im.jpg
            #txt_path = str(self.save_dir / 'labels' / p.stem) + (
            #    '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if self.save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if self.save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                        #with open(txt_path + '.txt', 'a') as f:
                        #    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if self.save_crop:
                            save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()


def main():
    check_requirements(exclude=('tensorboard', 'thop'))
    obc = subscriber()
    rospy.init_node('yolo', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == "__main__":
    main()
