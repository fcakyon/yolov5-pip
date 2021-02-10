import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml
from utils.general import check_file
from utils.plots import plot_study_txt
from yolov5 import test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="test.py")
    parser.add_argument(
        "--weights", nargs="+", type=str, default="yolov5s.pt", help="model.pt path(s)"
    )
    parser.add_argument(
        "--data", type=str, default="data/coco128.yaml", help="*.data path"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="size of each image batch"
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.001, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.6, help="IOU threshold for NMS"
    )
    parser.add_argument("--task", default="val", help="'val', 'test', 'study'")
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--single-cls", action="store_true", help="treat as single-class dataset"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-hybrid",
        action="store_true",
        help="save label+prediction hybrid results to *.txt",
    )
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="save a cocoapi-compatible JSON results file",
    )
    parser.add_argument("--project", default="runs/test", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.data = check_file(opt.data)  # check file
    print(opt)

    if opt.task in ["val", "test"]:  # run normally
        test(
            weights=opt.weights,
            data=opt.data,
            batch_size=opt.batch_size,
            image_size=opt.img_size,
            conf_thres=opt.conf_thres,
            iou_thres=opt.iou_thres,
            task=opt.task,
            single_cls=opt.single_cls,
            augment=opt.augment,
            verbose=opt.verbose,
            save_txt=opt.save_txt | opt.save_hybrid,
            save_hybrid=opt.save_hybrid,
            save_conf=opt.save_conf,
            save_json=opt.save_json,
        )

    elif opt.task == "study":  # run over a range of settings and save/plot
        for weights in ["yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"]:
            f = "study_%s_%s.txt" % (
                Path(opt.data).stem,
                Path(weights).stem,
            )  # filename to save to
            x_axis_size_list = list(range(320, 800, 64))  # x axis
            y = []  # y axis
            for image_size in x_axis_size_list:  # img-size
                print("\nRunning %s point %s..." % (f, image_size))
                r, _, t = test(
                    weights=opt.weights,
                    data=opt.data,
                    batch_size=opt.batch_size,
                    image_size=image_size,
                    conf_thres=opt.conf_thres,
                    iou_thres=opt.iou_thres,
                    task=opt.task,
                    save_json=opt.save_json,
                    plots=False,
                )
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt="%10.4g")  # save
        os.system("zip -r study.zip study_*.txt")
        plot_study_txt(f, x_axis_size_list)  # plot
