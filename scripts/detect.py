import argparse

import torch
from utils.general import strip_optimizer
from yolov5 import detect

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", nargs="+", type=str, default="yolov5s.pt", help="model.pt path(s)"
    )
    parser.add_argument(
        "--source", type=str, default="data/images", help="source"
    )  # file/folder, 0 for webcam
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --class 0, or --class 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument(
        "--project", default="runs/detect", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ["yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"]:
                detect(
                    weights=opt.weights,
                    source=opt.source,
                    img_size=opt.img_size,
                    conf_thres=opt.conf_thres,
                    iou_thres=opt.iou_thres,
                    device=opt.device,
                    view_img=opt.view_img,
                    save_txt=opt.save_txt,
                    save_conf=opt.save_conf,
                    classes=opt.classes,
                    agnostic_nms=opt.agnostic_nms,
                    augment=opt.augment,
                    update=opt.update,
                    project=opt.project,
                    name=opt.name,
                    exist_ok=opt.exist_ok,
                )
                strip_optimizer(opt.weights)
        else:
            detect(
                weights=opt.weights,
                source=opt.source,
                img_size=opt.img_size,
                conf_thres=opt.conf_thres,
                iou_thres=opt.iou_thres,
                device=opt.device,
                view_img=opt.view_img,
                save_txt=opt.save_txt,
                save_conf=opt.save_conf,
                classes=opt.classes,
                agnostic_nms=opt.agnostic_nms,
                augment=opt.augment,
                update=opt.update,
                project=opt.project,
                name=opt.name,
                exist_ok=opt.exist_ok,
            )
