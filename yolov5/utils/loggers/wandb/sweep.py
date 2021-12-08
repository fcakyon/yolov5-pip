import sys
from pathlib import Path

import wandb

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # YOLOv5 root directory

from yolov5.train import parse_opt, train
from yolov5.utils.callbacks import Callbacks
from yolov5.utils.general import increment_path
from yolov5.utils.torch_utils import select_device


def sweep():
    wandb.init()
    # Get hyp dict from sweep agent
    hyp_dict = vars(wandb.config).get("_items")

    # Workaround: get necessary opt args
    opt = parse_opt(known=True)
    opt.batch_size = hyp_dict.get("batch_size")
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok or opt.evolve))
    opt.epochs = hyp_dict.get("epochs")
    opt.nosave = True
    opt.data = hyp_dict.get("data")
    device = select_device(opt.device, batch_size=opt.batch_size)

    # train
    train(hyp_dict, opt, device, callbacks=Callbacks())


if __name__ == "__main__":
    sweep()
