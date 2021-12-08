# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Logging utils
"""

import os
import warnings
from pathlib import Path
from threading import Thread

import pkg_resources as pkg
import torch
from torch.utils.tensorboard import SummaryWriter

from yolov5.utils.general import colorstr, emojis
from yolov5.utils.loggers.neptune.neptune_utils import NeptuneLogger
from yolov5.utils.loggers.wandb.wandb_utils import WandbLogger
from yolov5.utils.plots import plot_images, plot_results
from yolov5.utils.torch_utils import de_parallel

LOGGERS = ('csv', 'tb', 'wandb', 'neptune')  # text-file, TensorBoard, Weights & Biases
RANK = int(os.getenv('RANK', -1))

try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
    if pkg.parse_version(wandb.__version__) >= pkg.parse_version('0.12.2') and RANK in [0, -1]:
        wandb_login_success = wandb.login(timeout=30)
        if not wandb_login_success:
            wandb = None
except (ImportError, AssertionError):
    wandb = None

try:
    import neptune.new as neptune
except ImportError:
    neptune = None

class Loggers():
    # YOLOv5 Loggers class
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS, mmdet_keys=False, class_names=None):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.logger = logger  # for printing results to console
        self.include = include
        if not mmdet_keys:
            self.keys = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                        'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',  # metrics
                        'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                        'x/lr0', 'x/lr1', 'x/lr2']  # params
        else:
            self.keys = ['train/loss_bbox', 'train/loss_obj', 'train/loss_cls',  # train loss
                        'val/precision', 'val/recall', 'val/bbox_mAP_50', 'val/bbox_mAP',
                        'val/loss_bbox', 'val/loss_obj', 'val/loss_cls',  # val loss
                        'learning_rate_0', 'learning_rate_1', 'learning_rate_2']
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv
        self.class_names = class_names
        if not mmdet_keys:
            self.class_name_keys = ['metrics/' + name + '_mAP_50' for name in class_names]
        else:
            self.class_name_keys = ['val/' + name + '_mAP_50' for name in class_names]
        self.s3_weight_folder = None if not opt.s3_upload_dir else "s3://" + str(Path(opt.s3_upload_dir.replace("s3://","")) / save_dir.name / "weights").replace(os.sep, '/')
        
        # Message
        if not wandb:
            prefix = colorstr('Weights & Biases: ')
            s = f"{prefix}run 'pip install wandb' to automatically track and visualize YOLOv5 🚀 runs"
            print(emojis(s))

        if not neptune:
            prefix = colorstr('Neptune AI: ')
            s = f"{prefix}run 'pip install neptune-client' to automatically track and visualize YOLOv5 🚀 runs"
            print(emojis(s))

        # TensorBoard
        s = self.save_dir
        if 'tb' in self.include and not self.opt.evolve:
            prefix = colorstr('TensorBoard: ')
            self.logger.info(f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/")
            self.tb = SummaryWriter(str(s))

        # W&B
        if wandb and 'wandb' in self.include:
            wandb_artifact_resume = isinstance(self.opt.resume, str) and self.opt.resume.startswith('wandb-artifact://')
            run_id = torch.load(self.weights).get('wandb_id') if self.opt.resume and not wandb_artifact_resume else None
            self.opt.hyp = self.hyp  # add hyperparameters
            self.wandb = WandbLogger(self.opt, run_id)
        else:
            self.wandb = None

        # Neptune
        if neptune and 'neptune' in self.include:
            self.neptune = NeptuneLogger(self.opt)
        else:
            self.neptune = None

    def on_pretrain_routine_end(self):
        # Callback runs on pre-train routine end
        paths = self.save_dir.glob('*labels*.jpg')  # training labels
        if self.wandb:
            self.wandb.log({"Labels": [wandb.Image(str(x), caption=x.name) for x in paths]})

    def on_train_batch_end(self, ni, model, imgs, targets, paths, plots, sync_bn):
        # Callback runs on train batch end
        if plots:
            if ni == 0:
                if not sync_bn:  # tb.add_graph() --sync known issue https://github.com/ultralytics/yolov5/issues/3754
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')  # suppress jit trace warning
                        self.tb.add_graph(torch.jit.trace(de_parallel(model), imgs[0:1], strict=False), [])
            if ni < 3:
                f = self.save_dir / f'train_batch{ni}.jpg'  # filename
                Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
            if self.wandb and ni == 10:
                files = sorted(self.save_dir.glob('train*.jpg'))
                self.wandb.log({'Mosaics': [wandb.Image(str(f), caption=f.name) for f in files if f.exists()]})

    def on_train_epoch_end(self, epoch):
        # Callback runs on train epoch end
        if self.wandb:
            self.wandb.current_epoch = epoch + 1
        if self.neptune and self.neptune.neptune_run:
            self.neptune.current_epoch = epoch + 1

    def on_val_image_end(self, pred, predn, path, names, im):
        # Callback runs on val image end
        if self.wandb:
            self.wandb.val_one_image(pred, predn, path, names, im)

    def on_val_end(self):
        # Callback runs on val end
        if self.wandb:
            files = sorted(self.save_dir.glob('val*.jpg'))
            self.wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in files]})

    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi):
        # Callback runs at the end of each fit (train+val) epoch
        x = {k: v for k, v in zip(self.keys + self.class_name_keys, vals)}  # dict
        if self.csv:
            file = self.save_dir / 'results.csv'
            n = len(x) + 1  # number of cols
            s = '' if file.exists() else (('%20s,' * n % tuple(['epoch'] + self.keys + self.class_name_keys)).rstrip(',') + '\n')  # add header
            with open(file, 'a') as f:
                f.write(s + ('%20.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')

        if self.tb:
            for k, v in x.items():
                self.tb.add_scalar(k, v, epoch)

        if self.wandb:
            self.wandb.log(x)
            self.wandb.end_epoch(best_result=best_fitness == fi)

        if self.neptune and self.neptune.neptune_run:
            self.neptune.log(x)
            self.neptune.end_epoch()

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        # Callback runs on model save event
        if self.wandb:
            if ((epoch + 1) % self.opt.save_period == 0 and not final_epoch) and self.opt.save_period != -1:
                self.wandb.log_model(last.parent, self.opt, epoch, fi, best_model=best_fitness == fi)
        if self.neptune and self.neptune.neptune_run and self.s3_weight_folder is not None:
            if not final_epoch and best_fitness == fi:
                self.neptune.neptune_run["weights"].track_files(self.s3_weight_folder)

    def on_train_end(self, last, best, plots, epoch, results):
        # Callback runs on training end
        if plots:
            plot_results(file=self.save_dir / 'results.csv')  # save results.png
        files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')], "results.html"]
        files = [(self.save_dir / f) for f in files if (self.save_dir / f).exists()]  # filter

        if self.tb:
            import cv2
            for f in files:
                if f.suffix != ".html":
                    self.tb.add_image(f.stem, cv2.imread(str(f))[..., ::-1], epoch, dataformats='HWC')

        if self.wandb:
            results = []
            for f in files:
                if f.suffix == ".html":
                    results.append(wandb.Html(str(f)))
                else:
                    results.append(wandb.Image(str(f), caption=f.name))
                    
            self.wandb.log({"Results": results})
            # Calling wandb.log. TODO: Refactor this into WandbLogger.log_model
            if not self.opt.evolve:
                wandb.log_artifact(str(best if best.exists() else last), type='model',
                                   name='run_' + self.wandb.wandb_run.id + '_model',
                                   aliases=['latest', 'best', 'stripped'])
                self.wandb.finish_run()
            else:
                self.wandb.finish_run()
                self.wandb = WandbLogger(self.opt)

        if self.neptune and self.neptune.neptune_run:
            for f in files:
                if f.suffix == ".html":
                    self.neptune.neptune_run['Results/{}'.format(f)].upload(neptune.types.File(str(f)))
                else:
                    self.neptune.neptune_run['Results/{}'.format(f)].log(neptune.types.File(str(f)))

            if self.s3_weight_folder is not None:
                self.neptune.neptune_run["weights"].track_files(self.s3_weight_folder)

            self.neptune.finish_run()
