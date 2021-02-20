import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from yolov5 import test  # import test.py to get mAP after each epoch
from yolov5.models.experimental import attempt_load
from yolov5.models.yolo import Model
from yolov5.utils.autoanchor import check_anchors
from yolov5.utils.datasets import create_dataloader
from yolov5.utils.general import (check_dataset, check_img_size, fitness,
                                  init_seeds, labels_to_class_weights,
                                  labels_to_image_weights, one_cycle,
                                  strip_optimizer)
from yolov5.utils.google_utils import attempt_download
from yolov5.utils.loss import compute_loss
from yolov5.utils.plots import (plot_evolution, plot_images, plot_labels,
                                plot_results)
from yolov5.utils.torch_utils import (ModelEMA, intersect_dicts, select_device,
                                      torch_distributed_zero_first)

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None
    logger.info(
        "Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)"
    )


def train(hyp, opt, device, tb_writer=None, wandb=None):
    logger.info(f"Hyperparameters {hyp}")
    save_dir, epochs, batch_size, total_batch_size, weights, rank = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.total_batch_size,
        opt.weights,
        opt.global_rank,
    )

    # Directories
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / "last.pt"
    best = wdir / "best.pt"
    results_file = save_dir / "results.txt"

    # Save run settings
    with open(save_dir / "hyp.yaml", "w") as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / "opt.yaml", "w") as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not opt.evolve  # create plots
    cuda = device.type != "cpu"
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict["train"]
    test_path = data_dict["val"]
    nc = 1 if opt.single_cls else int(data_dict["nc"])  # number of classes
    names = (
        ["item"]
        if opt.single_cls and len(data_dict["names"]) != 1
        else data_dict["names"]
    )  # class names
    assert len(names) == nc, "%g names found for nc=%g dataset in %s" % (
        len(names),
        nc,
        opt.data,
    )  # check

    # Model
    pretrained = weights.endswith(".pt")
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        # add yolov5 folder to system path
        here = Path(__file__).parent.absolute()
        yolov5_folder_dir = str(here)
        sys.path.insert(0, yolov5_folder_dir)
        # load checkpoint
        ckpt = torch.load(weights, map_location=device)
        # remove yolov5 folder from system path
        sys.path.remove(yolov5_folder_dir)
        if hyp.get("anchors"):
            ckpt["model"].yaml["anchors"] = round(hyp["anchors"])  # force autoanchor
        model = Model(opt.cfg or ckpt["model"].yaml, ch=3, nc=nc).to(device)  # create
        exclude = ["anchor"] if opt.cfg or hyp.get("anchors") else []  # exclude keys
        state_dict = ckpt["model"].float().state_dict()  # to FP32
        state_dict = intersect_dicts(
            state_dict, model.state_dict(), exclude=exclude
        )  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info(
            "Transferred %g/%g items from %s"
            % (len(state_dict), len(model.state_dict()), weights)
        )  # report
    else:
        model = Model(opt.cfg, ch=3, nc=nc).to(device)  # create

    # Freeze
    freeze = []  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print("freezing %s" % k)
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(
        round(nbs / total_batch_size), 1
    )  # accumulate loss before optimizing
    hyp["weight_decay"] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if opt.adam:
        optimizer = optim.Adam(
            pg0, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999)
        )  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(
            pg0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True
        )

    optimizer.add_param_group(
        {"params": pg1, "weight_decay": hyp["weight_decay"]}
    )  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
    logger.info(
        "Optimizer groups: %g .bias, %g conv.weight, %g other"
        % (len(pg2), len(pg1), len(pg0))
    )
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Logging
    if rank in [-1, 0] and wandb and wandb.run is None:
        opt.hyp = hyp  # add hyperparameters
        wandb_run = wandb.init(
            config=opt,
            resume="allow",
            project="YOLOv5" if opt.project == "runs/train" else Path(opt.project).stem,
            name=save_dir.stem,
            id=ckpt.get("wandb_id") if "ckpt" in locals() else None,
        )
    loggers = {"wandb": wandb}  # loggers dict

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            best_fitness = ckpt["best_fitness"]

        # Results
        if ckpt.get("training_results") is not None:
            with open(results_file, "w") as file:
                file.write(ckpt["training_results"])  # write results.txt

        # Epochs
        start_epoch = ckpt["epoch"] + 1
        if opt.resume:
            assert (
                start_epoch > 0
            ), "%s training to %g epochs is finished, nothing to resume." % (
                weights,
                epochs,
            )
        if epochs < start_epoch:
            logger.info(
                "%s has been trained for %g epochs. Fine-tuning for %g additional epochs."
                % (weights, ckpt["epoch"], epochs)
            )
            epochs += ckpt["epoch"]  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = int(model.stride.max())  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [
        check_img_size(x, gs) for x in opt.img_size
    ]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info("Using SyncBatchNorm()")

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    # Trainloader
    dataloader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size,
        gs,
        opt,
        hyp=hyp,
        augment=True,
        cache=opt.cache_images,
        rect=opt.rect,
        rank=rank,
        world_size=opt.world_size,
        workers=opt.workers,
        image_weights=opt.image_weights,
        quad=opt.quad,
    )
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert (
        mlc < nc
    ), "Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g" % (
        mlc,
        nc,
        opt.data,
        nc - 1,
    )

    # Process 0
    if rank in [-1, 0]:
        ema.updates = start_epoch * nb // accumulate  # set EMA updates
        testloader = create_dataloader(
            test_path,
            imgsz_test,
            total_batch_size,
            gs,
            opt,  # testloader
            hyp=hyp,
            cache=opt.cache_images and not opt.notest,
            rect=True,
            rank=-1,
            world_size=opt.world_size,
            workers=opt.workers,
            pad=0.5,
        )[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram("classes", c, 0)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)

    # Model parameters
    hyp["cls"] *= nc / 80.0  # scale hyp['cls'] to class count
    hyp["obj"] *= (
        imgsz ** 2 / 640.0 ** 2 * 3.0 / nl
    )  # scale hyp['obj'] to image size and output layers
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = (
        labels_to_class_weights(dataset.labels, nc).to(device) * nc
    )  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(
        round(hyp["warmup_epochs"] * nb), 1000
    )  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    logger.info(
        "Image sizes %g train, %g test\n"
        "Using %g dataloader workers\nLogging results to %s\n"
        "Starting training for %g epochs..."
        % (imgsz, imgsz_test, dataloader.num_workers, save_dir, epochs)
    )
    for epoch in range(
        start_epoch, epochs
    ):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = (
                    model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc
                )  # class weights
                iw = labels_to_image_weights(
                    dataset.labels, nc=nc, class_weights=cw
                )  # image weights
                dataset.indices = random.choices(
                    range(dataset.n), weights=iw, k=dataset.n
                )  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (
                    torch.tensor(dataset.indices)
                    if rank == 0
                    else torch.zeros(dataset.n)
                ).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(
            ("\n" + "%10s" * 8)
            % ("Epoch", "gpu_mem", "box", "obj", "cls", "total", "targets", "img_size")
        )
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (
            imgs,
            targets,
            paths,
            _,
        ) in (
            pbar
        ):  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = (
                imgs.to(device, non_blocking=True).float() / 255.0
            )  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(
                    1, np.interp(ni, xi, [1, nbs / total_batch_size]).round()
                )
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(
                        ni,
                        xi,
                        [
                            hyp["warmup_bias_lr"] if j == 2 else 0.0,
                            x["initial_lr"] * lf(epoch),
                        ],
                    )
                    if "momentum" in x:
                        x["momentum"] = np.interp(
                            ni, xi, [hyp["warmup_momentum"], hyp["momentum"]]
                        )

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [
                        math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]
                    ]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(
                        imgs, size=ns, mode="bilinear", align_corners=False
                    )

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(
                    pred, targets.to(device), model
                )  # loss scaled by batch_size
                if rank != -1:
                    loss *= (
                        opt.world_size
                    )  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.0

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = "%.3gG" % (
                    torch.cuda.memory_reserved() / 1e9
                    if torch.cuda.is_available()
                    else 0
                )  # (GB)
                s = ("%10s" * 2 + "%10.4g" * 6) % (
                    "%g/%g" % (epoch, epochs - 1),
                    mem,
                    *mloss,
                    targets.shape[0],
                    imgs.shape[-1],
                )
                pbar.set_description(s)

                # Plot
                if plots and ni < 3:
                    f = save_dir / f"train_batch{ni}.jpg"  # filename
                    Thread(
                        target=plot_images, args=(imgs, targets, paths, f), daemon=True
                    ).start()
                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(model, imgs)  # add model to tensorboard
                elif plots and ni == 3 and wandb:
                    wandb.log(
                        {
                            "Mosaics": [
                                wandb.Image(str(x), caption=x.name)
                                for x in save_dir.glob("train*.jpg")
                            ]
                        }
                    )

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x["lr"] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            if ema:
                ema.update_attr(
                    model,
                    include=[
                        "yaml",
                        "nc",
                        "hyp",
                        "gr",
                        "names",
                        "stride",
                        "class_weights",
                    ],
                )
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                results, maps, times = test(
                    data=opt.data,
                    batch_size=total_batch_size,
                    image_size=imgsz_test,
                    model=ema.ema,
                    single_cls=opt.single_cls,
                    dataloader=testloader,
                    save_dir=save_dir,
                    plots=plots and final_epoch,
                    log_imgs=opt.log_imgs if wandb else 0,
                )

            # Write
            with open(results_file, "a") as f:
                f.write(
                    s + "%10.4g" * 7 % results + "\n"
                )  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
            if len(opt.name) and opt.bucket:
                os.system(
                    "gsutil cp %s gs://%s/results/results%s.txt"
                    % (results_file, opt.bucket, opt.name)
                )

            # Log
            tags = [
                "train/box_loss",
                "train/obj_loss",
                "train/cls_loss",  # train loss
                "metrics/precision",
                "metrics/recall",
                "metrics/mAP_0.5",
                "metrics/mAP_0.5:0.95",
                "val/box_loss",
                "val/obj_loss",
                "val/cls_loss",  # val loss
                "x/lr0",
                "x/lr1",
                "x/lr2",
            ]  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb:
                    wandb.log({tag: x})  # W&B

            # Update best mAP
            fi = fitness(
                np.array(results).reshape(1, -1)
            )  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi

            # Save model
            save = (not opt.nosave) or (final_epoch and not opt.evolve)
            if save:
                with open(results_file, "r") as f:  # create checkpoint
                    ckpt = {
                        "epoch": epoch,
                        "best_fitness": best_fitness,
                        "training_results": f.read(),
                        "model": ema.ema,
                        "optimizer": None if final_epoch else optimizer.state_dict(),
                        "wandb_id": wandb_run.id if wandb else None,
                    }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    if rank in [-1, 0]:
        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in [last, best]:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if opt.bucket:
            os.system(f"gsutil cp {final} gs://{opt.bucket}/weights")  # upload

        # Plots
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb:
                files = [
                    "results.png",
                    "precision_recall_curve.png",
                    "confusion_matrix.png",
                ]
                wandb.log(
                    {
                        "Results": [
                            wandb.Image(str(save_dir / f), caption=f)
                            for f in files
                            if (save_dir / f).exists()
                        ]
                    }
                )
                if opt.log_artifacts:
                    wandb.log_artifact(
                        artifact_or_path=str(final), type="model", name=save_dir.stem
                    )

        # Test best.pt
        logger.info(
            "%g epochs completed in %.3f hours.\n"
            % (epoch - start_epoch + 1, (time.time() - t0) / 3600)
        )
        if opt.data.endswith("coco.yaml") and nc == 80:  # if COCO
            for conf, iou, save_json in (
                [0.25, 0.45, False],
                [0.001, 0.65, True],
            ):  # speed, mAP tests
                results, _, _ = test(
                    data=opt.data,
                    batch_size=total_batch_size,
                    image_size=imgsz_test,
                    conf_thres=conf,
                    iou_thres=iou,
                    model=attempt_load(final, device).half(),
                    single_cls=opt.single_cls,
                    dataloader=testloader,
                    save_dir=save_dir,
                    save_json=save_json,
                    plots=False,
                )

    else:
        dist.destroy_process_group()

    wandb.run.finish() if wandb and wandb.run else None
    torch.cuda.empty_cache()
    return results
