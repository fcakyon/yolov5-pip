import os
from pathlib import Path

import yaml
from yolov5 import __version__
from yolov5.utils.general import colorstr

try:
    import neptune.new as neptune
except ImportError:
    neptune = None


class NeptuneLogger:
    def __init__(self, opt, job_type='Training'):
        # Pre-training routine --
        self.job_type = job_type
        with open(opt.data) as f:
            data_dict = yaml.safe_load(f)  # data dict
        self.neptune, self.neptune_run = neptune, None

        if self.neptune and opt.neptune_token:
            self.neptune_run = neptune.init(api_token=opt.neptune_token,
                                            project=opt.neptune_project,
                                            name=Path(opt.save_dir).stem)
        if self.neptune_run:
            if self.job_type == 'Training':
                if not opt.resume:
                    self.neptune_run["opt"] = vars(opt)
                    self.track_dataset(opt)
                self.data_dict = self.setup_training(data_dict)
            # log yolov5 package version
            self.neptune_run["yolov5-version"].log(__version__)
            
            prefix = colorstr('neptune: ')
            print(f"{prefix}NeptuneAI logging initiated successfully.")

    def track_dataset(self, opt):
        has_yolo_s3_data_dir = False
        with open(opt.data, errors='ignore') as f:
            data_dict = yaml.safe_load(f)  # load data dict
            if data_dict.get("yolo_s3_data_dir") is not None:
                has_yolo_s3_data_dir = True 

        if has_yolo_s3_data_dir:
            yolo_s3_data_dir = data_dict["yolo_s3_data_dir"]
        elif opt.upload_dataset and opt.s3_upload_dir:
            yolo_s3_data_dir = "s3://" + str(Path(opt.s3_upload_dir.replace("s3://","")) / Path(opt.save_dir).name / 'data').replace(os.sep, '/')
        else:
            yolo_s3_data_dir = None
        if yolo_s3_data_dir is not None:
            self.neptune_run["data"].track_files(yolo_s3_data_dir)


    def setup_training(self, data_dict):
        self.log_dict, self.current_epoch = {}, 0  # Logging Constants
        return data_dict

    def log(self, log_dict):
        if self.neptune_run:
            for key, value in log_dict.items():
                self.log_dict[key] = value

    def end_epoch(self, best_result=False):
        if self.neptune_run:
            for key, value in self.log_dict.items():
                self.neptune_run[key].log(value)
                self.log_dict = {}

    def finish_run(self):
        if self.neptune_run:
            if self.log_dict:
                for key, value in self.log_dict.items():
                    self.neptune_run[key].log(value)
            self.neptune_run.stop()
