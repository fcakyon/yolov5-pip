import fire

from yolov5.benchmarks import run_cli as benchmarks
from yolov5.classify.predict import run as classify_predict
from yolov5.classify.train import run_cli as classify_train
from yolov5.classify.val import run as classify_val
from yolov5.detect import run as detect
from yolov5.export import run as export
from yolov5.segment.predict import run as segment_predict
from yolov5.segment.train import run_cli as segment_train
from yolov5.segment.val import run as segment_val
from yolov5.train import run_cli as train
from yolov5.val import run as val


def app() -> None:
    """Cli app."""
    fire.Fire(
        {
            "train": train,
            "val": val,
            "detect": detect,
            "export": export,
            "benchmarks": benchmarks,
            'classify': {'train': classify_train, 'val': classify_val, 'predict': classify_predict},
            'segment': {'train': segment_train, 'val': segment_val, 'predict': segment_predict},
        }
    )
