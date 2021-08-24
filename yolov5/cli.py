import fire

from yolov5.train import run as train
from yolov5.val import main as val
from yolov5.detect import main as detect
from yolov5.export import main as export


def app() -> None:
    """Cli app."""
    fire.Fire(
        {
            "train": train,
            "val": val,
            "detect": detect,
            "export": export,
        }
    )