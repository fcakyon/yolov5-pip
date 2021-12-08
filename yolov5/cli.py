import fire

from yolov5.detect import run as detect
from yolov5.export import run as export
from yolov5.train import run as train
from yolov5.val import run as val


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