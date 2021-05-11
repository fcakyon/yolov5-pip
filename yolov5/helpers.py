import logging
import pickle
import sys
from pathlib import Path

from yolov5.models.yolo import Model
from yolov5.utils.general import set_logging
from yolov5.utils.google_utils import attempt_download
from yolov5.utils.torch_utils import torch


def load_model(model_path, device=None, autoshape=True, verbose=False):
    """
    Creates a specified YOLOv5 model

    Arguments:
        model_path (str): path of the model
        config_path (str): path of the config file
        device (str): select device that model will be loaded (cpu, cuda)
        pretrained (bool): load pretrained weights into the model
        autoshape (bool): make model ready for inference
        verbose (bool): if False, yolov5 logs will be silent

    Returns:
        pytorch model

    (Adapted from yolov5.hubconf.create)
    """
    # set logging
    set_logging(verbose=verbose)

    # set device if not given
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    attempt_download(model_path)  # download if not found locally
    model = better_torch_load(model_path, map_location=torch.device(device))
    if isinstance(model, dict):
        model = model["model"]  # load model
    hub_model = Model(model.yaml).to(next(model.parameters()).device)  # create
    hub_model.load_state_dict(model.float().state_dict())  # load state_dict
    hub_model.names = model.names  # class names
    model = hub_model

    if autoshape:
        model = model.autoshape()

    return model


def better_torch_load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
    """
    Loads an object saved with torch.save from a file.

    Same interface with https://pytorch.org/docs/stable/generated/torch.load.html
    but better yolov5 model loading handling.

    Args:
        f: a file-like object (has to implement read, :methreadline, :methtell, and :methseek),
            or a string or os.PathLike object containing a file name
        map_location: a function, torch.device, string or a dict specifying how to remap storage
            locations
        pickle_module: module used for unpickling metadata and objects (has to
            match the pickle_module used to serialize file)
        pickle_load_args: (Python 3 only) optional keyword arguments passed over to
            pickle_module.load and pickle_module.Unpickler, e.g., errors=....

    Example:

    >>> better_torch_load('tensors.pt')
    # Load all tensors onto the CPU
    >>> better_torch_load('tensors.pt', map_location=torch.device('cpu'))
    # Load all tensors onto the CPU, using a function
    >>> better_torch_load('tensors.pt', map_location=lambda storage, loc: storage)
    # Load all tensors onto GPU 1
    >>> better_torch_load('tensors.pt', map_location=lambda storage, loc: storage.cuda(1))
    # Map tensors from GPU 1 to GPU 0
    >>> better_torch_load('tensors.pt', map_location={'cuda:1':'cuda:0'})
    # Load tensor from io.BytesIO object
    >>> with open('tensor.pt', 'rb') as f:
            buffer = io.BytesIO(f.read())
    >>> better_torch_load(buffer)
    # Load a module with 'ascii' encoding for unpickling
    >>> better_torch_load('module.pt', encoding='ascii')
    """
    # add yolov5 folder to system path
    here = Path(__file__).parent.absolute()
    yolov5_folder_dir = str(here)
    sys.path.insert(0, yolov5_folder_dir)
    # load torch model
    torch_model = torch.load(f, map_location=map_location)
    # remove yolov5 folder from system path
    sys.path.remove(yolov5_folder_dir)
    return torch_model


class YOLOv5:
    def __init__(self, model_path, device=None, load_on_init=True):
        self.model_path = model_path
        self.device = device
        if load_on_init:
            Path(model_path).parents[0].mkdir(parents=True, exist_ok=True)
            self.model = load_model(model_path=model_path, device=device, autoshape=True)
        else:
            self.model = None

    def load_model(self):
        """
        Load yolov5 weight.
        """
        Path(self.model_path).parents[0].mkdir(parents=True, exist_ok=True)
        self.model = load_model(model_path=self.model_path, device=self.device, autoshape=True)

    def predict(self, image_list, size=640, augment=False):
        """
        Perform yolov5 prediction using loaded model weights.

        Returns results as a yolov5.models.common.Detections object.
        """
        assert self.model is not None, "before predict, you need to call .load_model()"
        results = self.model(imgs=image_list, size=size, augment=augment)
        return results

if __name__ == "__main__":
    model_path = "yolov5/weights/yolov5s.pt"
    device = "cuda"
    model = load_model(model_path=model_path, config_path=None, device=device)

    from PIL import Image
    imgs = [Image.open(x) for x in Path("yolov5/data/images").glob("*.jpg")]
    results = model(imgs)
