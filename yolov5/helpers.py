from pathlib import Path
from yolov5.models.common import AutoShape, DetectMultiBackend
from yolov5.utils.general import LOGGER, logging
from yolov5.utils.torch_utils import torch

def load_model(model_path, device=None, autoshape=True, verbose=False):
    """
    Creates a specified YOLOv5 model

    Arguments:
        model_path (str): path of the model
        device (str): select device that model will be loaded (cpu, cuda)
        pretrained (bool): load pretrained weights into the model
        autoshape (bool): make model ready for inference
        verbose (bool): if False, yolov5 logs will be silent

    Returns:
        pytorch model

    (Adapted from yolov5.hubconf.create)
    """
    # set logging
    if not verbose:
        LOGGER.setLevel(logging.WARNING)

    # set device if not given
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif type(device) is str:
        device = torch.device(device)

    model = DetectMultiBackend(model_path, device=device)

    if autoshape:
        model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS
    return model.to(device)


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

    def video_predict(self, video_path, view_img=True, size=640, augment=False, line_thickness=3):
        from yolov5.utils.general import non_max_suppression
        from utils.plots import Annotator, colors
        import cv2

        cap = cv2.VideoCapture(video_path)         
        output = []
        while True:
            _, im = cap.read()
            im0s = im.copy()
            im = torch.from_numpy(im).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
            pred = self.model(im, size=size, augment=augment) 
            pred = non_max_suppression(pred, conf_thres=self.model.conf, iou_thres=self.model.iou, classes=self.model.classes, agnostic=self.model.agnostic)
            for _, det in enumerate(pred):
                for *xyxy, conf, cls in reversed(det):
                    annotator = Annotator(im0s, line_width=line_thickness, example=str(self.model.names))
                    output.append({"bbox": xyxy, "conf": conf, "cls": self.model.names[int(cls)]})
                    if view_img:  # Add bbox to image
                        label = "%s %.2f" % (self.model.names[int(cls)], conf)
                        annotator.box_label(xyxy, label, color=colors(int(cls), True))
                        
            # Stream results
            if view_img:
                im0 = annotator.result()
                cv2.imshow(str(Path(path).stem), im0)
                if cv2.waitKey(1) == ord("q"):
                    break
        cv2.destroyAllWindows()
        return output
        
if __name__ == "__main__":
    IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
    VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
    
    path = "yolov5/data/images/zidane.jpg"
    model_path = "yolov5/weights/yolov5s.pt"
    source = Path(path).suffix[1:]
    
    if source in IMG_FORMATS:
        from PIL import Image
        
        model = load_model(model_path=model_path, device="cpu")
        img = Image.open(path)
        result = model(img)

    elif source in VID_FORMATS:                    
        model = YOLOv5(model_path=model_path, device="cpu")
        result = model.video_predict(video_path=path, view_img=True)
        