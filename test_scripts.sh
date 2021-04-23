conda activate yolov5
pip install -e .
di=cpu # inference devices  # define device
# train
python yolov5/train.py --img 128 --batch 16 --weights yolov5/weights/yolov5s.pt --cfg models/yolov5s.yaml --epochs 1 --device $di
yolo_train --img 128 --batch 16 --weights yolov5/weights/yolov5s.pt --cfg models/yolov5s.yaml --epochs 1 --device $di
# detect
python yolov5/detect.py --weights yolov5/weights/yolov5s.pt --device $di
yolo_detect --weights yolov5/weights/yolov5s.pt --device $di
python yolov5/detect.py --weights runs/train/exp/weights/last.pt --device $di
yolo_detect --weights runs/train/exp/weights/last.pt --device $di
# test
python yolov5/test.py --img 128 --batch 16 --weights yolov5/weights/yolov5s.pt --device $di
yolo_test --img 128 --batch 16 --weights yolov5/weights/yolov5s.pt --device $di
python yolov5/test.py --img 128 --batch 16 --weights runs/train/exp/weights/last.pt --device $di
yolo_test --img 128 --batch 16 --weights runs/train/exp/weights/last.pt --device $di
# export
python yolov5/models/export.py --weights yolov5/weights/yolov5s.pt --device $di
yolo_export --weights yolov5/weights/yolov5s.pt --device $di