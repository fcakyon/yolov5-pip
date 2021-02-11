# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import unittest

import numpy as np
from PIL import Image
from yolov5 import YOLOv5

from tests.test_utils import TestConstants


class TestYolov5(unittest.TestCase):
    def test_load_model(self):
        # init model
        model_path = TestConstants.YOLOV5S_MODEL_PATH
        device = "cpu"
        yolov5 = YOLOv5(model_path, device, load_on_init=False)
        yolov5.load_model()

        # check if loaded
        self.assertNotEqual(yolov5.model, None)

    def test_load_model_on_init(self):
        # init model
        model_path = TestConstants.YOLOV5S_MODEL_PATH
        device = "cpu"
        yolov5 = YOLOv5(model_path, device, load_on_init=True)

        # check if loaded
        self.assertNotEqual(yolov5.model, None)

    def test_predict(self):

        # init model
        model_path = TestConstants.YOLOV5S_MODEL_PATH
        device = "cpu"
        yolov5 = YOLOv5(model_path, device, load_on_init=True)

        # prepare image
        image_path = TestConstants.ZIDANE_IMAGE_PATH
        image = Image.open(image_path)

        # perform inference
        results = yolov5.predict(image, augment=False)

        # compare
        self.assertEqual(box[:4].astype("int").tolist(), [336, 123, 346, 139])
        self.assertEqual(len(boxes), 80)
        self.assertEqual(len(masks), 80)


if __name__ == "__main__":
    unittest.main()
