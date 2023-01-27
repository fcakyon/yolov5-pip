import unittest

from tests.test_utils import TestConstants

DEVICE = "cpu"

class TestYolov5FromUltralytics(unittest.TestCase):       
    def test_load_model(self):
        import yolov5

        # init model
        model_path = TestConstants.YOLOV5S_MODEL_PATH
        model = yolov5.load(model_path, device=DEVICE)

        # check if loaded
        self.assertNotEqual(model, None)

    def test_predict(self):
        from PIL import Image

        import yolov5

        # init yolov5s model
        model_path = TestConstants.YOLOV5S_MODEL_PATH
        model = yolov5.load(model_path, device=DEVICE)

        # prepare image
        image_path = TestConstants.ZIDANE_IMAGE_PATH
        image = Image.open(image_path)

        # perform inference
        results = model(image, size=640, augment=False)

        # compare
        self.assertEqual(results.n, 1)
        self.assertEqual(len(results.names), 80)
        self.assertEqual(len(results.pred[0]), 4)
        
        # init yolov5l model
        model_path = TestConstants.YOLOV5L_MODEL_PATH
        model = yolov5.load(model_path, device=DEVICE)

        # prepare image
        image_path = TestConstants.BUS_IMAGE_PATH
        image = Image.open(image_path)
        # perform inference
        results = model(image, size=1280, augment=False)

        # compare
        self.assertEqual(results.n, 1)
        self.assertEqual(len(results.names), 80)
        self.assertEqual(len(results.pred[0]), 6)

        # init yolov5s model
        model_path = TestConstants.YOLOV5S_MODEL_PATH
        model = yolov5.load(model_path, device=DEVICE)

        # prepare images
        image_path1 = TestConstants.ZIDANE_IMAGE_PATH
        image_path2 = TestConstants.BUS_IMAGE_PATH
        image1 = Image.open(image_path1)
        image2 = Image.open(image_path2)

        # perform inference with multiple images and test augmentation
        results = model([image1, image2], size=1280, augment=True)

        # compare
        self.assertEqual(results.n, 2)
        self.assertEqual(len(results.names), 80)
        self.assertEqual(len(results.pred[0]), 4)
        self.assertEqual(len(results.pred[1]), 5)


class TestYolov5FromHuggingface(unittest.TestCase):      
    def test_load_model(self):
        import yolov5

        # init model
        model_path = TestConstants.YOLOV5S_HUB_ID
        model = yolov5.load(model_path, device=DEVICE)

        # check if loaded
        self.assertNotEqual(model, None)

    def test_predict(self):
        from PIL import Image

        import yolov5

        # init yolov5s model
        model_path = TestConstants.YOLOV5S_HUB_ID
        model = yolov5.load(model_path, device=DEVICE)

        # prepare image
        image_path = TestConstants.ZIDANE_IMAGE_PATH
        image = Image.open(image_path)

        # perform inference
        results = model(image, size=640, augment=False)

        # compare
        self.assertEqual(results.n, 1)
        self.assertEqual(len(results.names), 80)
        self.assertEqual(len(results.pred[0]), 4)

        # init yolov5s model
        model_path = TestConstants.YOLOV5S_HUB_ID
        model = yolov5.load(model_path, device=DEVICE)

        # prepare images
        image_path1 = TestConstants.ZIDANE_IMAGE_PATH
        image_path2 = TestConstants.BUS_IMAGE_PATH
        image1 = Image.open(image_path1)
        image2 = Image.open(image_path2)

        # perform inference with multiple images and test augmentation
        results = model([image1, image2], size=1280, augment=True)

        # compare
        self.assertEqual(results.n, 2)
        self.assertEqual(len(results.names), 80)
        self.assertEqual(len(results.pred[0]), 4)
        self.assertEqual(len(results.pred[1]), 5)



if __name__ == "__main__":
    unittest.main()
