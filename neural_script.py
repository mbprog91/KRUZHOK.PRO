from PIL import Image as PIL_Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import fastai
from fastai.vision import load_learner, Image, get_transforms
from io import BytesIO
import warnings
import argparse
parser = argparse.ArgumentParser(description='Process path')
warnings.simplefilter("ignore")

path_model = r'models/'


class ClassPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_learner(path_model)
        self.to_tensor = transforms.ToTensor()

    def predict(self, img_stream):
        return self.model.predict(self.process_image(img_stream))[0]

    def process_image(self, img_stream):
        # используем PIL, чтобы получить картинку из потока и изменить размер
        image = PIL_Image.open(img_stream).resize((256, 256))
        # переводим картинку в тензор и оборачиваем в объект Image, который использует fastai у себя внутри
        image = fastai.vision.Image(self.to_tensor(image))
        tfms = get_transforms(do_flip=True,
                              flip_vert=False,
                              max_rotate=10.0,
                              max_zoom=0.5,
                              max_lighting=0.9,
                              max_warp=0.2)
        image.apply_tfms(tfms[0], size=256)
        return image


model = ClassPredictor()


image_path = input()

class_ = model.predict(image_path)
print(class_)
