import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from .model import Net
import os


cwd = os.path.abspath(os.getcwd())
DEFAULT_DEVICE = 'cpu'
DEFAULT_WEIGHTS = os.path.join(cwd, 'deep_sort/deep/weights/original_ckpt.t7')

class Extractor(object):
    def __init__(self, weights=DEFAULT_WEIGHTS, device=DEFAULT_DEVICE):
        self.net = Net(reid=True)
        self.device = device
        state_dict = torch.load(weights, map_location=lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, imgs):
        def _resize(img, size):
            return cv2.resize(img.astype(np.float32)/255., size)

        img_batch = torch.cat([self.norm(_resize(img, self.size)).unsqueeze(0) for img in imgs], dim=0).float()
        return img_batch

    def __call__(self, imgs):
        img_batch = self._preprocess(imgs)
        with torch.no_grad():
            img_batch = img_batch.to(self.device)
            features = self.net(img_batch)
        return features.cpu().numpy()