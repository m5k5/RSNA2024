from torchmetrics import Metric
from torchmetrics.classification import MulticlassConfusionMatrix
import torch
import pydicom
import numpy as np
from PIL import Image

eps=1e-12

class BalancedAccuracy(Metric):
    def __init__(self, numClasses, **kwargs):
        super().__init__(**kwargs)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.mcm = MulticlassConfusionMatrix(num_classes=numClasses)
        self.eps=1e-12

    def update(self, preds, target):
        C = self.mcm(preds, target)
        per_class = torch.diag(C) / (C.sum(axis=1)+self.eps)
        score = per_class.mean()
        self.score += score
        self.total += 1

    def compute(self):
        return self.score.float() / self.total
    

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets ):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        
        #compute cross-entropy 
        CE = torch.nn.functional.cross_entropy(inputs, targets, reduction="none")
        CEExp = torch.exp(-CE)
        focal_loss = self.alpha * (1-CEExp)**self.gamma * CE
                       
        return focal_loss.mean()
    
def dicomToArray(path, IMG_SIZE):
    dicom = pydicom.read_file(path)
    data = pydicom.pixel_data_handlers.util.apply_modality_lut(dicom.pixel_array, dicom)
    data = pydicom.pixel_data_handlers.util.apply_windowing(data, dicom)
    # data = dicom.pixel_array
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    w, h = data.shape[0], data.shape[1]

    #Center crop
    if w>h:
        diff = w-h
        data = data[diff//2:diff//2+h, :]
    if h>w:
        diff = h-w
        data = data[:, diff//2:diff//2+w]

    data = data - np.min(data)
    data = data * 1.0/(np.max(data)+eps)

    w, h = data.shape[0], data.shape[1]

    # resize
    if not (w == IMG_SIZE[0] and h == IMG_SIZE[1]):
        data = np.array(Image.fromarray((data * 255).astype(np.uint8), mode="L").resize(IMG_SIZE))

    # data = (data * 255).astype(np.uint8)
    return data