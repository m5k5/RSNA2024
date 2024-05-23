from torchmetrics import Metric
from torchmetrics.classification import MulticlassConfusionMatrix
import torch

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