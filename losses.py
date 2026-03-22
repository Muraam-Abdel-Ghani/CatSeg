# -*- coding: utf-8 -*-
"""
loss function obtained from : https://github.com/Negin-Ghamsarian/DeepPyramidPlus_IJCARS/blob/main/utils/losses_MultiClass_ReduceMean.py
metric functions from: https://github.com/Negin-Ghamsarian/DeepPyramidPlus_IJCARS/blob/main/utils/Metrics_ReduceMean.py 
"""


import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


class Dice_MultiClass(nn.Module):
    
    def __init__(self, num_classes, ignore_first=True, apply_softmax=True):
        super(Dice_MultiClass, self).__init__()
        self.eps = 1
        self.ignore_first = ignore_first
        self.apply_softmax = apply_softmax
        self.num_classes = num_classes

        print(f'num_classes: {num_classes}')

        if ignore_first:

            self.intersection = torch.zeros(num_classes-1).to(device='cuda')
            self.cardinality = torch.zeros(num_classes-1).to(device='cuda')

        else:

            self.intersection = torch.zeros(num_classes).to(device='cuda')
            self.cardinality = torch.zeros(num_classes).to(device='cuda')



    def forward(self, input, target):
        '''
        input: torch.Tensor. Predicted tensor. Shape: [BxCxHxW]. Before softmax,
        it includes the raw outputs of the network. Softmax converts them into
        probabilities.
        target: torch.Tensor. Ground truth tensor. Shape: [BxHxW]
        target_one_hot: torch.Tensor. Conversion of target into a one hot tensor
        '''

        if self.apply_softmax:
            input = input.softmax(dim=1)
        input = torch.argmax(input, dim = 1).squeeze(1)



        input_one_hot = F.one_hot(input.long(), num_classes=self.num_classes).permute(0,3,1,2)
        
        target_one_hot = F.one_hot(target.long(), num_classes=self.num_classes).permute(0,3,1,2)

       
        if self.ignore_first:
            input_one_hot = input_one_hot[:, 1:]
            target_one_hot = target_one_hot[:, 1:]

            
        intersection = torch.sum(input_one_hot * target_one_hot, dim=(0, 2, 3))
        cardinality = torch.sum(input_one_hot + target_one_hot, dim=(0, 2, 3))

        self.intersection += intersection
        self.cardinality += cardinality
        
        dice = (2. * self.intersection + self.eps) / (self.cardinality + self.eps)

        
        return dice, torch.mean(dice) 


    
class IoU_MultiClass(nn.Module):
    
    def __init__(self, num_classes, ignore_first=True, apply_softmax=True):
        super(IoU_MultiClass, self).__init__()
        self.eps = 1
        self.ignore_first = ignore_first
        self.num_classes = num_classes
        self.apply_softmax = apply_softmax


        if ignore_first:

            self.intersection = torch.zeros(num_classes-1).to(device='cuda')
            self.denominator = torch.zeros(num_classes-1).to(device='cuda')

        else:

            self.intersection = torch.zeros(num_classes).to(device='cuda')
            self.denominator = torch.zeros(num_classes).to(device='cuda')

    def forward(self, input, target):
        '''
        input: torch.Tensor. Predicted tensor. Shape: [BxCxHxW]. Before softmax,
        it includes the raw outputs of the network. Softmax converts them into
        probabilities.
        target: torch.Tensor. Ground truth tensor. Shape: [BxHxW]
        target_one_hot: torch.Tensor. Conversion of target into a one hot tensor
        '''
        if self.apply_softmax:
            input = input.softmax(dim=1)
        input = torch.argmax(input, dim = 1)
        input = torch.squeeze(input, 1)

        input_one_hot = F.one_hot(input.long(), num_classes=self.num_classes).permute(0,3,1,2)
        
        target_one_hot = F.one_hot(target.long(), num_classes=self.num_classes).permute(0,3,1,2)

        if self.ignore_first:
            input_one_hot = input_one_hot[:, 1:]
            target_one_hot = target_one_hot[:, 1:]
            
        intersection = torch.sum(input_one_hot * target_one_hot, dim=(0, 2, 3))
        denominator = torch.sum(input_one_hot + target_one_hot, dim=(0, 2, 3)) - intersection


        self.intersection += intersection
        self.denominator += denominator

        IoU = (self.intersection + self.eps) / (self.denominator + self.eps)

        return IoU, torch.mean(IoU)
    

class Dice_CELoss(nn.Module):
    def __init__(self, ignore_first=True, apply_softmax=True):
        super(Dice_CELoss, self).__init__()
        self.eps = 1
        self.ignore_first = ignore_first
        self.apply_softmax = apply_softmax
        self.CE = nn.CrossEntropyLoss()

    def forward(self, output, target):
        '''
        input: torch.Tensor. Predicted tensor. Shape: [BxCxHxW]. Before softmax,
        it includes the raw outputs of the network. Softmax converts them into
        probabilities.
        target: torch.Tensor. Ground truth tensor. Shape: [BxHxW]
        target_one_hot: torch.Tensor. Conversion of target into a one hot tensor
        '''

        CE_loss = self.CE(output, target)

        if self.apply_softmax:
            output = output.softmax(dim=1)

        target_one_hot = F.one_hot(target.long(), num_classes=output.shape[1]).permute(0,3,1,2)

        if self.ignore_first:
            output = output[:, 1:]
            target_one_hot = target_one_hot[:, 1:]


        intersection= torch.sum(target_one_hot*output,dim=(2,3))
        cardinality= torch.sum(target_one_hot+output,dim=(2,3))

         
        dice=(2*intersection+self.eps)/(cardinality+self.eps)

        dice = torch.mean(torch.sum(dice, dim=1)/output.size(dim=1))

        loss = 0.8*CE_loss-0.2*torch.log(dice)
        return loss


# For CaDIS binary segmentation

class Dice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = F.sigmoid(inputs)
        inputs = (inputs > 0.5).float()
        inputs = torch.squeeze(inputs, 1)

        intersection = torch.sum(inputs * targets, dim=(1,2))
        total = torch.sum(inputs + targets, dim=(1,2))
        dice = (2*intersection + smooth)/(total + smooth)
        
        return torch.mean(dice)



class Dice_BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice_BCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.squeeze(inputs, 1)  # Now inputs is [4, 512, 512]

        BCE = F.binary_cross_entropy(inputs, targets.float(), reduction='mean')

        # Optional: binarize for dice only if needed, but keep BCE differentiable
        inputs_bin = (inputs > 0.5).float()
        inputs_bin = torch.squeeze(inputs_bin, 1)

        intersection = torch.sum(inputs_bin * targets, dim=(1, 2))
        total = torch.sum(inputs_bin + targets, dim=(1, 2))
        dice = (2 * intersection + smooth) / (total + smooth)

        Dice_BCE = 0.8 * BCE - 0.2 * torch.log(torch.mean(dice))
        return Dice_BCE

    
    
    
class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth = 1):

        inputs = torch.sigmoid(inputs)
        inputs = (inputs > 0.5).float()
        inputs = torch.squeeze(inputs, 1)
      
        intersection = torch.sum(inputs * targets, dim=(1,2))
        total = torch.sum(inputs + targets, dim=(1,2))
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)

        return torch.mean(IoU)
