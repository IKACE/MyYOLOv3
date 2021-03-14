import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2

# grid = np.arange(5)
# a, b = np.meshgrid(grid, grid)
# x_offset = torch.FloatTensor(a).view(-1,1)
# y_offset = torch.FloatTensor(b).view(-1,1)
# x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,3).view(-1,2).unsqueeze(0)
# print(x_y_offset.size())

# repeat is by dimension
# unsqueeze is adding a dimension
# view(-1) means dimension size is inferred
anchors = [(10,13), (16,30), (33,23)]
anchors = torch.FloatTensor(anchors)
anchors = anchors.repeat(5*5, 1)
print(anchors.size())
def output_transform(output, img_dim, anchors, num_classes, CUDA = False):
    # assume input has same height and weight
    batch_size = output.size(0)
    grid_size = output.size(2)
    stride = img_dim // grid_size
    bbox_attrs = 5 + num_classes
    anchorCount = len(anchors)
    output = output.view(batch_size, bbox_attrs*anchorCount, grid_size*grid_size)
    output = output.transpose(1,2).contiguous()
    output = output.view(batch_size, grid_size*grid_size*anchorCount, bbox_attrs)

    output[:,:,4] = torch.sigmoid(output[:,:,4])
    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)
    # prepare x and y coordinates separately
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    # concatnate x and y coordinates side by side, repeat by anchor count times, then add a dimension to the 0th to line up with output tensor
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,anchorCount).view(-1,2).unsqueeze(0)

    # sigmoid the  centre_X, centre_Y
    output[:,:,0] = torch.sigmoid(output[:,:,0])
    output[:,:,1] = torch.sigmoid(output[:,:,1])
    output[:,:,:2] += x_y_offset

    # solve height and weight of bounding box
    anchors = [(int(anchor[0])/stride, int(anchor[1])/stride) for anchor in anchors]
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    output[:,:,2:4] = torch.exp(output[:,:,2:4])*anchors

    # sigmod class scores
    output[:,:,5:5+num_classes] = torch.sigmoid(output[:,:,5:5+num_classes])

    # resize the bounding box to align with image
    output[:,:,:4] *= stride
    return output

def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (608,608))         
    img_ =  img[:,:,::-1].transpose((2,0,1))  #  H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_