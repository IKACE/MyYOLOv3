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

def unique(tensor):
    # delete redundant rows in a tensor
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes by solving intersection area and two input areas
    
    Note that box1 should only have one bounding box, but box2 could have multiple
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

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
    img = cv2.resize(img, (416,416))         
    img_ =  img[:,:,::-1].transpose((2,0,1))  #  H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

# def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
#     conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
#     prediction = prediction*conf_mask
#     box_corner = prediction.new(prediction.shape)
#     box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
#     box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
#     box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
#     box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
#     prediction[:,:,:4] = box_corner[:,:,:4]
#     batch_size = prediction.size(0)
#     write = False
#     for ind in range(batch_size):
#         image_pred = prediction[ind]          #image Tensor
#            #confidence threshholding 
#            #NMS
#         max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
#         max_conf = max_conf.float().unsqueeze(1)
#         max_conf_score = max_conf_score.float().unsqueeze(1)
#         seq = (image_pred[:,:5], max_conf, max_conf_score)
#         image_pred = torch.cat(seq, 1)
#         non_zero_ind =  (torch.nonzero(image_pred[:,4]))
#         try:
#             image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
#         except:
#             continue
        
#         #For PyTorch 0.4 compatibility
#         #Since the above code with not raise exception for no detection 
#         #as scalars are supported in PyTorch 0.4
#         if image_pred_.shape[0] == 0:
#             continue 
#         #Get the various classes detected in the image
#         img_classes = unique(image_pred_[:,-1]) # -1 index holds the class index

def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    # thresholding object scores
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask


    # transform bounding box coordinates to bottom left and top right coordinates
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    batch_size = prediction.size(0)

    write = False
    


    for ind in range(batch_size):
        image_pred = prediction[ind]          #image Tensor
       #confidence threshholding 
       #NMS
    
        max_conf, max_conf_idx = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_idx = max_conf_idx.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_idx)
        image_pred = torch.cat(seq, 1)

        # image_pred shape for ((bounding boxes), (x,y,w,h,object score, max conf, max conf idx))
        
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        # compatibility issue
        if image_pred_.shape[0] == 0:
            continue       
#        
  
        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1])  # -1 index holds the class index

        # img_classes store the unique classes that are detected in the img
        
        
        for cls in img_classes:
            #perform NMS

        
            #get the detections with one particular class

            # get image_pred that are equal to this class and zero out others
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            # get all nonzero elements index
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            # ge all nonzero elements
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top

            # torch.sort: A namedtuple of (values, indices) is returned, where the values are the sorted values and indices are the indices of the elements in the original input tensor.
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #Number of detections
            
            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at 
                #in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
            
                except IndexError:
                    break
            
                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       
            
                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class
            
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))

    # Each detections has 8 attributes, namely, index of the image in the batch to which the detection belongs to, 4 corner coordinates, objectness score, the score of class with maximum confidence, and the index of that class.
    try:
        return output
    except:
        return 0

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    img = letterbox_image(img, (inp_dim, inp_dim))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

