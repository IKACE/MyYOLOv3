from util import output_transform, get_test_input
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import sys

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
    
    def getAnchors(self):
        return self.anchors

class ShortcutLayer(nn.Identity):
    # skipFrom should be absolute index in the list
    def __init__(self, skipFrom):
        super(ShortcutLayer, self).__init__()
        self.skipFrom = skipFrom

    def getSkipFrom(self):
        return self.skipFrom

class RouteLayer(nn.Identity):
    # This layer for route does not generalized well, it is currently designed only for the YOLOv3 cfg.
    def __init__(self, mode, routeList):
        super(RouteLayer, self).__init__()
        self.mode = mode
        self.routeList = routeList
    
    def getInfo(self):
        return self.mode, self.routeList


class Darknet(nn.Module):
    def __init__(self, cfgPath):
        super(Darknet, self).__init__()
        self.netList = read_cfg(cfgPath)
        self.netConfig, self.moduleList = generate_modules(self.netList)
        self.netList = self.netList[1:]
        print(self.moduleList)

    def forward(self, x, CUDA):
        outputs = {}
        firstYOLO = True
        for idx, module in enumerate(self.moduleList):
            if self.netList[idx]["type"] == "convolutional":
                output = module(x)
            elif self.netList[idx]["type"] == "upsample":
                output = module(x)
            elif self.netList[idx]["type"] == "shortcut":
                output = outputs[idx-1] + outputs[module._modules["shortcut{0}".format(idx)].getSkipFrom()]
                # output = module(output)
            elif self.netList[idx]["type"] == "route":
                routeMode, routeList = module._modules["route{0}".format(idx)].getInfo()
                if routeMode == 1:
                    output = outputs[routeList[0]]
                elif routeMode == 2:
                    # TODO: which dim to concat? which is depth?
                    # B,C,H,W for input tensor in pytorch
                    output = torch.cat((outputs[routeList[0]], outputs[routeList[1]]), 1)
                # output = module(output)
            elif self.netList[idx]["type"] == "yolo":
                output = x
                anchors = module._modules["yolo{0}".format(idx)].getAnchors()
                img_dim = int(self.netConfig["width"])
                if int(self.netConfig["width"]) != int(self.netConfig["height"]):
                    raise Exception("Image height and width do not match")
                num_classes = int(self.netList[idx]["classes"])
                if firstYOLO == True:
                    firstYOLO = False
                    result = output_transform(output.data, img_dim, anchors, num_classes)
                else:
                    result = torch.cat((result, output_transform(output.data, img_dim, anchors, num_classes)), dim=1)
            outputs[idx] = output
            x = output
        return result

    def load_weights(self, weightfile):
        f = open(weightfile, "r")
        # must set int32 and float32, not int and float
        header = np.fromfile(f, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        weights = np.fromfile(f, dtype = np.float32)
        ptr = 0
        for idx, module in enumerate(self.moduleList):
            modelInfo = self.netList[idx]
            module_type = self.netList[idx]["type"]
            if module_type == "convolutional":
                conv = module[0]
                if "batch_normalize" in modelInfo and int(modelInfo["batch_normalize"]) == 1:
                    bn = module[1]
                    
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:        
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


                    






            

def read_cfg(cfgPath):
    """ Read network configuration file from cfgPath, and return a list of dictionary where each dic is the configuration for one neural network """
    netList = []
    with open(cfgPath, "r") as f:
        lines = f.readlines()
        lines = [line for line in lines if line[0] != '#']
        lines = [line.rstrip().lstrip() for line in lines]
        lines = [line for line in lines if len(line) != 0]
    # print(lines)
    netBlock = {}
    for line in lines:
        if line[0] == '[':
            if len(netBlock) != 0:
                netList.append(netBlock)
                netBlock = {}
            netBlock["type"] = line[1:-1]
        else:
            split_strs = line.split("=")
            netBlock[split_strs[0].rstrip()] = split_strs[1].lstrip()
    netList.append(netBlock)
    return netList

def generate_modules(netList):
    """Generate net configuration and nn.ModuleList from list obtained from cfg file"""
    in_channels = 3
    output_filters = []
    modules = nn.ModuleList()
    netConfig = netList[0]
    for idx, netBlock in enumerate(netList[1:]):
        module = nn.Sequential()
        if netBlock["type"] == "convolutional":
            try:
                out_channels = int(netBlock["filters"])
                kernel_size = int(netBlock["size"])
                stride = int(netBlock["stride"])
                padding = int(netBlock["pad"])
                # TODO: why changing?
                if padding:
                    padding = (kernel_size - 1) // 2
                else:
                    padding = 0
                activation = netBlock["activation"]
                batch_normalize = -1
                if "batch_normalize" in netBlock and int(netBlock["batch_normalize"]) == 1:
                    batch_normalize = int(netBlock["batch_normalize"])
                    bias = False
                else:
                    batch_normalize = -1
                    bias = True
            except KeyError:
                print(idx+1, " have generated KeyError")
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            module.add_module("conv{0}".format(idx), conv)
            if batch_normalize != -1:
                bn = nn.BatchNorm2d(out_channels)
                module.add_module("bn{0}".format(idx), bn)
            if activation == "leaky":
                relu = nn.LeakyReLU(0.1, True)
                module.add_module("leaky{0}".format(idx), relu)
        
        elif netBlock["type"] == "upsample":
            stride = netBlock["stride"]
            upsample = nn.Upsample(scale_factor=stride, mode="nearest")
            module.add_module("upsample{0}".format(idx), upsample)
        
        elif netBlock["type"] == "shortcut":
            skipFrom = int(netBlock["from"])
            shortcut = ShortcutLayer(idx+skipFrom)
            module.add_module("shortcut{0}".format(idx), shortcut)
            activation = netBlock["activation"]
            relu = nn.LeakyReLU(0.01, True)
            if activation == "leaky":
                module.add_module("leaky{0}".format(idx), relu)
            if output_filters[idx-1] != output_filters[idx+skipFrom]:
                raise Exception("Dimensions do not match for shortcut layer at layer idx", idx)
            out_channels = output_filters[idx-1]
        
        elif netBlock["type"] == "route":
            layers = netBlock["layers"]
            layers = layers.split(",")
            mode = len(layers)
            routeList = []
            if len(layers) == 1:
                routeFrom = int(layers[0])
                if routeFrom < 0:
                    out_channels = output_filters[idx+routeFrom]
                    routeList.append(idx+routeFrom)
                else:
                    out_channels = output_filters[routeFrom]
                    routeList.append(routeFrom)
            # this part is not generalized well, I just follow the format in the cfg file
            elif len(layers) == 2:
                left = int(layers[0])
                right = int(layers[1])
                if left >= 0 or right < 0:
                    raise Exception("Unrecognized format of route layer for layer idx", idx, "please contact developer for help")
                if right-idx >= 0:
                    # out_channels = output_filters[idx+left]
                    raise Exception("Routing index out of range for layer idx", idx, "please contact developer for help")
                else:
                    out_channels = output_filters[idx+left] + output_filters[right]
                    routeList.append(idx+left)
                    routeList.append(right)
            route = RouteLayer(mode, routeList)
            module.add_module("route{0}".format(idx), route)
        
        elif netBlock["type"] == "yolo":
            masks = netBlock["mask"].split(",")
            masks = [int(mask) for mask in masks]
            anchors = netBlock["anchors"].split(",")
            anchors = [(anchors[i].rstrip().lstrip(), anchors[i+1].rstrip().lstrip()) for i in range(len(anchors)//2)]
            anchors = [anchors[mask] for mask in masks]
            yolo = DetectionLayer(anchors)
            out_channels = output_filters[idx-1]
            module.add_module("yolo{0}".format(idx), yolo)
        in_channels = out_channels
        output_filters.append(out_channels)
        modules.append(module)
    return netConfig, modules
                





netList = read_cfg("cfg/yolov3.cfg")
netConfig, modules = generate_modules(netList)
# print(modules)
with open("modulelayout.txt", 'w') as f:
    original_stdout = sys.stdout
    sys.stdout = f
    print(modules)
    sys.stdout = original_stdout
model = Darknet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")
img = get_test_input()
# pred = model(img, torch.cuda.is_available())
pred = model(img, CUDA=False)

# TODO: size mismatch
print (pred.size())