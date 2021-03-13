import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np


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
    """Generate nn.ModuleList from list obtained from cfg file"""
    in_channels = 3
    output_filters = []
    modules = nn.ModuleList()
    for idx, netBlock in enumerate(netList[1:]):
        module = nn.Sequential()
        if netBlock["type"] == "convolutional":
            try:
                out_channels = int(netBlock["filters"])
                kernel_size = int(netBlock["size"])
                stride = int(netBlock["stride"])
                padding = int(netBlock["pad"])
                activation = netBlock["activation"]
                batch_normalize = -1
                if "batch_normalize" in netBlock:
                    batch_normalize = int(netBlock["batch_normalize"])
            except KeyError:
                print(idx+1, " have generated KeyError")
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,)
            module.add_module("conv{0}".format(idx), conv)
            if activation == "leaky":
                relu = nn.LeakyReLU(0.01, True)
                module.add_module("leaky{0}".format(idx), relu)
            if batch_normalize != -1:
                bn = nn.BatchNorm2d(out_channels)
                module.add_module("bn{0}".format(idx), bn)
        
        elif netBlock["type"] == "upsample":
            stride = netBlock["stride"]
            upsample = nn.Upsample(scale_factor=stride, mode="bilinear")
            module.add_module("upsample{0}".format(idx), upsample)
        
        elif netBlock["type"] == "shortcut":
            shortcut = nn.Identity()
            module.add_module("shortcut{0}".format(idx), shortcut)
            skipFrom = int(netBlock["from"])
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
            route = nn.Identity()
            module.add_module("route{0}".format(idx), route)
            if len(layers) == 1:
                routeFrom = int(layers[0])
                if routeFrom < 0:
                    out_channels = output_filters[idx+routeFrom]
                else:
                    out_channels = output_filters[routeFrom]
            # this part is not generalized well, I just follow the format in the cfg file
            elif len(layers) == 2:
                left = int(layers[0])
                right = int(layers[1])
                if left >= 0 or right < 0:
                    raise Exception("Unrecognized format of route layer for layer idx", idx, "please contact developer for help")
                if right-idx >= 0:
                    out_channels = output_filters[idx+left]
                else:
                    out_channels = output_filters[idx+left] + output_filters[right]
        
        elif netBlock["type"] == "yolo":
            yolo = nn.Identity()
            out_channels = output_filters[idx-1]
            module.add_module("yolo{0}".format(idx), yolo)
        in_channels = out_channels
        output_filters.append(out_channels)
        modules.append(module)
    return modules
                





netList = read_cfg("cfg/yolov3.cfg")
modules = generate_modules(netList)
print(modules)
