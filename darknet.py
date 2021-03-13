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
            netBlock[split_strs[0]] = split_strs[1]
    netList.append(netBlock)
    return netList

def generate_modules(netList):
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





netList = read_cfg("cfg/yolov3.cfg")
generate_modules(netList)
