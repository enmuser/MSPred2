"""
Module for high-level decoders in the MSPRed model.
Inspired by:
    https://github.com/JimmySuen/integral-human-pose:
    --> pytorch_projects/common_pytorch/base_modules/deconv_head.py
"""

import torch.nn as nn

BN_TRACK_STATS = False


class DeconvHead(nn.Module):
    """
    Deconvolutional decoder head for high-level decoders in MSPred # MSPred 中高级解码器的反卷积解码器头

    Args:
    -----
    in_channels: int
        Number of channels at the input of the head # 头部输入的通道数
    out_channels: int
        Number of output channels to predict # 要预测的输出通道数
    num_filters: int
        number of channels in decoder-head hidden layers # 解码器头隐藏层中的通道数
    num_layers: int
        Number of convolutional layers in the decoder head # 解码器头中的卷积层数
    period: int
        Firing period of the decoder head. It should match its corresponding RNN # 解码器头的发射周期。 它应该匹配其对应的 RNN
    """

    def __init__(self, in_channels, out_channels, num_filters=256, num_layers=6, period=1):
        """ Module initializer """ # 1. in_channels = 128, out_channels =1 , num_filters=64, num_layers=2, period=4
        super().__init__()         # 1. in_channels = 256, out_channels =1 , num_filters=64, num_layers=3, period=8
        self.in_channels = in_channels
        self.period = period
        self.counter = 0
        self.last_output = None

        self.upc_layers = nn.ModuleList()
        for i in range(num_layers): # num_layers = [2,3] =[0,1],[0,1,2]
            _in_channels = in_channels if i == 0 else num_filters # _in_channels = [128,64] //  _in_channels = [256,64,64]
            self.upc_layers.append( # 1. _in_channels = 128, num_filters = 64  2. _in_channels = 64, num_filters = 64
                nn.ConvTranspose2d(_in_channels, num_filters, kernel_size=4, stride=2, padding=1, # 1. _in_channels = 256, num_filters = 64  2. _in_channels = 64, num_filters = 64
                                   output_padding=0))                                             # 3. _in_channels = 64, num_filters = 64
            self.upc_layers.append(nn.BatchNorm2d(num_filters, track_running_stats=BN_TRACK_STATS))
            self.upc_layers.append(nn.ReLU(inplace=True))

        # 1x1 convolution #  num_filters = 64, out_channels = 1
        self.upc_layers.append(nn.Conv2d(num_filters, out_channels, kernel_size=1, padding=0))
        self.upc_layers.append(nn.Sigmoid())
        return

    def forward(self, x):
        """ forward pass"""
        should_fire = self.check_counters()
        if not should_fire:
            return self.last_output, should_fire
        if len(x.shape) == 2:
            x = x.view(-1, self.in_channels, 1, 1)
        for layer in self.upc_layers:
            x = layer(x)
        self.last_output = x
        return x, should_fire

    def check_counters(self):
        should_fire = (self.counter == 0)
        if(should_fire):
            self.reset_counter()
        else:
            self.counter = self.counter - 1
        return should_fire

    def reset_counter(self):
        self.counter = self.period - 1
        return

    def init_counter(self):
        self.counter = 0
        return
