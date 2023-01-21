"""
Implementation of Convolutional LSTM model, single ConvLSTM cell, and variational version of
a convolutional LSTM
"""

import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    """
    Singel Convolutional LSTM cell. Implements the basic LSTM gates,
    but using Convolutional layers, instead of Fully Connected
    Adapted from: https://github.com/ndrplz/ConvLSTM_pytorch

    Args:
    -----
    input_size: int
        Number of channels of the input
    hidden_size: int
        Number of channels of hidden state.
    kernel_size: int or tuple
        Size of the convolutional kernel.
    bias: bool
        Whether or not to add the bias.
    """

    def __init__(self, input_size, hidden_size, kernel_size=(3, 3), bias=True):
        """ Module initializer """ # 1.input_size = 64, hidden_size = 64, kernel_size=(3, 3) //// predictor: 1.input_size = 128, hidden_size = 128, kernel_size=(3, 3)
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size # kernel_size=(3, 3)
        assert len(kernel_size) == 2, f"Kernel size {kernel_size} has wrong shape"
        super().__init__()
        self.input_size = input_size # input_size = [64,64,64] //// predictor: input_size = [128,128,128,128]
        self.hidden_size = hidden_size # hidden_size = [64,64,64] ////  predictor: hidden_size = [128,128,128,128]

        self.kernel_size = kernel_size # kernel_size=(3, 3)
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2 # padding = (1,1)
        self.bias = bias # True

        self.conv = nn.Conv2d( #
                in_channels=self.input_size + self.hidden_size, # in_channels = [64,64,64] + [64,64,64] = [128,128,128] //// predictor: in_channels = [128,128,128,128] + [128,128,128,128] = [256,256,256,256]
                out_channels=4 * self.hidden_size,# out_channels = 256 //// predictor: out_channels = 4 * [128,128,128,128] = [512,512,512,512]
                kernel_size=self.kernel_size,# kernel_size=(3, 3)
                padding=self.padding,# padding = (1,1)
                bias=self.bias# True
            )

        self.hidden = None
        return

    def forward(self, x, state=None):
        """ x: 16 * 64 * 4 * 4 //// x = 16 * 128 * 16 * 16
        Forward pass of an input through the ConvLSTM cell

        Args:
        -----
        x: torch Tensor
            Feature maps to forward through ConvLSTM Cell. Shape is (B, input_size, H, W)
        state: tuple
            tuple containing the hidden and cell state. Both have shape (B, hidden_size, H, W)
        """
        if state is None:
            state = self.init_hidden(batch_size=x.shape[0], input_size=x.shape[-2:])
        hidden_state, cell_state = state
        # hidden_state = 16 * 64 * 4 * 4 / cell_state = 16 * 64 * 4 * 4 //// hidden_state = 16 * 128 * 16 * 16 / cell_state = 16 * 128 * 16 * 16
        # joinly computing all convs by stacking and spliting across channel dim.
        input = torch.cat([x, hidden_state], dim=1) # => 16 * 128 * 4 * 4 //// =>  16 * 256 * 16 * 16
        out_conv = self.conv(input) # out_conv = 16 * 256 * 4 * 4 <= 16 * 128 * 4 * 4 ////  out_conv = 16 * 512 * 16 * 16 <= 16 * 256 * 16 * 16
        cc_i, cc_f, cc_o, cc_g = torch.split(out_conv, self.hidden_size, dim=1) # 一分为四 #
        # cc_i, cc_f, cc_o, cc_g = 16 * 64 * 4 * 4，16 * 64 * 4 * 4，16 * 64 * 4 * 4，16 * 64 * 4 * 4
        # computing input, forget, update and output gates  # cc_i, cc_f, cc_o, cc_g = 16 * 128 * 16 * 16，16 * 128 * 16 * 16，16 * 128 * 16 * 16，16 * 128 * 16 * 16
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # updating hidden and cell state
        updated_cell_state = f * cell_state + i * g
        updated_hidden_state = o * torch.tanh(updated_cell_state)
        return updated_hidden_state, updated_cell_state # [16 * 64 * 4 * 4,16 * 64 * 4 * 4] //// [16 * 128 * 16 * 16,16 * 128 * 16 * 16]

    def init_hidden(self, batch_size, input_size, device):
        """ Initializing the hidden state of the cell """
        height, width = input_size # input_size = (16,16), batch_size = 16, hidden_size = 128
        state = (Variable(torch.zeros(batch_size, self.hidden_size, height, width, device=device)),
                 Variable(torch.zeros(batch_size, self.hidden_size, height, width, device=device)))
        return state # ((16,128,16,16),(16,128,16,16)) // ((16,128,8,8),(16,128,8,8)) //  ((16,128,4,4),(16,128,4,4)) //// prior/post: ((16,64,16,16),(16,64,16,16)) // ((16,64,8,8),(16,64,8,8)) //  ((16,64,4,4),(16,64,4,4))


class ConvRnnModels(nn.Module):
    """
    Father class to inherit different ConvLSTMS from

    Args:
    -----
    input_size: int
        Number of channels of the input
    hidden_size: int
        Number of channels of hidden state.
    output_size: int
        Number of channels of the output of the ConvLSTM
    num_layers: int
        number of ConvLSTM cells to appy
    kernel_size: int or tuple             # predictor
        Size of the convolutional kernel. # 1.input_size=138, hidden_size=128, output_size=128, num_layers=4, kernel_size=(3, 3),{'period': 1, 'use_output': True}
    bias: bool                            # 2.input_size=266, hidden_size=128, output_size=256, num_layers=4, kernel_size=(3, 3),{'period': 4, 'use_output': True}
        Whether or not to add the bias.   # 3.input_size=522, hidden_size=128, output_size=512, num_layers=4, kernel_size=(3, 3),{'period': 8, 'use_output': True}
    get_all: bool
        If True, returns the output of all layers
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers, kernel_size=(3, 3),
                 bias=True, get_all=True, **kwargs):  # 1. input_size = 148,hidden_size = 64, output_size = 10,num_layers=1,kernel_size=(3, 3),kwargs={'period': 1}
        """ Module initializer """                    # 2. input_size = 266,hidden_size = 64, output_size = 10,num_layers=1,kernel_size=(3, 3),kwargs={'period': 4}
        super().__init__()                            # 3. input_size = 512,hidden_size = 64, output_size = 10,num_layers=1,kernel_size=(3, 3),kwargs={'period': 8}
        self.input_size = input_size # input_size = [148,266,512]
        self.num_layers = num_layers # num_layers = 1
        self.kernel_size = self._check_kernel_size_consistency(kernel_size, num_layers) #kernel_size=(3, 3) => [(3,3)] //// predictor=[(3,3),(3,3),(3,3),(3,3)]
        self.hidden_size = self._extend_for_multilayer(hidden_size, num_layers) # hidden_size=64 => [64] //// predictor=[128,128,128,128]
        self.get_all = get_all # True
        self.bias = bias # True

        # firing frequency parameters
        self.last_output = None
        self.period = kwargs.get("period", 1) # period = [1,4,8] //// predictor= [1,4,8]
        self.counter = 0

        self.hidden = None
        return

    def forward(self, x, state):
        """ """
        raise NotImplementedError("Father class does not implement 'forward' method")

    @staticmethod
    def _check_kernel_size_consistency(kernel_size, num_layers):
        """ Making sure kernel size has one of the accepted types """
        if isinstance(kernel_size, tuple):
            kernel_size = [kernel_size] * num_layers
        if not (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size])):
            raise ValueError("'kernel_size' must be tuple or list of tuples")
        return kernel_size

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if isinstance(param, int):
            param = [param] * num_layers
        return param

    def reset_hidden(self):
        """ Setting the hidden state to None """
        self.hidden = None

    def init_hidden(self, batch_size, input_size, device):
        """ Initializing the hidden states with vectors of zeros """
        hidden = []
        for i in range(self.num_layers):
            hidden.append(self.cell_list[i].init_hidden(batch_size, input_size, device))
        self.hidden = hidden
        return hidden

    def check_counters(self):
        """ Checking if it is firing time (c=0), updating/resetting counter """ #检查它是否是触发时间（c = 0），更新/重置计数器
        should_fire = (self.counter == 0)
        if(should_fire):
            self.reset_counter()
        else:
            self.counter = self.counter - 1
        return should_fire

    def reset_counter(self):
        """ Resetting period counter """
        self.counter = self.period - 1
        return

    def init_counter(self):
        """ Resetting period counter """
        self.counter = 0
        return


class ConvLSTM(ConvRnnModels):
    """
    Full Convolutional LSTM module, which cascades one or more ConvLSTM cells.
    See ConvRnnModels class for parameters
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers, kernel_size=(3, 3), **kwargs):
        """ ConvLSTM module initializer """ # predictor: 1.input_size=138, hidden_size=128, output_size=128, num_layers=4, kernel_size=(3, 3),{'period': 1, 'use_output': True}
        super().__init__(input_size=input_size, hidden_size=hidden_size, output_size=output_size,  # predictor: 1.input_size=266, hidden_size=128, output_size=256, num_layers=4, kernel_size=(3, 3),{'period': 4, 'use_output': True}
                         num_layers=num_layers, kernel_size=kernel_size, **kwargs)
        cell_list = []
        for i in range(0, self.num_layers): # num_layers = 4 ,[0,1,2,3]
            cur_input_size = self.hidden_size[0] if i == 0 else self.hidden_size[i - 1]# predictor:  hidden_size =[128, 128, 128, 128] => cur_input_size = 128
            cell_list.append(ConvLSTMCell(
                    input_size=cur_input_size, # cur_input_size = 128
                    hidden_size=self.hidden_size[i], # hidden_size = [128,128,128,128]
                    kernel_size=self.kernel_size[i], # kernel_size = [(3,3),(3,3),(3,3),(3,3)]
                    bias=self.bias # True
                )
            ) # input_size = 138, hidden_size = 128, kernel_size = (3,3)//input_size = 266, hidden_size = 128, kernel_size = (3,3)//input_size=522, hidden_size = 128, kernel_size = (3,3)
        self.input_conv = nn.Conv2d(self.input_size, self.hidden_size[0], kernel_size=kernel_size, padding=1)
        self.cell_list = nn.ModuleList(cell_list)
        self.out_proj = nn.Sequential( # hidden_size = 128, output_size = [128,256,512]
                nn.Conv2d(in_channels=hidden_size, out_channels=output_size, kernel_size=3, padding=1),
                nn.Tanh()
            )
        return

    def forward(self, x, hidden_state=None):
        """
        Forward pass through all ConvLSTM Cells in the model

        Args:
        -----
        x: torch Tensor
            Feature maps to forward through ConvLSTM Cell. Shape is (B, input_size, H, W)

        Returns:
        --------
        output: torch Tensor
            Processed output of the ConvLSTMs layers. Shape is (B, C, H, W)
        """
        if(hidden_state is not None):  # updating hid-state of first RNN cell
            self.hidden[0] = hidden_state

        should_fire = self.check_counters()
        if (not should_fire):
            return self.last_output
        B, C, H, W = x.shape # 16 * 138 * 16 * 16 //// 16 * 266 * 8 * 8
        if self.hidden is None:
            _ = self.init_hidden(batch_size=B, input_size=(H, W))

        cur_input = self.input_conv(x) # 16 * 138 * 16 * 16 => 16 * 128 * 16 * 16
        # iterating through layers # 16 * 266 * 8 * 8 => 16 * 128 * 8 * 8
        for i in range(self.num_layers):
            self.hidden[i] = self.cell_list[i](x=cur_input, state=self.hidden[i])
            cur_input = self.hidden[i][0]  # cur layer output is next layer input

        output = self.out_proj(cur_input) # 16 * 128 * 16 * 16
        self.last_output = output # 16 * 128 * 16 * 16
        return output


class GaussianConvLSTM(ConvRnnModels):
    """
    ConvLSTM, but instead of predicting the image features at the next frame, it models
    the underlying prior distribution of the latent variables in a sort of VAE fashion.

    1: An initial convolution aligns the number of channels
    2: Aligned features are fed to a ConvLSTM for predicting next step's
    3: Mean and log-variance are computed using a convolutional layer
    4: Reparameterization trick using mean and log-var

    See parameters from LSTM class
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers, kernel_size=(3, 3), **kwargs):
        """                           # 1. input_size = 148,hidden_size = 64, output_size = 10,num_layers=1,kernel_size=(3, 3),kwargs={'period': 1}
        Initializer of the LSTM block # 2. input_size = 266,hidden_size = 64, output_size = 10,num_layers=1,kernel_size=(3, 3),kwargs={'period': 4}
        """                           # 3. input_size = 512,hidden_size = 64, output_size = 10,num_layers=1,kernel_size=(3, 3),kwargs={'period': 8}
        super().__init__(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                         num_layers=num_layers, kernel_size=kernel_size, **kwargs)
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_size = self.hidden_size[0] if i == 0 else self.hidden_size[i - 1] # hidden_size = [64] => cur_input_size = 64
            cell_list.append(ConvLSTMCell(
                    input_size=cur_input_size, # 64
                    hidden_size=self.hidden_size[i],# 64
                    kernel_size=self.kernel_size[i],# (3, 3)
                    bias=self.bias
                )
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.input_conv = nn.Conv2d(self.input_size, self.hidden_size[0], kernel_size=kernel_size, padding=1) # 148,64,(3,3),1 || 266,64,(3,3),1 || 512,64,(3,3),1
        self.mu_var_net = nn.Conv2d(self.hidden_size[-1], output_size * 2, kernel_size=kernel_size, padding=1)# 64,20,(3,3),1 || 64,20,(3,3),1 || 64,20,(3,3),1
        return

    def reparameterize(self, mu, logvar):
        """ Reparameterization trick """ #重新参数化技巧
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def forward(self, x):
        """ x : 1. 16 * 512 * 4 * 4
        Forward pass through GaussianConvLSTM #通过 Gaussian ConvLSTM 前向传递

        Args:
        -----
        x: torch Tensor
            Feature maps to forward through Gaussian ConvLSTM Cell. Shape is (B, input_size, H, W) #特征映射通过 Gaussian ConvLSTM Cell 进行转发。 形状为 (B, input_size, H, W)

        Returns:Inflated lat # 膨胀的纬度
        --------
        latent: torch Tensor
            Latent tensor. Shape is (B, output_size, H, W) # 潜在张量。 形状为 (B, output_size, H, W)
        mu, logvar: torch Tensors
            mean and log-variance of the prior/posterior distribution. Shapes are (B, output_size, H, W) # 先验/后验分布的均值和对数方差。 形状为 (B, output_size, H, W)
        """
        should_fire = self.check_counters()
        if (not should_fire):
            return self.last_output, should_fire

        B, C, H, W = x.shape # 16 * 512 * 4 * 4 //// 16 * 266 * 8 * 8 //// 16 * 148 * 16 * 16
        if self.hidden is None: # hidden = [((16 * 64 * 4 * 4),(16 * 64 * 4 * 4))]
            _ = self.init_hidden(batch_size=B, input_size=(H, W))

        # iterating through recurrent layers
        cur_input = self.input_conv(x) #  16 * 512 * 4 * 4 => 16 * 64 * 4 * 4 ////  16 * 266 * 8 * 8 =>  16 * 64 * 8 * 8 //// 16 * 148 * 16 * 16 => 16 * 64 * 16 * 16
        for i in range(self.num_layers):
            self.hidden[i] = self.cell_list[i](x=cur_input, state=self.hidden[i])
            cur_input = self.hidden[i][0]  # cur layer output is next layer input
        out_lstm = cur_input # 16 * 64 * 4 * 4

        # computing stats with corresponding layer chunking, and obtaining latent tensor # 用相应的层分块计算统计数据，并获得潜在张量
        stats = self.mu_var_net(out_lstm) # 16 * 64 * 4 * 4 => 16 * 20 * 4 * 4 //// 16 * 64 * 8 * 8 => 16 * 20 * 8 * 8 //// 16 * 64 * 16 * 16 => 16 * 20 * 16 * 16
        mu, logvar = torch.chunk(stats, chunks=2, dim=1) # mu = 16 * 10 * 4 * 4, logvar =  16 * 10 * 4 * 4 //// mu = 16 * 10 * 8 * 8, logvar =  16 * 10 * 8 * 8
        latent = self.reparameterize(mu, logvar) # latent = 16 * 10 * 4 * 4 ////   latent = 16 * 10 * 8 * 8

        self.last_output = (latent, mu, logvar) #  last_output = (16 * 10 * 4 * 4,16 * 10 * 4 * 4,16 * 10 * 4 * 4) //// last_output = (16 * 10 * 8 * 8,16 * 10 * 8 * 8,16 * 10 * 8 * 8)
        return (latent, mu, logvar), should_fire # (16 * 10 * 4 * 4,16 * 10 * 4 * 4,16 * 10 * 4 * 4),True ////  (16 * 10 * 8 * 8,16 * 10 * 8 * 8,16 * 10 * 8 * 8),True

#
