import numpy as np
import torch
import pickle
import random
import os
import time
import json
import datetime
import argparse
import torchdiffeq
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.integrate import odeint


from ..utils._utils import ColorCandidate, myprint, MultiSubplotDraw, draw_two_dimension


def sample_lhs(lb, ub, n, skip=100):
    # lb = np.array(lb)
    # ub = np.array(ub)
    large_n = skip * n

    sampler = qmc.LatinHypercube(d=1)
    sample = sampler.random(n=large_n)
    sample = qmc.scale(sample, lb, ub)
    sample = np.sort(np.squeeze(sample))

    sample = sample[::skip]

    return sample


def get_now_string():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


class MySin(nn.Module):
    def __init__(self):
        super().__init__()
        self.omega = nn.Parameter(torch.tensor([1.0]))
        # self.omega = 1.0

    def forward(self, x):
        return torch.sin(self.omega * x)


def activation_func(activation):
    return nn.ModuleDict({
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
        "selu": nn.SELU(),
        "sin": MySin(),
        "tanh": nn.Tanh(),
        "softplus": nn.Softplus(beta=1),  # MySoftplus(),
        "elu": nn.ELU(),
        "none": nn.Identity(),
    })[activation]


class ActivationBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activate_list_6 = ["sin", "tanh", "relu", "gelu", "softplus", "elu"]
        self.activate_list_5 = ["tanh", "relu", "gelu", "softplus", "elu"]
        self.activate_list_3 = ["gelu", "softplus", "elu"]
        self.activate_list_2 = ["gelu", "softplus"]
        assert self.config.activation in self.activate_list_6 + ["adaptive_6", "adaptive_3", "adaptive_5", "adaptive_2"]
        if "adaptive" in self.config.activation:
            if self.config.activation == "adaptive_6":
                self.activate_list = self.activate_list_6
            elif self.config.activation == "adaptive_3":
                self.activate_list = self.activate_list_3
            elif self.config.activation == "adaptive_5":
                self.activate_list = self.activate_list_5
            elif self.config.activation == "adaptive_2":
                self.activate_list = self.activate_list_2

            self.activates = nn.ModuleList([activation_func(item).to(config.device) for item in self.activate_list])
            if self.config.init_weights is None:
                self.activate_weights_raw = nn.Parameter(torch.rand(len(self.activate_list)).to(self.config.device),
                                                         requires_grad=True)
            else:
                if self.config.init_weights == "avg":
                    weights_raw = np.asarray([10.0 for item in self.activate_list])
                else:
                    assert self.config.init_weights in self.activate_list
                    weights_raw = np.asarray([item == self.config.init_weights for item in self.activate_list]) * 10.0
                assert self.config.init_weights_strategy in ["fixed", "trainable"]
                if self.config.init_weights_strategy == "fixed":
                    self.activate_weights_raw = torch.tensor(weights_raw).to(self.config.device)
                else:
                    self.activate_weights_raw = nn.Parameter(torch.tensor(weights_raw).to(self.config.device), requires_grad=True)
                # self.activate_weights_raw = torch.tensor([10.0, 10.0]).to(self.config.device)
            # else:
            #     self.activate_weights_raw = nn.Parameter(torch.rand(len(self.activate_list)).to(self.config.device), requires_grad=True)
            softmax = nn.Softmax(dim=0)
            self.activate_weights = softmax(self.activate_weights_raw)

        self.my_sin = activation_func("sin")
        self.my_softplus = activation_func("softplus")

        # self.activate_weights_6 = my_softmax(self.activate_weights_raw_6)
        # self.activate_weights_3 = my_softmax(self.activate_weights_raw_3)
        # self.activate_weights_5 = my_softmax(self.activate_weights_raw_5)
        # self.activate_weights_2 = my_softmax(self.activate_weights_raw_2)


    def forward(self, x):
        if self.config.activation == "gelu":
            return nn.functional.gelu(x)
        elif self.config.activation == "relu":
            return nn.functional.relu(x)
        elif self.config.activation == "tanh":
            return nn.functional.tanh(x)
        elif self.config.activation == "elu":
            return nn.functional.elu(x)
        elif self.config.activation == "sin":
            return self.my_sin(x)
        elif self.config.activation == "softplus":
            return self.my_softplus(x)
        elif self.config.activation == "selu":
            return nn.functional.selu(x)

        assert "adaptive" in self.config.activation, "activation = {} not satisfied".format(self.config.activation)
        activation_res = 0.0
        softmax = nn.Softmax(dim=0)
        self.activate_weights = softmax(self.activate_weights_raw)
        for i in range(len(self.activate_list)):
            tmp_sum = self.activate_weights[i] * self.activates[i](x)
            activation_res += tmp_sum
        # if self.config.activation == "adaptive_6":
        #     self.activate_weights_6 = my_softmax(self.activate_weights_raw_6)
        #     for i in range(len(self.activate_list_6)):
        #         tmp_sum = self.activate_weights_6[i] * self.activates_6[i](x)
        #         activation_res += tmp_sum
        # elif self.config.activation == "adaptive_3":  # adaptive_3
        #     self.activate_weights_3 = my_softmax(self.activate_weights_raw_3)
        #     for i in range(len(self.activate_list_3)):
        #         tmp_sum = self.activate_weights_3[i] * self.activates_3[i](x)
        #         activation_res += tmp_sum
        # elif self.config.activation == "adaptive_5":
        #     self.activate_weights_5 = my_softmax(self.activate_weights_raw_5)
        #     for i in range(len(self.activate_list_5)):
        #         tmp_sum = self.activate_weights_5[i] * self.activates_5[i](x)
        #         activation_res += tmp_sum
        # elif self.config.activation == "adaptive_2":
        #     self.activate_weights_2 = my_softmax(self.activate_weights_raw_2)
        #     for i in range(len(self.activate_list_2)):
        #         tmp_sum = self.activate_weights_2[i] * self.activates_2[i](x)
        #         activation_res += tmp_sum
        return activation_res


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, device):
        super(SpectralConv1d, self).__init__()
        self.device = device
        """ 
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat).to(self.device))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, dtype=torch.cfloat).to(self.device)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    def __init__(self, modes, width, output_dim, device):
        super(FNO1d, self).__init__()

        self.modes1 = modes
        self.width = width
        self.device = device
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width)  # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1, self.device)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1, self.device)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1, self.device)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1, self.device)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        # print("x1 shape: {}".format(x.shape))
        x = self.fc0(x)
        # print("x2 shape: {}".format(x.shape))
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float32)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, device):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat).to(self.device))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat).to(self.device))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat).to(self.device))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat).to(self.device))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        # print("input.shape", input.shape)
        # print("weights.shape", weights.shape)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        # print("x shape", x.shape)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        # print("x_ft shape", x_ft.shape)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device).to(self.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


def block_turing():
    return nn.Sequential(
        nn.Linear(1, 10),
        nn.Tanh(),
        nn.Linear(10, 10),
        nn.Tanh(),
        nn.Linear(10, 10),
        nn.Tanh(),
        nn.Linear(10, 1),
    )


class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, output_dim, device):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.device = device
        self.padding = 6  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(1 + 3, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.device)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.device)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.device)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.device)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        # x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


class ConfigTemplate:
    def __init__(self):
        self.model_name = "xxxxx"
        self.curve_names = None
        self.setup_seed(seed=0)
        self.params = None
        self.args = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 0
        self.pinn = 0

        self.T = None
        # self.T_unit = None
        self.T_N_all = None
        self.T_N_train = None
        self.T_N_test = None

        self.prob_dim = None
        self.y0 = None
        self.boundary_list = None
        # self.t = None
        # self.t_torch = None
        self.x_all, self.x_train, self.x_test = None, None, None
        self.y_all, self.y_train, self.y_test = None, None, None
        # self.x_train_torch = None
        self.truth_all, self.truth_train, self.truth_test = None, None, None
        self.truth_path = None

        self.modes = 12  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.width = 16
        self.fc_map_dim = 128

        self.batch_size = 16

        self.activation = ""
        self.cyclic = None
        self.stable = None
        self.boundary = None
        self.derivative = None
        self.skip_draw_flag = True
        self.loss_average_length = None
        self.init_weights = None
        self.init_weights_strategy = None
        self.scheduler = None

        self.loss_part_n = None

    def setup(self):
        assert self.T_N_all == self.T_N_train + self.T_N_test
        self.truth_path = "truth/{}_truth.pt".format(self.model_name)
        if not os.path.exists(os.path.dirname(self.truth_path)):
            os.makedirs(os.path.dirname(self.truth_path))
        if os.path.exists(self.truth_path):
            print("Truth exists. Loading ...")
            with open(self.truth_path, "rb") as f:
                data = pickle.load(f)
            # self.x_all = data["x_all"]
            # self.y_all = data["y_all"]
            self.x_train = data["x_train"]
            self.x_test = data["x_test"]
            self.y_train = data["y_train"]
            self.y_test = data["y_test"]
        else:
            print("Truth not found. Generating ...")
            # self.x_all = np.concatenate([np.asarray([0.0]), np.asarray(sample_lhs(0, self.T, self.T_N_all - 1))])
            self.x_train = np.concatenate([np.asarray([0.0]), np.asarray(sample_lhs(0, self.T, self.T_N_train - 1))])
            self.x_test = np.concatenate([np.asarray([0.0]), np.asarray(sample_lhs(0, self.T, self.T_N_test - 1))])
            print("self.x_train", self.x_train)
            print("self.x_test", self.x_test)
            self.y_train = odeint(self.pend, self.y0, self.x_train)
            self.y_test = odeint(self.pend, self.y0, self.x_test)
            # self.truth_all = [[x_item, y_item] for x_item, y_item in zip(self.x_all, self.y_all)]
            # truth_all_0 = self.truth_all[0: 1]
            # truth_all_except_0 = self.truth_all[1:]
            # random.shuffle(truth_all_except_0)
            # truth_train = truth_all_0 + sorted(truth_all_except_0[:self.T_N_train - 1], key=lambda x: x[0])
            # truth_test = sorted(truth_all_except_0[-self.T_N_test:], key=lambda x: x[0])
            # self.x_train = np.asarray([item[0] for item in truth_train])
            # self.x_test = np.asarray([item[0] for item in truth_test])
            # self.y_train = np.asarray([item[1] for item in truth_train])
            # self.y_test = np.asarray([item[1] for item in truth_test])
            truth_dic = {
                # "x_all": self.x_all,
                "x_train": self.x_train,
                "x_test": self.x_test,
                # "y_all": self.y_all,
                "y_train": self.y_train,
                "y_test": self.y_test,
            }
            with open(self.truth_path, "wb") as f:
                pickle.dump(truth_dic, f)
        print("x_train: {} {}, ..., {}".format(self.x_train.shape, self.x_train[:5], self.x_train[-5:]))
        print("x_test: {} {}, ..., {}".format(self.x_test.shape, self.x_test[:5], self.x_test[-5:]))
        print("y_train: {} {}, ..., {}".format(self.y_train.shape, self.y_train[:5], self.y_train[-5:]))
        print("y_test: {} {}, ..., {}".format(self.y_test.shape, self.y_test[:5], self.y_test[-5:]))

        self.prob_dim = len(self.curve_names)
        # self.x_train_torch = torch.tensor(self.x_train).reshape(1, -1, 1)
        # self.t_torch = torch.tensor(self.x_train, dtype=torch.float32).to(self.device)
        # self.x = torch.tensor(np.asarray([[[i * self.T_unit] * 1 for i in range(self.T_N)]]),
        #                       dtype=torch.float32).to(self.device)
        # self.truth = odeint(self.pend, self.y0, self.t)
        self.loss_average_length = int(0.1 * self.args.iteration)


    def pend(self, y, t):
        dydt = np.zeros([1])
        return dydt

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


class FourierModelTemplate(FNO1d):
    def __init__(self, config):
        self.time_string = get_now_string()
        self.config = config
        self.setup_seed(self.config.seed)
        super(FourierModelTemplate, self).__init__(modes=self.config.modes, width=self.config.width, output_dim=self.config.prob_dim, device=self.config.device)

        self.train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            torch.tensor(self.config.x_train, dtype=torch.float32).to(self.config.device),
            torch.tensor(self.config.y_train, dtype=torch.float32).to(self.config.device)), batch_size=self.config.T_N_train, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            torch.tensor(self.config.x_test, dtype=torch.float32).to(self.config.device),
            torch.tensor(self.config.y_test, dtype=torch.float32).to(self.config.device)), batch_size=self.config.T_N_test, shuffle=False)

        # self.fc0 = nn.Linear(2, self.config.width)  # input channel is 2: (a(x), x)
        # # self.layers = Layers(config=self.config, n=self.config.layer).to(self.config.device)
        # self.conv0 = SpectralConv1d(self.config).to(self.config.device)
        # self.conv1 = SpectralConv1d(self.config).to(self.config.device)
        # self.conv2 = SpectralConv1d(self.config).to(self.config.device)
        # self.conv3 = SpectralConv1d(self.config).to(self.config.device)
        # self.w0 = nn.Conv1d(self.config.width, self.config.width, 1).to(self.config.device)
        # self.w1 = nn.Conv1d(self.config.width, self.config.width, 1).to(self.config.device)
        # self.w2 = nn.Conv1d(self.config.width, self.config.width, 1).to(self.config.device)
        # self.w3 = nn.Conv1d(self.config.width, self.config.width, 1).to(self.config.device)
        # # self.mlp0 = MLP(self.config.width, self.config.width, self.config.width).to(self.config.device)
        # # self.mlp1 = MLP(self.config.width, self.config.width, self.config.width).to(self.config.device)
        # # self.mlp2 = MLP(self.config.width, self.config.width, self.config.width).to(self.config.device)
        # # self.mlp3 = MLP(self.config.width, self.config.width, self.config.width).to(self.config.device)
        # self.activate_block0 = ActivationBlock(self.config).to(self.config.device)
        # self.activate_block1 = ActivationBlock(self.config).to(self.config.device)
        # self.activate_block2 = ActivationBlock(self.config).to(self.config.device)
        # self.activate_block3 = ActivationBlock(self.config).to(self.config.device)
        #
        # self.fc1 = nn.Linear(self.config.width, self.config.fc_map_dim)
        # self.fc2 = nn.Linear(self.config.fc_map_dim, self.config.prob_dim)

        self.activate_block0 = ActivationBlock(self.config).to(self.config.device)
        self.activate_block1 = ActivationBlock(self.config).to(self.config.device)
        self.activate_block2 = ActivationBlock(self.config).to(self.config.device)
        self.activate_block3 = ActivationBlock(self.config).to(self.config.device)

        self.criterion = torch.nn.MSELoss().to(self.config.device)  # "sum"
        self.criterion_non_reduce = torch.nn.MSELoss(reduction="none").to(self.config.device)

        self.y_tmp_train = None
        self.y_tmp_test = None
        self.epoch_tmp = None
        self.loss_record_tmp = None
        self.real_loss_nmse_record_train_tmp = None
        self.real_loss_nmse_record_test_tmp = None
        self.time_record_tmp = None

        self.activation_weights_record = None

        self.figure_save_path_folder = "{0}/saves/figure/{1}_{2}/".format(self.config.args.main_path,
                                                                          self.config.model_name, self.time_string)
        self.train_save_path_folder = "{0}/saves/train/{1}_{2}/".format(self.config.args.main_path,
                                                                        self.config.model_name, self.time_string)
        if not os.path.exists(self.figure_save_path_folder):
            os.makedirs(self.figure_save_path_folder)
        if not os.path.exists(self.train_save_path_folder):
            os.makedirs(self.train_save_path_folder)
        self.default_colors = ColorCandidate().get_color_list(self.config.prob_dim, 0.5)
        self.default_colors_10 = ColorCandidate().get_color_list(10, 0.5)
        # self.default_colors = ["red", "blue", "green", "orange", "cyan", "purple", "pink", "indigo", "brown", "grey", "indigo", "olive"]

        if not os.path.exists(self.config.args.log_path):
            os.makedirs(os.path.dirname(self.config.args.log_path))
        myprint("using {}".format(str(self.config.device)), self.config.args.log_path)
        myprint("iteration = {}".format(self.config.args.iteration), self.config.args.log_path)
        myprint("epoch_step = {}".format(self.config.args.epoch_step), self.config.args.log_path)
        myprint("test_step = {}".format(self.config.args.test_step), self.config.args.log_path)
        myprint("model_name = {}".format(self.config.model_name), self.config.args.log_path)
        myprint("time_string = {}".format(self.time_string), self.config.args.log_path)
        myprint("seed = {}".format(self.config.seed), self.config.args.log_path)
        myprint("initial_lr = {}".format(self.config.args.initial_lr), self.config.args.log_path)
        myprint("cyclic = {}".format(self.config.cyclic), self.config.args.log_path)
        myprint("stable = {}".format(self.config.stable), self.config.args.log_path)
        myprint("derivative = {}".format(self.config.derivative), self.config.args.log_path)
        myprint("activation = {}".format(self.config.activation), self.config.args.log_path)
        myprint("boundary = {}".format(self.config.boundary), self.config.args.log_path)
        # myprint("early stop: {}".format("On" if self.config.args.early_stop else "Off"), self.config.args.log_path)

    def truth_loss(self):
        x_truth = torch.tensor(self.config.x_train, dtype=torch.float32).to(self.config.device)
        y_truth = torch.tensor(self.config.y_train.reshape([1, self.config.T_N_train, self.config.prob_dim]), dtype=torch.float32).to(self.config.device)
        tl, tl_list = self.loss(x_truth, y_truth)
        loss_print_part = " ".join(
            ["Loss_{0:d}:{1:.12f}".format(i + 1, loss_part.item()) for i, loss_part in enumerate(tl_list)])
        myprint("Ground truth has loss: Loss:{0:.12f} {1}".format(tl.item(), loss_print_part), self.config.args.log_path)

    #  MSE-loss of predicted value against truth
    def real_loss(self, y, y_truth):
        return None, None

    def early_stop(self):
        if not self.config.args.early_stop or len(self.loss_record_tmp) < 2 * self.config.args.early_stop_period:
            return False
        sum_old = sum(
            self.loss_record_tmp[- 2 * self.config.args.early_stop_period: - self.config.args.early_stop_period])
        sum_new = sum(self.loss_record_tmp[- self.config.args.early_stop_period:])
        if (sum_new - sum_old) / sum_old < - self.config.args.early_stop_tolerance:
            myprint("[Early Stop] epoch [{0:d}:{1:d}] -> [{1:d}:{2:d}] reduces {3:.4f} (tolerance = {4:.4f})".format(
                len(self.loss_record_tmp) - 2 * self.config.args.early_stop_period,
                len(self.loss_record_tmp) - self.config.args.early_stop_period,
                len(self.loss_record_tmp),
                (sum_old - sum_new) / sum_old,
                self.config.args.early_stop_tolerance
            ), self.config.args.log_path)
            return False
        else:
            myprint("[Early Stop] epoch [{0:d}:{1:d}] -> [{1:d}:{2:d}] reduces {3:.4f} (tolerance = {4:.4f})".format(
                len(self.loss_record_tmp) - 2 * self.config.args.early_stop_period,
                len(self.loss_record_tmp) - self.config.args.early_stop_period,
                len(self.loss_record_tmp),
                (sum_old - sum_new) / sum_old,
                self.config.args.early_stop_tolerance
            ), self.config.args.log_path)
            myprint("[Early Stop] Early Stop!", self.config.args.log_path)
            return True

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        # x = F.gelu(x)
        x = self.activate_block0(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        # x = F.gelu(x)
        x = self.activate_block1(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        # x = F.gelu(x)
        x = self.activate_block2(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        # x = F.gelu(x)
        x = self.activate_block3(x)
        x = self.fc2(x)
        return x

    # def ode_gradient(self, x, y):
    #     k = self.config.params
    #
    #     m_lacl = y[0, :, 0]
    #     m_tetR = y[0, :, 1]
    #     m_cl = y[0, :, 2]
    #     p_cl = y[0, :, 3]
    #     p_lacl = y[0, :, 4]
    #     p_tetR = y[0, :, 5]
    #
    #     m_lacl_t = torch.gradient(m_lacl, spacing=(self.config.t_torch,))[0]
    #     m_tetR_t = torch.gradient(m_tetR, spacing=(self.config.t_torch,))[0]
    #     m_cl_t = torch.gradient(m_cl, spacing=(self.config.t_torch,))[0]
    #     p_cl_t = torch.gradient(p_cl, spacing=(self.config.t_torch,))[0]
    #     p_lacl_t = torch.gradient(p_lacl, spacing=(self.config.t_torch,))[0]
    #     p_tetR_t = torch.gradient(p_tetR, spacing=(self.config.t_torch,))[0]
    #
    #     f_m_lacl = m_lacl_t - (k.beta * (k.rho + 1 / (1 + p_tetR ** k.n)) - m_lacl)
    #     f_m_tetR = m_tetR_t - (k.beta * (k.rho + 1 / (1 + p_cl ** k.n)) - m_tetR)
    #     f_m_cl = m_cl_t - (k.beta * (k.rho + 1 / (1 + p_lacl ** k.n)) - m_cl)
    #     f_p_cl = p_cl_t - (k.gamma * (m_lacl - p_cl))
    #     f_p_lacl = p_lacl_t - (k.gamma * (m_tetR - p_lacl))
    #     f_p_tetR = p_tetR_t - (k.gamma * (m_cl - p_tetR))
    #
    #     return torch.cat((f_m_lacl.reshape([-1, 1]), f_m_tetR.reshape([-1, 1]), f_m_cl.reshape([-1, 1]),
    #                       f_p_cl.reshape([-1, 1]), f_p_lacl.reshape([-1, 1]), f_p_tetR.reshape([-1, 1])), 1)

    def loss(self, x, y, iteration=-1):
        y0_pred = y[0, 0, :]
        y0_true = torch.tensor(self.config.y0, dtype=torch.float32).to(self.config.device)

        ode_n = self.ode_gradient(x, y)
        zeros_1D = torch.zeros([self.config.T_N]).to(self.config.device)
        zeros_nD = torch.zeros([self.config.T_N, self.config.prob_dim]).to(self.config.device)

        loss1 = self.criterion(y0_pred, y0_true)
        loss2 = 1.0 * (self.criterion(ode_n, zeros_nD))

        loss3 = (1.0 if self.config.boundary else 0.0) * (sum([
            self.criterion(torch.abs(y[:, :, i] - self.config.boundary_list[i][0]), y[:, :, i] - self.config.boundary_list[i][0]) +
            self.criterion(torch.abs(self.config.boundary_list[i][1] - y[:, :, i]), self.config.boundary_list[i][1] - y[:, :, i]) for i in range(self.config.prob_dim)]))
        #(self.criterion(torch.abs(y[:, :, 0] - 0), y[:, :, 0] - 0) + self.criterion(
            # torch.abs(0.65 - y[:, :, 0]), 0.65 - y[:, :, 0]) + self.criterion(torch.abs(y[:, :, 1] - 1.2),
            #                                                                   y[:, :, 1] - 1.2) + self.criterion(
            # torch.abs(4.0 - y[:, :, 1]), 4.0 - y[:, :, 1]))
        # loss4 = (1.0 if self.config.penalty else 0.0) * sum([penalty_func(torch.var(y[0, :, i])) for i in range(self.config.prob_dim)])
        # y_norm = torch.zeros(self.config.prob_dim).to(self.config.device)
        # for i in range(self.config.prob_dim):
        #     y_norm[i] = torch.var(
        #         (y[0, :, i] - torch.min(y[0, :, i])) / (torch.max(y[0, :, i]) - torch.min(y[0, :, i])))
        # loss4 = (1.0 if self.config.penalty else 0) * torch.mean(penalty_func(y_norm))
        # loss4 = self.criterion(1 / u_0, pt_all_zeros_3)
        # loss5 = self.criterion(torch.abs(u_0 - v_0), u_0 - v_0)

        loss = loss1 + loss2 + loss3
        loss_list = np.asarray([loss1.item(), loss2.item(), loss3.item()])
        return loss, loss_list

    def plot_activation_weights(self):
        # self.activation_weights_record
        activation_weights_save_path = "{}/activation_weights.png".format(self.train_save_path_folder)
        m = MultiSubplotDraw(row=2, col=2, fig_size=(16, 12), tight_layout_flag=True, show_flag=False, save_flag=True, save_path=activation_weights_save_path)
        activation_n = self.activation_weights_record.shape[2]
        for i in range(4):
            m.add_subplot(
                y_lists=[self.activation_weights_record[i, :, activation_id].flatten() for activation_id in range(activation_n)],
                x_list=range(1, self.config.args.iteration + 1),
                color_list=self.default_colors_10[:activation_n],
                legend_list=self.activate_block0.activate_list,
                line_style_list=["solid"] * activation_n,
                fig_title="activation block {}".format(i))
        m.draw()
        myprint("initial: \n{}".format(self.activation_weights_record[:, 0, :]), self.config.args.log_path)
        myprint("end: \n{}".format(self.activation_weights_record[:, -1, :]), self.config.args.log_path)


    def train_model(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.args.initial_lr)  # weight_decay=1e-4
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1 / (e / 1000 + 1))
        assert self.config.scheduler in ["cosine", "decade", "decade_pp", "fixed", "step"]
        if self.config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.args.iteration)
        elif self.config.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        elif self.config.scheduler == "decade":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1 / (e / 1000 + 1))
        elif self.config.scheduler == "decade_pp":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1 / (e / 10000 + 1))
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1.0)
        self.train()
        # assert mod in ["train", "test"]
        # if mod == "train":
        #     self.t_torch =

        start_time = time.time()
        start_time_0 = start_time
        loss_record = []
        # real_loss_mse_record = []
        real_loss_nmse_record_train = []
        real_loss_nmse_record_test = []
        time_record = []

        adaptive_weights_record_0 = []
        adaptive_weights_record_1 = []
        adaptive_weights_record_2 = []
        adaptive_weights_record_3 = []

        for epoch in range(1, self.config.args.iteration + 1):
            self.train()
            loss = 0.0
            train_nmse_loss = 0.0
            loss_list = np.zeros(self.config.loss_part_n)
            y_pred_train = None
            for x, _ in self.train_loader:
                x = x.to(self.config.device).reshape(1, -1, 1)
                # print("x shape: {} starting: {}".format(x.shape, x[:2]))
                optimizer.zero_grad()
                y_pred_train = self.forward(x)
                loss_tmp, loss_list_tmp = self.loss(torch.tensor(self.config.x_train, dtype=torch.float32).to(self.config.device), y_pred_train, epoch)
                _, real_loss_nmse_train_tmp = self.real_loss(
                    y=y_pred_train[0],
                    y_truth=torch.tensor(self.config.y_train[:, :]).to(self.config.device),
                )
                loss += loss_tmp.item()
                train_nmse_loss += real_loss_nmse_train_tmp.item()
                loss_list += loss_list_tmp
                loss_tmp.backward()
                optimizer.step()


            loss_record.append(loss)
            # real_loss_mse, real_loss_nmse = self.real_loss(y_pred)
            # real_loss_mse_record.append(real_loss_mse.item())
            real_loss_nmse_record_train.append(train_nmse_loss)

            test_nmse_loss = 0.0
            with torch.no_grad():
                for x, y in self.test_loader:
                    x = x.to(self.config.device).reshape(1, -1, 1)
                    y = y.to(self.config.device).reshape(-1, self.config.prob_dim)

                    y_pred_test = self.forward(x)
                    _, real_loss_nmse_test_tmp = self.real_loss(
                        y=y_pred_test[0],
                        y_truth=y,
                    )
                    test_nmse_loss += real_loss_nmse_test_tmp.item()
            real_loss_nmse_record_test.append(test_nmse_loss)

            if "adaptive" in self.config.activation:
                if "Turing" in self.config.model_name:
                    # myprint("Turing branch", self.config.args.log_path)
                    adaptive_weights_record_0.append(list(self.f_model.activate_block1.activate_weights.cpu().detach().numpy()))
                    adaptive_weights_record_1.append(list(self.f_model.activate_block2.activate_weights.cpu().detach().numpy()))
                    adaptive_weights_record_2.append(list(self.f_model.activate_block3.activate_weights.cpu().detach().numpy()))
                    adaptive_weights_record_3.append(list(self.f_model.activate_block4.activate_weights.cpu().detach().numpy()))
                else:
                    adaptive_weights_record_0.append(list(self.activate_block0.activate_weights.cpu().detach().numpy()))
                    adaptive_weights_record_1.append(list(self.activate_block1.activate_weights.cpu().detach().numpy()))
                    adaptive_weights_record_2.append(list(self.activate_block2.activate_weights.cpu().detach().numpy()))
                    adaptive_weights_record_3.append(list(self.activate_block3.activate_weights.cpu().detach().numpy()))

            # torch.autograd.set_detect_anomaly(True)
            # loss.backward(retain_graph=True)  # retain_graph=True
            # loss.backward()
            # optimizer.step()
            scheduler.step()

            now_time = time.time()
            time_record.append(now_time - start_time_0)

            if epoch % self.config.args.epoch_step == 0 or epoch == self.config.args.iteration:
                loss_print_part = " ".join(
                    ["Loss_{0:d}:{1:.12f}".format(i + 1, loss_part) for i, loss_part in enumerate(loss_list)])
                myprint(
                    "Epoch [{0:05d}/{1:05d}] Loss:{2:.12f} {3} NMSE-Loss(train): {4:.12f} NMSE-Loss(test): {5:.12f} Lr:{6:.12f} Time:{7:.6f}s ({8:.2f}min in total, {9:.2f}min remains)".format(
                        epoch, self.config.args.iteration, loss, loss_print_part, train_nmse_loss, test_nmse_loss,
                        optimizer.param_groups[0]["lr"], now_time - start_time, (now_time - start_time_0) / 60.0,
                        (now_time - start_time_0) / 60.0 / epoch * (self.config.args.iteration - epoch)), self.config.args.log_path)
                start_time = now_time

                if epoch % self.config.args.test_step == 0:
                    self.y_tmp_train = y_pred_train
                    self.y_tmp_test = y_pred_test
                    self.epoch_tmp = epoch
                    self.loss_record_tmp = loss_record
                    self.real_loss_nmse_record_test_tmp = real_loss_nmse_record_test
                    self.real_loss_nmse_record_train_tmp = real_loss_nmse_record_train
                    self.time_record_tmp = time_record

                    self.activation_weights_record = np.asarray([adaptive_weights_record_0, adaptive_weights_record_1, adaptive_weights_record_2, adaptive_weights_record_3])
                    self.draw_model()
                    # save_path_loss = "{}/{}_{}_loss.npy".format(self.train_save_path_folder, self.config.model_name, self.time_string)
                    # np.save(save_path_loss, np.asarray(loss_record))

                    myprint("saving training info ...", self.config.args.log_path)
                    train_info = {
                        "model_name": self.config.model_name,
                        "seed": self.config.seed,
                        "prob_dim": self.config.prob_dim,
                        "activation": self.config.activation,
                        "cyclic": self.config.cyclic,
                        "stable": self.config.stable,
                        "derivative": self.config.derivative,
                        "loss_average_length": self.config.loss_average_length,
                        "epoch": self.config.args.iteration,
                        "epoch_stop": self.epoch_tmp,
                        "initial_lr": self.config.args.initial_lr,
                        "loss_length": len(loss_record),
                        "loss": np.asarray(loss_record),
                        "real_loss_nmse_test": np.asarray(real_loss_nmse_record_test),
                        "real_loss_nmse_train": np.asarray(real_loss_nmse_record_train),
                        "time": np.asarray(time_record),
                        "y_predict_train": y_pred_train[0, :, :].cpu().detach().numpy(),
                        "y_predict_test": y_pred_test[0, :, :].cpu().detach().numpy(),
                        "truth_all": self.config.truth_all,
                        # "truth_train": self.config.truth_train,
                        # "truth_test": self.config.truth_test,
                        "x_all": np.asarray(self.config.x_all),
                        "x_train": np.asarray(self.config.x_train),
                        "x_test": np.asarray(self.config.x_test),
                        "y_all": np.asarray(self.config.y_all),
                        "y_train": np.asarray(self.config.y_train),
                        "y_test": np.asarray(self.config.y_test),
                        "y_train_shape": self.config.y_train.shape,
                        # "config": self.config,
                        "time_string": self.time_string,
                        # "weights_raw": np.asarray([
                        #     self.activate_block0.activate_weights_raw.cpu().detach().numpy(),
                        #     self.activate_block1.activate_weights_raw.cpu().detach().numpy(),
                        #     self.activate_block2.activate_weights_raw.cpu().detach().numpy(),
                        #     self.activate_block3.activate_weights_raw.cpu().detach().numpy(),
                        # ]),
                        # "weights": np.asarray([
                        #     self.activate_block0.activate_weights.cpu().detach().numpy(),
                        #     self.activate_block1.activate_weights.cpu().detach().numpy(),
                        #     self.activate_block2.activate_weights.cpu().detach().numpy(),
                        #     self.activate_block3.activate_weights.cpu().detach().numpy(),
                        # ]),
                        # "sin_weight": np.asarray([
                        #     self.activate_block0.activates[0].omega.cpu().detach().numpy(),
                        #     self.activate_block1.activates[0].omega.cpu().detach().numpy(),
                        #     self.activate_block2.activates[0].omega.cpu().detach().numpy(),
                        #     self.activate_block3.activates[0].omega.cpu().detach().numpy(),
                        # ]),
                        "activation_weights_record": self.activation_weights_record,
                    }
                    train_info_path_loss = "{}/{}_{}_info.npy".format(self.train_save_path_folder,
                                                                      self.config.model_name, self.time_string)
                    model_save_path = "{}/{}_{}_last.pt".format(self.train_save_path_folder,
                                                                      self.config.model_name, self.time_string)
                    with open(train_info_path_loss, "wb") as f:
                        pickle.dump(train_info, f)
                    torch.save({
                        "model_state_dict": self.state_dict(),
                        "info": train_info,
                    }, model_save_path)
                    if "adaptive" in self.config.activation:
                        self.plot_activation_weights()

                    if epoch == self.config.args.iteration or self.early_stop():
                        # myprint(str(train_info), self.config.args.log_path)
                        self.write_finish_log()

                        myprint("figure_save_path_folder: {}".format(self.figure_save_path_folder), self.config.args.log_path)
                        myprint("train_save_path_folder: {}".format(self.train_save_path_folder), self.config.args.log_path)
                        myprint("Finished.", self.config.args.log_path)
                        break

                    # myprint(str(train_info), self.config.args.log_path)

    def draw_model(self):
        if self.config.skip_draw_flag:
            myprint("(Skipped drawing)", self.config.args.log_path)
            return

        if "turing" not in self.config.model_name:
            y_draw_train = self.y_tmp_train[0].cpu().detach().numpy().swapaxes(0, 1)
            y_draw_test = self.y_tmp_test[0].cpu().detach().numpy().swapaxes(0, 1)
            x_draw_train = self.config.x_train
            x_draw_test = self.config.x_test
            y_draw_truth_train = self.config.y_train.swapaxes(0, 1)
            y_draw_truth_test = self.config.y_test.swapaxes(0, 1)
            save_path_train = "{}/{}_{}_epoch={}_train.png".format(self.figure_save_path_folder, self.config.model_name, self.time_string, self.epoch_tmp)
            save_path_test = "{}/{}_{}_epoch={}_test.png".format(self.figure_save_path_folder, self.config.model_name, self.time_string, self.epoch_tmp)
            # print(platform.system().lower())
            draw_two_dimension(
                y_lists=np.concatenate([y_draw_train, y_draw_truth_train], axis=0),
                x_list=x_draw_train,
                color_list=self.default_colors[: 2 * self.config.prob_dim],
                legend_list=self.config.curve_names + ["{}_true".format(item) for item in self.config.curve_names],
                line_style_list=["solid"] * self.config.prob_dim + ["dashed"] * self.config.prob_dim,
                fig_title="{}_{}_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp),
                fig_title_size=12,
                fig_size=(12, 6),
                show_flag=False,  # if platform.system().lower() == "darwin" else True,
                save_flag=True,  # if platform.system().lower() == "darwin" else False,
                save_path=save_path_train,
                save_dpi=200,
                legend_loc="center right",
            )
            myprint("Figure is saved to {}".format(save_path_train), self.config.args.log_path)
            draw_two_dimension(
                y_lists=np.concatenate([y_draw_test, y_draw_truth_test], axis=0),
                x_list=x_draw_test,
                color_list=self.default_colors[: 2 * self.config.prob_dim],
                legend_list=self.config.curve_names + ["{}_true".format(item) for item in self.config.curve_names],
                line_style_list=["solid"] * self.config.prob_dim + ["dashed"] * self.config.prob_dim,
                fig_title="{}_{}_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp),
                fig_title_size=12,
                fig_size=(12, 6),
                show_flag=False,  # if platform.system().lower() == "darwin" else True,
                save_flag=True,  # if platform.system().lower() == "darwin" else False,
                save_path=save_path_test,
                save_dpi=200,
                legend_loc="center right",
            )
            myprint("Figure is saved to {}".format(save_path_test), self.config.args.log_path)
        # self.draw_loss_multi(self.loss_record_tmp, [1.0, 0.5, 0.25, 0.125])
        # self.draw_loss_multi(self.real_loss_nmse_record_train_tmp, [1.0, 0.5, 0.25, 0.125])
        # self.draw_loss_multi(self.real_loss_nmse_record_test_tmp, [1.0, 0.5, 0.25, 0.125])

    def write_finish_log(self):
        with open(os.path.join(self.config.args.main_path, "saves/record.txt"), "a") as f:
            f.write("{0},{1},{2},{3:.2f},{4},{5:.6f},{6:.12f},{7:.12f},{8:.12f},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},{23},{24},{25},{26},{27},{28}\n".format(
                self.config.model_name,  # 0
                self.time_string,  # 1
                self.config.seed,  # 2
                self.time_record_tmp[-1] / 60.0,  # 3
                self.config.args.iteration,  # 4
                self.config.args.initial_lr,  # 5
                sum(self.loss_record_tmp[-self.config.loss_average_length:]) / self.config.loss_average_length,  # 6
                sum(self.real_loss_nmse_record_train_tmp[-self.config.loss_average_length:]) / self.config.loss_average_length,  # 7
                sum(self.real_loss_nmse_record_test_tmp[-self.config.loss_average_length:]) / self.config.loss_average_length,  # 8
                self.config.pinn,  # 9
                self.config.activation,  # 10
                self.config.stable,  # 11
                self.config.cyclic,  # 12
                self.config.derivative,  # 13
                self.config.boundary,  # 14
                self.config.loss_average_length,  # 15
                "{}-{}".format(self.config.args.iteration - self.config.loss_average_length, self.config.args.iteration),  # 16
                self.config.init_weights,  # 17
                self.config.init_weights_strategy,  # 18
                self.config.scheduler,  # 19
                self.config.T if self.config.T else None,  # 20
                self.config.T_N_train if self.config.T_N_train else None,  # 21
                self.config.T_N_test if self.config.T_N_test else None,  # 22
                self.activation_weights_record[0][-1][0] if self.config.activation in ["adaptive_5", "adaptive_6"] else None,  # 23
                self.activation_weights_record[0][-1][1] if self.config.activation in ["adaptive_5", "adaptive_6"] else None,  # 24
                self.activation_weights_record[0][-1][2] if self.config.activation in ["adaptive_5", "adaptive_6"] else None,  # 25
                self.activation_weights_record[0][-1][3] if self.config.activation in ["adaptive_5", "adaptive_6"] else None,  # 26
                self.activation_weights_record[0][-1][4] if self.config.activation in ["adaptive_5", "adaptive_6"] else None,  # 27
                self.activation_weights_record[0][-1][5] if self.config.activation in ["adaptive_6"] else None,  # 28
            ))

    def write_fail_log(self):
        with open(os.path.join(self.config.args.main_path, "saves/record.txt"), "a") as f:
            f.write("{0},{1},{2},{3:.2f},{4},{5:.6f},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18}\n".format(
                self.config.model_name,  # 0
                self.time_string,  # 1
                self.config.seed,  # 2
                0.0,  # self.time_record_tmp[-1] / 60.0,  # 3
                self.config.args.iteration,  # 4
                self.config.args.initial_lr,  # 5
                "fail",  # 6
                "fail",  # 7
                "fail",  # 8
                self.config.pinn,  # 9
                self.config.activation,  # 10
                self.config.stable,  # 11
                self.config.cyclic,  # 12
                self.config.derivative,  # 13
                self.config.boundary,  # 14
                self.config.loss_average_length,  # 15
                "{}-{}".format(self.config.args.iteration - self.config.loss_average_length, self.config.args.iteration),  # 16
                self.config.init_weights,
                self.config.init_weights_strategy,
            ))

    @staticmethod
    def draw_loss_multi(loss_list, last_rate_list):
        m = MultiSubplotDraw(row=1, col=len(last_rate_list), fig_size=(8 * len(last_rate_list), 6),
                             tight_layout_flag=True, show_flag=True, save_flag=False, save_path=None)
        for one_rate in last_rate_list:
            m.add_subplot(
                y_lists=[loss_list[-int(len(loss_list) * one_rate):]],
                x_list=range(len(loss_list) - int(len(loss_list) * one_rate) + 1, len(loss_list) + 1),
                color_list=["blue"],
                line_style_list=["solid"],
                fig_title="Loss - lastest ${}$% - epoch ${}$ to ${}$".format(int(100 * one_rate), len(loss_list) - int(
                    len(loss_list) * one_rate) + 1, len(loss_list)),
                fig_x_label="epoch",
                fig_y_label="loss",
            )
        m.draw()


class FourierModelTemplate3D(FNO3d):
    def __init__(self, config):
        self.time_string = get_now_string()
        self.config = config
        self.setup_seed(self.config.seed)
        self.modes1, self.modes2, self.modes3 = self.config.modes, self.config.modes, self.config.modes
        if self.config.params.N < self.modes2:
            self.modes2 = self.config.params.N
        if self.config.params.M < self.modes3:
            self.modes3 = self.config.params.M

        super(FourierModelTemplate3D, self).__init__(modes1=self.modes1, modes2=self.modes2, modes3=self.modes3, width=self.config.width, output_dim=self.config.prob_dim, device=self.config.device)

        self.train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            torch.tensor(self.config.x_train, dtype=torch.float32).to(self.config.device),
            torch.tensor(self.config.y_train, dtype=torch.float32).to(self.config.device)), batch_size=self.config.T_N_train, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            torch.tensor(self.config.x_test, dtype=torch.float32).to(self.config.device),
            torch.tensor(self.config.y_test, dtype=torch.float32).to(self.config.device)), batch_size=self.config.T_N_test, shuffle=False)

        # self.fc0 = nn.Linear(2, self.config.width)  # input channel is 2: (a(x), x)
        # # self.layers = Layers(config=self.config, n=self.config.layer).to(self.config.device)
        # self.conv0 = SpectralConv1d(self.config).to(self.config.device)
        # self.conv1 = SpectralConv1d(self.config).to(self.config.device)
        # self.conv2 = SpectralConv1d(self.config).to(self.config.device)
        # self.conv3 = SpectralConv1d(self.config).to(self.config.device)
        # self.w0 = nn.Conv1d(self.config.width, self.config.width, 1).to(self.config.device)
        # self.w1 = nn.Conv1d(self.config.width, self.config.width, 1).to(self.config.device)
        # self.w2 = nn.Conv1d(self.config.width, self.config.width, 1).to(self.config.device)
        # self.w3 = nn.Conv1d(self.config.width, self.config.width, 1).to(self.config.device)
        # # self.mlp0 = MLP(self.config.width, self.config.width, self.config.width).to(self.config.device)
        # # self.mlp1 = MLP(self.config.width, self.config.width, self.config.width).to(self.config.device)
        # # self.mlp2 = MLP(self.config.width, self.config.width, self.config.width).to(self.config.device)
        # # self.mlp3 = MLP(self.config.width, self.config.width, self.config.width).to(self.config.device)
        # self.activate_block0 = ActivationBlock(self.config).to(self.config.device)
        # self.activate_block1 = ActivationBlock(self.config).to(self.config.device)
        # self.activate_block2 = ActivationBlock(self.config).to(self.config.device)
        # self.activate_block3 = ActivationBlock(self.config).to(self.config.device)
        #
        # self.fc1 = nn.Linear(self.config.width, self.config.fc_map_dim)
        # self.fc2 = nn.Linear(self.config.fc_map_dim, self.config.prob_dim)

        self.activate_block0 = ActivationBlock(self.config).to(self.config.device)
        self.activate_block1 = ActivationBlock(self.config).to(self.config.device)
        self.activate_block2 = ActivationBlock(self.config).to(self.config.device)
        self.activate_block3 = ActivationBlock(self.config).to(self.config.device)

        self.criterion = torch.nn.MSELoss().to(self.config.device)  # "sum"
        self.criterion_non_reduce = torch.nn.MSELoss(reduction="none").to(self.config.device)

        self.y_tmp_train = None
        self.y_tmp_test = None
        self.epoch_tmp = None
        self.loss_record_tmp = None
        self.real_loss_nmse_record_train_tmp = None
        self.real_loss_nmse_record_test_tmp = None
        self.time_record_tmp = None

        self.activation_weights_record = None

        self.figure_save_path_folder = "{0}/saves/figure/{1}_{2}/".format(self.config.args.main_path,
                                                                          self.config.model_name, self.time_string)
        self.train_save_path_folder = "{0}/saves/train/{1}_{2}/".format(self.config.args.main_path,
                                                                        self.config.model_name, self.time_string)
        if not os.path.exists(self.figure_save_path_folder):
            os.makedirs(self.figure_save_path_folder)
        if not os.path.exists(self.train_save_path_folder):
            os.makedirs(self.train_save_path_folder)
        self.default_colors = ColorCandidate().get_color_list(self.config.prob_dim, 0.5)
        self.default_colors_10 = ColorCandidate().get_color_list(10, 0.5)
        # self.default_colors = ["red", "blue", "green", "orange", "cyan", "purple", "pink", "indigo", "brown", "grey", "indigo", "olive"]

        myprint("using {}".format(str(self.config.device)), self.config.args.log_path)
        myprint("iteration = {}".format(self.config.args.iteration), self.config.args.log_path)
        myprint("epoch_step = {}".format(self.config.args.epoch_step), self.config.args.log_path)
        myprint("test_step = {}".format(self.config.args.test_step), self.config.args.log_path)
        myprint("model_name = {}".format(self.config.model_name), self.config.args.log_path)
        myprint("time_string = {}".format(self.time_string), self.config.args.log_path)
        myprint("seed = {}".format(self.config.seed), self.config.args.log_path)
        myprint("initial_lr = {}".format(self.config.args.initial_lr), self.config.args.log_path)
        myprint("cyclic = {}".format(self.config.cyclic), self.config.args.log_path)
        myprint("stable = {}".format(self.config.stable), self.config.args.log_path)
        myprint("derivative = {}".format(self.config.derivative), self.config.args.log_path)
        myprint("activation = {}".format(self.config.activation), self.config.args.log_path)
        myprint("boundary = {}".format(self.config.boundary), self.config.args.log_path)
        # myprint("early stop: {}".format("On" if self.config.args.early_stop else "Off"), self.config.args.log_path)

    def truth_loss(self):
        x_truth = torch.tensor(self.config.x_train, dtype=torch.float32).to(self.config.device)
        y_truth = torch.tensor(self.config.y_train.reshape(
            [1, self.config.T_N_train, self.config.params.N, self.config.params.M, self.config.prob_dim]), dtype=torch.float32).to(self.config.device)

        # print("y_truth max:", torch.max(y_truth))
        # print("y_truth min:", torch.min(y_truth))
        tl, tl_list = self.loss(x_truth, y_truth)
        loss_print_part = " ".join(
            ["Loss_{0:d}:{1:.12f}".format(i + 1, loss_part.item()) for i, loss_part in enumerate(tl_list)])
        print("Ground truth has loss: Loss:{0:.12f} {1}".format(tl.item(), loss_print_part))

    #  MSE-loss of predicted value against truth
    def real_loss(self, y, y_truth):
        return None, None

    def early_stop(self):
        if not self.config.args.early_stop or len(self.loss_record_tmp) < 2 * self.config.args.early_stop_period:
            return False
        sum_old = sum(
            self.loss_record_tmp[- 2 * self.config.args.early_stop_period: - self.config.args.early_stop_period])
        sum_new = sum(self.loss_record_tmp[- self.config.args.early_stop_period:])
        if (sum_new - sum_old) / sum_old < - self.config.args.early_stop_tolerance:
            myprint("[Early Stop] epoch [{0:d}:{1:d}] -> [{1:d}:{2:d}] reduces {3:.4f} (tolerance = {4:.4f})".format(
                len(self.loss_record_tmp) - 2 * self.config.args.early_stop_period,
                len(self.loss_record_tmp) - self.config.args.early_stop_period,
                len(self.loss_record_tmp),
                (sum_old - sum_new) / sum_old,
                self.config.args.early_stop_tolerance
            ), self.config.args.log_path)
            return False
        else:
            myprint("[Early Stop] epoch [{0:d}:{1:d}] -> [{1:d}:{2:d}] reduces {3:.4f} (tolerance = {4:.4f})".format(
                len(self.loss_record_tmp) - 2 * self.config.args.early_stop_period,
                len(self.loss_record_tmp) - self.config.args.early_stop_period,
                len(self.loss_record_tmp),
                (sum_old - sum_new) / sum_old,
                self.config.args.early_stop_tolerance
            ), self.config.args.log_path)
            myprint("[Early Stop] Early Stop!", self.config.args.log_path)
            return True

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def forward(self, x):
        # print("x shape", x.shape)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        # x = F.gelu(x)
        x = self.activate_block0(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        # x = F.gelu(x)
        x = self.activate_block1(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        # x = F.gelu(x)
        x = self.activate_block2(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        # x = F.gelu(x)
        x = self.activate_block3(x)
        x = self.fc2(x)
        return x

    # def ode_gradient(self, x, y):
    #     k = self.config.params
    #
    #     m_lacl = y[0, :, 0]
    #     m_tetR = y[0, :, 1]
    #     m_cl = y[0, :, 2]
    #     p_cl = y[0, :, 3]
    #     p_lacl = y[0, :, 4]
    #     p_tetR = y[0, :, 5]
    #
    #     m_lacl_t = torch.gradient(m_lacl, spacing=(self.config.t_torch,))[0]
    #     m_tetR_t = torch.gradient(m_tetR, spacing=(self.config.t_torch,))[0]
    #     m_cl_t = torch.gradient(m_cl, spacing=(self.config.t_torch,))[0]
    #     p_cl_t = torch.gradient(p_cl, spacing=(self.config.t_torch,))[0]
    #     p_lacl_t = torch.gradient(p_lacl, spacing=(self.config.t_torch,))[0]
    #     p_tetR_t = torch.gradient(p_tetR, spacing=(self.config.t_torch,))[0]
    #
    #     f_m_lacl = m_lacl_t - (k.beta * (k.rho + 1 / (1 + p_tetR ** k.n)) - m_lacl)
    #     f_m_tetR = m_tetR_t - (k.beta * (k.rho + 1 / (1 + p_cl ** k.n)) - m_tetR)
    #     f_m_cl = m_cl_t - (k.beta * (k.rho + 1 / (1 + p_lacl ** k.n)) - m_cl)
    #     f_p_cl = p_cl_t - (k.gamma * (m_lacl - p_cl))
    #     f_p_lacl = p_lacl_t - (k.gamma * (m_tetR - p_lacl))
    #     f_p_tetR = p_tetR_t - (k.gamma * (m_cl - p_tetR))
    #
    #     return torch.cat((f_m_lacl.reshape([-1, 1]), f_m_tetR.reshape([-1, 1]), f_m_cl.reshape([-1, 1]),
    #                       f_p_cl.reshape([-1, 1]), f_p_lacl.reshape([-1, 1]), f_p_tetR.reshape([-1, 1])), 1)

    def loss(self, x, y, iteration=-1):
        y0_pred = y[0, 0, :]
        y0_true = torch.tensor(self.config.y0, dtype=torch.float32).to(self.config.device)

        ode_n = self.ode_gradient(x, y)
        zeros_1D = torch.zeros([self.config.T_N]).to(self.config.device)
        zeros_nD = torch.zeros([self.config.T_N, self.config.prob_dim]).to(self.config.device)

        loss1 = self.criterion(y0_pred, y0_true)
        loss2 = 1.0 * (self.criterion(ode_n, zeros_nD))

        loss3 = (1.0 if self.config.boundary else 0.0) * (sum([
            self.criterion(torch.abs(y[:, :, i] - self.config.boundary_list[i][0]), y[:, :, i] - self.config.boundary_list[i][0]) +
            self.criterion(torch.abs(self.config.boundary_list[i][1] - y[:, :, i]), self.config.boundary_list[i][1] - y[:, :, i]) for i in range(self.config.prob_dim)]))
        #(self.criterion(torch.abs(y[:, :, 0] - 0), y[:, :, 0] - 0) + self.criterion(
            # torch.abs(0.65 - y[:, :, 0]), 0.65 - y[:, :, 0]) + self.criterion(torch.abs(y[:, :, 1] - 1.2),
            #                                                                   y[:, :, 1] - 1.2) + self.criterion(
            # torch.abs(4.0 - y[:, :, 1]), 4.0 - y[:, :, 1]))
        # loss4 = (1.0 if self.config.penalty else 0.0) * sum([penalty_func(torch.var(y[0, :, i])) for i in range(self.config.prob_dim)])
        # y_norm = torch.zeros(self.config.prob_dim).to(self.config.device)
        # for i in range(self.config.prob_dim):
        #     y_norm[i] = torch.var(
        #         (y[0, :, i] - torch.min(y[0, :, i])) / (torch.max(y[0, :, i]) - torch.min(y[0, :, i])))
        # loss4 = (1.0 if self.config.penalty else 0) * torch.mean(penalty_func(y_norm))
        # loss4 = self.criterion(1 / u_0, pt_all_zeros_3)
        # loss5 = self.criterion(torch.abs(u_0 - v_0), u_0 - v_0)

        loss = loss1 + loss2 + loss3
        loss_list = np.asarray([loss1.item(), loss2.item(), loss3.item()])
        return loss, loss_list

    def plot_activation_weights(self):
        # self.activation_weights_record
        activation_weights_save_path = "{}/activation_weights.png".format(self.train_save_path_folder)
        m = MultiSubplotDraw(row=2, col=2, fig_size=(16, 12), tight_layout_flag=True, show_flag=False, save_flag=True, save_path=activation_weights_save_path)
        activation_n = self.activation_weights_record.shape[2]
        for i in range(4):
            m.add_subplot(
                y_lists=[self.activation_weights_record[i, :, activation_id].flatten() for activation_id in range(activation_n)],
                x_list=range(1, self.config.args.iteration + 1),
                color_list=self.default_colors_10[:activation_n],
                legend_list=self.activate_block0.activate_list,
                line_style_list=["solid"] * activation_n,
                fig_title="activation block {}".format(i))
        m.draw()
        myprint("initial: \n{}".format(self.activation_weights_record[:, 0, :]), self.config.args.log_path)
        myprint("end: \n{}".format(self.activation_weights_record[:, -1, :]), self.config.args.log_path)


    def train_model(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.args.initial_lr)  # weight_decay=1e-4
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1 / (e / 1000 + 1))
        assert self.config.scheduler in ["cosine", "decade", "decade_pp", "fixed", "step"]
        if self.config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.args.iteration)
        elif self.config.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        elif self.config.scheduler == "decade":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1 / (e / 1000 + 1))
        elif self.config.scheduler == "decade_pp":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1 / (e / 10000 + 1))
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1.0)
        self.train()
        # assert mod in ["train", "test"]
        # if mod == "train":
        #     self.t_torch =

        start_time = time.time()
        start_time_0 = start_time
        loss_record = []
        # real_loss_mse_record = []
        real_loss_nmse_record_train = []
        real_loss_nmse_record_test = []
        time_record = []

        adaptive_weights_record_0 = []
        adaptive_weights_record_1 = []
        adaptive_weights_record_2 = []
        adaptive_weights_record_3 = []

        for epoch in range(1, self.config.args.iteration + 1):
            self.train()
            loss = 0.0
            train_nmse_loss = 0.0
            loss_list = np.zeros(self.config.loss_part_n)
            y_pred_train = None
            for x, _ in self.train_loader:
                x = x.to(self.config.device).reshape(1, -1, 1)
                x_grid = x.view(1, -1, 1, 1, 1).repeat(1, 1, self.config.params.N, self.config.params.M, 1)
                # print("x shape: {} starting: {}".format(x.shape, x[:2]))
                optimizer.zero_grad()
                y_pred_train = self.forward(x_grid)
                loss_tmp, loss_list_tmp = self.loss(torch.tensor(self.config.x_train, dtype=torch.float32).to(self.config.device), y_pred_train, epoch)
                _, real_loss_nmse_train_tmp = self.real_loss(
                    y=y_pred_train[0],
                    y_truth=torch.tensor(self.config.y_train[:, :]).to(self.config.device),
                )
                loss += loss_tmp.item()
                train_nmse_loss += real_loss_nmse_train_tmp.item()
                loss_list += loss_list_tmp
                loss_tmp.backward()
                optimizer.step()


            loss_record.append(loss)
            # real_loss_mse, real_loss_nmse = self.real_loss(y_pred)
            # real_loss_mse_record.append(real_loss_mse.item())
            real_loss_nmse_record_train.append(train_nmse_loss)

            test_nmse_loss = 0.0
            with torch.no_grad():
                for x, y in self.test_loader:
                    x = x.to(self.config.device).reshape(1, -1, 1)
                    x_grid = x.view(1, -1, 1, 1, 1).repeat(1, 1, self.config.params.N, self.config.params.M, 1)
                    y = y.to(self.config.device).reshape(-1, self.config.params.N, self.config.params.M, self.config.prob_dim)

                    y_pred_test = self.forward(x_grid)
                    # print(y_pred_test[0])
                    # print(self.config.y_test[:, :])
                    _, real_loss_nmse_test_tmp = self.real_loss(
                        y=y_pred_test[0],
                        y_truth=torch.tensor(self.config.y_test[:, :]).to(self.config.device),
                    )
                    test_nmse_loss += real_loss_nmse_test_tmp.item()
            real_loss_nmse_record_test.append(test_nmse_loss)

            # if "adaptive" in self.config.activation:
            #     if "Turing" in self.config.model_name:
            #         # myprint("Turing branch", self.config.args.log_path)
            #         adaptive_weights_record_0.append(list(self.f_model.activate_block1.activate_weights.cpu().detach().numpy()))
            #         adaptive_weights_record_1.append(list(self.f_model.activate_block2.activate_weights.cpu().detach().numpy()))
            #         adaptive_weights_record_2.append(list(self.f_model.activate_block3.activate_weights.cpu().detach().numpy()))
            #         adaptive_weights_record_3.append(list(self.f_model.activate_block4.activate_weights.cpu().detach().numpy()))
            #     else:
            #         adaptive_weights_record_0.append(list(self.activate_block0.activate_weights.cpu().detach().numpy()))
            #         adaptive_weights_record_1.append(list(self.activate_block1.activate_weights.cpu().detach().numpy()))
            #         adaptive_weights_record_2.append(list(self.activate_block2.activate_weights.cpu().detach().numpy()))
            #         adaptive_weights_record_3.append(list(self.activate_block3.activate_weights.cpu().detach().numpy()))
            if "adaptive" in self.config.activation:
                adaptive_weights_record_0.append(list(self.activate_block0.activate_weights.cpu().detach().numpy()))
                adaptive_weights_record_1.append(list(self.activate_block1.activate_weights.cpu().detach().numpy()))
                adaptive_weights_record_2.append(list(self.activate_block2.activate_weights.cpu().detach().numpy()))
                adaptive_weights_record_3.append(list(self.activate_block3.activate_weights.cpu().detach().numpy()))
            # torch.autograd.set_detect_anomaly(True)
            # loss.backward(retain_graph=True)  # retain_graph=True
            # loss.backward()
            # optimizer.step()
            scheduler.step()

            now_time = time.time()
            time_record.append(now_time - start_time_0)

            if epoch % self.config.args.epoch_step == 0 or epoch == self.config.args.iteration:
                loss_print_part = " ".join(
                    ["Loss_{0:d}:{1:.12f}".format(i + 1, loss_part) for i, loss_part in enumerate(loss_list)])
                myprint(
                    "Epoch [{0:05d}/{1:05d}] Loss:{2:.12f} {3} NMSE-Loss(train): {4:.12f} NMSE-Loss(test): {5:.12f} Lr:{6:.12f} Time:{7:.6f}s ({8:.2f}min in total, {9:.2f}min remains)".format(
                        epoch, self.config.args.iteration, loss, loss_print_part, train_nmse_loss, test_nmse_loss,
                        optimizer.param_groups[0]["lr"], now_time - start_time, (now_time - start_time_0) / 60.0,
                        (now_time - start_time_0) / 60.0 / epoch * (self.config.args.iteration - epoch)), self.config.args.log_path)
                start_time = now_time

                if epoch % self.config.args.test_step == 0:
                    self.y_tmp_train = y_pred_train
                    self.y_tmp_test = y_pred_test
                    self.epoch_tmp = epoch
                    self.loss_record_tmp = loss_record
                    self.real_loss_nmse_record_test_tmp = real_loss_nmse_record_test
                    self.real_loss_nmse_record_train_tmp = real_loss_nmse_record_train
                    self.time_record_tmp = time_record

                    self.activation_weights_record = np.asarray([adaptive_weights_record_0, adaptive_weights_record_1, adaptive_weights_record_2, adaptive_weights_record_3])
                    self.draw_model()
                    # save_path_loss = "{}/{}_{}_loss.npy".format(self.train_save_path_folder, self.config.model_name, self.time_string)
                    # np.save(save_path_loss, np.asarray(loss_record))

                    myprint("saving training info ...", self.config.args.log_path)
                    train_info = {
                        "model_name": self.config.model_name,
                        "seed": self.config.seed,
                        "prob_dim": self.config.prob_dim,
                        "activation": self.config.activation,
                        "cyclic": self.config.cyclic,
                        "stable": self.config.stable,
                        "derivative": self.config.derivative,
                        "loss_average_length": self.config.loss_average_length,
                        "epoch": self.config.args.iteration,
                        "epoch_stop": self.epoch_tmp,
                        "initial_lr": self.config.args.initial_lr,
                        "loss_length": len(loss_record),
                        "loss": np.asarray(loss_record),
                        "real_loss_nmse_test": np.asarray(real_loss_nmse_record_test),
                        "real_loss_nmse_train": np.asarray(real_loss_nmse_record_train),
                        "time": np.asarray(time_record),
                        "y_predict_train": y_pred_train[0, :, :].cpu().detach().numpy(),
                        "y_predict_test": y_pred_test[0, :, :].cpu().detach().numpy(),
                        "truth_all": self.config.truth_all,
                        # "truth_train": self.config.truth_train,
                        # "truth_test": self.config.truth_test,
                        "x_all": np.asarray(self.config.x_all),
                        "x_train": np.asarray(self.config.x_train),
                        "x_test": np.asarray(self.config.x_test),
                        "y_all": np.asarray(self.config.y_all),
                        "y_train": np.asarray(self.config.y_train),
                        "y_test": np.asarray(self.config.y_test),
                        "y_train_shape": self.config.y_train.shape,
                        # "config": self.config,
                        "time_string": self.time_string,
                        # "weights_raw": np.asarray([
                        #     self.activate_block0.activate_weights_raw.cpu().detach().numpy(),
                        #     self.activate_block1.activate_weights_raw.cpu().detach().numpy(),
                        #     self.activate_block2.activate_weights_raw.cpu().detach().numpy(),
                        #     self.activate_block3.activate_weights_raw.cpu().detach().numpy(),
                        # ]),
                        # "weights": np.asarray([
                        #     self.activate_block0.activate_weights.cpu().detach().numpy(),
                        #     self.activate_block1.activate_weights.cpu().detach().numpy(),
                        #     self.activate_block2.activate_weights.cpu().detach().numpy(),
                        #     self.activate_block3.activate_weights.cpu().detach().numpy(),
                        # ]),
                        # "sin_weight": np.asarray([
                        #     self.activate_block0.activates[0].omega.cpu().detach().numpy(),
                        #     self.activate_block1.activates[0].omega.cpu().detach().numpy(),
                        #     self.activate_block2.activates[0].omega.cpu().detach().numpy(),
                        #     self.activate_block3.activates[0].omega.cpu().detach().numpy(),
                        # ]),
                        "activation_weights_record": self.activation_weights_record,
                    }
                    train_info_path_loss = "{}/{}_{}_info.npy".format(self.train_save_path_folder,
                                                                      self.config.model_name, self.time_string)
                    model_save_path = "{}/{}_{}_last.pt".format(self.train_save_path_folder,
                                                                      self.config.model_name, self.time_string)
                    with open(train_info_path_loss, "wb") as f:
                        pickle.dump(train_info, f)
                    torch.save({
                        "model_state_dict": self.state_dict(),
                        "info": train_info,
                    }, model_save_path)
                    if "adaptive" in self.config.activation:
                        self.plot_activation_weights()

                    if epoch == self.config.args.iteration or self.early_stop():
                        # myprint(str(train_info), self.config.args.log_path)
                        self.write_finish_log()

                        myprint("figure_save_path_folder: {}".format(self.figure_save_path_folder), self.config.args.log_path)
                        myprint("train_save_path_folder: {}".format(self.train_save_path_folder), self.config.args.log_path)
                        myprint("Finished.", self.config.args.log_path)
                        break

                    # myprint(str(train_info), self.config.args.log_path)

    def draw_model(self):
        if self.config.skip_draw_flag:
            myprint("(Skipped drawing)", self.config.args.log_path)
            return

        if "turing" in self.config.model_name.lower():
            y_draw_train = self.y_tmp_train[0].cpu().detach().numpy()
            y_draw_test = self.y_tmp_test[0].cpu().detach().numpy()
            x_draw_train = self.config.x_train
            x_draw_test = self.config.x_test
            y_draw_truth_train = self.config.y_train
            y_draw_truth_test = self.config.y_test
            save_path_train = "{}/{}_{}_epoch={}_train.png".format(self.figure_save_path_folder, self.config.model_name,
                                                                   self.time_string, self.epoch_tmp)
            save_path_test = "{}/{}_{}_epoch={}_test.png".format(self.figure_save_path_folder, self.config.model_name,
                                                                 self.time_string, self.epoch_tmp)
            save_path_truth_train = "{}/{}_{}_truth_train.png".format(self.figure_save_path_folder, self.config.model_name,
                                                                 self.time_string)
            save_path_truth_test = "{}/{}_{}_truth_test.png".format(self.figure_save_path_folder, self.config.model_name,
                                                                 self.time_string)
            assert "1d" in self.config.model_name.lower() or "2d" in self.config.model_name.lower()
            if not os.path.exists(save_path_truth_train) or not os.path.exists(save_path_truth_test):
                myprint("Generating truth figure...", self.config.args.log_path)
                if "2d" in self.config.model_name.lower():
                    y_predict_train = y_draw_truth_train[-1]
                    y_predict_test = y_draw_truth_test[-1]
                    # y_truth_train = y_draw_truth_train[-1]
                    # y_truth_test = y_draw_truth_test[-1]

                    u_last_train = y_predict_train[:, :, 0]
                    u_last_test = y_predict_test[:, :, 0]
                    v_last_train = y_predict_train[:, :, 1]
                    v_last_test = y_predict_test[:, :, 1]

                    u_max_train = u_last_train.max()
                    u_max_test = u_last_test.max()
                    u_min_train = u_last_train.min()
                    u_min_test = u_last_test.min()
                    v_max_train = v_last_train.max()
                    v_max_test = v_last_test.max()
                    v_min_train = v_last_train.min()
                    v_min_test = v_last_train.min()
                else:
                    y_predict_train = y_draw_truth_train
                    y_predict_test = y_draw_truth_test
                    # y_truth_train = y_draw_truth_train
                    # y_truth_test = y_draw_truth_test

                    u_last_train = y_predict_train[:, :, 0, 0]
                    u_last_test = y_predict_test[:, :, 0, 0]
                    v_last_train = y_predict_train[:, :, 0, 1]
                    v_last_test = y_predict_test[:, :, 0, 1]

                    u_max_train = y_predict_train[-1, :, :, 0].max()
                    u_max_test = y_predict_test[-1, :, :, 0].max()
                    u_min_train = y_predict_train[-1, :, :, 0].min()
                    u_min_test = y_predict_test[-1, :, :, 0].min()
                    v_max_train = y_predict_train[-1, :, :, 1].max()
                    v_max_test = y_predict_test[-1, :, :, 1].max()
                    v_min_train = y_predict_train[-1, :, :, 1].min()
                    v_min_test = y_predict_test[-1, :, :, 1].min()
                m = MultiSubplotDraw(row=1, col=2, fig_size=(12, 6), tight_layout_flag=True, show_flag=False,
                                     save_flag=True, save_path=save_path_truth_train, save_dpi=200)
                m.add_subplot_turing(
                    matrix=u_last_train,
                    v_max=u_max_train,  # u_last_true.max(),
                    v_min=u_min_train,  # u_last_true.min()
                    fig_title_size=30,
                    number_label_size=30,
                    colorbar=False,
                    fig_title="$\hat{U}$",
                    x_ticks_set_flag=True,
                    y_ticks_set_flag=True,
                    x_ticks=range(0, u_last_train.shape[1], 5) if "2d" in self.config.model_name.lower() else range(0,
                                                                                                                    u_last_train.shape[
                                                                                                                        1],
                                                                                                                    20),
                    y_ticks=range(0, u_last_train.shape[0], 5) if "2d" in self.config.model_name.lower() else range(0,
                                                                                                                    self.config.T_N_train,
                                                                                                                    int(self.config.T_N_train / self.config.T * 2)),
                    y_ticklabels=None if "2d" in self.config.model_name.lower() else range(0, self.config.T, 2),
                    y_label="Y" if "2d" in self.config.model_name.lower() else "Time",
                    invert=True if "2d" in self.config.model_name.lower() else False,
                    # y_label_rotate=0,
                    # fig_title="{}_{}_U_pred_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp)
                )
                m.add_subplot_turing(
                    matrix=v_last_train,
                    v_max=v_max_train,  # v_last_true.max()
                    v_min=v_min_train,  # v_last_true.min()
                    fig_title_size=30,
                    number_label_size=30,
                    colorbar=False,
                    fig_title="$\hat{V}$",
                    x_ticks_set_flag=True,
                    y_ticks_set_flag=True,
                    x_ticks=range(0, u_last_train.shape[1], 5) if "2d" in self.config.model_name.lower() else range(0,
                                                                                                                    u_last_train.shape[
                                                                                                                        1],
                                                                                                                    20),
                    y_ticks=range(0, u_last_train.shape[0], 5) if "2d" in self.config.model_name.lower() else range(0,
                                                                                                                    self.config.T_N_train,
                                                                                                                    int(self.config.T_N_train / self.config.T * 2)),
                    y_ticklabels=None if "2d" in self.config.model_name.lower() else range(0, self.config.T, 2),
                    y_label="Y" if "2d" in self.config.model_name.lower() else "Time",
                    invert=True if "2d" in self.config.model_name.lower() else False,
                    # y_label_rotate=0,
                    # fig_title="{}_{}_V_pred_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp)
                )
                m.draw()

                save_path_truth_train_raw_u = save_path_truth_train.replace(".png", "_u.pkl")
                save_path_truth_train_raw_v = save_path_truth_train.replace(".png", "_v.pkl")

                with open(save_path_truth_train_raw_u, "wb") as f:
                    pickle.dump(u_last_train, f)
                with open(save_path_truth_train_raw_v, "wb") as f:
                    pickle.dump(v_last_train, f)



                m = MultiSubplotDraw(row=1, col=2, fig_size=(12, 6), tight_layout_flag=True, show_flag=False,
                                     save_flag=True, save_path=save_path_truth_test, save_dpi=200)
                m.add_subplot_turing(
                    matrix=u_last_test,
                    v_max=u_max_test,  # u_last_true.max(),
                    v_min=u_min_test,  # u_last_true.min()
                    fig_title_size=30,
                    number_label_size=30,
                    colorbar=False,
                    fig_title="$\hat{U}$",
                    x_ticks_set_flag=True,
                    y_ticks_set_flag=True,
                    x_ticks=range(0, u_last_test.shape[1], 5) if "2d" in self.config.model_name.lower() else range(0,
                                                                                                                   u_last_test.shape[
                                                                                                                       1],
                                                                                                                   20),
                    y_ticks=range(0, u_last_test.shape[0], 5) if "2d" in self.config.model_name.lower() else range(0,
                                                                                                                   self.config.T_N_test,
                                                                                                                   int(self.config.T_N_test / self.config.T * 2)),
                    y_ticklabels=None if "2d" in self.config.model_name.lower() else range(0, self.config.T, 2),
                    y_label="Y" if "2d" in self.config.model_name.lower() else "Time",
                    invert=True if "2d" in self.config.model_name.lower() else False,
                    # y_label_rotate=0,
                    # fig_title="{}_{}_U_pred_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp)
                )
                m.add_subplot_turing(
                    matrix=v_last_test,
                    v_max=v_max_test,  # v_last_true.max()
                    v_min=v_min_test,  # v_last_true.min()
                    fig_title_size=30,
                    number_label_size=30,
                    colorbar=False,
                    fig_title="$\hat{V}$",
                    x_ticks_set_flag=True,
                    y_ticks_set_flag=True,
                    x_ticks=range(0, u_last_test.shape[1], 5) if "2d" in self.config.model_name.lower() else range(0,
                                                                                                                   u_last_test.shape[
                                                                                                                       1],
                                                                                                                   20),
                    y_ticks=range(0, u_last_test.shape[0], 5) if "2d" in self.config.model_name.lower() else range(0,
                                                                                                                   self.config.T_N_test,
                                                                                                                   int(self.config.T_N_test / self.config.T * 2)),
                    y_ticklabels=None if "2d" in self.config.model_name.lower() else range(0, self.config.T, 2),
                    y_label="Y" if "2d" in self.config.model_name.lower() else "Time",
                    invert=True if "2d" in self.config.model_name.lower() else False,
                    # y_label_rotate=0,
                    # fig_title="{}_{}_V_pred_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp)
                )
                m.draw()

                save_path_truth_test_raw_u = save_path_truth_test.replace(".png", "_u.pkl")
                save_path_truth_test_raw_v = save_path_truth_test.replace(".png", "_v.pkl")

                with open(save_path_truth_test_raw_u, "wb") as f:
                    pickle.dump(u_last_test, f)
                with open(save_path_truth_test_raw_v, "wb") as f:
                    pickle.dump(v_last_test, f)

            if "2d" in self.config.model_name.lower():
                y_predict_train = y_draw_train[-1]
                y_predict_test = y_draw_test[-1]
                # y_truth_train = y_draw_truth_train[-1]
                # y_truth_test = y_draw_truth_test[-1]

                u_last_train = y_predict_train[:, :, 0]
                u_last_test = y_predict_test[:, :, 0]
                v_last_train = y_predict_train[:, :, 1]
                v_last_test = y_predict_test[:, :, 1]

                u_max_train = u_last_train.max()
                u_max_test = u_last_test.max()
                u_min_train = u_last_train.min()
                u_min_test = u_last_test.min()
                v_max_train = v_last_train.max()
                v_max_test = v_last_test.max()
                v_min_train = v_last_train.min()
                v_min_test = v_last_train.min()
            else:
                y_predict_train = y_draw_train
                y_predict_test = y_draw_test
                # y_truth_train = y_draw_truth_train
                # y_truth_test = y_draw_truth_test

                u_last_train = y_predict_train[:, :, 0, 0]
                u_last_test = y_predict_test[:, :, 0, 0]
                v_last_train = y_predict_train[:, :, 0, 1]
                v_last_test = y_predict_test[:, :, 0, 1]

                u_max_train = y_predict_train[-1, :, :, 0].max()
                u_max_test = y_predict_test[-1, :, :, 0].max()
                u_min_train = y_predict_train[-1, :, :, 0].min()
                u_min_test = y_predict_test[-1, :, :, 0].min()
                v_max_train = y_predict_train[-1, :, :, 1].max()
                v_max_test = y_predict_test[-1, :, :, 1].max()
                v_min_train = y_predict_train[-1, :, :, 1].min()
                v_min_test = y_predict_test[-1, :, :, 1].min()
            m = MultiSubplotDraw(row=1, col=2, fig_size=(12, 6), tight_layout_flag=True, show_flag=False,
                                 save_flag=True, save_path=save_path_train, save_dpi=200)
            m.add_subplot_turing(
                matrix=u_last_train,
                v_max=u_max_train,  # u_last_true.max(),
                v_min=u_min_train,  # u_last_true.min()
                fig_title_size=30,
                number_label_size=30,
                colorbar=False,
                fig_title="$\hat{U}$",
                x_ticks_set_flag=True,
                y_ticks_set_flag=True,
                x_ticks=range(0, u_last_train.shape[1], 5) if "2d" in self.config.model_name.lower() else range(0, u_last_train.shape[1], 20),
                y_ticks=range(0, u_last_train.shape[0], 5) if "2d" in self.config.model_name.lower() else range(0, self.config.T_N_train, int(self.config.T_N_train / self.config.T * 2)),
                y_ticklabels=None if "2d" in self.config.model_name.lower() else range(0,  self.config.T, 2),
                y_label="Y" if "2d" in self.config.model_name.lower() else "Time",
                invert=True if "2d" in self.config.model_name.lower() else False,
                # y_label_rotate=0,
                # fig_title="{}_{}_U_pred_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp)
            )
            m.add_subplot_turing(
                matrix=v_last_train,
                v_max=v_max_train,  # v_last_true.max()
                v_min=v_min_train,  # v_last_true.min()
                fig_title_size=30,
                number_label_size=30,
                colorbar=False,
                fig_title="$\hat{V}$",
                x_ticks_set_flag=True,
                y_ticks_set_flag=True,
                x_ticks=range(0, u_last_train.shape[1], 5) if "2d" in self.config.model_name.lower() else range(0, u_last_train.shape[1], 20),
                y_ticks=range(0, u_last_train.shape[0], 5) if "2d" in self.config.model_name.lower() else range(0, self.config.T_N_train, int(self.config.T_N_train / self.config.T * 2)),
                y_ticklabels=None if "2d" in self.config.model_name.lower() else range(0,  self.config.T, 2),
                y_label="Y" if "2d" in self.config.model_name.lower() else "Time",
                invert=True if "2d" in self.config.model_name.lower() else False,
                # y_label_rotate=0,
                # fig_title="{}_{}_V_pred_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp)
            )
            m.draw()

            save_path_train_raw_u = save_path_train.replace(".png", "_u.pkl")
            save_path_train_raw_v = save_path_train.replace(".png", "_v.pkl")

            with open(save_path_train_raw_u, "wb") as f:
                pickle.dump(u_last_train, f)
            with open(save_path_train_raw_v, "wb") as f:
                pickle.dump(v_last_train, f)

            m = MultiSubplotDraw(row=1, col=2, fig_size=(12, 6), tight_layout_flag=True, show_flag=False,
                                 save_flag=True, save_path=save_path_test, save_dpi=200)
            m.add_subplot_turing(
                matrix=u_last_test,
                v_max=u_max_test,  # u_last_true.max(),
                v_min=u_min_test,  # u_last_true.min()
                fig_title_size=30,
                number_label_size=30,
                colorbar=False,
                fig_title="$\hat{U}$",
                x_ticks_set_flag=True,
                y_ticks_set_flag=True,
                x_ticks=range(0, u_last_test.shape[1], 5) if "2d" in self.config.model_name.lower() else range(0,
                                                                                                                u_last_test.shape[
                                                                                                                    1],
                                                                                                                20),
                y_ticks=range(0, u_last_test.shape[0], 5) if "2d" in self.config.model_name.lower() else range(0,
                                                                                                                self.config.T_N_test,
                                                                                                                int(self.config.T_N_test / self.config.T * 2)),
                y_ticklabels=None if "2d" in self.config.model_name.lower() else range(0, self.config.T, 2),
                y_label="Y" if "2d" in self.config.model_name.lower() else "Time",
                invert=True if "2d" in self.config.model_name.lower() else False,
                # y_label_rotate=0,
                # fig_title="{}_{}_U_pred_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp)
            )
            m.add_subplot_turing(
                matrix=v_last_test,
                v_max=v_max_test,  # v_last_true.max()
                v_min=v_min_test,  # v_last_true.min()
                fig_title_size=30,
                number_label_size=30,
                colorbar=False,
                fig_title="$\hat{V}$",
                x_ticks_set_flag=True,
                y_ticks_set_flag=True,
                x_ticks=range(0, u_last_test.shape[1], 5) if "2d" in self.config.model_name.lower() else range(0,
                                                                                                                u_last_test.shape[
                                                                                                                    1],
                                                                                                                20),
                y_ticks=range(0, u_last_test.shape[0], 5) if "2d" in self.config.model_name.lower() else range(0,
                                                                                                                self.config.T_N_test,
                                                                                                                int(self.config.T_N_test / self.config.T * 2)),
                y_ticklabels=None if "2d" in self.config.model_name.lower() else range(0, self.config.T, 2),
                y_label="Y" if "2d" in self.config.model_name.lower() else "Time",
                invert=True if "2d" in self.config.model_name.lower() else False,
                # y_label_rotate=0,
                # fig_title="{}_{}_V_pred_epoch={}".format(self.config.model_name, self.time_string, self.epoch_tmp)
            )
            m.draw()

            save_path_test_raw_u = save_path_test.replace(".png", "_u.pkl")
            save_path_test_raw_v = save_path_test.replace(".png", "_v.pkl")

            with open(save_path_test_raw_u, "wb") as f:
                pickle.dump(u_last_test, f)
            with open(save_path_test_raw_v, "wb") as f:
                pickle.dump(v_last_test, f)

        # self.draw_loss_multi(self.loss_record_tmp, [1.0, 0.5, 0.25, 0.125])
        # self.draw_loss_multi(self.real_loss_nmse_record_train_tmp, [1.0, 0.5, 0.25, 0.125])
        # self.draw_loss_multi(self.real_loss_nmse_record_test_tmp, [1.0, 0.5, 0.25, 0.125])

    def write_finish_log(self):
        with open(os.path.join(self.config.args.main_path, "saves/record.txt"), "a") as f:
            f.write("{0},{1},{2},{3:.2f},{4},{5:.6f},{6:.12f},{7:.12f},{8:.12f},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},{23},{24},{25}\n".format(
                self.config.model_name,  # 0
                self.time_string,  # 1
                self.config.seed,  # 2
                self.time_record_tmp[-1] / 60.0,  # 3
                self.config.args.iteration,  # 4
                self.config.args.initial_lr,  # 5
                sum(self.loss_record_tmp[-self.config.loss_average_length:]) / self.config.loss_average_length,  # 6
                sum(self.real_loss_nmse_record_train_tmp[-self.config.loss_average_length:]) / self.config.loss_average_length,  # 7
                sum(self.real_loss_nmse_record_test_tmp[-self.config.loss_average_length:]) / self.config.loss_average_length,  # 8
                self.config.pinn,  # 9
                self.config.activation,  # 10
                self.config.stable,  # 11
                self.config.cyclic,  # 12
                self.config.derivative,  # 13
                self.config.boundary,  # 14
                self.config.loss_average_length,  # 15
                "{}-{}".format(self.config.args.iteration - self.config.loss_average_length, self.config.args.iteration),  # 16
                self.config.init_weights,  # 17
                self.config.init_weights_strategy,  # 18
                self.config.scheduler,  # 19
                self.config.T if self.config.T else None,  # 20
                self.config.T_N_train if self.config.T_N_train else None,  # 21
                self.config.T_N_test if self.config.T_N_test else None,  # 22
                self.activation_weights_record[0][-1][0] if self.config.activation in ["adaptive_5", "adaptive_6"] else None,  # 23
                self.activation_weights_record[0][-1][1] if self.config.activation in ["adaptive_5", "adaptive_6"] else None,  # 24
                self.activation_weights_record[0][-1][2] if self.config.activation in ["adaptive_5", "adaptive_6"] else None,  # 25
                self.activation_weights_record[0][-1][3] if self.config.activation in ["adaptive_5", "adaptive_6"] else None,  # 26
                self.activation_weights_record[0][-1][4] if self.config.activation in ["adaptive_5", "adaptive_6"] else None,  # 27
                self.activation_weights_record[0][-1][5] if self.config.activation in ["adaptive_6"] else None,  # 28
            ))

    def write_fail_log(self):
        with open(os.path.join(self.config.args.main_path, "saves/record.txt"), "a") as f:
            f.write("{0},{1},{2},{3:.2f},{4},{5:.6f},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18}\n".format(
                self.config.model_name,  # 0
                self.time_string,  # 1
                self.config.seed,  # 2
                0.0,  # self.time_record_tmp[-1] / 60.0,  # 3
                self.config.args.iteration,  # 4
                self.config.args.initial_lr,  # 5
                "fail",  # 6
                "fail",  # 7
                "fail",  # 8
                self.config.pinn,  # 9
                self.config.activation,  # 10
                self.config.stable,  # 11
                self.config.cyclic,  # 12
                self.config.derivative,  # 13
                self.config.boundary,  # 14
                self.config.loss_average_length,  # 15
                "{}-{}".format(self.config.args.iteration - self.config.loss_average_length, self.config.args.iteration),  # 16
                self.config.init_weights,
                self.config.init_weights_strategy,
            ))

    @staticmethod
    def draw_loss_multi(loss_list, last_rate_list):
        m = MultiSubplotDraw(row=1, col=len(last_rate_list), fig_size=(8 * len(last_rate_list), 6),
                             tight_layout_flag=True, show_flag=True, save_flag=False, save_path=None)
        for one_rate in last_rate_list:
            m.add_subplot(
                y_lists=[loss_list[-int(len(loss_list) * one_rate):]],
                x_list=range(len(loss_list) - int(len(loss_list) * one_rate) + 1, len(loss_list) + 1),
                color_list=["blue"],
                line_style_list=["solid"],
                fig_title="Loss - lastest ${}$% - epoch ${}$ to ${}$".format(int(100 * one_rate), len(loss_list) - int(
                    len(loss_list) * one_rate) + 1, len(loss_list)),
                fig_x_label="epoch",
                fig_y_label="loss",
            )
        m.draw()







if __name__ == "__main__":
    pass
