from ._template import *


class Parameters:
    beta = 0.01
    gamma = 0.05
    N = 100.0


class TrainArgs:
    iteration = 50000  # 20000 -> 50000
    epoch_step = 100  # 1000
    test_step = epoch_step * 10
    initial_lr = 0.001
    main_path = "../.."
    log_path = None
    early_stop = False
    early_stop_period = test_step // 2
    early_stop_tolerance = 0.01


class Config(ConfigTemplate):
    def __init__(self):
        super(Config, self).__init__()
        self.model_name = "SIR_Fourier"
        self.curve_names = ["S", "I", "R"]
        self.params = Parameters
        self.args = TrainArgs

        self.T = 100
        self.T_N_all = 5500
        self.T_N_train = 5000
        self.T_N_test = 500
        self.y0 = np.asarray([50.0, 40.0, 10.0])
        self.loss_part_n = 4
        # self.boundary_list = np.asarray([[0.0, 6.0], [0.0, 6.0], [0.0, 6.0]])
        self.boundary_list = np.asarray([[0, 50.00], [0.63, 73.5], [10, 99.37]])

        self.setup()

    def pend(self, y, t):
        k = self.params
        dydt = np.asarray([
            - self.params.beta * y[0] * y[1],
            self.params.beta * y[0] * y[1] - self.params.gamma * y[1],
            self.params.gamma * y[1]
        ])
        return dydt


def penalty_func(x):
    return 1 * (- torch.tanh((x - 0.004) * 300) + 1)


class FourierModel(FourierModelTemplate):
    def __init__(self, config):
        super(FourierModel, self).__init__(config)
        self.truth_loss()

    def real_loss(self, y, y_truth):
        # truth = torch.tensor(self.config.y_train[:, :]).to(self.config.device)
        y, y_truth = y[:, 2], y_truth[:, 2]
        real_loss_mse = self.criterion(y, y_truth)
        real_loss_nmse = torch.mean(self.criterion_non_reduce(y, y_truth) / (y_truth ** 2))
        return real_loss_mse, real_loss_nmse

    def ode_gradient(self, x, y):
        k = self.config.params
        S = y[0, :, 0]
        I = y[0, :, 1]
        R = y[0, :, 2]
        S_t = torch.gradient(S, spacing=(x,))[0]
        I_t = torch.gradient(I, spacing=(x,))[0]
        R_t = torch.gradient(R, spacing=(x,))[0]
        f_S = S_t - (- self.config.params.beta * S * I)
        f_I = I_t - (self.config.params.beta * S * I - self.config.params.gamma * I)
        f_R = R_t - (self.config.params.gamma * I)
        return torch.cat((f_S.reshape([-1, 1]), f_I.reshape([-1, 1]), f_R.reshape([-1, 1])), 1)

    def loss(self, x, y, iteration=-1):
        y0_pred = y[0, 0, :]
        y0_true = torch.tensor(self.config.y0, dtype=torch.float32).to(self.config.device)

        ode_n = self.ode_gradient(x, y)
        zeros_1D = torch.zeros([len(x)]).to(self.config.device)
        zeros_nD = torch.zeros([len(x), self.config.prob_dim]).to(self.config.device)

        loss1 = self.criterion(y0_pred, y0_true)
        loss2 = 1.0 * (self.criterion(ode_n, zeros_nD))

        boundary_iteration = int(0.3 * self.config.args.iteration)  # 1.0 if self.config.boundary and iteration > boundary_iteration else 0.0
        loss3 = (1.0 if self.config.boundary and iteration > boundary_iteration else 0.0) * (sum([
            self.criterion(torch.abs(y[:, :, i] - self.config.boundary_list[i][0]),
                           y[:, :, i] - self.config.boundary_list[i][0]) +
            self.criterion(torch.abs(self.config.boundary_list[i][1] - y[:, :, i]),
                           self.config.boundary_list[i][1] - y[:, :, i]) for i in range(self.config.prob_dim)]))

        # y_norm = torch.zeros(self.config.prob_dim).to(self.config.device)
        # for i in range(self.config.prob_dim):
        #     y_norm[i] = torch.var(
        #         (y[0, :, i] - torch.min(y[0, :, i])) / (torch.max(y[0, :, i]) - torch.min(y[0, :, i])))
        # loss4 = (1.0 if self.config.cyclic else 0) * torch.mean(penalty_cyclic_func(y_norm))

        loss4 = (1.0 if self.config.cyclic else 0) * sum(
            [penalty_func(torch.var(y[0, :, i])) for i in range(self.config.prob_dim)])
        # loss4 = (1.0 if self.config.cyclic else 0) * sum(
        #     [penalty_func(torch.var(y[0, :, i])) for i in range(self.config.prob_dim)])

        loss = loss1 + loss2 + loss3 + loss4
        loss_list = np.asarray([loss1.item(), loss2.item(), loss3.item(), loss4.item()])
        return loss, loss_list


class PINNModel(FourierModel):
    def __init__(self, config):
        config.model_name = config.model_name.replace("Fourier", "PINN")
        super(PINNModel, self).__init__(config)

        del self.conv0
        del self.conv1
        del self.conv2
        del self.conv3

        del self.w0
        del self.w1
        del self.w2
        del self.w3

        self.fc1 = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

    def forward(self, x):
        x1_new = self.fc1(x)
        x2_new = self.fc2(x)
        x3_new = self.fc3(x)
        x = torch.cat((x1_new, x2_new, x3_new), -1)
        return x


