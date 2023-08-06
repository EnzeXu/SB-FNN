from ._template import *


class Parameters:
    beta = 10
    rho = 1e-6
    gamma = 1
    n = 3


class TrainArgs:
    iteration = 50000  # 20000 -> 50000
    epoch_step = 100  # 1000
    test_step = epoch_step * 10
    initial_lr = 0.005
    main_path = "../.."
    log_path = None
    early_stop = False
    early_stop_period = test_step // 2
    early_stop_tolerance = 0.01


class Config(ConfigTemplate):
    def __init__(self):
        super(Config, self).__init__()
        self.model_name = "REP6_Fourier"
        self.curve_names = ["m_lacI", "m_tetR", "m_cI", "p_cI", "p_lacI", "p_tetR"]
        self.params = Parameters
        self.args = TrainArgs

        self.T = 20#5#10#15#20
        self.T_N_all = 5500#1375#2750#4125#5500
        self.T_N_train = 5000#1250#2500#3750#5000
        self.T_N_test = 500#125#250#275#500
        self.y0 = np.asarray([9.0983668, 0.90143886, 0.15871683, 7.2439572, 2.85342356, 0.15983291])
        self.loss_part_n = 4
        # self.boundary_list = np.asarray([[0.0, 6.0], [0.0, 6.0], [0.0, 6.0]])
        self.boundary_list = np.asarray([[0.06, 9.57], [0.06, 9.57], [0.06, 9.57], [0.15, 8.93], [0.15, 8.93], [0.15, 8.93]])

        self.setup()

    def pend(self, y, t):
        k = self.params
        dydt = 1.0 * np.asarray([
            k.beta * (k.rho + 1 / (1 + y[5] ** k.n)) - y[0],
            k.beta * (k.rho + 1 / (1 + y[3] ** k.n)) - y[1],
            k.beta * (k.rho + 1 / (1 + y[4] ** k.n)) - y[2],
            k.gamma * (y[0] - y[3]),
            k.gamma * (y[1] - y[4]),
            k.gamma * (y[2] - y[5])
        ])
        return dydt


def penalty_func(x):
    return 0.5 * (- torch.tanh((x - 0.035) * 250) + 1)


class FourierModel(FourierModelTemplate):
    def __init__(self, config):
        super(FourierModel, self).__init__(config)
        self.truth_loss()

    def real_loss(self, y, y_truth):
        # truth = torch.tensor(self.config.y_train[:, :]).to(self.config.device)
        real_loss_mse = self.criterion(y, y_truth)
        real_loss_nmse = torch.mean(self.criterion_non_reduce(y, y_truth) / (y_truth ** 2))
        return real_loss_mse, real_loss_nmse

    def ode_gradient(self, x, y):
        k = self.config.params

        m_lacl = y[0, :, 0]
        m_tetR = y[0, :, 1]
        m_cl = y[0, :, 2]
        p_cl = y[0, :, 3]
        p_lacl = y[0, :, 4]
        p_tetR = y[0, :, 5]

        m_lacl_t = torch.gradient(m_lacl, spacing=(x,))[0]
        m_tetR_t = torch.gradient(m_tetR, spacing=(x,))[0]
        m_cl_t = torch.gradient(m_cl, spacing=(x,))[0]
        p_cl_t = torch.gradient(p_cl, spacing=(x,))[0]
        p_lacl_t = torch.gradient(p_lacl, spacing=(x,))[0]
        p_tetR_t = torch.gradient(p_tetR, spacing=(x,))[0]

        ratio = 1.0
        f_m_lacl = m_lacl_t - ratio * (k.beta * (k.rho + 1 / (1 + p_tetR ** k.n)) - m_lacl)
        f_m_tetR = m_tetR_t - ratio * (k.beta * (k.rho + 1 / (1 + p_cl ** k.n)) - m_tetR)
        f_m_cl = m_cl_t - ratio * (k.beta * (k.rho + 1 / (1 + p_lacl ** k.n)) - m_cl)
        f_p_cl = p_cl_t - ratio * (k.gamma * (m_lacl - p_cl))
        f_p_lacl = p_lacl_t - ratio * (k.gamma * (m_tetR - p_lacl))
        f_p_tetR = p_tetR_t - ratio * (k.gamma * (m_cl - p_tetR))

        return torch.cat((f_m_lacl.reshape([-1, 1]), f_m_tetR.reshape([-1, 1]), f_m_cl.reshape([-1, 1]),
                          f_p_cl.reshape([-1, 1]), f_p_lacl.reshape([-1, 1]), f_p_tetR.reshape([-1, 1])), 1)

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

        y_norm = torch.zeros(self.config.prob_dim).to(self.config.device)
        for i in range(self.config.prob_dim):
            y_norm[i] = torch.var(
                (y[0, :, i] - torch.min(y[0, :, i])) / (torch.max(y[0, :, i]) - torch.min(y[0, :, i])))

        loss4 = (1.0 if self.config.cyclic else 0) * torch.mean(penalty_func(y_norm))

        # print("y_norm:", y_norm)
        # print("penalty_func(y_norm):", penalty_func(y_norm))

        # loss4 = (1.0 if self.config.cyclic else 0) * sum(
        #     [penalty_func(torch.var(y[0, :, i])) for i in range(self.config.prob_dim)])
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

        self.fc4 = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        self.fc5 = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        self.fc6 = nn.Sequential(
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
        x4_new = self.fc4(x)
        x5_new = self.fc5(x)
        x6_new = self.fc6(x)
        x = torch.cat((x1_new, x2_new, x3_new, x4_new, x5_new, x6_new), -1)
        return x


