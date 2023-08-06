from ._template import *


class Parameters:
    N = 25
    M = 25
    d1 = 1
    d2 = 40
    c1 = 0.1  # 0.1
    c2 = 0.9  # 0.9
    c_1 = 1
    c3 = 1
    l = 0.8
    w = 0.8


class TrainArgs:
    iteration = 5000  # 20000 -> 50000
    epoch_step = 1  # 1000
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
        self.model_name = "Turing2D_Fourier"
        self.curve_names = ["U", "V"]
        self.params = Parameters
        self.args = TrainArgs

        self.T = 1
        self.T_N_all = 2000
        self.T_N_train = 1000
        self.T_N_test = 1000

        self.T_before = 30
        # self.T_N_before = 10000
        self.T_unit_before = 2e-3
        self.T_N_before = int(self.T_before / self.T_unit_before)
        # self.y0 = np.asarray([9.0983668, 0.90143886, 0.15871683, 7.2439572, 2.85342356, 0.15983291])
        self.loss_part_n = 3
        # self.boundary_list = np.asarray([[0.0, 6.0], [0.0, 6.0], [0.0, 6.0]])
        self.boundary_list = np.asarray([[0.1, 6.0], [0.2, 1.5]])

        self.setup()

    def setup(self):
        assert self.T_N_all == self.T_N_train + self.T_N_test
        self.truth_path = "truth/{}_truth.pt".format(self.model_name)
        self.prob_dim = len(self.curve_names)
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
            self.y0 = data["y0"]
        else:
            print("Truth not found. Generating ...")
            # self.x_all = np.concatenate([np.asarray([0.0]), np.asarray(sample_lhs(0, self.T, self.T_N_all - 1))])
            self.x_train = np.concatenate([np.asarray([0.0]), np.asarray(sample_lhs(0, self.T, self.T_N_train - 1))])
            self.x_test = np.concatenate([np.asarray([0.0]), np.asarray(sample_lhs(0, self.T, self.T_N_test - 1))])
            # print("self.x_train", self.x_train)
            # print("self.x_test", self.x_test)
            # self.y_train = odeint(self.pend, self.y0, self.x_train)
            # self.y_test = odeint(self.pend, self.y0, self.x_test)
            y0_before = torch.rand([self.params.N, self.params.M, self.prob_dim]) + 2.0
            t_before = np.asarray([i * self.T_unit_before for i in range(self.T_N_before)])
            # print(y0_before.shape, t_before.shape)
            y_before = torchdiffeq.odeint(self.pend, y0_before, torch.tensor(t_before), method='euler')
            # self.draw_turing(y_before[-1])
            # self.y0 = (y_before[-1] + 0.1).cpu().numpy()
            self.noise_rate = 0.05
            noise = (np.random.rand(self.params.N, self.params.M, self.prob_dim) - 0.5) * self.noise_rate
            self.y0 = y_before[-1] + 0.1  # (y_before[-1] * (1.0 + noise) + 0.1).cpu().numpy()
            self.y_train = torchdiffeq.odeint(self.pend, torch.tensor(self.y0), torch.tensor(self.x_train), method='euler').cpu().numpy()
            self.y_test = torchdiffeq.odeint(self.pend, torch.tensor(self.y0), torch.tensor(self.x_test), method='euler').cpu().numpy()

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
                "y0": self.y0,
            }
            with open(self.truth_path, "wb") as f:
                pickle.dump(truth_dic, f)
        print("x_train: {} {}, ..., {}".format(self.x_train.shape, self.x_train[:5], self.x_train[-5:]))
        print("x_test: {} {}, ..., {}".format(self.x_test.shape, self.x_test[:5], self.x_test[-5:]))
        # print("y_train: {} {}, ..., {}".format(self.y_train.shape, self.y_train[:5], self.y_train[-5:]))
        # print("y_test: {} {}, ..., {}".format(self.y_test.shape, self.y_test[:5], self.y_test[-5:]))


        # self.x_train_torch = torch.tensor(self.x_train).reshape(1, -1, 1)
        # self.t_torch = torch.tensor(self.x_train, dtype=torch.float32).to(self.device)
        # self.x = torch.tensor(np.asarray([[[i * self.T_unit] * 1 for i in range(self.T_N)]]),
        #                       dtype=torch.float32).to(self.device)
        # self.truth = odeint(self.pend, self.y0, self.t)
        self.loss_average_length = int(0.1 * self.args.iteration)
    # def setup(self):
    #     assert self.T_N_all == self.T_N_train + self.T_N_test
    #     self.truth_path = "truth/{}_truth.pt".format(self.model_name)
    #
    #     self.prob_dim = len(self.curve_names)
    #     # self.x_train_torch = torch.tensor(self.x_train).reshape(1, -1, 1)
    #     # self.t_torch = torch.tensor(self.x_train, dtype=torch.float32).to(self.device)
    #     # self.x = torch.tensor(np.asarray([[[i * self.T_unit] * 1 for i in range(self.T_N)]]),
    #     #                       dtype=torch.float32).to(self.device)
    #     # self.truth = odeint(self.pend, self.y0, self.t)
    #     self.loss_average_length = int(0.1 * self.args.iteration)
    #     self.setup_seed(0)
    #     self.prob_dim = len(self.curve_names)
    #
    #     # np.save(truth_path, self.truth.cpu().detach().numpy())
    #
    #     # print("y0:")
    #     # # self.draw_turing(self.y0)
    #     # print("Truth:")
    #     # print("Truth U: max={0:.6f} min={1:.6f}".format(torch.max(self.truth_torch[:, :, :, 0]).item(),
    #     #                                                 torch.min(self.truth_torch[:, :, :, 0]).item()))
    #     # print("Truth V: max={0:.6f} min={1:.6f}".format(torch.max(self.truth_torch[:, :, :, 1]).item(),
    #     #                                                 torch.min(self.truth_torch[:, :, :, 1]).item()))
    #     if os.path.exists(self.truth_path):
    #         print("Truth exists. Loading ...")
    #         with open(self.truth_path, "rb") as f:
    #             data = pickle.load(f)
    #         self.x_all = data["x_all"]
    #         self.y_all = data["y_all"]
    #         self.x_train = data["x_train"]
    #         self.x_test = data["x_test"]
    #         self.y_train = data["y_train"]
    #         self.y_test = data["y_test"]
    #     else:
    #         print("Truth not found. Generating ...")
    #         self.y0_before = torch.rand([self.params.N, self.params.M, self.prob_dim]).to(self.device) + 2.0
    #         self.t_before = np.asarray([i * self.T_unit_before for i in range(self.T_N_before)])
    #         # self.t = np.asarray([i * self.T_unit_before for i in range(self.T_N_before)])
    #         # self.t_torch = torch.tensor(self.t, dtype=torch.float32).to(self.device)
    #         # x = torch.zeros([1, self.T_N, self.params.N, self.params.M, 1]).to(self.device)
    #         # self.x = FNO3d.get_grid(x.shape, x.device)
    #         self.noise_rate = 0.05
    #         truth_before = torchdiffeq.odeint(self.pend, self.y0_before.cpu(), torch.tensor(self.t_before),
    #                                           method='euler').to(self.device)
    #         noise = (torch.rand([self.params.N, self.params.M, self.prob_dim]).to(self.device) - 0.5) * self.noise_rate
    #         self.y0 = truth_before[
    #                       -1] + 0.12345  # torch.abs(truth_before[-1] * (1.0 + noise) + 0.1) # torch.abs(truth_before[-1] * (1.0 + noise) + 0.2)
    #         self.truth_torch = torchdiffeq.odeint(self.pend, self.y0.cpu(), torch.tensor(self.t), method='euler').to(
    #             self.device)
    #         # self.x_all = np.concatenate([np.asarray([0.0]), np.asarray(sample_lhs(0, self.T, self.T_N_all - 1))])
    #         self.x_all = np.linspace(0, self.T, self.T_N_all)
    #         self.y_all = odeint(self.pend, self.y0, self.x_all)
    #         self.truth_all = [[x_item, y_item] for x_item, y_item in zip(self.x_all, self.y_all)]
    #         truth_all_0 = self.truth_all[0: 1]
    #         truth_all_except_0 = self.truth_all[1:]
    #         random.shuffle(truth_all_except_0)
    #         truth_train = truth_all_0 + sorted(truth_all_except_0[:self.T_N_train - 1], key=lambda x: x[0])
    #         truth_test = sorted(truth_all_except_0[-self.T_N_test:], key=lambda x: x[0])
    #         self.x_train = np.asarray([item[0] for item in truth_train])
    #         self.x_test = np.asarray([item[0] for item in truth_test])
    #         self.y_train = np.asarray([item[1] for item in truth_train])
    #         self.y_test = np.asarray([item[1] for item in truth_test])
    #         truth_dic = {
    #             "x_all": self.x_all,
    #             "x_train": self.x_train,
    #             "x_test": self.x_test,
    #             "y_all": self.y_all,
    #             "y_train": self.y_train,
    #             "y_test": self.y_test,
    #         }
    #         with open(self.truth_path, "wb") as f:
    #             pickle.dump(truth_dic, f)
    #     print("x_train: {} {}, ..., {}".format(self.x_train.shape, self.x_train[:5], self.x_train[-5:]))
    #     print("x_test: {} {}, ..., {}".format(self.x_test.shape, self.x_test[:5], self.x_test[-5:]))
    #     print("y_train: {} {}, ..., {}".format(self.y_train.shape, self.y_train[:5], self.y_train[-5:]))
    #     print("y_test: {} {}, ..., {}".format(self.y_test.shape, self.y_test[:5], self.y_test[-5:]))
    #     # self.draw_turing(self.truth[-1])
    #     self.truth = self.truth_torch.cpu().detach().numpy()
    #     # self.draw_turing(self.truth_torch[-1])
    #     # turing_1d_all = self.truth_torch.reshape([-1, self.params.N, 2])
    #     # self.draw_turing_1d(turing_1d_all)
    #     self.loss_average_length = int(0.1 * self.args.iteration)

    def pend(self, t, y):
        shapes = y.shape
        reaction_part = torch.zeros([shapes[0], shapes[1], 2])
        reaction_part[:, :, 0] = self.params.c1 - self.params.c_1 * y[:, :, 0] + self.params.c3 * (y[:, :, 0] ** 2) * y[
                                                                                                                      :,
                                                                                                                      :,
                                                                                                                      1]
        reaction_part[:, :, 1] = self.params.c2 - self.params.c3 * (y[:, :, 0] ** 2) * y[:, :, 1]

        y_from_left = torch.roll(y, 1, 1)
        y_from_left[:, :1] = y[:, :1]
        y_from_right = torch.roll(y, -1, 1)
        y_from_right[:, -1:] = y[:, -1:]

        y_from_top = torch.roll(y, 1, 0)
        y_from_top[:1, :] = y[:1, :]
        y_from_bottom = torch.roll(y, -1, 0)
        y_from_bottom[-1:, :] = y[-1:, :]

        diffusion_part = torch.zeros([shapes[0], shapes[1], 2])
        diffusion_part[:, :, 0] = self.params.d1 * (
                ((y_from_left[:, :, 0] + y_from_right[:, :, 0] - y[:, :, 0] * 2) / (self.params.l ** 2)) + (
                (y_from_top[:, :, 0] + y_from_bottom[:, :, 0] - y[:, :, 0] * 2) / (self.params.w ** 2)))
        diffusion_part[:, :, 1] = self.params.d2 * (
                ((y_from_left[:, :, 1] + y_from_right[:, :, 1] - y[:, :, 1] * 2) / (self.params.l ** 2)) + (
                (y_from_top[:, :, 1] + y_from_bottom[:, :, 1] - y[:, :, 1] * 2) / (self.params.w ** 2)))
        return reaction_part + diffusion_part

    @staticmethod
    def draw_turing(map):
        # map: N * M * 2
        u = map[:, :, 0].cpu().detach().numpy()
        v = map[:, :, 1].cpu().detach().numpy()
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(121)
        im1 = ax1.imshow(u, cmap=plt.cm.jet, aspect='auto')
        ax1.set_title("u")
        cb1 = plt.colorbar(im1, shrink=1)

        ax2 = fig.add_subplot(122)
        im2 = ax2.imshow(v, cmap=plt.cm.jet, aspect='auto')
        ax2.set_title("v")
        cb2 = plt.colorbar(im2, shrink=1)
        plt.tight_layout()
        plt.show()


# def penalty_func(x):
#     return 1 * (- torch.tanh((x - 0.004) * 300) + 1)


class FourierModel(FourierModelTemplate3D):
    def __init__(self, config):
        super(FourierModel, self).__init__(config)
        self.truth_loss()

    def real_loss(self, y, y_truth):
        # # truth = torch.tensor(self.config.y_train[:, :]).to(self.config.device)
        # real_loss_mse = self.criterion(y, y_truth)
        # real_loss_nmse = torch.mean(self.criterion_non_reduce(y, y_truth) / (y_truth ** 2))
        # return real_loss_mse, real_loss_nmse
        truth = y_truth[:, :].to(self.config.device)
        # print("y", y.shape)
        # print("truth", truth.shape)
        real_loss_mse = self.criterion(y, truth)
        real_loss_nmse = torch.mean(self.criterion_non_reduce(y, truth) / (truth ** 2))
        return real_loss_mse, real_loss_nmse

    def ode_gradient(self, x, y):
        y = y[0]
        shapes = y.shape
        reaction_part = torch.zeros([shapes[0], shapes[1], shapes[2], 2]).to(self.config.device)
        reaction_part[:, :, :, 0] = self.config.params.c1 - self.config.params.c_1 * y[:, :, :,
                                                                                     0] + self.config.params.c3 * (
                                            y[:, :, :, 0] ** 2) * y[:, :, :, 1]
        reaction_part[:, :, :, 1] = self.config.params.c2 - self.config.params.c3 * (y[:, :, :, 0] ** 2) * y[:, :, :, 1]

        y_from_left = torch.roll(y, 1, 2)
        y_from_left[:, :, :1] = y[:, :, :1]
        y_from_right = torch.roll(y, -1, 2)
        y_from_right[:, :, -1:] = y[:, :, -1:]

        y_from_top = torch.roll(y, 1, 1)
        y_from_top[:, :1, :] = y[:, :1, :]
        y_from_bottom = torch.roll(y, -1, 1)
        y_from_bottom[:, -1:, :] = y[:, -1:, :]

        diffusion_part = torch.zeros([shapes[0], shapes[1], shapes[2], 2]).to(self.config.device)
        diffusion_part[:, :, :, 0] = self.config.params.d1 * (((y_from_left[:, :, :, 0] + y_from_right[:, :, :, 0] - y[
                                                                                                                     :,
                                                                                                                     :,
                                                                                                                     :,
                                                                                                                     0] * 2) / (
                                                                       self.config.params.l ** 2)) + ((y_from_top[:,
                                                                                                       :, :,
                                                                                                       0] + y_from_bottom[
                                                                                                            :, :, :,
                                                                                                            0] - y[
                                                                                                                 :,
                                                                                                                 :,
                                                                                                                 :,
                                                                                                                 0] * 2) / (
                                                                                                              self.config.params.w ** 2)))
        diffusion_part[:, :, :, 1] = self.config.params.d2 * (((y_from_left[:, :, :, 1] + y_from_right[:, :, :, 1] - y[
                                                                                                                     :,
                                                                                                                     :,
                                                                                                                     :,
                                                                                                                     1] * 2) / (
                                                                       self.config.params.l ** 2)) + ((y_from_top[:,
                                                                                                       :, :,
                                                                                                       1] + y_from_bottom[
                                                                                                            :, :, :,
                                                                                                            1] - y[
                                                                                                                 :,
                                                                                                                 :,
                                                                                                                 :,
                                                                                                                 1] * 2) / (
                                                                                                              self.config.params.w ** 2)))

        y_t_theory = reaction_part + diffusion_part

        y_t = torch.gradient(y, spacing=(x,), dim=0)[0]

        return y_t - y_t_theory

    def loss(self, x, y, iteration=-1):
        y0_pred = y[0, 0, :]
        y0_true = torch.tensor(self.config.y0, dtype=torch.float32).to(self.config.device)

        ode_n = self.ode_gradient(x, y)
        zeros_1D = torch.zeros([len(x)]).to(self.config.device)
        zeros_nD = torch.zeros([self.config.T_N_train, self.config.params.N, self.config.params.M, self.config.prob_dim]).to(
            self.config.device)
        # print("y0_pred.shape", y0_pred.shape)
        # print("y0_true.shape", y0_true.shape)
        # print("ode_n.shape", ode_n.shape)
        # print("zeros_nD.shape", zeros_nD.shape)
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

        # loss4 = (1.0 if self.config.cyclic else 0) * sum(
        #     [penalty_func(torch.var(y[0, :, i])) for i in range(self.config.prob_dim)])
        # loss4 = (1.0 if self.config.cyclic else 0) * sum(
        #     [penalty_func(torch.var(y[0, :, i])) for i in range(self.config.prob_dim)])

        loss = loss1 + loss2 + loss3
        loss_list = np.asarray([loss1.item(), loss2.item(), loss3.item()])
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

        self.sequences_u = nn.Sequential(*[block_turing() for _ in range(self.config.params.N * self.config.params.M)])
        self.sequences_v = nn.Sequential(*[block_turing() for _ in range(self.config.params.N * self.config.params.M)])

    def forward(self, x):
        shapes = x.shape
        # print(shapes)
        results_u = torch.zeros([shapes[0], shapes[1], shapes[2], shapes[3], 1]).to(self.config.device)
        results_v = torch.zeros([shapes[0], shapes[1], shapes[2], shapes[3], 1]).to(self.config.device)
        for n in range(self.config.params.N):
            for m in range(self.config.params.M):
                results_u[0, :, n, m, :] = self.sequences_u[n * self.config.params.M + m](x[0, :, n, m, :])
                results_v[0, :, n, m, :] = self.sequences_v[n * self.config.params.M + m](x[0, :, n, m, :])
        y = torch.cat((results_u, results_v), -1)
        return y


