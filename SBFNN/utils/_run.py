import argparse
import json
import torch
import numpy as np

from ._utils import myprint


class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        # print(type(o))
        if isinstance(o, torch.device):
            return str(o)
        if isinstance(o, list):
            return "list[length={}]".format(len(o))
        if isinstance(o, np.ndarray):
            return str(o.shape) + " ... " + str(o[:2]) + " ..."
        if isinstance(o, type):
            return str(o)
        if isinstance(o, torch.Tensor):
            return o.shape
        return json.JSONEncoder.default(self, o)


def run(config, fourier_model, pinn_model):
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="logs/test.txt", help="log path")
    parser.add_argument("--main_path", default="./", help="main_path")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--pinn", type=int, default=0, help="0=off 1=on")
    parser.add_argument("--activation",
                        choices=["gelu", "elu", "relu", "sin", "tanh", "softplus", "adaptive_6", "adaptive_3",
                                 "adaptive_5", "adaptive_2", "selu"],
                        type=str, default="gelu", help="activation plan")
    parser.add_argument("--cyclic", type=int, choices=[0, 1, 2], default=0, help="0=off 1=on")
    parser.add_argument("--stable", type=int, choices=[0, 1], default=0, help="0=off 1=on")
    parser.add_argument("--derivative", type=int, choices=[0, 1], default=0, help="0=off 1=on")
    parser.add_argument("--boundary", type=int, choices=[0, 1, 2], default=0, help="0=off 1=on")
    parser.add_argument("--skip_draw_flag", type=int, default=1, choices=[0, 1], help="0=off 1=on")
    parser.add_argument("--test", type=int, default=0, help="test mode will take a very small epoch for debugging")
    parser.add_argument("--init_lr", type=float, default=None, help="forced initial learning rate (it will take the initial_lr variable in Config if not set here)")
    parser.add_argument("--init_weights", type=str, default=None,
                        choices=[None, "avg", "gelu", "elu", "relu", "sin", "tanh", "softplus"], help="init_weights")
    parser.add_argument("--init_weights_strategy", type=str, default="trainable", help="init_weights_strategy")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "decade", "decade_pp", "fixed", "step"],
                        help="scheduler")
    opt = parser.parse_args()

    conf = config()
    conf.__dict__.update(vars(opt))
    conf.args.main_path = opt.main_path
    conf.args.log_path = opt.log_path
    if opt.init_lr:
        conf.args.initial_lr = opt.init_lr

    conf_print = conf.__dict__.copy()
    del conf_print["truth_all"]
    print(json.dumps(conf_print, indent=4, cls=CustomEncoder))
    if opt.test:
        conf.args.iteration = 2  # 10000
        conf.args.epoch_step = 1  # 1000  # 1000  # 1000
        conf.args.test_step = 2  # 10000
        if not opt.pinn:
            model = fourier_model(conf).to(conf.device)
        else:
            model = pinn_model(conf).to(conf.device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        myprint("model name: {} number of parameters: {}".format(model.config.model_name, num_params), conf.args.log_path)
        model.train_model()
        return

    if not opt.pinn:
        model = fourier_model(conf).to(conf.device)
    else:
        model = pinn_model(conf).to(conf.device)

    try:
        model.train_model()
    except Exception as e:
        print("[Error]", e)
        model.write_fail_log()