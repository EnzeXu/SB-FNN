SB-FNN: Systems-Biology Fourier-enhanced Neural Network
===


# Contents

* [1 Introduction](#1-introduction)
* [2 Citation](#2-citation)
* [3 Structure of the Repository](#3-structure-of-the-repository)
* [4 Getting Started](#4-getting-started)
  * [4.1 Preparations](#41-preparations)
  * [4.2 Install Packages](#42-install-packages)
  * [4.3 Run Training](#43-run-training)
  * [4.4 Apply Your ODE/PDE Equations to SB-FNN](#44-apply-your-odepde-equations-to-sb-fnn)
* [5 Questions](#5-questions)



# 1. Introduction
This study proposes a novel machine learning neural network, named Systems-Biology Fourier-enhanced Neural Network (SB-FNN).

# 2. Citation

If you use our code or datasets from `https://github.com/EnzeXu/ChemGNN_Dataset` for academic research, please cite the following paper:

Paper BibTeX:

```
@article{xxx2023xxxxxx,
  title        = {xxxxx},
  author       = {xxxxx},
  journal      = {arXiv preprint arXiv:xxxx.xxxx},
  year         = {2023}
}
```



# 3. Structure of the Repository


```
SB-FNN
┌── SBFNN/
├────── models/
├────────── _template.py
├────────── model_rep3.py
├────────── model_rep6.py
├────────── model_sir.py
├────────── model_asir.py
├────────── model_turing1d.py
├────────── model_turing2d.py
├────── utils/
├────────── __init__.py
├────────── _run.py
├────────── _utils.py
├── LICENSE
├── README.md
├── requirements.txt
└── run.py
```

- `ChemGNN/models/`: folder contains the model scripts
- `ChemGNN/utils/`: folder contains the utility scripts
- `LICENSE`: license file
- `README.md`: readme file
- `requirements.txt`: main dependent packages (please follow section 3.1 to install all dependent packages)
- `run.py`: training script



# 4. Getting Started

This project is developed using Python 3.9+ and is compatible with macOS, Linux, and Windows operating systems.

## 4.1 Preparations

(1) Clone the repository to your workspace.

```shell
~ $ git clone https://github.com/EnzeXu/SB-FNN.git
```

(2) Navigate into the repository.
```shell
~ $ cd SB-FNN
~/SB-FNN $
```

(3) Create a new virtual environment and activate it. In this case we use Virtualenv environment (Here we assume you have installed the `virtualenv` package using you source python script), you can use other virtual environments instead (like conda).

For macOS or Linux operating systems:
```shell
~/SB-FNN $ python -m venv ./venv/
~/SB-FNN $ source venv/bin/activate
(venv) ~/SB-FNN $ 
```

For Windows operating systems:

```shell
~/SB-FNN $ python -m venv ./venv/
~/SB-FNN $ .\venv\Scripts\activate
(venv) ~/SB-FNN $ 
```

You can use the command deactivate to exit the virtual environment at any time.

## 4.2 Install Packages

```shell
(venv) ~/SB-FNN $ pip install -r requirements.txt
```

## 4.3 Run Training

(1) Choose the model you want to run in `run.py`. Model `rep3` is set as default:
```python
...
# Please choose a model from ['rep3', 'rep6', 'sir', 'asir', 'turing1d', 'turing2d']
model_name = "rep3"
...
```

(2) Run Training. Please follow the following instructions or use command `python run.py --help` to parse all necessary arguments.

```python
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
```

Run training in your terminal.

```shell
(venv) ~/SB-FNN $ python run.py #
```

(2) Collect the auto-generated training results in `saves/figure/` and `saves/train/`.
```shell
(venv) ~/SB-FNN $ ls saves/train/MODEL_NAME_YYYYMMDD_HHMMSS_f/
(venv) ~/SB-FNN $ ls saves/figure/MODEL_NAME_YYYYMMDD_HHMMSS_f/
```

## 4.4 Apply Your ODE/PDE Equations to SB-FNN

Please follow six given models in `SBFNN/models` as examples to create your own model based on SB-FNN.

# 5. Questions

If you have any questions, please contact xezpku@gmail.com.


