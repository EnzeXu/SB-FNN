from SBFNN import run

available_model_name_list = ['rep3', 'rep6', 'sir', 'asir', 'turing1d', 'turing2d']

# Please choose a model from ['rep3', 'rep6', 'sir', 'asir', 'turing1d', 'turing2d']
model_name = "rep3"

if model_name.lower() == "rep3":
    from SBFNN.models.model_rep3 import Config, FourierModel, PINNModel
elif model_name.lower() == "rep6":
    from SBFNN.models.model_rep6 import Config, FourierModel, PINNModel
elif model_name.lower() == "sir":
    from SBFNN.models.model_sir import Config, FourierModel, PINNModel
elif model_name.lower() == "asir":
    from SBFNN.models.model_asir import Config, FourierModel, PINNModel
elif model_name.lower() == "turing1d":
    from SBFNN.models.model_turing1d import Config, FourierModel, PINNModel
elif model_name.lower() == "turing2d":
    from SBFNN.models.model_turing2d import Config, FourierModel, PINNModel
else:
    print(f"In run.py, the available model name list is {available_model_name_list} ('{model_name}' is chosen). Please pick an available model from the list.")
    exit(-1)

run(Config, FourierModel, PINNModel)
