from models import wsl_model


def create_model(opts):
    if opts.model_type == 'model_wsl':
        model = wsl_model.CNNModel(opts)

    else:
        raise NotImplementedError

    return model
