from ray import tune
class MI_ESTIMATOR:
    def __init__ (self):
        self.config = {
            "use_tanh": False,
            "ff_residual_connection": False,
            "ff_activation": 'relu',
            # "ff_layers": tune.grid_search([1, 3, 5, 7]),
            # "hidden_dim": tune.grid_search([16, 32, 64, 128]),
            "ff_layers": 7,
            "hidden_dim": None,
            "ff_layer_norm": False,
            "optimize_mu": True,
            "cond_modes": 128,
            "marg_modes": 128,
            "init_std": 0.01,
            "average": "var",
            "cov_diagonal": "var",
            "cov_off_diagonal": "var",
            "start_epoch": 10,
            # "start_epoch": 0,
            "weight": 0.01,
            "lr": 0.001
            }
