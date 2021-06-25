import copy
import logging
from collections import defaultdict

import numpy as np
import torch

from FedML.fedml_core.instances.client import Client
from FedML.fedml_api.standalone.fedavg.silo_fedavg import SiloFedAvg
from FedML.fedml_api.standalone.fedopt.optrepo import OptRepo


class SiloFedOpt(SiloFedAvg):
    def __init__(self, dataset, model_trainer, log_fn=logging.info, early_stopper=None, history_save_fn=None):
        """
        dataset: array like containing:
            local_trn_data_dict: dict with key: client_id, value: torch DataLoader
            local_tst_data_dict: dict with key: client_id, value: torch DataLoader
            local_val_data_dict: dict with key: client_id, value: torch DataLoader
        model_trainer: fedml_core.instances.model_trainer that defines the Clients local train and test behaviour
        Optional:
            log_fn: a logging function
            early_stopper: an EarlyStopping class that can be called on a metric and provides attributes: best_score, improved, stop
            history_save_fn: a function that gets self.history (dict) and a identifying string as an argument and saves it based on the function definition
        """
        super().__init__(dataset, model_trainer, log_fn, early_stopper, history_save_fn)
        self._instantiate_opt()
        self.log_fn(f'### Further initializations for {self.__class__.__name__} environment (END) ###')

    def _instantiate_opt(self):
        self.opt = OptRepo.name2cls(self.model_trainer.args.server_optimizer)(self.model_trainer.model.parameters(), lr=self.model_trainer.args.server_lr)


    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        
        # server optimizer
        self.opt.zero_grad()
        opt_state = self.opt.state_dict()
        self._set_model_global_grads(averaged_params)
        self._instantiate_opt()
        self.opt.load_state_dict(opt_state)
        self.opt.step()

    def _set_model_global_grads(self, new_state):
        new_model = copy.deepcopy(self.model_trainer.model)
        new_model.load_state_dict(new_state)
        with torch.no_grad():
            for parameter, new_parameter in zip(
                self.model_trainer.model.parameters(), new_model.parameters()
            ):
                parameter.grad = parameter.data - new_parameter.data
                # because we go to the opposite direction of the gradient
        model_state_dict = self.model_trainer.model.state_dict()
        new_model_state_dict = new_model.state_dict()
        for k in dict(self.model_trainer.model.named_parameters()).keys():
            new_model_state_dict[k] = model_state_dict[k]
        self.model_trainer.set_model_params(new_model_state_dict)