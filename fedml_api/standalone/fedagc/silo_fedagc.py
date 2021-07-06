import copy
import logging
from collections import defaultdict

import numpy as np
import torch

from FedML.fedml_core.instances.client import Client
from FedML.fedml_api.standalone.fedavg.silo_fedavg import SiloFedAvg

# code from https://github.com/benjs/nfnets_pytorch/blob/master/nfnets/optim.py
def unitwise_norm(x):
    if (len(torch.squeeze(x).shape)) <= 1: # Scalars, vectors
        axis = 0
        keepdims = False
    elif len(x.shape) in [2,3]: # Linear layers
        # Original code: IO
        # Pytorch: OI
        axis = 1
        keepdims = True
    elif len(x.shape) == 4: # Conv kernels
        # Original code: HWIO
        # Pytorch: OIHW
        axis = [1, 2, 3]
        keepdims = True
    else:
        raise ValueError(f'Got a parameter with len(shape) not in [1, 2, 3, 4]! {x}')

    return torch.sqrt(torch.sum(torch.square(x), axis=axis, keepdim=keepdims))

class SiloFedAGC(SiloFedAvg):
    def __init__(self, dataset, model_trainer, log_fn=logging.info, early_stopper=None, history_save_fn=None, clipping_thresh=0, server_lr=1):
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
        self.server_lr = server_lr
        self.clipping_threshold = clipping_thresh
        self.eps = 1e-3
        self.log_fn(f'### Further initializations for {self.__class__.__name__} environment (END) ###')

    def _aggregate(self, local_weights):
        init_params = self.model_trainer.get_model_params()
        w = 1/len(local_weights)

        for k in init_params.keys():
            weight_norm = torch.maximum(unitwise_norm(init_params[k]), torch.tensor(self.eps).to(init_params[k].device))
            max_norm = weight_norm * self.clipping_threshold
            for _, local_params in local_weights:
                local_grad = init_params[k] - local_params[k]
                grad_norm = torch.maximum(unitwise_norm(local_grad), torch.tensor(1e-6).to(local_grad.device))
                trigger_mask = grad_norm > max_norm
                clipped_grad = local_grad * (max_norm/grad_norm)

                local_grad = torch.where(trigger_mask, clipped_grad, local_grad)
                averaged_grad = w * local_grad

            init_params[k].add_(averaged_grad, alpha=-self.server_lr)

                
        self.model_trainer.set_model_params(init_params)