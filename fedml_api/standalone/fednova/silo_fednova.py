import copy
import logging
from collections import defaultdict

import numpy as np
import torch

from FedML.fedml_core.instances.client import Client
from FedML.fedml_api.standalone.fedavg.silo_fedavg import SiloFedAvg


class SiloFedNova(SiloFedAvg):
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
        self.log_fn(f'### Initializing {self.__class__.__name__} environment (START) ###')
        self.total_trainsample_num = sum([sample_dict['train'] for sample_dict in self.client_samples.values()])

        self.log_fn(f'### Initializing {self.__class__.__name__} environment (END) ###')

    def train(self, communication_rounds, validation_frequency):
        """Pretty similar to SiloFedAvg Only adjusting the aggregation and optimizer
        """
        self.log_fn(f'### {self.__class__.__name__} Training (START) ###')
        w_global = self.model_trainer.get_model_params()
        for global_epoch in range(communication_rounds):
            self.log_fn(f'# Communication round {global_epoch + 1} #')

            initial_parameters = copy.deepcopy(self.model_trainer.model.to(self.model_trainer.device).state_dict())
            norm_grads = []
            tau_effs = []
            self.global_momentum_buffer = dict()

            for client_id, client in self.clients.items():
                ratio = torch.FloatTensor([client.get_sample_number('train')/self.total_trainsample_num]).to(self.model_trainer.device)
                client.model_trainer.args.ratio = ratio
                _, _ = client.train(w_global) #TODO Do anything with local train metrics?
                norm_grads.append(copy.deepcopy(client.model_trainer.norm_grad))
                tau_effs.append(client.model_trainer.tau_eff)

            # update global weights
            w_global = self._aggregate(initial_parameters, norm_grads, tau_effs)
            self.model_trainer.set_model_params(w_global)
            self.log_fn('# Globally set new model parameters for testing #')

            if global_epoch % validation_frequency == 0:
                #global_trn_metrics = self.test('trn', global_epoch)
                #for metric, value in global_trn_metrics.items():
                #    self.history['trn_'+metric].append(value)
                global_val_metrics = self.test('val', global_epoch)
                for metric, value in global_val_metrics.items():
                    self.history['val_'+metric].append(value)

                if self.model_trainer.save_model != None:
                    self.model_trainer.save_model()
                self._save_histories()

                #EarlyStopping
                if self.early_stopper != None:
                    self.early_stopper(self.history[self.early_stopper.metric][-1])

                    if self.early_stopper.improved:
                        if self.model_trainer.save_model != None:
                            self.model_trainer.save_model('_best')
                    if self.early_stopper.stop:
                        self.log_fn(f'Early stopping the Federated training since we had no improvement of {self.early_stopper.metric} for {self.early_stopper.patience} rounds. Training was stopped after {global_epoch+1} communication rounds.')
                        break
            if self.model_trainer.debug:
                break

        self.log_fn(f'### {self.__class__.__name__} Training (END) ###')
        return self.history, self._get_local_histories()

    def _aggregate(self, params, norm_grads, tau_effs, tau_eff=0):
        # get tau_eff
        if tau_eff == 0:
            tau_eff = sum(tau_effs)
        # get cum grad
        # cum_grad = tau_eff * sum(norm_grads) 
        cum_grad = norm_grads[0]
        for k in norm_grads[0].keys():
            for i in range(0, len(norm_grads)):
                if i == 0:
                    cum_grad[k] = norm_grads[i][k] * tau_eff
                else:
                    cum_grad[k] += norm_grads[i][k] * tau_eff
        # update params
        for k in params.keys():
            if self.model_trainer.args.gmf != 0:
                if k not in self.global_momentum_buffer:
                    buf = self.global_momentum_buffer[k] = torch.clone(cum_grad[k]).detach()
                    buf.div_(self.model_trainer.args.lr)
                else:
                    buf = self.global_momentum_buffer[k]
                    buf.mul_(self.model_trainer.args.gmf).add_(1/self.model_trainer.args.lr, cum_grad[k])
                params[k].sub_(self.model_trainer.args.lr, buf)
            else:
                params[k].sub_(cum_grad[k])

        return params