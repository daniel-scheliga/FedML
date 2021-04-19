import copy
import logging
from collections import defaultdict

import numpy as np
import torch

from FedML.fedml_core.instances.client import Client

class SiloFedAvg(object):
    """
    In this SiloFedAvg all clients participate in the training every communication round.
    Better not exceed ~15 Clients for scalability reasons.
    """
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
        self.log_fn = log_fn
        self.log_fn('### Initializing FedAvg environment (START) ###')

        local_trn_data_dict, local_tst_data_dict, local_val_data_dict = dataset

        self.model_trainer = model_trainer

        self.clients = {}
        self._setup_clients(local_trn_data_dict, local_tst_data_dict, local_val_data_dict, model_trainer)

        self.early_stopper = early_stopper

        self.history = defaultdict(list)
        self.history_save_fn = history_save_fn

        self.log_fn('### Initializing FedAvg environment (END) ###')

    def _setup_clients(self, local_trn_data_dict, local_tst_data_dict, local_val_data_dict, model_trainer):
        self.log_fn('# Setting up Clients (START) #')
        for client_id in local_trn_data_dict.keys():
            self.clients[client_id] = Client(client_id, local_trn_data_dict[client_id], local_tst_data_dict[client_id], local_val_data_dict[client_id],
                                             model_trainer, self.log_fn)
        self.num_clients = len(self.clients)
        self.log_fn(f'# A total of {self.num_clients} clients were initialized. #')
        self.log_fn('# Setting up Clients (END) #')

    def train(self, communication_rounds, validation_frequency):
        self.log_fn('### FedAvg Training (START) ###')

        w_global = self.model_trainer.get_model_params()
        for global_epoch in range(communication_rounds):
            self.log_fn(f'# Communication round {global_epoch + 1} #')

            local_weights = []
            for client_id, client in self.clients.items():
                # train on new dataset
                w, _ = client.train(w_global) #TODO Do anything with local train metrics?
                local_weights.append((client.get_sample_number('train'), copy.deepcopy(w)))

            # update global weights
            w_global = self._aggregate(local_weights)
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

        self.log_fn('### FedAvg Training (END) ###')
        return self.history, self._get_local_histories()

    def test(self, split, round=-1, return_locals=False):
        self.log_fn('### FedAvg Testing Model (START) ###')
        client_metrics = defaultdict(list)
        client_samples = []
        for client_id, client in self.clients.items():
            client.model_trainer.set_model_params(self.model_trainer.get_model_params())
            local_metrics = client.test(split)
            for metric, value in local_metrics.items():
                client_metrics[metric].append(value)
            client_samples.append(client.get_sample_number(split))

        acc_metrics = {}
        for metric, values in client_metrics.items():
            acc_metrics[metric+'_weighted'] = np.average(values, weights=client_samples)
            acc_metrics[metric] = np.mean(values)

        metric_strings = [f'{metric}: {value}' for metric, value in acc_metrics.items()]

        ms = ' | '.join(metric_strings)
        if round >= 0:
            self.log_fn(f'# Testing the current global model in communication round {round} on {split} data: {ms} #')
        else:
            self.log_fn(f'# Evaluation of FedAvg global model on {split} data: {ms} #')

        self.log_fn('### FedAvg Testing Model (END) ###')
        if return_locals:
            client_metrics['IDs'] = list(self.clients.keys())
            return acc_metrics, client_metrics
        else:
            return acc_metrics


    def _aggregate(self, local_weights):
        training_num = 0
        for idx in range(len(local_weights)):
            (sample_num, averaged_params) = local_weights[idx]
            training_num += sample_num

        (sample_num, averaged_params) = local_weights[0]
        for k in averaged_params.keys():
            for i in range(0, len(local_weights)):
                local_sample_number, local_model_params = local_weights[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _get_local_histories(self):
        return {client_id: client.local_history for client_id, client in self.clients.items()}

    def _save_histories(self):
        if self.history_save_fn != None:
            self.history_save_fn(self.history, 'GLOBAL')
            local_histories = self._get_local_histories()
            for client_id, history in local_histories.items():
                self.history_save_fn(history, 'LOCAL_'+client_id)
