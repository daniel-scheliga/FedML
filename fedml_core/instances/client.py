import logging
from collections import defaultdict


class Client:
    def __init__(self, client_name, trn_data, tst_data, val_data, model_trainer, log_fn = logging.info):
        """
        client_name: Expects client_name / identifier string (unique)
        trn/tst/val_data: Dataloder (pref. torch) for training, test, and validation data.
        model_trainer: fedml_core.instances.model_trainer that defines the Clients local train and test behaviour
        Optional:
        log_fn: a logging function
        """
        self.client_name = client_name
        self.trn_data = trn_data
        self.tst_data = tst_data
        self.val_data = val_data
        self.trn_samples = len(trn_data.dataset)
        self.tst_samples = len(tst_data.dataset)
        self.val_samples = 0 if val_data == None else len(val_data.dataset)

        self.model_trainer = model_trainer
        #model_trainer.set_id(client_name+'_trainer') # We only have one trainer anyways, who is training everyone
        self.local_history = defaultdict(list)

        self.log_fn = log_fn

        self.log_fn(f'# Client {client_name} initialized. local train samples: {self.trn_samples} | local test samples: {self.tst_samples} | local val samples: {self.val_samples} #')

    def get_sample_number(self, split, get_truth=False):
        """returns the number of samples in a given split of a dataset

        Args:
            split ([str]): one of trn, tst, val
            get_truth (bool, optional): returns the actual number of samples if True (else val_samples may be modified if no val_data was provided to the client). Defaults to False.

        Returns:
            [int]: number of samples
        """
        if split == 'train' or split == 'trn':
            return self.trn_samples
        elif split == 'test' or split == 'tst':
            return self.tst_samples
        elif split == 'validation' or split == 'val':
            if self.val_samples == 0: logging.warning('The Client was not provided with explicit validation data. Using testdata instead.')
            return self.val_samples if self.val_samples > 0 else self.tst_samples
        else:
            self.log_fn('# This data split is not defined for Clients. Choose one of trn, tst, val #')

    def train(self, w_global):
        self.log_fn(f'# Client {self.client_name} training (START) #')
        self.model_trainer.set_model_params(w_global)
        metrics = self.model_trainer.train(self.trn_data)
        weights = self.model_trainer.get_model_params()
        for metric, value in metrics.items():
            self.local_history[metric+'_trn_local'].extend(value)
        self.log_fn(f'# Client {self.client_name} training (END) #')
        return weights, metrics

    def test(self, split='tst'):
        self.log_fn(f'# Client {self.client_name} now testing on local {split} data... #')
        if split == 'train' or split == 'trn':
            data = self.trn_data
        elif split == 'test' or split == 'tst':
            data = self.tst_data
        elif split == 'validation' or split == 'val':
            if self.val_samples > 0:
                data = self.val_data 
            else:
                logging.warning('The Client was not provided with explicit validation data. Using testdata instead.')
                data = self.tst_data

        metrics = self.model_trainer.test(data)

        for metric, value in metrics.items():
            self.local_history[metric+f'_{split}'].append(value)

        return metrics
