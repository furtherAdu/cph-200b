import gc
import os
import sys
from collections import defaultdict
import numpy as np
import scipy.stats as stats
import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.loss.cox import neg_partial_log_likelihood
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.data_dict import clinical_feature_type
from src.torch import CoxRiskTorch
from src.directory import log_dir
from src.metrics import cox_partial_likelihood

pl.seed_everything(40)


def get_trainer(model_name, checkpoint_callback, monitor='val_loss', mode='min', **kwargs):
    # set up logging
    logger = CSVLogger(save_dir=log_dir, name=model_name)

    trainer_kwargs = dict(
        precision="bf16-mixed",
        accelerator='auto',
        logger=logger,
        callbacks=[
            EarlyStopping(monitor=monitor, mode=mode, patience=5),
            checkpoint_callback
        ],
        log_every_n_steps=1,
        max_epochs=100,
    )

    trainer_kwargs.update(kwargs)

    # set up trainer
    trainer = pl.Trainer(
        **trainer_kwargs
    )

    return trainer


def get_checkpoint_callback(model_name, dir_path, monitor='val_loss', mode='min'):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        dirpath=dir_path,
        filename=model_name + '-{epoch:002d}-{val_loss:.2f}',
        save_top_k=1)

    return checkpoint_callback


def get_log_dir_path(model_name):
    dir_path = os.path.join(log_dir, model_name)
    if not os.path.isdir(dir_path):
        version = '0'
    else:
        version = str(int(sorted(os.listdir(dir_path))[-1].replace('version_', '')) + 1)
    dir_path = os.path.join(dir_path, f'version_{version}')

    return dir_path


class MyProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        # if not sys.stdout.isatty():
        bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        # if not sys.stdout.isatty():
        bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar


class CoxRiskLightning(pl.LightningModule):
    def __init__(self, clinical_features, time_col='time', event_col='DEATH_EVENT', init_lr=1e-1,
                 interaction_features=[], dataset_name=None):
        super(CoxRiskLightning, self).__init__()
        self.save_hyperparameters()

        self.init_lr = init_lr

        # define model
        self.n_features = len(clinical_features) + len(interaction_features)
        self.model = CoxRiskTorch(self.n_features)

        # model input names
        self.clinical_features = clinical_features
        self.interaction_features = interaction_features
        self.feature_names = clinical_features + interaction_features
        self.time_col = time_col
        self.event_col = event_col
        self.dataset_name = dataset_name

        # metrics & loss
        self.cindex = ConcordanceIndex()
        self.auc = Auc()
        # self.loss = neg_partial_log_likelihood
        self.loss = cox_partial_likelihood
        self.metric_dict = defaultdict(dict)

        # structures to hold outputs until epoch end
        self.outputs = defaultdict(list)
        self.hessian = torch.zeros((self.n_features, self.n_features))

        # init weights
        self.model.apply(self.init_weights)

        self.eps = 1e-8

    def forward(self, x):
        return self.model(x).squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.init_lr)
        scheduler = LinearLR(optimizer)
        return [optimizer], [scheduler]

    def step(self, batch, batch_idx, stage):
        x, t, c = self.get_xtc(batch)
        y_hat = self.forward(x)
        loss = self.loss(y_hat, c, t)

        # log loss
        if stage not in ['predict', 'test']:
            self.log(f'{stage}_loss', loss, prog_bar=True, sync_dist=True)

        # store outputs
        self.outputs[stage].append({"y_hat": y_hat, "c": c, "t": t})

        return loss

    def on_epoch_end(self, stage):
        # concat outputs
        y_hat = torch.cat([o["y_hat"] for o in self.outputs[stage]]).squeeze()
        c = torch.cat([o["c"] for o in self.outputs[stage]]).bool().squeeze()
        t = torch.cat([o["t"] for o in self.outputs[stage]]).squeeze()

        # calculate metrics
        auc = self.auc(y_hat, c, t).mean()
        cindex = self.cindex(y_hat, c, t)

        metric_dict = {
            f"{stage}_auc": auc.item(),
            f"{stage}_cindex": cindex.item(),
        }

        if stage not in ['predict']:  # log metrics
            self.log_dict(metric_dict, prog_bar=True, sync_dist=True)
        # else:  # calculate p value and return
        #     coefficient_pvals = self.get_coefficient_pvals()
        #     metric_dict.update({f'{stage}_{k}': v for k, v in coefficient_pvals.items()})

        self.metric_dict[stage].update(metric_dict)

        # clear outputs
        self.outputs[stage] = []

        # clean space
        gc.collect()

    def get_xtc(self, batch):
        x = self.get_features(batch).squeeze()
        t = batch[self.time_col].squeeze()
        c = batch[self.event_col].squeeze()

        return x, t, c  # features, time to event, censoring/event occurrence

    def get_features(self, batch):
        clinical_features_dict = {}
        interaction_features_dict = {}
        if not self.dataset_name:
            self.dataset_name = self.trainer.datamodule.dataset_name

        if self.clinical_features:
            for feature_name in self.clinical_features:
                clinical_features_dict.update(
                    self.vectorize_feature(batch,
                                           feature_name=feature_name,
                                           feature_type=clinical_feature_type[self.dataset_name][feature_name])
                )

        if self.interaction_features:
            for fname_1, fname_2 in self.interaction_features:
                if fname_1 in clinical_features_dict:
                    f1 = clinical_features_dict[fname_1]
                else:
                    f1 = self.vectorize_feature(batch,
                                                feature_name=fname_1,
                                                feature_type=clinical_feature_type[self.dataset_name][fname_1])

                if fname_2 in clinical_features_dict:
                    f2 = clinical_features_dict[fname_2]
                else:
                    f2 = self.vectorize_feature(batch,
                                                feature_name=fname_2,
                                                feature_type=clinical_feature_type[self.dataset_name][fname_2])

                interaction_features_dict.update({f'{fname_1}_{fname_2}': f1 * f2})

        if self.clinical_features or self.interaction_features:
            # concatenate all features
            features = torch.stack(
                list({**clinical_features_dict, **interaction_features_dict}.values())
            ).to(self.device).bfloat16().T  # size: (B, len(clinical_features) + len(interaction_features)
        else:
            features = None

        return features

    def vectorize_feature(self, batch, feature_name, feature_type):
        # vectorize clinical feature as numpy array
        vectorized_feature = self.trainer.datamodule.vectorizer.transform(
            {feature_name: self.safely_to_numpy(batch[feature_name])}, feature_type=feature_type
        )

        if feature_type == 'categorical':
            vectorized_feature[feature_name] = np.argmax(vectorized_feature[feature_name], axis=1, keepdims=True)

        # send back to torch
        vectorized_feature = {k: torch.from_numpy(v.squeeze(1)) for k, v in vectorized_feature.items()}

        return vectorized_feature

    def get_coefficient_pvals(self, X):
        """ X is a torch tensor of size (samples, features) """
        try:
            covariance_matrix = torch.cov(X.T)
            standard_errors = torch.sqrt(torch.diag(covariance_matrix))
            z_scores = self.model.risk.weight.squeeze() / standard_errors
            p_values = 2 * (1 - stats.norm.cdf(torch.abs(z_scores).cpu().detach().numpy())) # two-tailed
            return {f"pval_{x}": p_values[i] for i, x in enumerate(self.feature_names)}

        except RuntimeError as e:  # Handle singular matrix errors
            print(f"Error computing covariance matrix: {e}")
            return {f"pval_{x}": float('nan') for i, x in enumerate(self.feature_names)}

    @staticmethod
    def init_weights(m, nonlinearity='linear'):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)

    @staticmethod
    def safely_to_numpy(tensor):
        return tensor.to(torch.float).cpu().numpy()

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "predict")

    def on_train_epoch_end(self):
        self.on_epoch_end('train')

    def on_validation_epoch_end(self):
        self.on_epoch_end('val')

    def on_test_epoch_end(self):
        self.on_epoch_end('test')

    def on_predict_epoch_end(self):
        self.on_epoch_end('predict')

    def predict_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.outputs['predict']]).squeeze()
        return y_hat

    def on_train_epoch_start(self):
        self.on_epoch_start('train')

    def on_validation_epoch_start(self):
        self.on_epoch_start('val')

    def on_test_epoch_start(self):
        self.on_epoch_start('test')

    def on_predict_epoch_start(self):
        self.on_epoch_start('predict')
