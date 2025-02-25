import gc
import os
import sys
from collections import defaultdict
import numpy as np
import scipy.stats as stats
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import LinearLR
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.loss.cox import neg_partial_log_likelihood
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.data_dict import clinical_feature_type
from src._torch import CoxRiskTorch, CounterfactualRegressionTorch, DragonNetTorch
from src.directory import log_dir
from src.metrics import cox_partial_likelihood, mmd, targeted_regularization_loss, dragonnet_loss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

pl.seed_everything(40)


def get_trainer(model_name, checkpoint_callback, monitor='val_loss', mode='min', max_epochs=100, **kwargs):
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
        max_epochs=max_epochs,
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

def get_logger(model_name, project_name='CPH_200B', wandb_entity='furtheradu', dir_path='..',**kwargs):
    if kwargs.get('disable_wandb'):
        print("wandb logging is disabled.")
        return None
    else:
        logger = pl.loggers.WandbLogger(
            project=project_name,
            entity=wandb_entity,
            group=model_name,
            dir=dir_path,
            **kwargs
        )
        return logger

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
        self.loss = cox_partial_likelihood
        self.metric_dict = defaultdict(dict)

        # structures to hold outputs until epoch end
        self.outputs = defaultdict(list)

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

    def on_epoch_start(self, stage):
        pass

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


class CounterfactualRegressionLightning(pl.LightningModule):
    def __init__(self, 
                 input_features, 
                 r_hidden_dim=200,
                 h_hidden_dim=100,
                 alpha=1.0,
                 learning_rate=1e-3,
                 complexity_lambda=1.0,
                 outcome_type='continuous'):
        super().__init__()
        valid_outcome_types = ['continuous', 'binary']
        assert outcome_type in valid_outcome_types, f'outcome_type must be in {valid_outcome_types}'
        self.save_hyperparameters()

        self.alpha = alpha
        self.learning_rate = learning_rate
        self.input_features = input_features
        self.input_dim = len(self.input_features)
        self.r_hidden_dim = r_hidden_dim
        self.h_hidden_dim = h_hidden_dim
        self.outcome_type = outcome_type
        self.complexity_lambda = complexity_lambda
        
        if self.outcome_type == 'continuous':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            # self.loss_fn = log_loss
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        
        # structures to hold outputs/metrics
        self.outputs = defaultdict(list)
        self.metric_dict = defaultdict(dict)

        self.model = CounterfactualRegressionTorch(input_dim=self.input_dim,
                                                  r_hidden_dim=self.r_hidden_dim,
                                                  h_hidden_dim=self.h_hidden_dim)

        # init weights
        self.model.apply(self.init_weights)

    def forward(self, x, t):
        return self.model(x,t)

    def configure_optimizers(self):
        h_weight_decay = 1e-2
        hypothesis_params = []
        other_params = []

        for name, param in self.named_parameters():
            if 'h0' in name or 'h1' in name:
                hypothesis_params.append(param)
            else:
                other_params.append(param)

        # define optimizer with different weight decay for each group
        optimizer = optim.Adam([
            {'params': hypothesis_params, 'weight_decay': h_weight_decay},
            {'params': other_params, 'weight_decay': 0}
        ], lr=self.learning_rate)

        return optimizer
        
    def get_factual_loss(self, y_pred, Y, T, threshold=.5):
        u = self.trainer.datamodule.u
        w_i = (T / (2 * u) + (1 - T) / (2 * (1 - u)))
        # if self.outcome_type == 'binary':
        #     y_pred = (y_pred > threshold).float()
        loss = self.loss_fn(y_pred, Y)
        factual_loss = (w_i * loss).mean() # [~torch.any(loss.isnan(),dim=1)]
        return factual_loss

    @staticmethod
    def safely_to_numpy(tensor):
        return tensor.to(torch.float).cpu().numpy()
    
    @staticmethod
    def init_weights(m, nonlinearity='relu'):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)

    @staticmethod
    def get_model_input(batch):
        return batch['X'].squeeze(dim=1).double(), batch['Y'].double(), batch['T'].double()
    
    def step(self, batch, batch_idx, stage):
        X, Y, T = self.get_model_input(batch)

        out = self.model(X,T)
        phi_x, y0, y1 = out['phi_x'], out['y0'], out['y1']
        
        y_pred = y1 * T + y0 * (1 - T)
        phi_x_0 = phi_x[(T == 0).squeeze()]
        phi_x_1 = phi_x[(T == 1).squeeze()]

        factual_loss = self.get_factual_loss(y_pred,Y,T)
        IPM_loss = mmd(phi_x_0, phi_x_1)
        model_complexity_loss = torch.Tensor([0]) * self.complexity_lambda
        loss = self.alpha * IPM_loss + factual_loss + model_complexity_loss
        
        # store outputs
        self.outputs[stage].append({'X':X, 
                                    'T':T,
                                    'Y':Y,
                                    "phi_x":phi_x,
                                    'y0':y0,
                                    'y1':y1})

        # log metrics
        if stage not in ['predict', 'test']:
            log_kwargs = dict(prog_bar=True, sync_dist=True)
            self.log(f'{stage}_IPM_loss', IPM_loss, **log_kwargs)
            self.log(f'{stage}_factual_loss', factual_loss, **log_kwargs)
            self.log(f'{stage}_loss', loss, **log_kwargs)

        return loss
    
    def on_epoch_end(self, stage):
        # concat outputs
        outputs_vars = ['X', 'T', 'Y', 'phi_x', 'y0', 'y1']
        X, T, Y, phi_x, y0, y1 = [torch.cat([o[x] for o in self.outputs[stage]]).squeeze() 
                               for x in outputs_vars]

        # calculate performance metrics
        y_pred = y1 * T + y0 * (1 - T)
        factual_loss = self.get_factual_loss(y_pred,Y,T)
        tau = (y1 - y0).mean()
        
        metric_dict = {
            f'{stage}_{self.loss_fn._get_name()}': factual_loss,
            f'{stage}_tau': tau
        }

        if stage not in ['predict']:  # log metrics
            log_kwargs = dict(prog_bar=True, sync_dist=True)
            self.log_dict(metric_dict, **log_kwargs)

        self.metric_dict[stage].update(metric_dict)
        
        # send tensors to numpy
        X, T, phi_x = [self.safely_to_numpy(x) for x in [X, T, phi_x]]        
        
        # get t-SNE embeddings
        embeddings = {}
        for x_name, x in dict(zip(['X', 'phi_x'], [X, phi_x])).items():
            embeddings[x_name] = TSNE(random_state=40).fit_transform(x)
        
        if self.logger.experiment and stage not in ['train']:
            for k, embedding in embeddings.items():
                # get plot name
                key = f'tSNE {k}, CFR alpha {self.alpha}, epoch{self.current_epoch}'
                
                # make figure
                fig = plt.figure(figsize=(10, 8))
                ax = plt.scatter(embedding[:, 0], embedding[:, 1], c=T) # TODO: suppress printing
                plt.colorbar(ax, label='treatment')
                plt.title(key)
                                
                # log figure
                self.logger.log_image(key=key, images=[fig])
                
                # suppress displaying
                plt.close(fig)

        # clear outputs
        self.outputs[stage] = []

        # clean space
        gc.collect()
        
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "predict")

    # def on_train_epoch_end(self):
    #     self.on_epoch_end('train')

    # def on_validation_epoch_end(self):
    #     self.on_epoch_end('val')

    def on_test_epoch_end(self):
        self.on_epoch_end('test')

    # def on_predict_epoch_end(self):
    #     self.on_epoch_end('predict')
    
class DragonNetLightning(pl.LightningModule):
    def __init__(self, 
                 input_features, 
                 sr_hidden_dim=200,
                 co_hidden_dim=100,
                 alpha=1.0,
                 beta=1.0,
                 learning_rate=1e-5,
                 target_reg=False):
        super().__init__()
        self.save_hyperparameters()

        self.alpha = alpha
        self.beta = beta
        self.learning_rate = learning_rate
        self.input_features = input_features
        self.input_dim = len(self.input_features)
        self.sr_hidden_dim = sr_hidden_dim
        self.co_hidden_dim = co_hidden_dim
        self.target_reg = target_reg
        
        # structures to hold outputs/metrics
        self.outputs = defaultdict(list)
        self.metric_dict = defaultdict(dict)

        self.model = DragonNetTorch(input_dim=self.input_dim,
                                    sr_hidden_dim=self.sr_hidden_dim,
                                    co_hidden_dim=self.co_hidden_dim)
        
        if self.target_reg:
            self.epsilon = nn.Parameter(torch.randn(1) / 10, requires_grad=True)
        
        # init weights
        self.model.apply(self.init_weights)

    def forward(self, x, t):
        return self.model(x,t)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=.9)
        return optimizer

    @staticmethod
    def safely_to_numpy(tensor):
        return tensor.to(torch.float).cpu().numpy()
    
    @staticmethod
    def init_weights(m, nonlinearity='relu'):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)

    @staticmethod
    def get_model_input(batch):
        return batch['X'].squeeze(dim=1).double(), batch['Y'].double(), batch['T'].double()

    def get_losses(self, y1, y0, t, Y, T):
        y1 = y1.squeeze()
        y0 = y0.squeeze()
        t = t.squeeze()
        Y = Y.squeeze()
        T = T.squeeze()
        
        dragon_net_loss = dragonnet_loss(y1, y0, t, Y, T, alpha=self.alpha)
        if self.target_reg:
            target_reg_loss = targeted_regularization_loss(y1, y0, t, Y, T, eps=self.epsilon, beta=self.beta)
        else:
            target_reg_loss = torch.Tensor([0])
        loss = dragon_net_loss + target_reg_loss
        return {'loss': loss, 'dragon_net_loss': dragon_net_loss, 'targeted_regularization_loss':target_reg_loss}
    
    def step(self, batch, batch_idx, stage):
        X, Y, T = self.get_model_input(batch)
        
        # get outputs
        out = self.model(X)
        t, y0, y1 = out['t'], out['y0'], out['y1']
        
        # get loss
        losses = self.get_losses(y1, y0, t, Y, T)
        loss = losses['loss']
        dragon_net_loss = losses['dragon_net_loss']
        target_reg_loss = losses['targeted_regularization_loss']
        
        # store outputs
        self.outputs[stage].append({'y0':y0,
                                    'y1':y1,
                                    't':t,
                                    'T':T,
                                    'Y':Y})

        # log metrics
        if stage not in ['predict', 'test']:
            log_kwargs = dict(prog_bar=True, sync_dist=True)
            self.log(f'{stage}_dragon_net_loss', dragon_net_loss, **log_kwargs)
            self.log(f'{stage}_target_regularization_loss', target_reg_loss, **log_kwargs)
            self.log(f'{stage}_loss', loss, **log_kwargs)

        return loss
    
    def on_epoch_end(self, stage):
        # concat outputs
        outputs_vars = ['Y', 'T', 'y1', 'y0', 't']
        Y, T, y1, y0, t = [torch.cat([o[x] for o in self.outputs[stage]]).squeeze() for x in outputs_vars]
        
        # get losses
        losses = self.get_losses(y1, y0, t, Y, T)
        loss = losses['loss']
        dragon_net_loss = losses['dragon_net_loss']
        target_reg_loss = losses['targeted_regularization_loss']
        
         # exclude data points with propensity score outside [.01,.99]
        included = (.01 <= t).squeeze() * (t <= .99).squeeze()
        y0 = y0[included]
        y1 = y1[included]
        T = T[included]

        # calculate performance metrics
        tau = (y1 - y0).mean()

        metric_dict = {
            f'{stage}_tau': tau,
            f'{stage}_loss': loss,
            f'{stage}_dragon_net_loss': dragon_net_loss,
            f'{stage}_targeted_regularization_loss': target_reg_loss
        }

        if stage not in ['predict']:  # log metrics
            log_kwargs = dict(prog_bar=True, sync_dist=True)
            self.log_dict(metric_dict, **log_kwargs)

        self.metric_dict[stage].update(metric_dict)

        # clear outputs
        self.outputs[stage] = []

        # clean space
        gc.collect()
        
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "predict")

    # def on_train_epoch_end(self):
    #     self.on_epoch_end('train')

    # def on_validation_epoch_end(self):
    #     self.on_epoch_end('val')

    def on_test_epoch_end(self):
        self.on_epoch_end('test')

    # def on_predict_epoch_end(self):
    #     self.on_epoch_end('predict')
