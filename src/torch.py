import os.path

import numpy as np
import torch
from torch import nn
from math import ceil
from tqdm import tqdm
from torchsurv.metrics.cindex import ConcordanceIndex
from sklearn.model_selection import train_test_split
from src.data_dict import clinical_feature_type, feature_config, dataset_stats
from torchsurv.loss.cox import neg_partial_log_likelihood
from src.metrics import cox_partial_likelihood
from torch.optim.lr_scheduler import ExponentialLR, LinearLR


class CoxRiskTorch(nn.Module):
    def __init__(self, num_features):
        super(CoxRiskTorch, self).__init__()

        # init model components
        self.risk = nn.Linear(num_features, 1)  # risk

    def forward(self, x):
        return self.risk(x)


class DeepSurvival(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim=64,
                 save_path='best_model.pth',
                 dataset_name='unos',
                 event_col="Censor (Censor = 1)",
                 time_col="Survival Time",
                 clinical_features=[]):

        super(DeepSurvival, self).__init__()
        
        # define necessaries
        self.time_col = time_col
        self.event_col = event_col
        self.dataset_name = dataset_name
        self.dataset_stats = dataset_stats[self.dataset_name]
        self.input_features = feature_config[self.dataset_name] if not clinical_features else clinical_features
        self.save_path = save_path

        # define metrics
        self.cindex = ConcordanceIndex()
        self.loss = cox_partial_likelihood
        # self.loss = neg_partial_log_likelihood

        # define model
        activation = nn.ReLU()
        layer_kwargs = dict(bias=True)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, **layer_kwargs),
            activation,
            nn.Linear(hidden_dim, hidden_dim // 2, **layer_kwargs),
            activation,
            nn.Linear(hidden_dim // 2, 1, **layer_kwargs),
        )

        # model settings
        self.net.apply(self.init_weights)
        self.double()

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def init_weights(m, nonlinearity='relu'):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)

    def fit(self, x, t, c, epochs=1000, lr=3e-4, weight_decay=1e-3, batch_size=1024, patience=5):
        # make dir as necessary
        if not os.path.isdir(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path))

        # set up training utils
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ExponentialLR(optimizer, gamma=.99)
        best_val_loss = float('inf')
        epochs_no_improve = 0
        train_loss, val_loss = float('inf'), float('inf')

        # vectorize features
        x = self.vectorize(x)

        # check vectorization successful
        non_numerical_features = np.array(self.input_features)[x.mean(axis=0) > .1]
        assert all([clinical_feature_type[self.dataset_name][feature] != 'numerical'
                    for feature in non_numerical_features]), f'X columns must be in ord: {self.input_features}'

        # adjust t
        n_samples_negative_time = len(t[t < 0])
        if n_samples_negative_time > 0:
            print(f'{n_samples_negative_time} negative time samples found. Adjusting by setting t == 0.')
            t[t < 0] = 0

        # get train and val data
        X_train, X_val, T_train, T_val, C_train, C_val = train_test_split(x, t, c,
                                                                          test_size=.2, random_state=40)

        # convert to tensors
        X_train, T_train, C_train = self.get_xtc(X_train, T_train, C_train)
        X_val, T_val, C_val = self.get_xtc(X_val, T_val, C_val)
        
        # get batches
        train_batch_idxs = np.array_split(np.arange(len(X_train)), ceil(len(X_train) / batch_size))
        val_batch_idxs = np.array_split(np.arange(len(X_val)), ceil(len(X_val) / batch_size))
        n_val_batches = len(val_batch_idxs)

        # train
        for epoch in (cbar := tqdm(range(epochs))):
            cbar.set_description(f'Training DeepSurvival, epoch {epoch}/{epochs},'
                                 f' val_loss:{val_loss}, train_loss:{train_loss}')

            self.train()
            for batch in train_batch_idxs:
                risk_scores = self(self.vectorize(X_train[batch]))
                train_loss = self.loss(risk_scores, C_train[batch].long(), T_train[batch])
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
            scheduler.step()

            # validation check
            self.eval()  # set to eval mode
            with torch.no_grad():
                val_loss = 0
                for batch in val_batch_idxs:
                    val_risk_scores = self(X_val[batch])
                    val_loss += self.loss(val_risk_scores, C_val[batch].long(), T_val[batch]) / n_val_batches

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.state_dict(), self.save_path)  # save best model
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'Early stopping triggered at epoch {epoch + 1}')
                    self.load_state_dict(torch.load(self.save_path, weights_only=True))  # load best weights
                    break

    def predict(self, x, max_years=10):  # times in months
        self.eval()

        x = self.to_tensor(self.vectorize(x))

        with torch.no_grad():
            risk_scores = self(x).detach().numpy().flatten()

        survival_probs = []
        months = np.arange(0, 12 * max_years + 1, 6)
        for time in months:
            # Approximate survival probability based on risk scores
            survival_prob = np.exp(-np.exp(risk_scores) * time / 12)
            survival_probs.append(survival_prob)

        self.train()

        return np.array(survival_probs).T  # transposed for correct shape

    def get_cindex(self, x, t, c):
        self.eval()

        x = self.vectorize(x)
        t[t < 0] = 0  # adjust t
        x, t, c = self.get_xtc(x, t, c)
        x, t, c = x.squeeze(), t.squeeze(), c.squeeze()

        with torch.no_grad():
            y_hat = self(x)
            cindex = self.cindex(y_hat, c.bool(), t)

        self.train()
        return cindex

    def vectorize(self, x):
        for i, feature in enumerate(self.input_features):
            if feature in self.dataset_stats:
                f_mean = self.dataset_stats[feature]['mean']
                f_std = self.dataset_stats[feature]['std']
                x[:, i] = (x[:, i] - f_mean) / f_std
        return x

    def get_xtc(self, x, t, c):
        x = self.to_tensor(self.vectorize(x).squeeze())
        t = self.to_tensor(t).reshape(-1, 1)
        c = self.to_tensor(c).reshape(-1, 1)
        return x, t, c  # features, time to event, censoring/event occurrence


    @staticmethod
    def to_tensor(x):
        x = torch.tensor(x, dtype=torch.double)
        return x