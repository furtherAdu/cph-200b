import os.path
import numpy as np
from math import ceil
from tqdm import tqdm

import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, LinearLR
from torchsurv.metrics.cindex import ConcordanceIndex

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from scipy.sparse import hstack

from src.data_dict import clinical_feature_type, feature_config
from src.utils import descriptive_stats
from src.metrics import cox_partial_likelihood


class CoxRiskTorch(nn.Module):
    def __init__(self, num_features):
        super(CoxRiskTorch, self).__init__()

        # init model components
        self.risk = nn.Linear(num_features, 1)  # risk

    def forward(self, x):
        return self.risk(x)


class DeepSurvival(nn.Module):
    def __init__(self,
                 hidden_dim=64,
                 save_path='best_model.pth',
                 dataset_name='unos',
                 dataset_stats=descriptive_stats,
                 clinical_features=[],
                 importance_weighting=False):

        super(DeepSurvival, self).__init__()
        
        # define necessaries
        self.dataset_name = dataset_name
        self.dataset_stats = dataset_stats[self.dataset_name]
        self.input_features = feature_config[self.dataset_name] if not clinical_features else clinical_features
        self.save_path = save_path

        # define metrics
        self.cindex = ConcordanceIndex()
        self.loss = cox_partial_likelihood
        self.importance_weighting = importance_weighting
        self.loss_kwargs = {}
        self.domain_classifier = None
        self.T_stats = {}

        # define model hyperparams
        self.input_dim = None
        self.hidden_dim = hidden_dim
        self.activation = nn.ReLU()
        self.layer_kwargs = dict(bias=True)
        self.net = None

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def init_weights(m, nonlinearity='relu'):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)

    def fit(self, x, t, c, epochs=1000, lr=3e-3, weight_decay=1e-3, batch_size=1024, patience=5):
        # make dir as necessary
        if not os.path.isdir(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path))

        # vectorize features
        x = self.vectorize_x(x)
        self.T_stats['mean'] = t.mean()
        self.T_stats['std'] = t.std()

        # init model
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, **self.layer_kwargs),
            self.activation,
            nn.Linear(self.hidden_dim, self.hidden_dim // 2, **self.layer_kwargs),
            self.activation,
            nn.Linear(self.hidden_dim // 2, 1, **self.layer_kwargs),
        )

        # model settings
        self.net.apply(self.init_weights)
        self.double()

        # set up training utils
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ExponentialLR(optimizer, gamma=.99)
        best_val_loss = torch.Tensor([float('inf')])
        train_loss, val_loss = torch.Tensor([float('inf')]), torch.Tensor([float('inf')])
        epochs_no_improve = 0

        # adjust t
        n_samples_negative_time = len(t[t < 0])
        if n_samples_negative_time > 0:
            print(f'{n_samples_negative_time} negative time samples found. Adjusting by setting t == 0.')
            t[t < 0] = 0

        # get train and val data
        X_train, X_val, T_train, T_val, C_train, C_val = train_test_split(x, t, c,
                                                                          test_size=.2, random_state=40)

        # init domain classifier
        if self.importance_weighting:
            self.domain_classifier = self.get_domain_classifier(X=self.get_domain_classifier_input(X_train, T_train),
                                                                y=C_train)

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
                                 f' val_loss:{val_loss.item()}, train_loss:{train_loss.item()}')

            self.train()
            for batch in train_batch_idxs:
                risk_scores = self(X_train[batch])

                if self.importance_weighting:
                    importance_weights = self.get_importance_weights(X_train[batch], T_train[batch])
                    self.loss_kwargs['weights'] = importance_weights

                train_loss = self.loss(risk_scores, C_train[batch].long(), T_train[batch], **self.loss_kwargs)
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

                    if self.importance_weighting:
                        importance_weights = self.get_importance_weights(X_val[batch], T_val[batch])
                        self.loss_kwargs['weights'] = importance_weights

                    val_loss += self.loss(val_risk_scores, C_val[batch].long(), T_val[batch], **self.loss_kwargs) / n_val_batches

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

        x = self.to_tensor(self.vectorize_x(x))

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

        x = self.vectorize_x(x)
        t[t < 0] = 0  # adjust t
        x, t, c = self.get_xtc(x, t, c)
        x, t, c = x.squeeze(), t.squeeze(), c.squeeze()

        with torch.no_grad():
            y_hat = self(x)
            cindex = self.cindex(y_hat, c.bool(), t)

        self.train()
        return cindex

    def vectorize_x(self, x):
        vectorized_features = []

        for i, feature in enumerate(self.input_features):
            feature_type = clinical_feature_type[self.dataset_name][feature]
            if feature_type == 'numerical':
                f_mean = self.dataset_stats[feature]['mean']
                f_std = self.dataset_stats[feature]['std']
                # x[:, i] = (x[:, i] - f_mean) / f_std
                vectorized_feature = np.expand_dims((x[:, i] - f_mean) / f_std, axis=-1)
            elif feature_type == 'categorical':
                enc = OneHotEncoder()
                vectorized_feature = enc.fit_transform(x[:, i].reshape(-1,1))
            
            vectorized_features.append(vectorized_feature)

        # update input dimension
        vectorized_features = hstack(vectorized_features).toarray()
        self.input_dim = vectorized_features.shape[1]

        return vectorized_features

    def get_xtc(self, x, t, c):
        x = self.to_tensor(x)
        t = self.to_tensor(t).reshape(-1, 1)
        c = self.to_tensor(c).reshape(-1, 1)
        return x, t, c  # features, time to event, censoring/event occurrence

    def get_domain_classifier(self, X, y):
        domain_classifier = LogisticRegression()
        domain_classifier.fit(X, y)
        return domain_classifier

    def get_domain_classifier_input(self, X, T):
        X = self.safely_to_numpy(X) if torch.is_tensor(X) else X
        T = self.safely_to_numpy(T) if torch.is_tensor(T) else T

        # standardize T
        T = (T - self.T_stats['mean']) / self.T_stats['std']

        input = np.hstack([X, T.reshape(-1,1)])
        return input

    def get_importance_weights(self, X, T):
        input = self.get_domain_classifier_input(X, T)
        domain_prediction = self.domain_classifier.predict_proba(input) 
        importance_weights = domain_prediction[:, 0] / domain_prediction[:, 1]  # T/S, where T is unlabeled and S is labeled
        return torch.Tensor(importance_weights)[:, None]

    @staticmethod
    def to_tensor(x):
        x = torch.tensor(x, dtype=torch.double)
        return x

    @staticmethod
    def safely_to_numpy(tensor):
        return tensor.to(torch.float).cpu().numpy()
    
