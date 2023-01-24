import torch
import pytorch_lightning as pl
import numpy


class BaseLightningModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.num_users = params['num_users']
        self.num_items = params['num_items']
        self.latent_dim = params['latent_dim']
        self.mu = params['global_mean']
        self.loss_fn = torch.nn.MSELoss()

    def training_step(self, batch, batch_idx):
        X, y = batch
        pred = self(X)
        train_loss = self.loss_fn(pred, y)
        self.log('training loss', train_loss, on_step=True, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        val_loss = self.loss_fn(pred, y)
        self.log('validation loss', val_loss, on_step=True, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


class MatrixLightningModel(BaseLightningModel):
    def __init__(self, params):
        super().__init__(params)
        self.user_embedding = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.item_embedding = torch.nn.Embedding(self.num_items, self.latent_dim)

        self.user_bias = torch.nn.Embedding(self.num_users, 1)
        self.user_bias.weight.data = torch.zeros(self.num_users, 1).float()
        self.item_bias = torch.nn.Embedding(self.num_items, 1)
        self.item_bias.weight.data = torch.zeros(self.num_items, 1).float()

    def forward(self, x):
        user_indices = x[:, 0]
        item_indices = x[:, 1]
        user_vec = self.user_embedding(user_indices)
        item_vec = self.item_embedding(item_indices)
        dot = torch.mul(user_vec, item_vec).sum(dim=1)
        rating = dot + self.mu + torch.squeeze(self.user_bias(user_indices)) + torch.squeeze(
            self.item_bias(item_indices))
        return rating


class AutoEncoderLightningModel(BaseLightningModel):
    def __init__(self, params):
        super(params).__init__()
        # first layer: nb_neurons: number of hidden nodes
        self.fc1 = torch.nn.Linear(nb_features, self.latent_dim)
        # second layer: nb_neurons input size, and nb_neurons/2 number of hidden nodes
        self.fc2 = torch.nn.Linear(self.latent_dim, numpy.int(self.latent_dim / 2))
        # second layer: nb_neurons/2 input size, and nb_neurons number of hidden nodes
        self.fc3 = torch.nn.Linear(numpy.int(self.latent_dim2), self.latent_dim)
        # Output layes, size output is the same of size input
        self.fc4 = torch.nn.Linear(self.latent_dim, nb_features)
        # Activation function
        self.activation = torch.nn.Sigmoid()

    # Forward propagation
    def forward(self, x):
        # Activate the Encoded vector of the first FC
        x = self.activation(self.fc1(x))
        # Activate the Encoded vector of the second FC
        x = self.activation(self.fc2(x))
        # Activate the Encoded vector of the third FC
        x = self.activation(self.fc3(x))
        # No activation function on the output layer to get the reconstructed vector
        x = self.fc4(x)
        return x
