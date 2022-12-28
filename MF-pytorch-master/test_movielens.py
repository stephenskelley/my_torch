import torch
from torch.utils.data import DataLoader, Dataset
import pandas
from BiasMFRecommender import BiasMF


class RateDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, index):
        return self.df.user[index], self.df.movie[index], self.df.rating[index]

    def __len__(self):
        return self.df.shape[0]


COLS = ['user', 'movie', 'rating', 'timestamp']
df_train = pandas.read_csv("./data/ml-100k/u1.base", sep='\t', names=COLS).drop(columns=['timestamp']).astype(int)
df_test = pandas.read_csv("./data/ml-100k/u1.test", sep='\t', names=COLS).drop(columns=['timestamp']).astype(int)
train_data = RateDataset(df_train)
test_data = RateDataset(df_test)

params = {'num_users': df_train.user.max(), 'num_items': df_train.movie.max(),
          'global_mean': df_train.rating.mean(), 'latent_dim': 25}
model = BiasMF(params)
#device = torch.device('mps')
#model.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
num_epoch = 50

for epoch in range(num_epoch):
    for bid, batch in enumerate(train_loader):
        u, i, r = batch[0]-1, batch[1]-1, batch[2]
        r = r.float()
        # forward pass
        preds = model(u, i)
        loss = criterion(preds, r)
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epoch}], Loss: {loss.item():.4f}')
torch.save(model.state_dict(), "./saved_models/matrix_movielens.pth")