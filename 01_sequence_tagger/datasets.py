from torch.utils.data import Dataset
#===========================================================================
# subclass of torch.utils.data.Dataset
class NERDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n_samples = len(X)       
    def __getitem__(self, index):
        return self.X[index], self.y[index]   
    def __len__(self):
        return self.n_samples
#=============================================================================
