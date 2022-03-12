import torch
import matplotlib.pyplot as plt

class KNNRegression:
    """
    Implementation of KNN in torch. Expects data to be in the following form:
        X: [B x D] tensor
        y: [B] tensor

    Note that knn generally requires 0-1 normalization of data
    Design decision: Handle the normalization at the query step
    """
    def __init__(self, D, max_datapoints, K, sigma=1.0, device='cpu'):
        """
        Args:
            D: Dimension of features
            max_datapoints: maximum number of data to store
            K: number of points to use to query
            sigma: sharpness param on exponential weighting (some constant > 0)
            device: device to store data on
        """
        self.D = D
        self.max_datapoints = max_datapoints
        self.K = K

        self.data = torch.zeros(self.max_datapoints, self.D)
        self.labels = torch.zeros(self.max_datapoints, 1)
        self.sigma = sigma

        self.device = device

    def insert(self, X, y):
        """
        Add new data to the knn buffer
        Args:
            X: The data/feature(s) [N x D]
            y: The label(s) [N x 1]
        """
        print(f"Input X has shape: {X.shape}")
        print(f"Input y has shape: {y.shape}")

        if len(X.shape) == 1:
            self.insert(X.unsqueeze(0), y.unsqueeze(0))

        N = X.shape[0]  # Number of points

        if N >= self.max_datapoints:
            self.data = X[-self.max_datapoints:].to(self.device)
            self.labels = y[-self.max_datapoints:].to(self.device)
        else:
            data_shape = self.data[N:].shape
            self.data = torch.cat([self.data[N:], X.to(self.device)], dim=0)
            self.labels = torch.cat([self.labels[N:], y.to(self.device)], dim=0)

        self.xmin = self.data.min(dim=0)[0]
        self.xmax = self.data.max(dim=0)[0]

    def forward(self, X):
        """
        Perform KNN inference with new datapoints
        """
        
        # scaled_xd = (self.data - self.xmin.view(1, self.D)) / (self.xmax - self.xmin).view(1, self.D)
        # scaled_xq = (X - self.xmin.view(1, self.D)) / (self.xmax - self.xmin).view(1, self.D)
        scaled_xd = self.data
        scaled_xq = X

        dists = torch.linalg.norm(scaled_xd.view(1, self.data.shape[0], self.D) - scaled_xq.view(X.shape[0], 1, self.D), dim=-1) #[Q x D]
        top_idxs = torch.topk(dists, k=self.K, largest=False, dim=-1)[1]
        
        labels = self.labels[top_idxs]
        weights = (-self.sigma * dists[torch.arange(X.shape[0]).unsqueeze(1), top_idxs]).exp()
        weights /= weights.sum(dim=-2, keepdim=True)

        # import pdb;pdb.set_trace()

        return (labels*weights).sum(dim=-1)

    def to(self, device):
        self.device = device
        self.data = self.data.to(device)
        self.labels = self.labels.to(device)
        self.xmin = self.xmin.to(device)
        self.xmax = self.xmax.to(device)
        return self
        
if __name__ == '__main__':
    import time

    N1 = 50000
    N2 = 10
    D = 16

    X = torch.randn(N1, D)
    y = X.sum(dim=-1, keepdim=True)
    y += torch.randn_like(y) * 0.1

    knn = KNNRegression(D=X.shape[-1], max_datapoints=50000, K=25, sigma=10.0)
    knn.insert(X, y)
    knn = knn.to('cuda')
    knn.insert(X, y)

    print(knn.data.shape)
    
    X2 = torch.randn(N2, D).to('cuda')
    y2 = X2.sum(dim=-1, keepdim=True).to('cuda')

    ts = time.time()
    yp = knn.forward(X2.to('cuda'))
    te = time.time() - ts

    print('TOOK {:.4f}s'.format(te))

    print(torch.stack([y2, yp], dim=-1))
