from learned_cost_map.trainer.utils import  get_FFM_freqs, FourierFeatureMapping
import torch

def main():
    # data = torch.Tensor([0.3])
    # data = torch.Tensor([0.2, 0.6, 0.8])
    data = torch.Tensor([[0.1], [0.3], [0.5], [0.7]])
    # data = torch.Tensor([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.5, 0.6, 0.7], [0.7, 0.8, 0.9]])

    data_size = data.shape[-1]
    scale = 10.0
    num_features = 16
    B = get_FFM_freqs(data_size, scale, num_features)

    fourier_data = FourierFeatureMapping(data, B)
    import pdb;pdb.set_trace()

if __name__=="__main__":
    main()