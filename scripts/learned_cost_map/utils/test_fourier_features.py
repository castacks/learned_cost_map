from learned_cost_map.trainer.utils import  get_FFM_freqs, FourierFeatureMapping
import torch

def main():
    import pdb;pdb.set_trace()
    data = torch.Tensor([0.3])
    data = torch.Tensor([0.2, 0.6, 0.8])

    data_size = data.shape[0]
    scale = 10.0
    num_features = 16
    B = get_FFM_freqs(data_size, scale, num_features)

    fourier_data = FourierFeatureMapping(data, B)


if __name__=="__main__":
    main()