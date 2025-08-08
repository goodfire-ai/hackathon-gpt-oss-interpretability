import torch


class TopKSAE(torch.nn.Module):
    def __init__(
        self,
        d_in,
        d_hidden,
        k,
        device: torch.device,
        normalization_constant: int = 1,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.device = device
        self.dtype = dtype
        self.norm_constant = normalization_constant
        self.encoder_linear = torch.nn.Linear(d_in, d_hidden, bias=False)
        self.decoder = torch.nn.Embedding(d_hidden, d_in)
        self.decoder_bias = torch.nn.Parameter(torch.zeros(d_in))
        self.to(self.device, self.dtype)
        self.k = k

    def _encode_pre_topk(self, x):
        x = x - self.decoder_bias
        x = torch.nn.functional.relu(self.encoder_linear(x))
        return x

    def encode(self, x):
        return torch.topk(self._encode_pre_topk(x), self.k, dim=-1)

    def decode(self, f):
        return (
            torch.einsum("ij,ijk->ik", f.values, self.decoder(f.indices))
            + self.decoder_bias
        )

    def forward(self, x):
        x = x * self.norm_constant
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f


def load_sae(file_path, device, dtype):
    state_dict = torch.load(file_path, weights_only=True, map_location=device)
    sae = TopKSAE(d_in=2880, d_hidden=2880 * 16, dtype=dtype, device=device, k=32, normalization_constant=0.016724765300750732)
    sae.load_state_dict(state_dict)
    return sae
