import torch
from torch import nn


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.max_len = max_len
        self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=hidden_size * 4,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, num_items + 1)

    def _causal_mask(self, seq_len: int, device: torch.device):
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.item_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(self.norm(x))

        padding_mask = input_ids.eq(0)
        hidden = self.encoder(x, mask=self._causal_mask(seq_len, input_ids.device), src_key_padding_mask=padding_mask)

        lengths = (~padding_mask).sum(dim=1).clamp(min=1)
        last_hidden = hidden[torch.arange(bsz, device=input_ids.device), lengths - 1]
        return self.output(last_hidden)
