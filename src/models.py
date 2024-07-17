import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops.layers.torch import Rearrange
from transformers import AutoModel

class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)


class SubjectSpecificLayer(nn.Module):
    def __init__(self, num_subjects: int, hid_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_subjects, hid_dim)

    def forward(self, X: torch.Tensor, subject_idx: torch.Tensor) -> torch.Tensor:
        subject_embedding = self.embedding(subject_idx)
        return X + subject_embedding.unsqueeze(2)

class ComplexModelWithSubject(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, in_channels: int, num_subjects: int, hid_dim: int = 128) -> None:
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
            ConvBlock(hid_dim, hid_dim * 2),
        )
        self.subject_layer = SubjectSpecificLayer(num_subjects, hid_dim * 2)
        self.lstm = nn.LSTM(hid_dim * 2, hid_dim, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim * 2, num_classes),
        )

    def forward(self, X: torch.Tensor, subject_idx: torch.Tensor) -> torch.Tensor:
        X = self.conv_blocks(X)
        X = self.subject_layer(X, subject_idx)
        X, _ = self.lstm(X.permute(0, 2, 1))
        X = X.permute(0, 2, 1)
        return self.head(X)


class CheckpointTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        transformer_model_name: str = "bert-base-uncased",
        hid_dim: int = 768,  # This should match the hidden size of the transformer model
        dropout_rate: float = 0.3  # Dropout rate
    ) -> None:
        super().__init__()

        self.input_linear = nn.Linear(in_channels, hid_dim)  # Linearly transform input to hidden dimension
        self.transformer = AutoModel.from_pretrained(transformer_model_name)
        self.fc = nn.Sequential(
            nn.Linear(hid_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass
        Args:
            X (b, c, t): Input tensor
        Returns:
            X (b, num_classes): Output tensor
        """
        # Assuming X has shape [batch_size, in_channels, seq_len]
        X = X.permute(0, 2, 1)  # Reshape to [batch_size, seq_len, in_channels]
        X = self.input_linear(X)  # Transform to [batch_size, seq_len, hid_dim]

        def custom_forward(*inputs):
            return self.transformer(inputs_embeds=inputs[0])[0]  # Use hidden states

        transformer_out = checkpoint.checkpoint(custom_forward, X)
        X = transformer_out[:, 0, :]  # Use CLS token output
        X = self.fc(X)

        return X