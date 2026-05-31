import torch
import torch.nn as nn

class NetraDriveTemporalGRU(nn.Module):
    """
    Custom Temporal Cognitive Estimator.
    Processes sequential 1D biometric metrics vectors over time 
    to classify fine-grained states like microsleep or cognitive exhaustion.
    """
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, num_layers: int = 2, num_classes: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional Gated Recurrent Unit for sequential context tracking
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        # Deep fully-connected scoring layer mapping to behavior classes
        # (0: Focused, 1: Distracted, 2: Drowsy/Drooping, 3: Active Microsleep)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, Sequence_Length, Input_Dim)
        # Initialize hidden states natively inside the pass execution loop
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate through sequence layers
        out, _ = self.gru(x, h0)
        
        # Extract and pass the terminal time-step state matrix output
        final_sequence_output = out[:, -1, :]
        logits = self.classifier(final_sequence_output)
        return logits