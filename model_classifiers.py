import torch
import torch.nn as nn


# --- MediaPipe LSTM classifier ---------------------------------------------
class MediaPipeLSTM(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_layers=2,
                 num_classes=14, dropout=0.3, projection_dim=None):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        if projection_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(input_size, projection_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            lstm_input_size = projection_dim
        else:
            self.projection = None
            lstm_input_size = input_size
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        if self.projection is not None:
            batch, seq_len, features = x.shape
            x = x.reshape(batch * seq_len, features)
            x = self.projection(x)
            x = x.reshape(batch, seq_len, -1)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        forward_hidden = h_n[-2]   
        backward_hidden = h_n[-1]  
        
        # Concatenate forward and backward
        hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        return self.classifier(hidden)


# --- MediaPipe GRU classifier ----------------------------------------------
class MediaPipeGRU(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_layers=2,
                 num_classes=14, dropout=0.3, projection_dim=None):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
            
        if projection_dim is not None:
                self.projection = nn.Sequential(
                    nn.Linear(input_size, projection_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                gru_input_size = projection_dim
        else:
                self.projection = None
                gru_input_size = input_size
            
        self.gru = nn.GRU(
                input_size=gru_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True,
            )
            
        self.classifier = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(hidden_size, num_classes)
            )

    def forward(self, x):
        if self.projection is not None:
            batch, seq_len, features = x.shape
            x = x.reshape(batch * seq_len, features)
            x = self.projection(x)
            x = x.reshape(batch, seq_len, -1)
        _, h_n = self.gru(x)
        forward_hidden = h_n[-2]
        backward_hidden = h_n[-1]
        hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        return self.classifier(hidden)


# --- Temporal convolution residual block ----------------------------------
class _TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, 
                     padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, 
                     padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.downsample(x)


# --- MediaPipe TCN classifier ----------------------------------------------
class MediaPipeTCN(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_levels=4, kernel_size=3,
                 num_classes=14, dropout=0.3, projection_dim=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        if projection_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(input_size, projection_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            tcn_input_size = projection_dim
        else:
            self.projection = None
            tcn_input_size = input_size

        self.input_conv = nn.Conv1d(tcn_input_size, hidden_size, kernel_size=1)
        
        channels = [hidden_size] * num_levels
        layers = []
        in_ch = hidden_size
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(_TemporalConvBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        if self.projection is not None:
            batch, seq_len, features = x.shape
            x = x.reshape(batch * seq_len, features)
            x = self.projection(x)
            x = x.reshape(batch, seq_len, -1)
        x = x.transpose(1, 2)
        x = self.input_conv(x)
        x = self.tcn(x)
        x = x.mean(dim=2)
        return self.classifier(x)
