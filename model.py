import torch
import torch.nn as nn
from topomodelx.nn.simplicial.sccn_layer import SCCNLayer
from torch_geometric.nn import global_max_pool

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    

class TemporalSCCN_approach1(nn.Module):
    def __init__(
        self,
        in_channel,
        hidden_channels,
        out_channels,
        n_layers=2
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        # 1. Feature Encoders (Project inputs to hidden dim)
        self.proj_0 = nn.Linear(in_channel, hidden_channels)
        self.proj_1 = nn.Linear(in_channel, hidden_channels)
        self.proj_2 = nn.Linear(in_channel, hidden_channels)

        # 2. SCCN Layers
        self.conv_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.conv_layers.append(
                SCCNLayer(
                    channels=hidden_channels,
                    max_rank=2 ,
                    update_func='sigmoid'
                )
            )

        # 3. Temporal Updaters (GRU for ROLAND)
        self.LSTM = nn.LSTM(hidden_channels, hidden_channels, batch_first=True, bias=True)
        
        # 4. Final Linear transformation
        self.predict = nn.Linear(3*hidden_channels, 2)

    def forward(self, temporal_snapshots):
        """
        Args:
            temporal_snapshots: List of dictionaries. Each dict contains:
                {
                    'features': {0: x0, 1: x1, 2: x2},
                    'incidences': {1: B1, 2: B2},
                    'adjacencies': {0: A0, 1: A1, 2: A2}
                }
        """
        h_0, c_0 = None, None
        
        for t, snapshot_data in enumerate(temporal_snapshots):
            if t != len(temporal_snapshots) - 1:
                features = snapshot_data[0]
                incidences = snapshot_data[1]
                adjacencies = snapshot_data[2]
                
                for key in incidences:
                    incidences[key] = incidences[key].to(device)
                for key in adjacencies:
                    adjacencies[key] = adjacencies[key].to(device)
                
                x_0 = features[0].to(device)
                x_1 = features[1].to(device)
                x_2 = features[2].to(device)
                # --- Project Features ---
                x_0 = self.proj_0(x_0)
                x_1 = self.proj_1(x_1)
                x_2 = self.proj_2(x_2)
                x_0 = torch.sigmoid(x_0)
                x_1 = torch.sigmoid(x_1)
                x_2 = torch.sigmoid(x_2)
                
                # Repack into dictionary for SCCNLayer
                # SCCNLayer expects a dict: {0: tensor, 1: tensor, 2: tensor}
                x_dict = {'rank_0': x_0, 'rank_1': x_1, 'rank_2': x_2}

                # --- Spatial Convolution (SCCN) ---
                for layer in self.conv_layers:
                    # TopoModelX SCCNLayer.forward(features, incidences, adjacencies)
                    x_dict = layer(x_dict, incidences, adjacencies)
                
                # Extract Convolved Features
                z_0, z_1, z_2 = x_dict['rank_0'], x_dict['rank_1'], x_dict['rank_2']

                # --- Temporal Update (ROLAND) ---
                # Initialize hidden states if first step
                if h_0 is None and c_0 is None:
                    h_0 = torch.zeros_like(z_0)
                    c_0 = torch.zeros_like(z_0)
                
                # Update Memory
                output, (h_0, c_0) = self.LSTM(z_0.unsqueeze(1), (h_0.unsqueeze(0), c_0.unsqueeze(0)))
                h_0, c_0 = h_0.squeeze(0), c_0.squeeze(0)
                del features, incidences, adjacencies, x_0, x_1, x_2, x_dict, z_0, z_1, z_2
                torch.cuda.empty_cache()
            
        output_max = output.max(dim=0, keepdim=True).values.squeeze(1)  # Global Max Pooling over nodes
        output_mean = output.mean(dim=0, keepdim=True).squeeze(1)  # Global Mean Pooling over nodes
        output_min = output.min(dim=0, keepdim=True).values.squeeze(1)  # Global Min Pooling over nodes
        #print("Output shapes:", output_max.shape, output_mean.shape, output_min.shape)
        new_output = torch.cat((output_max, output_mean, output_min), dim=1)
        output = self.predict(new_output)
        return torch.nn.functional.softmax(output, dim=-1)  # Return probabilities
    
    
class TemporalSCCN_approach2(nn.Module):
    def __init__(
        self,
        in_channel,
        hidden_channels,
        out_channels,
        n_layers=2
    ):
        super().__init__()
        self.num_layers = n_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        # 1. Feature Encoders (Project inputs to hidden dim)
        self.proj_0 = nn.Linear(in_channel, hidden_channels)
        self.proj_1 = nn.Linear(in_channel, hidden_channels)
        self.proj_2 = nn.Linear(in_channel, hidden_channels)

        # 2. SCCN Layers
        self.conv_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.conv_layers.append(
                SCCNLayer(
                    channels=hidden_channels,
                    max_rank=2 ,
                    update_func='sigmoid'
                )
            )

        # 3. Temporal Updaters (GRU for ROLAND)
        self.temp_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.temp_layers.append(
                nn.LSTM(hidden_channels, hidden_channels, batch_first=True, bias=True)
            )
        
        # 4. Final Linear transformation
        self.predict = nn.Linear(hidden_channels, 2)
        
    def forward(self, temporal_snapshots):
        """
        Args:
            temporal_snapshots: List of dictionaries. Each dict contains:
                {
                    'features': {0: x0, 1: x1, 2: x2},
                    'incidences': {1: B1, 2: B2},
                    'adjacencies': {0: A0, 1: A1, 2: A2}
                }
        """
        h_0, c_0 = None, None
        h_1, c_1 = None, None
        for t, snapshot_data in enumerate(temporal_snapshots):
            if t != len(temporal_snapshots) - 1:
                features = snapshot_data[0]
                incidences = snapshot_data[1]
                adjacencies = snapshot_data[2]
                
                for key in incidences:
                    incidences[key] = incidences[key].to(device)
                for key in adjacencies:
                    adjacencies[key] = adjacencies[key].to(device)
                
                x_0 = features[0].to(device)
                x_1 = features[1].to(device)
                x_2 = features[2].to(device)
                # --- Project Features ---
                x_0 = self.proj_0(x_0)
                x_1 = self.proj_1(x_1)
                x_2 = self.proj_2(x_2)
                x_0 = torch.sigmoid(x_0)
                x_1 = torch.sigmoid(x_1)
                x_2 = torch.sigmoid(x_2)
                
                # Repack into dictionary for SCCNLayer
                # SCCNLayer expects a dict: {0: tensor, 1: tensor, 2: tensor}
                x_dict = {'rank_0': x_0, 'rank_1': x_1, 'rank_2': x_2}

                # --- Spatial Convolution and  Temporal Update (SCCN + ROLAND) ---
                for layer_num in range(self.num_layers):
                    # TopoModelX SCCNLayer.forward(features, incidences, adjacencies)
                    x_dict = self.conv_layers[layer_num](x_dict, incidences, adjacencies)
                
                    # Extract Convolved Features
                    z_0, z_1, z_2 = x_dict['rank_0'], x_dict['rank_1'], x_dict['rank_2']

                    # Initialize hidden states if first step
                    if h_0 is None and c_0 is None:
                        h_0 = torch.zeros_like(z_0)
                        c_0 = torch.zeros_like(z_0)
                    if h_1 is None and c_1 is None:
                        h_1 = torch.zeros_like(z_0)
                        c_1 = torch.zeros_like(z_0)
                
                    # Update Memory
                    if layer_num == 0:
                        output, (h_0, c_0) = self.temp_layers[layer_num](z_0.unsqueeze(1), (h_0.unsqueeze(0), c_0.unsqueeze(0)))
                        h_0, c_0 = h_0.squeeze(0), c_0.squeeze(0)
                    else:
                        output, (h_1, c_1) = self.temp_layers[layer_num](z_0.unsqueeze(1), (h_1.unsqueeze(0), c_1.unsqueeze(0)))
                        h_1, c_1 = h_1.squeeze(0), c_1.squeeze(0)
                    
                    x_dict = {'rank_0': output.squeeze(1), 'rank_1': z_1, 'rank_2': z_2}
                
        output = output.mean(dim=0, keepdim=True).squeeze(1)  # Global Max Pooling over nodes
        output = self.predict(output)
        return torch.nn.functional.softmax(output, dim=-1)  # Return probabilities