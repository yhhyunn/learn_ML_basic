# “””
Netlist CSV → DeepSNAP HeteroGraph Conversion

Converts netlist CSV data into a DeepSNAP HeteroGraph for
heterogeneous GNN training (anomaly detection on circuit nets).

Requirements:
pip install torch torch-geometric deepsnap networkx pandas numpy

Graph Structure (Bipartite):
Node Types: ‘instance’, ‘net’
Edge Types: (‘instance’, ‘drives’, ‘net’), (‘net’, ‘loads’, ‘instance’)

Usage:
# With your CSV file:
hetero_graph, metadata = csv_to_deepsnap_hetero(‘your_netlist.csv’)

```
# With sample data (for testing):
python deepsnap_netlist_converter.py
```

“””

import pandas as pd
import numpy as np
import networkx as nx
import torch
from deepsnap.hetero_graph import HeteroGraph
from deepsnap.dataset import GraphDataset

# ============================================================

# 1. Sample Data Generator

# ============================================================

def create_sample_data(n_rows=300):
“”“Generate realistic sample netlist CSV data.”””
np.random.seed(42)

```
blocks = ['A16', 'B08', 'C32', 'D04']
sub_blocks = ['INV', 'BUF', 'NAND', 'NOR', 'MUX', 'DFF', 'AOI']

rows = []
net_counter = 0

for _ in range(n_rows):
    blk = np.random.choice(blocks)
    sub1, sub2 = np.random.choice(sub_blocks, 2)
    idx1, idx2 = np.random.randint(1, 30, size=2)
    
    from_inst = f"{blk}/{sub1}/{idx1}"
    to_inst = f"{blk}/{sub2}/{idx2}"
    while to_inst == from_inst:
        to_inst = f"{blk}/{np.random.choice(sub_blocks)}/{np.random.randint(1, 30)}"
    
    net_name = f"{blk}/NET{net_counter}"
    net_counter += 1
    
    line_res = np.random.lognormal(mean=3.0, sigma=0.8)
    line_cap = np.random.lognormal(mean=2.0, sigma=0.6)
    loading_cap = np.random.lognormal(mean=2.5, sigma=0.7)
    tR = np.random.lognormal(mean=3.5, sigma=0.5)
    tF = np.random.lognormal(mean=3.5, sigma=0.5)
    
    rows.append({
        'FromInst': from_inst, 'Net': net_name, 'ToInst': to_inst,
        'Line_res': round(line_res, 2), 'Line_cap': round(line_cap, 2),
        'Loading_cap': round(loading_cap, 2),
        'tR(ps)': round(tR, 2), 'tF(ps)': round(tF, 2),
    })

# Inject anomalies (~10%)
n_anomalous = n_rows // 10
for i in np.random.choice(n_rows, n_anomalous, replace=False):
    rows[i]['Line_res'] *= np.random.uniform(3, 8)
    rows[i]['tR(ps)'] *= np.random.uniform(2, 5)
    rows[i]['tF(ps)'] *= np.random.uniform(2, 5)
    rows[i]['Loading_cap'] *= np.random.uniform(2, 4)

return pd.DataFrame(rows)
```

# ============================================================

# 2. Feature Engineering

# ============================================================

def compute_node_features(df, all_instances, all_nets):
“””
Compute node features for instance and net nodes.

```
Instance features: [fan_out, fan_in, avg_driven_res, avg_driven_tR, avg_driven_tF]
Net features:      [avg_line_res, avg_line_cap, total_loading_cap, 
                    max_tR, max_tF, fan_out, RC_product]
"""
# --- Instance node features ---
inst_features = {}
for inst in all_instances:
    as_driver = df[df['FromInst'] == inst]
    as_receiver = df[df['ToInst'] == inst]
    feat = np.array([
        len(as_driver),                                                  # fan_out
        len(as_receiver),                                                # fan_in
        as_driver['Line_res'].mean() if len(as_driver) > 0 else 0,      # avg driven resistance
        as_driver['tR(ps)'].mean() if len(as_driver) > 0 else 0,        # avg driven rise delay
        as_driver['tF(ps)'].mean() if len(as_driver) > 0 else 0,        # avg driven fall delay
    ], dtype=np.float32)
    inst_features[inst] = feat

# --- Net node features ---
net_features = {}
for net in all_nets:
    net_rows = df[df['Net'] == net]
    avg_res = net_rows['Line_res'].mean()
    total_lcap = net_rows['Loading_cap'].sum()
    feat = np.array([
        avg_res,                          # avg line resistance
        net_rows['Line_cap'].mean(),      # avg line capacitance
        total_lcap,                       # total loading capacitance
        net_rows['tR(ps)'].max(),         # worst-case rise delay
        net_rows['tF(ps)'].max(),         # worst-case fall delay
        len(net_rows),                    # fan_out (number of receivers)
        avg_res * total_lcap,             # RC product (signal integrity proxy)
    ], dtype=np.float32)
    net_features[net] = feat

return inst_features, net_features
```

def normalize_features(features_dict):
“”“Log-transform + standard scaling on feature dictionary.”””
all_feats = np.stack(list(features_dict.values()))
log_feats = np.log1p(np.abs(all_feats))  # log1p to handle zeros
mean = log_feats.mean(axis=0)
std = log_feats.std(axis=0) + 1e-8
normalized = (log_feats - mean) / std

```
result = {}
for i, key in enumerate(features_dict.keys()):
    result[key] = normalized[i]

return result, mean, std
```

def compute_edge_features(df):
“”“Normalize edge features.”””
edge_feat_cols = [‘Line_res’, ‘Line_cap’, ‘Loading_cap’, ‘tR(ps)’, ‘tF(ps)’]
raw = df[edge_feat_cols].values.astype(np.float32)

```
log_raw = np.log1p(raw)
mean = log_raw.mean(axis=0)
std = log_raw.std(axis=0) + 1e-8
normalized = (log_raw - mean) / std

return normalized, mean, std
```

# ============================================================

# 3. CSV → NetworkX → DeepSNAP HeteroGraph

# ============================================================

def csv_to_deepsnap_hetero(csv_path=None, df=None):
“””
Convert netlist CSV to DeepSNAP HeteroGraph.

```
DeepSNAP's HeteroGraph requires a NetworkX graph where:
  - Each node has 'node_type' (str) and 'node_feature' (torch.Tensor)
  - Each edge has 'edge_type' (str) and 'edge_feature' (torch.Tensor)

Args:
    csv_path: Path to CSV file (columns: FromInst, Net, ToInst, 
              Line_res, Line_cap, Loading_cap, tR(ps), tF(ps))
    df: Or pass a DataFrame directly

Returns:
    hetero_graph: DeepSNAP HeteroGraph object
    metadata: Dict with node/edge mappings and normalization stats
"""
if df is None:
    df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} rows from CSV")

# --- Identify unique nodes ---
all_instances = sorted(set(df['FromInst'].unique()) | set(df['ToInst'].unique()))
all_nets = sorted(df['Net'].unique())

print(f"Unique instances: {len(all_instances)}")
print(f"Unique nets:      {len(all_nets)}")

# --- Compute and normalize features ---
inst_features_raw, net_features_raw = compute_node_features(df, all_instances, all_nets)
inst_features_norm, inst_mean, inst_std = normalize_features(inst_features_raw)
net_features_norm, net_mean, net_std = normalize_features(net_features_raw)
edge_features_norm, edge_mean, edge_std = compute_edge_features(df)

# --- Build NetworkX graph with DeepSNAP-compatible attributes ---
# DeepSNAP reads: node_type (str), node_feature (tensor), 
#                 edge_type (str), edge_feature (tensor)
G = nx.DiGraph()

# Add instance nodes
for inst in all_instances:
    G.add_node(
        inst,
        node_type='instance',
        node_feature=torch.tensor(inst_features_norm[inst], dtype=torch.float),
    )

# Add net nodes
for net in all_nets:
    G.add_node(
        net,
        node_type='net',
        node_feature=torch.tensor(net_features_norm[net], dtype=torch.float),
    )

# Add edges with features
for idx, row in df.iterrows():
    edge_feat = torch.tensor(edge_features_norm[idx], dtype=torch.float)
    
    # FromInst --drives--> Net
    G.add_edge(
        row['FromInst'], row['Net'],
        edge_type='drives',
        edge_feature=edge_feat,
    )
    
    # Net --loads--> ToInst
    G.add_edge(
        row['Net'], row['ToInst'],
        edge_type='loads',
        edge_feature=edge_feat.clone(),
    )

print(f"NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# --- Convert to DeepSNAP HeteroGraph ---
hetero_graph = HeteroGraph(G)

# --- Metadata for later reference ---
metadata = {
    'instance_names': all_instances,
    'net_names': all_nets,
    'inst_feature_names': ['fan_out', 'fan_in', 'avg_driven_res', 'avg_driven_tR', 'avg_driven_tF'],
    'net_feature_names': ['avg_line_res', 'avg_line_cap', 'total_loading_cap', 
                          'max_tR', 'max_tF', 'fan_out', 'RC_product'],
    'edge_feature_names': ['Line_res', 'Line_cap', 'Loading_cap', 'tR', 'tF'],
    'inst_norm': {'mean': inst_mean, 'std': inst_std},
    'net_norm': {'mean': net_mean, 'std': net_std},
    'edge_norm': {'mean': edge_mean, 'std': edge_std},
    'raw_df': df,
}

return hetero_graph, metadata
```

# ============================================================

# 4. HeteroGraph Inspection & Summary

# ============================================================

def inspect_hetero_graph(hg, metadata):
“”“Print detailed summary of the DeepSNAP HeteroGraph.”””
print(”\n” + “=” * 65)
print(”       DeepSNAP HeteroGraph Summary”)
print(”=” * 65)

```
# Node types
print(f"\n--- Node Types ---")
for ntype in hg.node_types:
    nfeat = hg.node_feature[ntype]
    print(f"  '{ntype}':")
    print(f"    Count:             {nfeat.shape[0]}")
    print(f"    Feature dimension: {nfeat.shape[1]}")
    print(f"    Feature names:     {metadata.get(ntype + '_feature_names', metadata.get('inst_feature_names' if ntype == 'instance' else 'net_feature_names'))}")
    print(f"    Feature stats:     mean={nfeat.mean().item():.4f}, std={nfeat.std().item():.4f}")

# Edge types (message types)
print(f"\n--- Edge / Message Types ---")
for mtype in hg.message_types:
    eidx = hg.edge_index[mtype]
    print(f"  {mtype}:")
    print(f"    Edge count:        {eidx.shape[1]}")
    if mtype in hg.edge_feature:
        efeat = hg.edge_feature[mtype]
        print(f"    Feature dimension: {efeat.shape[1]}")
        print(f"    Feature stats:     mean={efeat.mean().item():.4f}, std={efeat.std().item():.4f}")

# Overall
print(f"\n--- Overall ---")
total_nodes = sum(hg.node_feature[nt].shape[0] for nt in hg.node_types)
total_edges = sum(hg.edge_index[mt].shape[1] for mt in hg.message_types)
print(f"  Total nodes:   {total_nodes}")
print(f"  Total edges:   {total_edges}")
print(f"  Node types:    {hg.node_types}")
print(f"  Message types: {hg.message_types}")

print("=" * 65)
```

# ============================================================

# 5. Dataset Split for Anomaly Detection

# ============================================================

def create_dataset_for_anomaly_detection(hetero_graph, task=‘node’):
“””
Create a GraphDataset from the HeteroGraph.

```
For anomaly detection (unsupervised), we use 'node' task 
with reconstruction-based objectives, so the split is on nodes.

Args:
    hetero_graph: DeepSNAP HeteroGraph
    task: 'node' for node-level anomaly detection

Returns:
    dataset: GraphDataset
    train_set, val_set, test_set: Split datasets
"""
dataset = GraphDataset(
    graphs=[hetero_graph],
    task=task,
)

# Split: 70% train, 15% val, 15% test
train_set, val_set, test_set = dataset.split(
    transductive=True,
    split_ratio=[0.7, 0.15, 0.15],
    split_types=None,  # split all node types
    shuffle=True,
)

print("\n--- Dataset Split ---")
print(f"  Task: {task}")
print(f"  Train graphs: {len(train_set)}")
print(f"  Val graphs:   {len(val_set)}")
print(f"  Test graphs:  {len(test_set)}")

# Inspect split indices
for split_name, split_data in [("Train", train_set), ("Val", val_set), ("Test", test_set)]:
    g = split_data[0]
    if hasattr(g, 'node_label_index') and g.node_label_index:
        for ntype, idx in g.node_label_index.items():
            print(f"  {split_name} - {ntype} nodes: {len(idx)}")

return dataset, train_set, val_set, test_set
```

# ============================================================

# 6. Example: Access Data for GNN Training

# ============================================================

def demo_data_access(hetero_graph):
“”“Demonstrate how to access HeteroGraph data for GNN training.”””
print(”\n” + “=” * 65)
print(”       Data Access Demo (for GNN Training Loop)”)
print(”=” * 65)

```
# Node features (dict: node_type -> tensor)
print("\n--- Node Features ---")
for ntype in hetero_graph.node_types:
    feat = hetero_graph.node_feature[ntype]
    print(f"  hetero_graph.node_feature['{ntype}'].shape = {feat.shape}")

# Edge indices (dict: message_type -> tensor [2, num_edges])
print("\n--- Edge Indices ---")
for mtype in hetero_graph.message_types:
    eidx = hetero_graph.edge_index[mtype]
    print(f"  hetero_graph.edge_index[{mtype}].shape = {eidx.shape}")

# Edge features (dict: message_type -> tensor)
print("\n--- Edge Features ---")
for mtype in hetero_graph.message_types:
    if mtype in hetero_graph.edge_feature:
        efeat = hetero_graph.edge_feature[mtype]
        print(f"  hetero_graph.edge_feature[{mtype}].shape = {efeat.shape}")

# Example: Get specific data for one message type
mtype = hetero_graph.message_types[0]
print(f"\n--- Example: Message Type {mtype} ---")
print(f"  Source indices: {hetero_graph.edge_index[mtype][0, :5]}...")
print(f"  Target indices: {hetero_graph.edge_index[mtype][1, :5]}...")

print("\n--- GNN Training Loop Pattern ---")
print("""
# In your training loop:
node_features = hetero_graph.node_feature  # dict
edge_indices  = hetero_graph.edge_index    # dict
edge_features = hetero_graph.edge_feature  # dict

# Pass to HeteroConv layer:
# out = hetero_conv(node_features, edge_indices, edge_features)
""")
print("=" * 65)
```

# ============================================================

# 7. Main

# ============================================================

if **name** == “**main**”:
print(”=” * 65)
print(”  Netlist CSV → DeepSNAP HeteroGraph Pipeline”)
print(”=” * 65)

```
# --- Step 1: Load data ---
# To use your own CSV:
#   hetero_graph, metadata = csv_to_deepsnap_hetero('your_netlist.csv')
print("\n[Step 1] Loading sample data...")
df = create_sample_data(n_rows=300)
print(f"  Sample:\n{df.head(3).to_string(index=False)}")

# --- Step 2: Convert to HeteroGraph ---
print("\n[Step 2] Converting to DeepSNAP HeteroGraph...")
hetero_graph, metadata = csv_to_deepsnap_hetero(df=df)

# --- Step 3: Inspect ---
print("\n[Step 3] Inspecting HeteroGraph...")
inspect_hetero_graph(hetero_graph, metadata)

# --- Step 4: Dataset split ---
print("\n[Step 4] Creating dataset split...")
dataset, train_set, val_set, test_set = create_dataset_for_anomaly_detection(hetero_graph)

# --- Step 5: Demo data access ---
print("\n[Step 5] Data access demo...")
demo_data_access(hetero_graph)

print("\n✅ Pipeline complete!")
print("   Next step: Build a HeteroGNN model for anomaly detection")
print("   (Graph Autoencoder with HeteroSAGEConv layers)")
```






import matplotlib.pyplot as plt
import networkx as nx

# hetero_graph = HeteroGraph(nx_graph)  # 이미 만들어둔 HeteroGraph
G = hetero_graph.G  # 원본 NetworkX 그래프 접근

# 레이아웃 선택
pos = nx.spring_layout(G, k=2, seed=42)  # k를 키우면 노드 간격 넓어짐

# 노드 타입별 색상 분리
instance_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'instance']
net_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'net']

plt.figure(figsize=(20, 14))

# 타입별로 다른 색/모양으로 그리기
nx.draw_networkx_nodes(G, pos, nodelist=instance_nodes,
                       node_color='#065A82', node_shape='s', node_size=300, alpha=0.9)
nx.draw_networkx_nodes(G, pos, nodelist=net_nodes,
                       node_color='#19D3A2', node_shape='o', node_size=300, alpha=0.9)

nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, arrowsize=10)

# 노드 이름 전부 표시
nx.draw_networkx_labels(G, pos, font_size=6, font_color='black')

plt.legend(['Instance', 'Net'], loc='upper left')
plt.title("Circuit Netlist Bipartite Graph")
plt.axis('off')
plt.tight_layout()
plt.savefig('graph_vis.png', dpi=150, bbox_inches='tight')
plt.show()






import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

from deepsnap.hetero_graph import HeteroGraph
from torch_geometric.nn import HeteroConv, SAGEConv


# ================================================================
# 1. CSV → NetworkX → DeepSNAP HeteroGraph (변경 없음)
# ================================================================

def build_hetero_graph(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=['FromInst', 'Net', 'ToInst'], keep='first')
    
    G = nx.DiGraph()
    
    inst_driven_res = defaultdict(list)
    inst_driven_tR = defaultdict(list)
    inst_driven_tF = defaultdict(list)
    inst_fan_out = defaultdict(int)
    inst_fan_in = defaultdict(int)
    
    net_line_res = defaultdict(list)
    net_line_cap = defaultdict(list)
    net_loading_cap = defaultdict(list)
    net_tR = defaultdict(list)
    net_tF = defaultdict(list)
    net_fan_out = defaultdict(int)
    
    for _, row in df.iterrows():
        from_inst = row['FromInst']
        net_name = row['Net']
        to_inst = row['ToInst']
        line_res = row['Line_res']
        line_cap = row['Line_cap']
        loading_cap = row['Loading_cap']
        tR = row['tR(ps)']
        tF = row['tF(ps)']
        
        inst_driven_res[from_inst].append(line_res)
        inst_driven_tR[from_inst].append(tR)
        inst_driven_tF[from_inst].append(tF)
        inst_fan_out[from_inst] += 1
        inst_fan_in[to_inst] += 1
        
        net_line_res[net_name].append(line_res)
        net_line_cap[net_name].append(line_cap)
        net_loading_cap[net_name].append(loading_cap)
        net_tR[net_name].append(tR)
        net_tF[net_name].append(tF)
        net_fan_out[net_name] += 1
    
    all_instances = set(df['FromInst']).union(set(df['ToInst']))
    all_nets = set(df['Net'])
    
    for inst in all_instances:
        driven_res = inst_driven_res.get(inst, [0])
        driven_tR = inst_driven_tR.get(inst, [0])
        driven_tF = inst_driven_tF.get(inst, [0])
        feature = [
            inst_fan_out.get(inst, 0),
            inst_fan_in.get(inst, 0),
            np.mean(driven_res),
            np.mean(driven_tR),
            np.mean(driven_tF),
        ]
        G.add_node(inst, node_type='instance',
                   node_feature=torch.tensor(feature, dtype=torch.float32),
                   name=inst)
    
    for net in all_nets:
        avg_res = np.mean(net_line_res[net])
        avg_cap = np.mean(net_line_cap[net])
        total_load = np.sum(net_loading_cap[net])
        max_tR = np.max(net_tR[net])
        max_tF = np.max(net_tF[net])
        fo = net_fan_out[net]
        rc_product = avg_res * total_load
        feature = [avg_res, avg_cap, total_load, max_tR, max_tF, fo, rc_product]
        G.add_node(net, node_type='net',
                   node_feature=torch.tensor(feature, dtype=torch.float32),
                   name=net)
    
    for _, row in df.iterrows():
        edge_feat = torch.tensor([
            row['Line_res'], row['Line_cap'], row['Loading_cap'],
            row['tR(ps)'], row['tF(ps)']
        ], dtype=torch.float32)
        G.add_edge(row['FromInst'], row['Net'],
                   edge_type='drives', edge_feature=edge_feat)
        G.add_edge(row['Net'], row['ToInst'],
                   edge_type='loads', edge_feature=edge_feat)
    
    hetero_graph = HeteroGraph(G)
    return hetero_graph, G


# ================================================================
# 2. [NEW] Subgraph 분석 & 분류
# ================================================================

def analyze_components(nx_graph):
    """Connected component 분석 및 크기별 분류"""
    undirected = nx_graph.to_undirected()
    components = list(nx.connected_components(undirected))
    
    sizes = [len(c) for c in components]
    
    print(f"=== Component Analysis ===")
    print(f"Total components: {len(components)}")
    print(f"Total nodes: {sum(sizes)}")
    print(f"Size distribution:")
    print(f"  1 node:     {sum(1 for s in sizes if s == 1)}")
    print(f"  2-3 nodes:  {sum(1 for s in sizes if 2 <= s <= 3)}")
    print(f"  4-6 nodes:  {sum(1 for s in sizes if 4 <= s <= 6)}")
    print(f"  7-15 nodes: {sum(1 for s in sizes if 7 <= s <= 15)}")
    print(f"  16+ nodes:  {sum(1 for s in sizes if s >= 16)}")
    print(f"  Largest:    {max(sizes)} nodes")
    print(f"  Median:     {np.median(sizes):.0f} nodes")
    
    return components, sizes


def classify_nodes_by_component(nx_graph, min_gnn_size=5):
    """
    노드를 component 크기에 따라 GNN 대상 / MLP fallback 대상으로 분류
    
    Args:
        min_gnn_size: 이 크기 이상의 component만 GNN 학습 대상
                      (2-hop이 의미 있으려면 최소 5개 노드 필요)
    Returns:
        gnn_nodes: GNN 학습 대상 노드 set
        mlp_nodes: MLP fallback 대상 노드 set
    """
    undirected = nx_graph.to_undirected()
    components = list(nx.connected_components(undirected))
    
    gnn_nodes = set()
    mlp_nodes = set()
    
    for comp in components:
        if len(comp) >= min_gnn_size:
            gnn_nodes.update(comp)
        else:
            mlp_nodes.update(comp)
    
    print(f"\nNode classification:")
    print(f"  GNN target (component >= {min_gnn_size}): {len(gnn_nodes)} nodes")
    print(f"  MLP fallback (component < {min_gnn_size}): {len(mlp_nodes)} nodes")
    
    return gnn_nodes, mlp_nodes


# ================================================================
# 3. Feature 정규화 (전체 데이터 기준으로 통일)
# ================================================================

class FeatureNormalizer:
    """학습 시 fit, 추론 시 transform 분리"""
    
    def __init__(self):
        self.mean = {}
        self.std = {}
    
    def fit_transform(self, hetero_graph):
        for ntype in hetero_graph.node_feature:
            feat = hetero_graph.node_feature[ntype]
            self.mean[ntype] = feat.mean(dim=0, keepdim=True)
            self.std[ntype] = feat.std(dim=0, keepdim=True) + 1e-8
            hetero_graph.node_feature[ntype] = (feat - self.mean[ntype]) / self.std[ntype]
        return hetero_graph
    
    def transform(self, x, ntype):
        return (x - self.mean[ntype]) / self.std[ntype]
    
    def inverse_transform(self, x, ntype):
        return x * self.std[ntype] + self.mean[ntype]


# ================================================================
# 4. [NEW] Adaptive GNN Encoder (layer 수 조정 가능)
# ================================================================

class AdaptiveHeteroEncoder(nn.Module):
    """
    Subgraph 크기에 맞게 layer 수를 선택할 수 있는 Encoder.
    작은 그래프에서는 1-layer, 큰 그래프에서는 2-layer 사용.
    """
    
    def __init__(self, in_channels_dict, hidden_dim, latent_dim):
        super().__init__()
        
        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(in_ch, hidden_dim)
            for ntype, in_ch in in_channels_dict.items()
        })
        
        self.conv1 = HeteroConv({
            ('instance', 'drives', 'net'): SAGEConv(hidden_dim, hidden_dim),
            ('net', 'loads', 'instance'): SAGEConv(hidden_dim, hidden_dim),
        }, aggr='mean')
        
        self.conv2 = HeteroConv({
            ('instance', 'drives', 'net'): SAGEConv(hidden_dim, latent_dim),
            ('net', 'loads', 'instance'): SAGEConv(hidden_dim, latent_dim),
        }, aggr='mean')
        
        # 1-layer만 쓸 때를 위한 projection
        self.one_layer_proj = nn.ModuleDict({
            ntype: nn.Linear(hidden_dim, latent_dim)
            for ntype in in_channels_dict
        })
        
        self.bn1 = nn.ModuleDict({
            ntype: nn.BatchNorm1d(hidden_dim)
            for ntype in in_channels_dict
        })
    
    def forward(self, x_dict, edge_index_dict, num_layers=2):
        h_dict = {ntype: self.input_proj[ntype](x) for ntype, x in x_dict.items()}
        
        # Layer 1
        h_dict = self.conv1(h_dict, edge_index_dict)
        h_dict = {k: self.bn1[k](F.relu(v)) for k, v in h_dict.items()}
        
        if num_layers == 1:
            z_dict = {k: self.one_layer_proj[k](v) for k, v in h_dict.items()}
        else:
            z_dict = self.conv2(h_dict, edge_index_dict)
        
        return z_dict


# ================================================================
# 5. [NEW] MLP Fallback (작은 subgraph용)
# ================================================================

class MLPAnomalyDetector(nn.Module):
    """
    작은 subgraph 노드용 — message passing 없이 feature만으로 학습.
    GNN과 동일한 autoencoder 구조지만 graph 구조를 안 씀.
    """
    
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
    
    def anomaly_score(self, x):
        x_hat, _ = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=1)


# ================================================================
# 6. Feature Decoder (변경 없음)
# ================================================================

class FeatureDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels_dict):
        super().__init__()
        self.decoders = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(latent_dim, latent_dim * 2),
                nn.ReLU(),
                nn.Linear(latent_dim * 2, out_ch),
            )
            for ntype, out_ch in out_channels_dict.items()
        })
    
    def forward(self, z_dict):
        return {ntype: self.decoders[ntype](z) for ntype, z in z_dict.items()}


# ================================================================
# 7. [UPDATED] Hybrid Graph Autoencoder
# ================================================================

class HybridAutoencoder(nn.Module):
    """GNN (큰 subgraph) + MLP (작은 subgraph) 하이브리드"""
    
    def __init__(self, in_channels_dict, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = AdaptiveHeteroEncoder(in_channels_dict, hidden_dim, latent_dim)
        self.decoder = FeatureDecoder(latent_dim, in_channels_dict)
        
        # 노드 타입별 MLP fallback
        self.mlp_fallbacks = nn.ModuleDict({
            ntype: MLPAnomalyDetector(in_ch, hidden_dim, latent_dim)
            for ntype, in_ch in in_channels_dict.items()
        })
    
    def forward(self, x_dict, edge_index_dict, num_layers=2):
        z_dict = self.encoder(x_dict, edge_index_dict, num_layers)
        x_hat_dict = self.decoder(z_dict)
        return x_hat_dict, z_dict
    
    def compute_loss(self, x_dict, x_hat_dict):
        total_loss = 0
        for ntype in x_dict:
            total_loss += F.mse_loss(x_hat_dict[ntype], x_dict[ntype])
        return total_loss
    
    def anomaly_score(self, x_dict, x_hat_dict):
        scores = {}
        for ntype in x_dict:
            scores[ntype] = ((x_dict[ntype] - x_hat_dict[ntype]) ** 2).mean(dim=1)
        return scores


# ================================================================
# 8. DeepSNAP → PyG dict 변환
# ================================================================

def hetero_graph_to_pyg_dicts(hetero_graph):
    x_dict = {}
    for ntype in hetero_graph.node_feature:
        x_dict[ntype] = hetero_graph.node_feature[ntype]
    
    edge_index_dict = {}
    for etype in hetero_graph.edge_index:
        edge_index_dict[etype] = hetero_graph.edge_index[etype]
    
    return x_dict, edge_index_dict


# ================================================================
# 9. [NEW] 노드 인덱스 매핑
# ================================================================

def build_node_index_maps(nx_graph, hetero_graph):
    """
    NetworkX 노드 이름 → DeepSNAP 내부 인덱스 매핑.
    anomaly score를 노드 이름과 연결하기 위해 필요.
    """
    maps = {}
    for ntype in hetero_graph.node_feature:
        # DeepSNAP은 node_type별로 0부터 인덱싱
        typed_nodes = [n for n, d in nx_graph.nodes(data=True)
                       if d.get('node_type') == ntype]
        maps[ntype] = typed_nodes  # index i → node name
    return maps


# ================================================================
# 10. [NEW] 학습 + 평가 통합
# ================================================================

def train_epoch(model, x_dict, edge_index_dict, optimizer):
    model.train()
    optimizer.zero_grad()
    x_hat_dict, z_dict = model(x_dict, edge_index_dict)
    loss = model.compute_loss(x_dict, x_hat_dict)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_mlp_epoch(mlp_model, x, optimizer):
    mlp_model.train()
    optimizer.zero_grad()
    x_hat, z = mlp_model(x)
    loss = F.mse_loss(x_hat, x)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def compute_all_scores(model, x_dict, edge_index_dict,
                       mlp_models, mlp_x_dict,
                       node_maps, gnn_nodes, mlp_nodes):
    """GNN score + MLP score를 합쳐서 전체 net 노드의 anomaly score 산출"""
    model.eval()
    
    # --- GNN scores (큰 component) ---
    x_hat_dict, z_dict = model(x_dict, edge_index_dict)
    gnn_scores = model.anomaly_score(x_dict, x_hat_dict)
    
    # --- MLP scores (작은 component) ---
    mlp_scores = {}
    for ntype, mlp in mlp_models.items():
        mlp.eval()
        if ntype in mlp_x_dict and mlp_x_dict[ntype].shape[0] > 0:
            mlp_scores[ntype] = mlp.anomaly_score(mlp_x_dict[ntype])
    
    # --- 합치기: 모든 net 노드의 최종 score ---
    net_names = node_maps['net']
    all_scores = {}
    
    gnn_net_names = [n for n in net_names if n in gnn_nodes]
    mlp_net_names = [n for n in net_names if n in mlp_nodes]
    
    # GNN net 인덱스 → score
    gnn_net_indices = [i for i, n in enumerate(net_names) if n in gnn_nodes]
    for local_idx, global_idx in enumerate(gnn_net_indices):
        name = net_names[global_idx]
        all_scores[name] = gnn_scores['net'][local_idx].item()
    
    # MLP net 인덱스 → score
    if 'net' in mlp_scores:
        mlp_net_indices = [i for i, n in enumerate(net_names) if n in mlp_nodes]
        for local_idx, global_idx in enumerate(mlp_net_indices):
            name = net_names[global_idx]
            all_scores[name] = mlp_scores['net'][local_idx].item()
    
    return all_scores


# ================================================================
# 11. [UPDATED] 메인 실행
# ================================================================

def main():
    csv_path = 'your_netlist.csv'  # ← 너의 CSV 경로
    
    # ---- Step 1: 그래프 구성 ----
    print("Building graph from CSV...")
    hetero_graph, nx_graph = build_hetero_graph(csv_path)
    
    # ---- Step 2: Component 분석 ----
    components, sizes = analyze_components(nx_graph)
    gnn_nodes, mlp_nodes = classify_nodes_by_component(nx_graph, min_gnn_size=5)
    
    # ---- Step 3: GNN용 / MLP용 subgraph 분리 ----
    gnn_subgraph = nx_graph.subgraph(gnn_nodes).copy()
    mlp_subgraph = nx_graph.subgraph(mlp_nodes).copy()
    
    print(f"\nGNN subgraph: {gnn_subgraph.number_of_nodes()} nodes, "
          f"{gnn_subgraph.number_of_edges()} edges")
    print(f"MLP subgraph: {mlp_subgraph.number_of_nodes()} nodes")
    
    # ---- Step 4: 정규화 (전체 기준) ----
    normalizer = FeatureNormalizer()
    hetero_graph = normalizer.fit_transform(hetero_graph)
    
    # ---- Step 5: GNN 데이터 준비 ----
    if gnn_subgraph.number_of_nodes() > 0:
        gnn_hetero = HeteroGraph(gnn_subgraph)
        # 정규화는 전체 기준 mean/std 사용
        for ntype in gnn_hetero.node_feature:
            gnn_hetero.node_feature[ntype] = normalizer.transform(
                gnn_hetero.node_feature[ntype], ntype)
        
        x_dict, edge_index_dict = hetero_graph_to_pyg_dicts(gnn_hetero)
        in_channels_dict = {ntype: x.shape[1] for ntype, x in x_dict.items()}
        node_maps = build_node_index_maps(gnn_subgraph, gnn_hetero)
    else:
        print("No components large enough for GNN!")
        x_dict, edge_index_dict = {}, {}
        in_channels_dict = {}
        node_maps = {}
    
    # ---- Step 6: MLP 데이터 준비 ----
    mlp_x_dict = {}
    mlp_node_maps = {}
    for ntype in ['instance', 'net']:
        typed_nodes = [n for n, d in mlp_subgraph.nodes(data=True)
                       if d.get('node_type') == ntype]
        if typed_nodes:
            feats = torch.stack([mlp_subgraph.nodes[n]['node_feature']
                                for n in typed_nodes])
            mlp_x_dict[ntype] = normalizer.transform(feats, ntype)
            mlp_node_maps[ntype] = typed_nodes
    
    # ---- Step 7: 모델 생성 ----
    hidden_dim = 32
    latent_dim = 16
    
    # GNN model
    if in_channels_dict:
        gnn_model = HybridAutoencoder(in_channels_dict, hidden_dim, latent_dim)
        gnn_optimizer = torch.optim.Adam(gnn_model.parameters(),
                                          lr=0.005, weight_decay=1e-5)
        print(f"\nGNN parameters: {sum(p.numel() for p in gnn_model.parameters()):,}")
    
    # MLP models (노드 타입별)
    mlp_models = nn.ModuleDict()
    mlp_optimizers = {}
    for ntype, x in mlp_x_dict.items():
        mlp_models[ntype] = MLPAnomalyDetector(x.shape[1], hidden_dim, latent_dim)
        mlp_optimizers[ntype] = torch.optim.Adam(
            mlp_models[ntype].parameters(), lr=0.005, weight_decay=1e-5)
    
    if mlp_x_dict:
        total_mlp_params = sum(p.numel() for p in mlp_models.parameters())
        print(f"MLP parameters: {total_mlp_params:,}")
    
    # ---- Step 8: 학습 ----
    num_epochs = 300
    best_gnn_loss = float('inf')
    best_mlp_loss = {ntype: float('inf') for ntype in mlp_x_dict}
    patience = 30
    gnn_patience_counter = 0
    
    print(f"\n=== Training ===")
    
    for epoch in range(1, num_epochs + 1):
        # GNN 학습
        gnn_loss = 0
        if in_channels_dict:
            gnn_loss = train_epoch(gnn_model, x_dict, edge_index_dict, gnn_optimizer)
            if gnn_loss < best_gnn_loss:
                best_gnn_loss = gnn_loss
                gnn_patience_counter = 0
                torch.save(gnn_model.state_dict(), 'best_gnn_model.pt')
            else:
                gnn_patience_counter += 1
        
        # MLP 학습
        mlp_loss = {}
        for ntype in mlp_x_dict:
            ml = train_mlp_epoch(mlp_models[ntype], mlp_x_dict[ntype],
                                  mlp_optimizers[ntype])
            mlp_loss[ntype] = ml
            if ml < best_mlp_loss[ntype]:
                best_mlp_loss[ntype] = ml
                torch.save(mlp_models[ntype].state_dict(),
                           f'best_mlp_{ntype}.pt')
        
        if epoch % 30 == 0:
            mlp_str = ", ".join(f"MLP-{k}: {v:.6f}" for k, v in mlp_loss.items())
            print(f"Epoch {epoch:3d} | GNN: {gnn_loss:.6f} | {mlp_str}")
        
        if gnn_patience_counter >= patience and in_channels_dict:
            print(f"GNN early stopping at epoch {epoch}")
            break
    
    # ---- Step 9: 평가 ----
    print(f"\n=== Evaluation ===")
    
    # Best model 로드
    if in_channels_dict:
        gnn_model.load_state_dict(torch.load('best_gnn_model.pt'))
        gnn_model.eval()
    
    for ntype in mlp_x_dict:
        mlp_models[ntype].load_state_dict(torch.load(f'best_mlp_{ntype}.pt'))
        mlp_models[ntype].eval()
    
    # 전체 net anomaly scores
    all_net_scores = {}
    
    # GNN scores
    if in_channels_dict:
        with torch.no_grad():
            x_hat_dict, z_dict = gnn_model(x_dict, edge_index_dict)
            gnn_scores = gnn_model.anomaly_score(x_dict, x_hat_dict)
        
        if 'net' in node_maps:
            for i, name in enumerate(node_maps['net']):
                all_net_scores[name] = gnn_scores['net'][i].item()
    
    # MLP scores
    if 'net' in mlp_x_dict:
        with torch.no_grad():
            mlp_sc = mlp_models['net'].anomaly_score(mlp_x_dict['net'])
        for i, name in enumerate(mlp_node_maps.get('net', [])):
            all_net_scores[name] = mlp_sc[i].item()
    
    # ---- Step 10: 결과 출력 ----
    if not all_net_scores:
        print("No net scores computed!")
        return
    
    scores_array = np.array(list(all_net_scores.values()))
    names_array = list(all_net_scores.keys())
    
    print(f"\nTotal nets scored: {len(scores_array)}")
    print(f"  GNN-scored: {len([n for n in names_array if n in gnn_nodes])}")
    print(f"  MLP-scored: {len([n for n in names_array if n in mlp_nodes])}")
    print(f"\nAnomaly Score Statistics:")
    print(f"  Mean:   {scores_array.mean():.6f}")
    print(f"  Std:    {scores_array.std():.6f}")
    print(f"  Median: {np.median(scores_array):.6f}")
    print(f"  Max:    {scores_array.max():.6f}")
    
    # Top-K
    top_k = min(20, len(names_array))
    sorted_indices = np.argsort(scores_array)[::-1][:top_k]
    
    print(f"\n=== Top {top_k} Anomalous Nets ===")
    print(f"{'Rank':<6}{'Net Name':<35}{'Score':<12}{'Method':<8}")
    print("-" * 61)
    for rank, idx in enumerate(sorted_indices, 1):
        name = names_array[idx]
        method = "GNN" if name in gnn_nodes else "MLP"
        print(f"{rank:<6}{name:<35}{scores_array[idx]:<12.6f}{method:<8}")
    
    # IQR threshold
    q75 = np.percentile(scores_array, 75)
    q25 = np.percentile(scores_array, 25)
    iqr = q75 - q25
    threshold = q75 + 1.5 * iqr
    
    anomalous_mask = scores_array > threshold
    print(f"\n=== Anomaly Summary (threshold: {threshold:.6f}) ===")
    print(f"Anomalous nets: {anomalous_mask.sum()} / {len(scores_array)}")
    
    anomalous_nets = [(names_array[i], scores_array[i])
                      for i in range(len(names_array)) if anomalous_mask[i]]
    anomalous_nets.sort(key=lambda x: x[1], reverse=True)
    
    for name, score in anomalous_nets:
        method = "GNN" if name in gnn_nodes else "MLP"
        print(f"  [{method}] {name}: {score:.6f}")
    
    return gnn_model, mlp_models, all_net_scores


if __name__ == '__main__':
    gnn_model, mlp_models, scores = main()
    
    
    
    
    
    
    
    
    
“””
CSV Comparison Script (old vs new)

- Validates column structure between old and new CSV files
- Compares rows by a unique key column
- Carries over ‘Check’ and ‘Comment’ values from old to new for fully matching rows
  “””

import polars as pl
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────

# Configuration

# ──────────────────────────────────────────────────────────────────────────────

# 비교 시 항상 제외할 컬럼들 (사용자 메타 컬럼)

META_COLUMNS = [“Check”, “Comment”]

# 추가로 비교에서 제외할 컬럼이 있다면 여기에 (필요 없으면 빈 리스트)

EXTRA_EXCLUDE_COLUMNS: list[str] = []

# Row 매칭의 기준이 되는 unique key 컬럼

KEY_COLUMN = “Name”  # ← 실제 key 컬럼명으로 변경

# ──────────────────────────────────────────────────────────────────────────────

# Validation

# ──────────────────────────────────────────────────────────────────────────────

def validate_columns(old: pl.DataFrame, new: pl.DataFrame) -> list[str]:
“””
old/new 컬럼 구조를 검증하고, 비교에 사용할 컬럼 리스트를 반환한다.

```
Rules:
  - len(new) > len(old)  → Error (새 컬럼이 추가된 상태로는 정확한 비교 불가)
  - len(old) > len(new)  → old가 new 컬럼을 모두 포함해야 함. 그렇지 않으면 Error.
                           포함하면 new 컬럼 기준으로 비교 진행.
  - len(old) == len(new) → 컬럼 집합이 동일해야 함.
"""
old_cols = set(old.columns)
new_cols = set(new.columns)

if len(new.columns) > len(old.columns):
    extra = new_cols - old_cols
    raise ValueError(
        f"new CSV has more columns than old CSV "
        f"(old={len(old.columns)}, new={len(new.columns)}). "
        f"Extra columns in new: {sorted(extra)}"
    )

if len(old.columns) > len(new.columns):
    if not new_cols.issubset(old_cols):
        missing = new_cols - old_cols
        raise ValueError(
            f"old CSV does not contain all columns of new CSV. "
            f"Missing in old: {sorted(missing)}"
        )
    # new 컬럼 기준으로 비교
    return list(new.columns)

# 같은 개수일 때 — 컬럼 집합도 같아야 함
if old_cols != new_cols:
    raise ValueError(
        f"Column sets differ. "
        f"Only in old: {sorted(old_cols - new_cols)}, "
        f"Only in new: {sorted(new_cols - old_cols)}"
    )
return list(new.columns)
```

def validate_key_column(
old: pl.DataFrame, new: pl.DataFrame, key: str
) -> None:
“”“Key 컬럼 존재 여부와 unique 여부 검증.”””
for label, df in [(“old”, old), (“new”, new)]:
if key not in df.columns:
raise ValueError(f”Key column ‘{key}’ not found in {label} CSV.”)
n_total = df.height
n_unique = df.select(pl.col(key).n_unique()).item()
if n_total != n_unique:
raise ValueError(
f”Key column ‘{key}’ is not unique in {label} CSV “
f”({n_total} rows, {n_unique} unique values).”
)

# ──────────────────────────────────────────────────────────────────────────────

# Comparison

# ──────────────────────────────────────────────────────────────────────────────

def compare_and_carry_over(
old: pl.DataFrame,
new: pl.DataFrame,
key: str,
common_columns: list[str],
exclude_columns: list[str],
) -> tuple[pl.DataFrame, dict]:
“””
Key 기준으로 old/new를 매칭하고, 비교 대상 컬럼 값이 모두 일치하는 row에 한해
old의 Check/Comment 값을 new에 적용한다.

```
Returns:
    updated_new: Check/Comment가 carry-over된 new DataFrame
    stats: 통계 정보 dict
"""
# 비교 대상 컬럼 = common_columns − (META + EXTRA_EXCLUDE) − key
compare_cols = [
    c for c in common_columns
    if c not in exclude_columns and c != key
]

if not compare_cols:
    raise ValueError("No columns left to compare after exclusions.")

# old에서 필요한 컬럼만 추출 (key + 비교 컬럼 + 메타 컬럼)
old_meta_cols = [c for c in META_COLUMNS if c in old.columns]
old_subset = old.select([key] + compare_cols + old_meta_cols)

# 비교 컬럼명에 suffix 붙여서 join (충돌 방지)
old_for_join = old_subset.rename(
    {c: f"{c}__old" for c in compare_cols}
    | {c: f"{c}__old" for c in old_meta_cols}
)

# new에 old를 left join
joined = new.join(old_for_join, on=key, how="left")

# 모든 비교 컬럼에서 old == new 인지 확인하는 마스크 생성
#   - null/null도 일치로 간주하기 위해 eq_missing 사용
match_expr = pl.lit(True)
for c in compare_cols:
    match_expr = match_expr & pl.col(c).eq_missing(pl.col(f"{c}__old"))

# key가 old에 존재해야 매칭 가능 → key__old가 null이 아닌 경우만
# (left join이므로 old에 없으면 모든 __old 컬럼이 null)
# → match_expr만으로는 "둘 다 null"인 컬럼이 많을 때 False positive 위험이 있어,
#   old에 해당 key가 실재했는지 별도로 확인
old_keys = set(old.select(key).to_series().to_list())
joined = joined.with_columns(
    pl.col(key).is_in(list(old_keys)).alias("__key_in_old")
)

fully_matched = pl.col("__key_in_old") & match_expr

# Check/Comment carry-over: 일치하는 row에 한해 old 값으로 덮어쓰기
update_exprs = []
for meta in META_COLUMNS:
    if meta in new.columns and f"{meta}__old" in joined.columns:
        update_exprs.append(
            pl.when(fully_matched)
            .then(pl.col(f"{meta}__old"))
            .otherwise(pl.col(meta))
            .alias(meta)
        )
    elif meta not in new.columns and f"{meta}__old" in joined.columns:
        # new에 메타 컬럼이 없으면 매칭된 row만 값 채우고 나머지는 null
        update_exprs.append(
            pl.when(fully_matched)
            .then(pl.col(f"{meta}__old"))
            .otherwise(None)
            .alias(meta)
        )

if update_exprs:
    joined = joined.with_columns(update_exprs)

# 통계 산출
n_new = new.height
n_matched = joined.filter(fully_matched).height
n_key_hit = joined.filter(pl.col("__key_in_old")).height
n_key_miss = n_new - n_key_hit
n_value_diff = n_key_hit - n_matched

stats = {
    "total_new_rows": n_new,
    "key_found_in_old": n_key_hit,
    "key_not_in_old": n_key_miss,
    "fully_matched_rows": n_matched,
    "value_mismatch_rows": n_value_diff,
    "carried_over_columns": [m for m in META_COLUMNS if m in old.columns],
    "compared_columns": compare_cols,
}

# 정리: 임시 컬럼 제거하고 new의 원래 컬럼 순서 유지
drop_cols = [c for c in joined.columns
             if c.endswith("__old") or c == "__key_in_old"]
result = joined.drop(drop_cols)

# new에 원래 없던 META 컬럼이 추가됐다면 그대로 유지, 아니면 원래 순서로
final_cols = list(new.columns)
for meta in META_COLUMNS:
    if meta in result.columns and meta not in final_cols:
        final_cols.append(meta)
result = result.select(final_cols)

return result, stats
```

# ──────────────────────────────────────────────────────────────────────────────

# Main

# ──────────────────────────────────────────────────────────────────────────────

def run(old_path: str | Path, new_path: str | Path, out_path: str | Path) -> None:
old = pl.read_csv(old_path)
new = pl.read_csv(new_path)

```
common_columns = validate_columns(old, new)
validate_key_column(old, new, KEY_COLUMN)

exclude = META_COLUMNS + EXTRA_EXCLUDE_COLUMNS
updated_new, stats = compare_and_carry_over(
    old=old,
    new=new,
    key=KEY_COLUMN,
    common_columns=common_columns,
    exclude_columns=exclude,
)

updated_new.write_csv(out_path)

print("=" * 60)
print("CSV Comparison Summary")
print("=" * 60)
for k, v in stats.items():
    if isinstance(v, list):
        print(f"  {k}: {len(v)} columns")
    else:
        print(f"  {k}: {v}")
print(f"\nOutput written to: {out_path}")
```

if **name** == “**main**”:
run(
old_path=“old.csv”,
new_path=“new.csv”,
out_path=“new_updated.csv”,
)



“””
Batch CSV Comparison: dir0 (old) vs dir1 (new) — hardcoded config version

For each filename that exists in both dir0 and dir1 (top-level only):

- Load each pair as old/new
- Run the comparison logic from csv_compare
- Write the carry-over result into dir1 with a suffix indicating the source

Per-file errors are logged and skipped; the batch continues.

Usage:
python batch_compare_hardcoded.py <dir0> <dir1>
“””

import argparse
import sys
import traceback
from pathlib import Path

import polars as pl

import csv_compare

# ──────────────────────────────────────────────────────────────────────────────

# Hardcoded configuration

# ──────────────────────────────────────────────────────────────────────────────

# 파일명별 key 컬럼 매핑. 매핑에 없는 파일은 DEFAULT_KEYS를 사용한다.

KEY_COLUMNS_MAP: dict[str, list[str]] = {
“netlist.csv”:   [“Name”, “Net”],
“parasitic.csv”: [“Instance”, “Pin”],
# 필요한 파일 추가…
}

# 위 매핑에 없는 파일에 적용할 기본 key

DEFAULT_KEYS: list[str] = [“Name”]

# 모든 파일에서 비교 전에 제거할 컬럼 (없으면 빈 리스트)

EXTRA_EXCLUDE_COLUMNS: list[str] = []

def resolve_keys(filename: str) -> list[str]:
“”“Get KEY_COLUMNS for this filename, falling back to default.”””
return KEY_COLUMNS_MAP.get(filename, DEFAULT_KEYS)

# ──────────────────────────────────────────────────────────────────────────────

# File discovery

# ──────────────────────────────────────────────────────────────────────────────

def find_common_csv_files(
dir0: Path, dir1: Path
) -> tuple[list[str], list[str], list[str]]:
“””
Find CSV files that exist in both directories (top-level, by filename).

```
Returns:
    common: filenames present in both
    only_in_dir0: filenames only in dir0
    only_in_dir1: filenames only in dir1
"""
files0 = {p.name for p in dir0.iterdir()
          if p.is_file() and p.suffix.lower() == ".csv"}
files1 = {p.name for p in dir1.iterdir()
          if p.is_file() and p.suffix.lower() == ".csv"}
common = sorted(files0 & files1)
only0 = sorted(files0 - files1)
only1 = sorted(files1 - files0)
return common, only0, only1
```

# ──────────────────────────────────────────────────────────────────────────────

# Per-file comparison

# ──────────────────────────────────────────────────────────────────────────────

def compare_one_file(
old_path: Path,
new_path: Path,
out_path: Path,
keys: list[str],
extra_exclude: list[str],
) -> dict:
“””
Run the full compare pipeline for one file pair.
Takes per-file keys/exclude as arguments instead of reading module globals.
“””
old = pl.read_csv(old_path)
new = pl.read_csv(new_path)

```
old = csv_compare.drop_excluded_columns(old, extra_exclude)
new = csv_compare.drop_excluded_columns(new, extra_exclude)

common_columns = csv_compare.validate_columns(old, new)
csv_compare.validate_key_columns(old, new, keys)

updated_new, stats = csv_compare.compare_and_carry_over(
    old=old,
    new=new,
    keys=keys,
    common_columns=common_columns,
)

updated_new.write_csv(out_path)
return stats
```

# ──────────────────────────────────────────────────────────────────────────────

# Batch driver

# ──────────────────────────────────────────────────────────────────────────────

def build_output_path(dir1: Path, filename: str, dir0_label: str) -> Path:
“””
e.g. netlist.csv + dir0_label=‘baseline’ → dir1/netlist__vs_baseline.csv
“””
stem = Path(filename).stem
suffix = Path(filename).suffix
return dir1 / f”{stem}_*vs*{dir0_label}{suffix}”

def run_batch(dir0: Path, dir1: Path) -> None:
common, only0, only1 = find_common_csv_files(dir0, dir1)
dir0_label = dir0.name

```
print("=" * 70)
print(f"Batch CSV Comparison")
print(f"  dir0 (old): {dir0}")
print(f"  dir1 (new): {dir1}")
print("=" * 70)
print(f"Common files     : {len(common)}")
print(f"Only in dir0     : {len(only0)}  {only0 if only0 else ''}")
print(f"Only in dir1     : {len(only1)}  {only1 if only1 else ''}")
print("-" * 70)

success: list[str] = []
failed: list[tuple[str, str]] = []

for filename in common:
    keys = resolve_keys(filename)
    old_path = dir0 / filename
    new_path = dir1 / filename
    out_path = build_output_path(dir1, filename, dir0_label)

    print(f"\n[{filename}]  keys={keys}")
    try:
        stats = compare_one_file(
            old_path=old_path,
            new_path=new_path,
            out_path=out_path,
            keys=keys,
            extra_exclude=EXTRA_EXCLUDE_COLUMNS,
        )
        print(f"  matched={stats['fully_matched_rows']}/{stats['total_new_rows']}  "
              f"key_miss={stats['key_not_in_old']}  "
              f"value_diff={stats['value_mismatch_rows']}")
        print(f"  → {out_path.name}")
        success.append(filename)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"  [SKIP] {msg}")
        failed.append((filename, msg))

print("\n" + "=" * 70)
print(f"Batch summary: {len(success)} succeeded, {len(failed)} skipped")
print("=" * 70)
if failed:
    print("Skipped files:")
    for fname, reason in failed:
        print(f"  - {fname}: {reason}")
```

# ──────────────────────────────────────────────────────────────────────────────

# CLI

# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
p = argparse.ArgumentParser(
description=“Batch-compare CSV files between two directories.”
)
p.add_argument(“dir0”, type=Path, help=“Directory with old CSVs”)
p.add_argument(“dir1”, type=Path,
help=“Directory with new CSVs (output also goes here)”)
args = p.parse_args()

```
for d in (args.dir0, args.dir1):
    if not d.is_dir():
        p.error(f"Not a directory: {d}")
return args
```

def main() -> int:
args = parse_args()
try:
run_batch(args.dir0, args.dir1)
except Exception:
traceback.print_exc()
return 1
return 0

if **name** == “**main**”:
sys.exit(main())