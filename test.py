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
