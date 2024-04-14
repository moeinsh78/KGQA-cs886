import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


def load_kg_edges_df():
    kg_relations = pd.read_table("dataset/MetaQA/MetaQA-3/kb.txt", delimiter = "|", header=None)
    kg_relations = kg_relations.rename(columns = {0:"head", 1:"relation", 2:"tail"})
    return kg_relations




Graph = nx.Graph()
edges = load_kg_edges_df()
counter = 0
for _, edge in edges.iterrows():
    counter += 1
    if counter > 20:
        break
    Graph.add_edge(edge['head'], edge['tail'], label=edge['relation'])


pos = nx.spring_layout(Graph, seed=42, k=0.9)
labels = nx.get_edge_attributes(Graph, 'label')
plt.figure(figsize=(12, 10))
nx.draw(Graph, pos, with_labels=True, font_size=10, node_size=700, node_color='lightblue', edge_color='gray', alpha=0.6)
nx.draw_networkx_edge_labels(Graph, pos, edge_labels=labels, font_size=12, label_pos=0.3, verticalalignment='baseline')
plt.title('Knowledge Graph')
plt.show(block=True)
