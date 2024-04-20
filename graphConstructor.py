import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter


def get_description(head, relation, tail):
    if relation == "directed_by":
        return "Movie \'{}\' was directed by \'{}\'.".format(head, tail)
    elif relation == "has_genre":
        return "Movie \'{}\' has genre {}.".format(head, tail)
    elif relation == "has_imdb_rating":
        return "Movie \'{}\' is rated {} in imdb.".format(head, tail)
    elif relation == "has_imdb_votes":
        return "Movie \'{}\' is voted {} in imdb.".format(head, tail)
    elif relation == "has_tags":
        return "Movie \'{}\' has the tag \'{}\'.".format(head, tail)
    elif relation == "in_language":
        return "Movie \'{}\' is in {} language.".format(head, tail)
    elif relation == "release_year":
        return "Movie \'{}\' was released in {}.".format(head, tail)
    elif relation == "starred_actors":
        return "Actor \'{}\' starred in \'{}\'.".format(tail, head)
    elif relation == "written_by":
        return "Movie \'{}\' was written by \'{}\'.".format(head, tail)
    else:
        print("ERROR!: Relation type {} unknown!".format(relation))

def load_kg_edges_df(edge_list_file):
    kg_relations = pd.read_table(edge_list_file, delimiter = "|", header=None)
    kg_relations = kg_relations.rename(columns = {0:"head", 1:"relation", 2:"tail"})
    kg_relations['description'] = kg_relations.apply(lambda d: get_description(d["head"], d["relation"], d["tail"]), axis = 1)

    # # Relation distribution in the knowledge graph
    # print(kg_relations.groupby(["relation"]).size())
    
    return kg_relations


def get_bfs_edge_list(graph, source, depth, expand_ending_nodes = False, give_edge = False):
    ending_node_relations = ["release_year", "in_language", "has_tags", "has_genre", "has_imdb_rating", "has_imdb_votes"]
    bfs_edge_description_list = []
    to_be_expanded = [source]
    visited = set()
    curr_depth = 0
    while curr_depth < depth:
        to_expand_count = len(to_be_expanded)
        for _ in range(to_expand_count):
            curr_node = to_be_expanded.pop(0)
            if curr_node in visited:
                continue
            neighbors = list(nx.bfs_edges(graph, curr_node, depth_limit=1))
            visited.add(curr_node)
            for pair in neighbors:
                # Node to be expanded is always in the second position
                if pair[1] in visited:
                    continue
                if (expand_ending_nodes) or (graph.edges[pair[0], pair[1], 0]["label"] not in ending_node_relations):
                    to_be_expanded.append(pair[1])
                for i in range(graph.number_of_edges(pair[0], pair[1])):
                    if give_edge:
                        edge_info = [pair[0], graph.edges[pair[0], pair[1], i]["label"], pair[1]]
                        bfs_edge_description_list.append(edge_info)
                    else: 
                        bfs_edge_description_list.append(graph.edges[pair[0], pair[1], i]["description"])
        
        curr_depth += 1


    return bfs_edge_description_list


def build_knowledge_graph(edge_list_file):
    # Graph Construction
    graph = nx.MultiGraph()
    edges = load_kg_edges_df(edge_list_file)
    for _, edge in edges.iterrows():
        graph.add_edge(edge['head'], edge['tail'], label=edge['relation'], description=edge["description"])

    return graph


def visualize_graph(source, depth):
    complete_graph = build_knowledge_graph(edge_list_file = "dataset/MetaQA/MetaQA-3/kb.txt")
    sample_edge_list = get_bfs_edge_list(complete_graph, source, depth, give_edge=True)

    sample_graph = nx.MultiGraph()
    for edge in sample_edge_list:
        head = edge[0]
        tail = edge[2]
        if (len(head) > 15):
            head = head[:15] + ".."
        if (len(tail) > 15):
            tail = tail[:15] + ".."
        sample_graph.add_edge(head, tail, label=edge[1])
    pos = nx.spring_layout(sample_graph, seed=55, k=0.9)
    labels = nx.get_edge_attributes(sample_graph, 'label')
    plt.figure(figsize=(12, 10))
    nx.draw(sample_graph, pos, with_labels=True, font_size=12, node_size=5000, node_color='green', edge_color='gray', alpha=0.6)
    nx.draw_networkx_edge_labels(sample_graph, pos, edge_labels=labels, font_size=8, label_pos=0.5, verticalalignment='baseline')
    plt.title('Knowledge Graph')
    plt.show(block=True)



# graph = build_knowledge_graph(edge_list_file = "dataset/MetaQA/MetaQA-3/kb.txt")
# edge_list = get_bfs_edge_list(graph, source = "Grown Ups 2", depth = 1, expand_ending_nodes = False)
# print(edge_list)

# load_kg_edges_df("./dataset/MetaQA/MetaQA-3/kb.txt")
visualize_graph(source="Jean Rochefort", depth=2)
