import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter


def get_description(head, relation, tail):
    if relation == "directed_by":
        return "The movie \'{}\' was directed by \'{}\'.".format(head, tail)
    elif relation == "has_genre":
        return "The movie \'{}\' has genre {}.".format(head, tail)
    elif relation == "has_imdb_rating":
        return "The movie \'{}\' is rated {} in imdb.".format(head, tail)
    elif relation == "has_imdb_votes":
        return "The movie \'{}\' is voted {} in imdb.".format(head, tail)
    elif relation == "has_tags":
        return "The movie \'{}\' has the tag \'{}\'.".format(head, tail)
    elif relation == "in_language":
        return "The movie \'{}\' is in {} language.".format(head, tail)
    elif relation == "release_year":
        return "The movie \'{}\' was released in {}.".format(head, tail)
    elif relation == "starred_actors":
        return "The actor \'{}\' starred in \'{}\'.".format(tail, head)
    elif relation == "written_by":
        return "The movie \'{}\' was written by \'{}\'.".format(head, tail)
    else:
        print("ERROR!: Relation type {} unknown!".format(relation))

def load_kg_edges_df():
    kg_relations = pd.read_table("dataset/MetaQA/MetaQA-3/kb.txt", delimiter = "|", header=None)
    kg_relations = kg_relations.rename(columns = {0:"head", 1:"relation", 2:"tail"})
    kg_relations['description'] = kg_relations.apply(lambda d: get_description(d["head"], d["relation"], d["tail"]), axis = 1)

    # # Relation distribution in the knowledge graph
    # print(kg_relations)
    
    return kg_relations


def get_bfs_edge_list(graph, source, depth, expand_ending_nodes = False):
    ending_node_relations = ["release_year", "in_language", "has_tags", "has_genre", "has_imdb_rating", "has_imdb_votes"]
    bfs_edges_list = []
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
            # print("Neighbors: ", neighbors)
            visited.add(curr_node)
            for pair in neighbors:
                # Node to be expanded is always in the second position
                if pair[1] in visited:
                    continue
                if (expand_ending_nodes) or (graph.edges[pair[0], pair[1]]["label"] not in ending_node_relations):
                    to_be_expanded.append(pair[1])
                bfs_edges_list.append(tuple((pair[0], graph.edges[pair[0], pair[1]]["label"], pair[1])))
        
        curr_depth += 1


    # for item in bfs_edges_list:
    #     print(item)
    # print(len(bfs_edges_list))
    return bfs_edges_list


def traverse_node_neighborhood(source_node, depth):
    # Graph Construction
    graph = nx.Graph()
    edges = load_kg_edges_df()
    for _, edge in edges.iterrows():
        graph.add_edge(edge["head"], edge["tail"], label=edge["relation"])

    top20_degree_nodes = sorted([tuple((node, graph.degree(node))) for node in graph.nodes], key=itemgetter(1), reverse=True)[:20]
    # print(top20_degree_nodes)

    edge_list = get_bfs_edge_list(graph, source_node, depth, expand_ending_nodes = False)
    return edge_list



traverse_node_neighborhood("The Human Comedy", 3)


# pos = nx.spring_layout(Graph, seed=42, k=0.9)
# labels = nx.get_edge_attributes(Graph, 'label')
# plt.figure(figsize=(12, 10))
# nx.draw(Graph, pos, with_labels=True, font_size=10, node_size=700, node_color='lightblue', edge_color='gray', alpha=0.6)
# nx.draw_networkx_edge_labels(Graph, pos, edge_labels=labels, font_size=12, label_pos=0.3, verticalalignment='baseline')
# plt.title('Knowledge Graph')
# plt.show(block=True)
