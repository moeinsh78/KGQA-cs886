from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd




def find_query_embedding(query):
    # Just to extract the named entity in the question
    q = query[query.find("[") + 1: query.find("]")]
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    # encoded_input = tokenizer(query, return_tensors='pt')
    encoded_input = tokenizer(q, return_tensors='pt', max_length=512, padding=True, truncation=True)
    query_vector_embedding = model(**encoded_input).last_hidden_state[:, 0, :].detach().numpy()
    return np.concatenate(query_vector_embedding, axis=0)


def load_entity_embeddings():
    entity_embeddings = np.load("entity_embeddings.npy")
    print("Entity Embeddings Shape:", entity_embeddings.shape)
    return entity_embeddings


def load_entities_dict():
    entities = pd.read_table("dataset/MetaQA/MetaQA-3/entity/kb_entity_dict.txt", delimiter = "\t", header=None)
    entities = entities.rename(columns = {0: "entity_ids", 1: "entity_labels"})
    entity_id_label_dict = entities.set_index('entity_ids').to_dict()['entity_labels']
    entity_label_id_dict = entities.set_index('entity_labels').to_dict()['entity_ids']
    return entity_id_label_dict, entity_label_id_dict



def get_most_similar_entity_ids(n = 3):
    entities_dict, entities_dict_reverse = load_entities_dict()

    query_embedding = find_query_embedding(query = "[Mona McKinnon] appears in which film")
    entity_embeddings = load_entity_embeddings()
    
    # Calculate cosine similarities between the query vector and the dataset
    similarities = cosine_similarity(entity_embeddings, [query_embedding]).flatten()

    print(similarities)
    print(type(similarities))
    
    # Find the most similar vector(s)
    # top_n_indices = np.argpartition(similarities, -n)[-n:]
    top_n_indices = np.argsort(similarities)[-n:]
    top_n_indices
    print("Most Similar Indices:", top_n_indices)
    print("Top Scores:", similarities[top_n_indices])
    print("Most Similar Labels:")
    top_n_indices[:] = top_n_indices[::-1]
    labels = []
    for ind in top_n_indices:
        labels.append(entities_dict[ind])

    print(labels)    
    return top_n_indices, labels


get_most_similar_entity_ids(n = 3)
