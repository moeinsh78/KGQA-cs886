from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import pandas as pd
import numpy as np



def load_kb_dict():
    entities = pd.read_table("dataset/MetaQA/MetaQA-3/entity/kb_entity_dict.txt", delimiter = "\t", header=None)
    entities = entities.rename(columns = {0: "entity_ids", 1: "entity_labels"})
    return entities


entities = load_kb_dict()
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained("distilbert-base-uncased")


batch_size = 32
embeddings = []


entities_dict = {k:v for v,k in enumerate(list(set(entities.entity_ids)))}
entity_labels_list = list(entities.entity_labels)
encoded_entity_labels = tokenizer.batch_encode_plus(entity_labels_list, return_tensors='pt', max_length=512, padding=True, truncation=True)


# for i in tqdm(range(0, len(encoded_entity_labels['input_ids']), batch_size)):
for i in tqdm(range(0, len(encoded_entity_labels['input_ids']), batch_size)):
    batch_input_ids = encoded_entity_labels['input_ids'][i:i + batch_size]
    batch_attention_mask = encoded_entity_labels['attention_mask'][i:i + batch_size]
    
    outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
    
    batch_embeddings = outputs.last_hidden_state[:,0,:]
    print(batch_embeddings)
    batch_embeddings = batch_embeddings.detach().numpy()
    # print(batch_embeddings.shape)

#     positions = []
#     for j in range(i, i + batch_size):
#         if j >= len(entities):
#             break
#         key = entities_dict[entities.iloc[j]['entity_ids']]
#         p = getPositionEncoding(key, d=768)
#         positions.append(p)

#     positional_encoding = np.concatenate(positions, axis = 0)
#     batch_embeddings += positional_encoding
    embeddings.append(batch_embeddings)

embeddings = np.concatenate(embeddings, axis=0)

np.save('./entity_embeddings.npy', embeddings)



# text1 = "What does [Grégoire Colin] appear in"
# encoded_input1 = tokenizer.batch_encode_plus(text1, return_tensors='pt')
# output1 = model(**encoded_input1)


# text2 = "What does [Grégoire Colin] appear in"
# encoded_input2 = tokenizer(text2, return_tensors='pt')
# output2 = model(**encoded_input2)

# print(output2.last_hidden_state)