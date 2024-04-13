from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import pandas

# text1 = "What does [Grégoire Colin] appear in"
# encoded_input1 = tokenizer.batch_encode_plus(text1, return_tensors='pt')
# output1 = model(**encoded_input1)

# print(type(output1.last_hidden_state))


# text2 = "What does [Grégoire Colin] appear in"
# encoded_input2 = tokenizer(text2, return_tensors='pt')
# output2 = model(**encoded_input2)

# print(output2.last_hidden_state)

batch_size = 32
embeddings = []



def encode_labels(entity_labels, tokenizer):
    encoded = tokenizer.batch_encode_plus(entity_labels, return_tensors='pt', max_length=512, padding=True, truncation=True)
    return encoded


def load_kb_dict():
    entity_labels = pandas.read_table("dataset/MetaQA/MetaQA-3/entity/kb_entity_dict.txt", delimiter = "\t", header=None)
    entity_labels = entity_labels.rename(columns = {0: "entity_ids", 1: "entity_labels"})
    return entity_labels



entity_labels = load_kb_dict()

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained("distilbert-base-uncased")


encoded_entity_labels = encode_labels(entity_labels, tokenizer)
print(encoded_entity_labels)

for i in tqdm(range(0, len(encoded_entity_labels['entity_ids']), batch_size)):
    batch_input_ids = encoded_entity_labels['input_ids'][i:i + batch_size]
    batch_attention_mask = encoded_entity_labels['attention_mask'][i:i + batch_size]
    
    outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
    
    batch_embeddings = outputs.last_hidden_state[:,0,:]
    batch_embeddings = batch_embeddings.cpu().numpy()
#     positions = []
#     for j in range(i, i + batch_size):
#         if j >= len(df):
#             break
#         k = videos_dict[df.iloc[j]['video_id']]
#         p = getPositionEncoding(k, d=768)
#         positions.append(p)

#     positional_encoding = np.concatenate(positions, axis = 0)
#     batch_embeddings += positional_encoding
#     embeddings.append(batch_embeddings)

# embeddings = np.concatenate(embeddings, axis=0)