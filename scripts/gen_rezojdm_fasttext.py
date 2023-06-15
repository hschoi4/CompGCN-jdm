import fasttext
import fasttext.util
import io
import pandas as pd
import torch
fasttext.util.download_model('fr', if_exists='ignore')  # French

ft = fasttext.load_model('cc.fr.300.bin')
ft = fasttext.util.reduce_model(ft, 100)

ent2id = {}
with open("./data/RezoJDM16k/entities.txt", 'r') as f:
    for line in f.readlines():
        tokens = line.strip().split()
        _id = int(tokens.pop(0))
        _ent = ' '.join(tokens)
        ent2id[_ent] = _id

id2ent = {idx: ent for ent, idx in ent2id.items()}


dim = ft['meuf'].shape[0]

vectors = torch.zeros((len(id2ent), dim), dtype=torch.float32)
for _id, _ent in id2ent.items():
    # If there is space in the word, use sentence else use word
    vectors[_id] = torch.tensor(ft.get_sentence_vector(_ent) if ' ' in _ent else ft.get_word_vector(_ent))

# We store the vectors back in the data folder
torch.save(vectors, './data/RezoJDM16k/vectors_entity_fasttext.torch')
