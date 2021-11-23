from keras_bert import extract_embeddings
import numpy as np
# np.set_printoptions(threshold=np.inf)

class ExtractBertEmb:
    # extract the features of sentences;
    # shape for one sentence (extracted by BERT): (length + 2, 768);
    # the output shape: (length, 768);
    def __init__(self):
        self.model_path = r'./BERT'

    def extract(self, sentences):
        feats = extract_embeddings(self.model_path, sentences)
        return feats

txts = ['望穿秋水','盈盈秋水']
print(txts)

a = []
n = 5000# 一次性读入服务器会爆
for j in [txts[i:i + n] for i in range(0, len(txts), n)]:
    extractor = ExtractBertEmb()
    b = extractor.extract(j)
    a.append(b)


with open(r'vector_avg_all.pickle', 'wb')as f:
    pickle.dump(a, f)







