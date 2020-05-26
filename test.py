import main
import pickle
from gensim.models import Word2Vec
import numpy as np


input_file = './BIC_code.txt'
commits, commit_count, longest_len = main.read_commits(input_file)

model = Word2Vec(commits,
                 size=1,
                 window=5,
                 min_count=1,
                 workers=10,
                 sg=1)
model.save("word2vec_sg.model")



vectorized_commits = main.vectorize_commits(np.asarray(commits),
                                            commit_count,
                                            longest_len,
                                            model.wv)

file = open('commits2vec.vec', 'wb')

pickle.dump(vectorized_commits, file)

file.close()