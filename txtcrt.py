import nltk
from nltk.corpus import brown, conll2000
from gensim.models import Word2Vec
import multiprocessing
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Activation, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
import numpy as np 
import collections
from nltk.tokenize import TweetTokenizer

tknzr = TweetTokenizer()
s0 = """ Sáng ngày 11/3, sau thông tin "bệnh nhân 34" dương tính, anh này được hướng dẫn tự cách ly tại nhà. 

Ngày 13/3 bệnh nhân được chuyển cách ly tập trung tại Quận 10, lấy mẫu xét nghiệm và chưa có triệu chứng, sức khoẻ ổn định. 

Bệnh viện Bệnh nhiệt đới thành phố Hồ Chí Minh lấy mẫu xét nghiệm dương tính vào tối ngày 13/3. Sau đó, mẫu bệnh phẩm được chuyển cho Viện Pasteur thành phố Hồ Chí Minh vào 23 giờ cùng ngày.

Sáng nay, kết quả cho thấy dương tính nCoV bằng kỹ thuật realtime RT-PCR. """

words = tknzr.tokenize(s0)
print(words)


#sentences = brown.sents()
#print(sentences[:3])

EMB_DIM = 300

w2v = Word2Vec(words, size= EMB_DIM, window=5, min_count=0, negative=15, iter=10, workers=multiprocessing.cpu_count())
word_vectors = w2v.wv
result = word_vectors.similar_by_vector("RT-PCR")
print(result)

#train_words = conll2000.tagged_words("train.txt")
#test_words = conll2000.tagged_words("test.txt")
#print(train_words[:10])



# def get_tag_vocabulary(tagged_word):
#     tag2id = {}
#     for item in tagged_word:
#         tag = item[1]
#         tag2id.setdefault(tag, len(tag2id))
#     return tag2id

# word2id = {k: v.index for  k ,v in word_vectors.vocab.items()}
# tag2id = get_tag_vocabulary(train_words)
# print(word2id)
# print(tag2id)
