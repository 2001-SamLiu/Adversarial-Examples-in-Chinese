import jieba
import re
sentence = "我来自上海交通大学，今年21岁！出生于深圳"
seq = re.split('\！|\，',sentence)
# for s in seq:
    # print(s)
word_list = []

for s in seq:
    word_list += (jieba.cut(s, use_paddle=True))
for word in word_list:    
    print(word)
def _get_masked(words):
    len_text = len(words)
    masked_words = []
    for i in range(len_text - 1):
        masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
    # list of words
    masked_words.append(words[:len_text-1]+['[UNK]'])
    return masked_words 
masked_words = _get_masked(word_list)
texts = [' '.join(words) for words in masked_words]
for text in texts:
    print(text)