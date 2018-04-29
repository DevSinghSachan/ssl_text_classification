import sys
import numpy as np

infile = sys.argv[1]
max_len = int(sys.argv[2])

def read_data(loc_):
    data = list()
    with open(loc_) as fp:
        for line in fp:
            text = line.strip()
            data.append(text)
    return data


train_text = read_data(infile)
data_len = list(map(lambda x: len(x.split()), train_text))

print("Median words: {}".format(np.median(data_len)))
print("Mean words: {}".format(np.mean(data_len)))
print("Max words: {}".format(np.max(data_len)))
print("Min words: {}".format(np.min(data_len)))
print("Std words: {}".format(np.std(data_len)))
print("> {} words: {}".format(max_len, len(list(filter(lambda x: x > max_len, data_len)))))
