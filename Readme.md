* `w2v.py`: This module can be used to read a pretrained embeddings file from a tool such as word2vec and can save the embeddings of our vocab words in numpy serialized file.
```bash
python -m utils.w2v --input data/ICD10 --data demo --embeddings embedding_vec.txt
```
In the above command, "embedding_vec.txt" file is obtained after training the model with word2vec tool.
