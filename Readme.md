* `w2v.py`: This module can be used to read a pretrained embeddings file from a tool such as word2vec and can save the embeddings of our vocab words in numpy serialized file.
```bash
python -m utils.w2v --input data/ICD10 --data demo --embeddings embedding_vec.txt
```
In the above command, "embedding_vec.txt" file is obtained after training the model with word2vec tool.


* Visualize the top-K embeddings:
```bash
# ML Objective  
python extract_embeddings.py --input temp/aclImdbSimple_pretrained/data/ --data demo --model temp/aclImdbSimple_pretrained/model/aclImdbSimple_pretrained_ce.pt --output_dir temp/aclImdbSimple_pretrained/model/ --gpu 1

# Mixed Objective
python extract_embeddings.py --input temp/aclImdbSimple_pretrained_mixed/data/ --data demo --model temp/aclImdbSimple_pretrained_mixed/model/aclImdbSimple_pretrained_mixed.pt --output_dir temp/aclImdbSimple_pretrained_mixed/model/ --gpu 1
```

