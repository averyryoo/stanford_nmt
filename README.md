# stanford_nmt
An implementation of a Stanford project on Neural Machine Translation

link: https://nlp.stanford.edu/pubs/luong-manning-iwslt15.pdf

## Arguments

```
--train               train model
--translate           translate a word or phrase
--data_dir            dataset directory
--ckpt_dir            checkpoint directory
--out                 name of output checkpoint
--batch_size          input batch size
--input_lang          source language 
--target_lang         target language
--max_len             max length of input tensor
--limit_size          maximum size of dataset
--dev_split           split for validation dataset
--epochs              number of training epochs
--embed_dims          number of embedding dimensions
--enc_units           number of encoding and decoding units
--dropout             neuronal dropout in training
--learn_rate          training learning rate             
```

Train command example
```
python main.py --train --batch_size 32 --epochs 10 --learn_rate 0.001
```

### Disclaimer
Unfortunately I did not have a reliable training environment (as Colab and Kaggle were extremely wonky), so training and testing of translation is still in progress