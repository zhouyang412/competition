import os
import json
import random
import logging
import pandas as pd
from transformers import BertTokenizer

# 转换数据为bert的正确的输入

class InputExample:
    def __init__(self,
                 set_type,
                 text,
                 label=None):
        self.set_type = set_type
        self.text = text
        self.label = label
        
class BaseFeature:
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids):
        # BERT 输入
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        
class BertFeature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 label=None):
        super(BertFeature, self).__init__(token_ids=token_ids,
                                         attention_masks=attention_masks,
                                         token_type_ids=token_type_ids)
        self.label = label

        
class Processor:

    @staticmethod
    def read_data(file_path):
        df = pd.read_csv(file_path, sep="\t")
        return df
    
    @staticmethod
    def get_examples(file_path):
        df = Processor.read_data(file_path)
        examples = []
        df_columns = df.columns.tolist()
        
        if 'label' in df_columns:
            set_type = 'train'
        else:
            set_type = 'test'
        
        if set_type == 'train':
            for idx, (label, text) in enumerate(zip(df.label, df.comment)):
                examples.append(InputExample(set_type=set_type,
                                             text=text,
                                             label=label))
        else:
            for idx, text in enumerate(df.comment):
                label = None
                examples.append(InputExample(set_type=set_type,
                                             text=text,
                                             label=label))            
            
        return examples
    
def convert_input_example(example: InputExample, tokenizer: BertTokenizer,
                        max_seq_len):
    set_type = example.set_type
    text = example.text
    label = example.label

    encode_dict = tokenizer.encode_plus(text=text,
                                        max_length=max_seq_len,
                                        pad_to_max_length=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True,
                                        truncation=True,
                                        padding=True)


    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    out_len = len(encode_dict['input_ids'])
    pad_len = max_seq_len - out_len

    token_ids  = encode_dict['input_ids'] + [0] * pad_len
    attention_masks  = encode_dict['attention_mask'] + [0] * pad_len
    token_type_ids  = encode_dict['token_type_ids'] + [0] * pad_len

    feature = BertFeature(
        # bert inputs
        token_ids=token_ids,
        attention_masks=attention_masks,
        token_type_ids=token_type_ids,
    )

    return feature

def convert_examples_to_features(examples, max_seq_len, bert_path):

    tokenizer = BertTokenizer(os.path.join(bert_path, 'vocab.txt'))

    features = []

    for i, example in enumerate(examples):
        
        feature = convert_input_example(
            example=example,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer
        )

        if feature is None:
            continue

        features.append(feature)

    return features

if __name__ == '__main__':
    file_path = './data/train.csv'
    bert_path = '/Users/zy/bert_wwm_ext/'
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    train_examples = Processor.get_examples(file_path)
    train_features = convert_examples_to_features(examples=train_examples, max_seq_len=156, bert_path=bert_path)