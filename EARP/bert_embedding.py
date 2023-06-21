# 初始化
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel

class bert_embedding:
    def __init__(self):
        self.model_name = 'bert-base-chinese'
        self.MODEL_PATH = 'D:/Bert/bert-base-chinese/'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model_config = BertConfig.from_pretrained(self.model_name)
        # model_config.output_hidden_states = True
        # model_config.output_attentions = True
        self.bert_model = BertModel.from_pretrained(self.MODEL_PATH, config=self.model_config)

        # self.conv1d = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=256, stride=96)
        self.linear = torch.nn.Linear(768,50)
        # self.describe_type_list = describe_type_list
        self.describe_type_list = None

        # self.describe_embedding, self.type_embedding = self.entities_embedding()

    def entities_embedding(self):
        describe_embedding_list = []
        type_embedding_list=[]
        for row in self.describe_type_list:
            describe = row[0]
            type = row[1]
            describe_embedding_list.append(self.token_embedding(describe))
            # type_embedding_list.append(self.token_embedding(type))
        return torch.cat(describe_embedding_list).data, torch.cat(type_embedding_list).data


    def token_embedding(self,text):
        token = self.tokenizer(text, return_tensors='pt')
        emb = self.bert_model(**token).last_hidden_state
        out = self.linear(emb.sum(dim=1, keepdim=True)).data
        return out.view(-1)

def get_embedding(describe_type_list):
    bert = bert_embedding(describe_type_list)
    describe_embedding = bert.describe_embedding.tolist()
    type_embedding = bert.type_embedding.tolist()
    return  describe_embedding, type_embedding


if __name__ == '__main__':
    # 读取数据
    indices = np.loadtxt('./indices_embeddings/entity_map.txt')
    entity_desb_type = pd.read_csv('./rich_data/entity_desb_type.txt')
    event_desb_type = pd.read_csv('./rich_data/event_desb_type.txt')

    rich_dict = dict()
    # 加载事件信息
    for idx, row in event_desb_type.iterrows():
        item = row[0]
        des = row[1]
        typ = row[2]
        rich_dict[item] = ([des, typ])

    # 加载实体信息
    # typs：多种类型信息
    for idx, row in entity_desb_type.iterrows():
        item = row[0]
        des = row[1]
        typs = row[2]
        rich_dict[item] = ([des, typs])

    des_typ_list = []
    for row in indices:
        ent = row[0]
        if ent in rich_dict:
            des = rich_dict[0]
            typ = rich_dict[1]
            des_typ_list.append([des, typ])
        else:
            des_typ_list.append(['none', 'none'])

    np.savetxt('./rich_data/des_typ_list.txt', des_typ_list, fmt='%s')

    # describe_type_list = np.loadtxt('./rich_data/des_typ_list.txt',dtype=str,delimiter='\t',encoding='utf-8')
    # print((describe_type_list))

    # 初始化模型
    bert = bert_embedding()
    none_embedding = bert.token_embedding('none').tolist()
    print(none_embedding)

    # 描述embedding
    describe_embedding_list = []

    i = 0
    for row in des_typ_list:
        describe = row[1]
        if describe == 'none':
            describe_embedding_list.append(none_embedding)
        else:
            describe_embedding_list.append(bert.token_embedding(describe).tolist())
        # print(row[0])
        print(i)
        i+=1
    np.savetxt('./rich_data/describe.txt',np.array(describe_embedding_list))

    # 类型embedding
    # 考虑多种类型
    types_embedding_list = []

    i = 0
    for row in des_typ_list:
        # 类型分割
        types = row[2].split('|')
        type_list = []
        # 对每一种类型做bert embedding词嵌入
        for t in types:
            if t == 'none':
                type_list.append(none_embedding)
            else:
                type_list.append(bert.token_embedding(t).tolist())
        # embedding相加求平均
        type_embedding = sum(type_list) / len(type_list)
        print(i)
        i += 1

    np.savetxt('./rich_data/types.txt',np.array(types_embedding_list))