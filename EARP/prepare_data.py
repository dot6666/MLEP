import pandas as pd
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import random
import os

# from bert_embedding import get_embedding

class TrainSet(Dataset):
    def __init__(self):
        super(TrainSet, self).__init__()
        # self.raw_data, self.entity_dic, self.relation_dic = self.load_texd()
        self.raw_data, self.entity_to_index, self.relation_to_index = self.load_text()
        self.entity_num, self.relation_num = len(self.entity_to_index), len(self.relation_to_index)
        self.triple_num = self.raw_data.shape[0]
        triple_num = 1019879
        relation_num = 13
        entity_num = 145895+1438
        print(f'Train set: {self.entity_num} indices_embeddings, {self.relation_num} relations, {self.triple_num} triplets.')
        self.pos_data = self.convert_word_to_index(self.raw_data)
        # related_dic = {entity_id: {related_entity_id_1, related_entity_id_2...}}
        self.related_dic = self.get_related_entity()

        # 加载描述和类型信息
        # self.describe_embedding, self.type_embedding = self.load_rich_embedding()

        self.relation_entity = self.get_relation_entity()
        self.triple_paths_dict = self.PCRA()

        self.neg_data = self.generate_neg()


    def __len__(self):
        return self.triple_num

    def __getitem__(self, item):
        return [self.pos_data[item], self.neg_data[item]]

    def load_text(self):

        raw_data = pd.read_csv('dataset/train_data.txt', sep='\t',
                               keep_default_na=False, encoding='utf-8',dtype=str)
        # raw_data = raw_data.applymap(lambda x: x.strip())   # 移除头尾字符

        entity_list = list(set(raw_data['head'])|set(raw_data['tail']))
        relation_list = list(set(raw_data['relation']))

        entity_list.sort()
        relation_list.sort()

        entity_dic = dict([(word, idx) for idx, word in enumerate(entity_list)])    # 生成dict：{word1:idx1, word2:idx2,...}
        relation_dic = dict([(word, idx) for idx, word in enumerate(relation_list)])
        # raw_data:[ [head, relation, tail],
        #            [head, relation, tail],
        #            [head, relation, tail],
        #            ...
        #                                   ]
        #           numpy.ndarray

        # 保存
        # entity_map = list(entity_dic.items())
        # np.savetxt('./indices_embeddings/entity_map.txt',np.array(entity_map),fmt='%s')
        # relation_map = list(relation_dic.items())
        # np.savetxt('./indices_embeddings/relation_map.txt', np.array(relation_map),fmt='%s')

        return raw_data.values, entity_dic, relation_dic

    def convert_word_to_index(self, data):
        index_list = np.array([
            [self.entity_to_index[triple[0]], self.relation_to_index[triple[1]], self.entity_to_index[triple[2]]] for
            triple in data])
        return index_list

    def generate_neg(self):
        """
        generate negative sampling
        :return: same shape as positive sampling
        """
        neg_candidates, i = [], 0
        neg_data = []
        population = list(range(self.entity_num))

        # 每次选取10000个 从中选取1个进行替换
        for idx, triple in enumerate(self.pos_data):
            while True:
                if i == len(neg_candidates):
                    i = 0
                    neg_candidates = random.choices(population=population, k=int(1e4))  # 选取10000次 返回一个list
                neg, i = neg_candidates[i], i + 1
                if random.randint(0, 1) == 0:
                    # replace head
                    if neg not in self.related_dic[triple[2]]:
                        neg_data.append([neg, triple[1], triple[2]])
                        break
                else:
                    # replace tail
                    if neg not in self.related_dic[triple[0]]:
                        neg_data.append([triple[0], triple[1], neg])
                        break

        return np.array(neg_data)

    def get_related_entity(self):
        """
        get related indices_embeddings
        :return: {entity_id: {related_entity_id_1, related_entity_id_2...}}
        """
        related_dic = dict()
        for triple in self.pos_data:
            if related_dic.get(triple[0]) is None:
                related_dic[triple[0]] = {triple[2]}
            else:
                related_dic[triple[0]].add(triple[2])
            if related_dic.get(triple[2]) is None:
                related_dic[triple[2]] = {triple[0]}
            else:
                related_dic[triple[2]].add(triple[0])
        return related_dic

    # 找到连接的关系和尾节点
    def get_relation_entity(self):
        relation_entity_dict = dict()
        for triple in self.pos_data:
            if relation_entity_dict.get(triple[0]) is None:
                relation_entity_dict[triple[0]] = {(triple[1], triple[2])}
            else:
                relation_entity_dict[triple[0]].add((triple[1], triple[2]))
            if relation_entity_dict.get(triple[2]) is None:
                relation_entity_dict[triple[2]] = {(triple[1], triple[0])}
            else:
                relation_entity_dict[triple[2]].add((triple[1], triple[0]))

        return  relation_entity_dict

    # 寻找路径
    def PCRA(self):

        # 存储三元组对应的path，和他们的weight
        # triple_paths_dict：{(head,tail):[([relation1, relation2, ...],weight),(...)]}
        triple_paths_dict = dict()

        for triple in self.pos_data:
            head_tail = (triple[0], triple[2])

            # 记录遍历过的头节点
            visited_entities = list()
            visited_entities.append(triple[0])
            # 初始化空列表
            triple_paths_dict[head_tail] = list()
            # 初始化空路径
            relation_path = list()
            # 设置最大深度为3
            self.dfs(head_tail, triple_paths_dict, visited_entities, relation_path, 1,3,1)

        return triple_paths_dict

    def dfs(self,head_tail, triple_paths_dict, visited_entities, relation_path, depth, max_depth, prob):

        target = head_tail[1]
        cur_entity = visited_entities[-1]
        relation_entity = self.relation_entity[cur_entity]
        for relation, tail in relation_entity:
            ## 需要操作的两种情况
            # 1、找到目标节点，但是dept必须大于1（否则就是triple本身）
            # 2、没找到目标节点，进入下一层寻找，tail需没有遍历过，depth需小于maxdepth

            new_prob = prob / len(relation_entity)
            # 添加一个阈值小于直接丢弃
            if(new_prob < 0.001):
                break

            if(tail == target and depth>1):
                relation_path.append(relation)
                triple_paths_dict[head_tail].append((relation_path.copy(),new_prob))
                relation_path.pop()
                continue

            elif(tail!=target and tail not in visited_entities and depth<max_depth):
                visited_entities.append(tail)
                relation_path.append(relation)
                self.dfs(head_tail, triple_paths_dict, visited_entities, relation_path, depth+1, max_depth, new_prob)
                visited_entities.pop()
                relation_path.pop()
        return

    def load_rich_embedding(self):
        describe_embedding_file = ('./rich_data/describe.txt')
        type_embedding_file = ('./rich_data/type.txt')

        describe_embedding = np.loadtxt(describe_embedding_file)
        type_embedding = np.loadtxt(type_embedding_file)

        # describe_embedding = np.random.rand(10)
        # type_embedding = np.random.rand(10)

        return describe_embedding, type_embedding


class TestSet(Dataset):
    def __init__(self):
        super(TestSet, self).__init__()
        self.raw_data = self.load_text()
        self.data = self.raw_data
        num = 95265
        print(f"Test set: {len(self.data)} triplets")

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.data.shape[0]

    def load_text(self):
        raw_data = pd.read_csv('dataset/test_data.txt', sep='\t',
                               keep_default_na=False, encoding='utf-8',dtype=str)
        # raw_data = raw_data.applymap(lambda x: x.strip())
        return raw_data.values

    def convert_word_to_index(self, entity_to_index, relation_to_index, data):
        index_list = np.array(
            [[entity_to_index[triple[0]], relation_to_index[triple[1]], entity_to_index[triple[2]]] for triple in data])
        self.data = index_list

    def generate_classify_data(self, entity_num, related_dic):
        """
        generate negative sampling
        :return: same shape as positive sampling
        """
        neg_candidates, i = [], 0
        neg_data = []
        population = list(range(entity_num))

        # 每次选取10000个 从中选取1个进行替换
        for idx, triple in enumerate(self.data):
            while True:
                if i == len(neg_candidates):
                    i = 0
                    neg_candidates = random.choices(population=population, k=int(1e4))  # 选取10000次 返回一个list
                neg, i = neg_candidates[i], i + 1
                if random.randint(0, 1) == 0:
                    # replace head
                    if neg not in related_dic[triple[2]]:
                        neg_data.append([neg, triple[1], triple[2]])
                        break
                else:
                    # replace tail
                    if neg not in related_dic[triple[0]]:
                        neg_data.append([triple[0], triple[1], neg])
                        break

        return self.data, np.array(neg_data)



