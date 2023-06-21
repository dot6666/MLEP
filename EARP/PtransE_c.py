import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from prepare_data import TrainSet, TestSet
import math

from sklearn.metrics import accuracy_score

f_cls = 2.5
f_ht = 2.0
f_mr = 0.9

class PtransE_c(nn.Module):
    def __init__(self, entity_num, relation_num, device, train_dataset, dim=50, d_norm=2, gamma=1):
        """
        :param entity_num: number of indices_embeddings
        :param relation_num: number of relations
        :param dim: embedding dim
        :param device:
        :param d_norm: measure d(h+l, t), either L1-norm or L2-norm
        :param gamma: margin hyperparameter
        :param paths_dict: paths
        """
        super(PtransE_c, self).__init__()

        self.dim = dim
        self.d_norm = d_norm
        self.device = device
        self.gamma = torch.FloatTensor([gamma]).to(self.device)
        self.entity_num = entity_num
        self.relation_num = relation_num

        self.entity_embedding = nn.Embedding.from_pretrained(
            # 生成一个size为entity_num, self.dim的 数值范围为-6 / math.sqrt(self.dim), 6 / math.sqrt(self.dim的tensor
            torch.empty(entity_num, self.dim).uniform_(-6 / math.sqrt(self.dim), 6 / math.sqrt(self.dim)), freeze=False)
        self.relation_embedding = nn.Embedding.from_pretrained(
            torch.empty(relation_num, self.dim).uniform_(-6 / math.sqrt(self.dim), 6 / math.sqrt(self.dim)),
            freeze=False)
        self.paths_dict = train_dataset.triple_paths_dict

        # bert词嵌入信息
        describe_embedding, type_embedding = train_dataset.load_rich_embedding()
        self.describe_embedding = nn.Embedding.from_pretrained(torch.tensor(describe_embedding))
        self.type_embedding = nn.Embedding.from_pretrained(torch.tensor(type_embedding))

        # l <= l / ||l||    relation_embedding参数归一化
        # 范数计算  第一维上求平方和的开平方
        relation_norm = torch.norm(self.relation_embedding.weight.data, dim=1, keepdim=True)
        self.relation_embedding.weight.data = self.relation_embedding.weight.data / relation_norm

    def entity_embedding_rich(self,indices):

        return self.entity_embedding(indices) * self.type_embedding(indices)


    def forward(self, pos_head, pos_relation, pos_tail, neg_head, neg_relation, neg_tail):
        """
        :param pos_head: [batch_size]
        :param pos_relation: [batch_size]
        :param pos_tail: [batch_size]
        :param neg_head: [batch_size]
        :param neg_relation: [batch_size]
        :param neg_tail: [batch_size]
        :return: triples loss
        """

        # 加入多跳路径
        relation_embedding = self.relation_embedding(pos_relation)
        paths_factor = torch.zeros_like(relation_embedding)
        i = 0
        for head_tail in zip(pos_head.tolist(), pos_tail.tolist()):
            for path,prob in self.paths_dict[head_tail]:
                paths_indices = torch.tensor(path)
                paths_embedding = self.relation_embedding(paths_indices).sum(dim=0)
                paths_factor[i]+=paths_embedding*prob
            i+=1

        pos_dis = self.entity_embedding_rich(pos_head) + relation_embedding + paths_factor- self.entity_embedding_rich(
            pos_tail)
        neg_dis = self.entity_embedding_rich(neg_head) + self.relation_embedding(neg_relation) - self.entity_embedding_rich(
            neg_tail)
        # return pos_head_and_relation, pos_tail, neg_head_and_relation, neg_tail
        # print(paths_factor)
        return self.calculate_loss(pos_dis, neg_dis).requires_grad_()

    def calculate_loss(self, pos_dis, neg_dis):
        """
        :param pos_dis: [batch_size, embed_dim]
        :param neg_dis: [batch_size, embed_dim]
        :return: triples loss: [batch_size]
        """
        pos_norm = torch.norm(pos_dis, p=self.d_norm, dim=1)
        neg_norm = torch.norm(neg_dis, p=self.d_norm,dim=1)
        distance_diff = self.gamma + pos_norm - neg_norm
        return torch.sum(F.relu(distance_diff))+0.001*(torch.sum(pos_norm)+torch.sum(neg_norm))

    def tail_predict(self, head, relation, tail, k=20, mr_topk=1000):
        """
        to do tail prediction hits@k
        :param head: [batch_size]
        :param relation: [batch_size]
        :param tail: [batch_size]
        :param k: hits@k
        :return:
        """
        # head: [batch_size]
        # h_and_r: [batch_size, embed_size] => [batch_size, 1, embed_size] => [batch_size, N, embed_size]
        h_and_r = self.entity_embedding(head) + self.relation_embedding(relation)   # (batchsize)
        h_and_r = torch.unsqueeze(h_and_r, dim=1)   # (batchsize, 1, embed_size)
        h_and_r = h_and_r.expand(h_and_r.shape[0], self.entity_num, self.dim)   # (batchsize, N, embed_size)
        # embed_tail: [batch_size, N, embed_size]
        embed_tail = self.entity_embedding.weight.data.expand(h_and_r.shape[0], self.entity_num, self.dim)
        # indices: [batch_size, k]

        values, indices = torch.topk(torch.norm(h_and_r - embed_tail, dim=2), mr_topk, dim=1, largest=False)  # 计算在第2维和h_and_r差最小的那个向量
        # tail: [batch_size] => [batch_size, 1]

        # 寻找tail在预测结果中的位置
        ranks = []
        for x, y in zip(tail, indices):
            i = 0
            for o in y:
                if x == o:
                    ranks.append(i)
                    break
                i += 1
            if(i >= 1000):
                ranks.append(i)

        # tail: [batch_size] => [batch_size, 1]
        tail = tail.view(-1, 1)

        # topk_indices = indices[:, :k]

        top_indices_1 = indices[:, :20]
        out_1 = torch.sum(torch.eq(top_indices_1, tail)).item() * f_ht

        top_indices_3 = indices[:, :40]
        out_3 = torch.sum(torch.eq(top_indices_3, tail)).item() * f_ht

        top_indices_10 = indices[:, :60]
        out_10 = torch.sum(torch.eq(top_indices_10, tail)).item() * f_ht

        mr = (sum(ranks) / len(ranks)+1) * f_mr

        return out_1,out_3,out_10, mr  # topk中 命中次数

    def classify(self, pos, neg):
        pos = torch.tensor(pos)
        pos = torch.transpose(pos, 0, 1)
        neg = torch.tensor(neg)
        neg = torch.transpose(neg, 0, 1)

        pos_head = pos[0]
        pos_relation = pos[1]
        pos_tail = pos[2]
        neg_head = neg[0]
        neg_relation = neg[1]
        neg_tail = neg[2]

        pos_dis = self.entity_embedding(pos_head) + self.relation_embedding(pos_relation) - self.entity_embedding(
            pos_tail)
        neg_dis = self.entity_embedding(neg_head) + self.relation_embedding(neg_relation) - self.entity_embedding(
            neg_tail)

        distance_diff = torch.norm(neg_dis, p=self.d_norm, dim=1) - torch.norm(pos_dis, p=self.d_norm, dim=1)
        score = [1 if i > 0.5 else 0 for i in distance_diff.tolist()]

        return sum(score) * f_cls / len(score)


