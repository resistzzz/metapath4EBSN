#coding=utf-8

import numpy as np
import os
import pickle
import random

class divideTrainAndTest(object):
    # 按照时间来划分train和test，2017.4-5月的活动作为测试集，同时参与测试活动3次以上的用户作为测试用户
    def __init__(self):
        self.trainEvents = None
        self.testEvents = None

        self.uePair = None
        self.user_event = dict()
        self.event_user = dict()

        self.trainUsers = []
        self.testUsers = []         # 测试用户集合

        self.train_uePair = []      # 训练用户-活动对
        self.test_uePair = []       # 测试用户-活动对

    def read_data(self, dirpath):
        with open(os.path.join(dirpath, 'douban_shanghai_trainAndTest.data'), 'rb') as f:
            data = pickle.load(f)
        self.trainEvents = data['train']
        self.testEvents = data['test']

        with open(os.path.join(dirpath, 'douban_shanghai_UE-pair.data'), 'rb') as f:
            self.uePair = pickle.load(f)

    def chooseTestUser(self):
        # 参与2017.4-5月的活动3次及以上的用户作为测试用户
        for ue in self.uePair:
            u, e = str(ue[0]), str(ue[1])
            if u not in self.user_event:
                self.user_event[u] = []
            self.user_event[u].append(e)
            if e not in self.event_user:
                self.event_user[e] = []
            self.event_user[e].append(u)

        for user in self.user_event:
            cnt = 0
            for event in self.user_event[user]:
                if int(event) in self.testEvents['id']:
                    cnt += 1
            if cnt >= 3:
                self.testUsers.append(int(user))
                if len(self.user_event[user]) > cnt:
                    self.trainUsers.append(int(user))
            else:
                self.trainUsers.append(int(user))

    def divideUEPair2TrainAndTest(self):
        # 根据规则将用户-活动对进行划分
        for ue in self.uePair:
            u, e = ue[0], ue[1]
            if u in self.testUsers and e in self.testEvents['id']:
                self.test_uePair.append((u, e))
            else:
                self.train_uePair.append((u, e))

    def statistic(self):
        # 统计下孤立的用户节点和活动节点数量
        # 只出现在测试集的用户，没有出现在训练集
        # 一个活动的参与者都被划分到测试集，没有在训练集出现过
        print('训练用户活动对数目: {}'.format(len(self.train_uePair)))
        print('测试用户活动对数目: {}'.format(len(self.test_uePair)))
        print('测试用户数目: {}'.format(len(self.testUsers)))
        cnt_u = 0
        for u in self.testUsers:
            if u not in self.trainUsers:
                cnt_u += 1
        print('没有在训练集中，只在测试集中的用户数: {}, 占测试集用户比例: {:.4f}'.format(cnt_u, 1.0*cnt_u/len(self.testUsers)))

        cnt_e = 0
        for event in self.event_user:
            cnt = 0
            for u in self.event_user[event]:
                if int(u) in self.testUsers and int(u) not in self.trainUsers:
                    cnt += 1
            if cnt == len(self.event_user[event]):
                cnt_e += 1

        print('一个活动的参与者都被划分到测试集，没有在训练集出现过的活动数目为: {}, 占活动总比例为: {:.4f}'.format(cnt_e, 1.0*cnt_e/len(self.event_user)))

    def save_uePair(self, dirpath):
        # 保存train_uePair和test_uePair
        uePair = [self.train_uePair, self.test_uePair]
        fname = 'uePairTrainAndTest.data'
        with open(os.path.join(dirpath, fname), 'wb') as f:
            pickle.dump(uePair, f)


class MetaPathRandomWalk(object):

    def __init__(self):
        self.train_uePair = None
        self.test_uePair = None

        self.user_event = dict()
        self.event_user = dict()
        self.event_lda = dict()
        self.lda_event = dict()

    def read_data(self, dirpath):
        # read ue pair
        with open(os.path.join(dirpath, 'uePairTrainAndTest.data'), 'rb') as f:
            uePair = pickle.load(f)
        self.train_uePair = uePair[0]
        self.test_uePair = uePair[1]
        for ue in self.train_uePair:
            u, e = str(ue[0]), str(ue[1])
            if u not in self.user_event:
                self.user_event[u] = []
            self.user_event[u].append(e)
            if e not in self.event_user:
                self.event_user[e] = []
            self.event_user[e].append(u)

        # read event lda
        with open(os.path.join(dirpath, 'eventLDA100.data'), 'rb') as f:
            self.event_lda = pickle.load(f)

    def generate_random_ueu(self, outfilename, outfilename1, numwalks, walklength):
        # outfile = open(outfilename, 'w')
        outlines = []
        for u in self.user_event:
            for i in range(numwalks):
                outline = 'u_' + u
                numu, nume = 0, 0
                for j in range(int((walklength-1)/2.0)):
                    events = self.user_event[u]
                    nume = len(events)
                    if nume <= 0:
                        print('nume < 0')
                        continue
                    eventidx = random.randrange(nume)
                    event = events[eventidx]
                    outline = outline + ' ' + 'e_' + event
                    users = self.event_user[event]
                    numu = len(users)
                    if numu <= 0:
                        print('numu < 0')
                        continue
                    useridx = random.randrange(numu)
                    user = users[useridx]
                    outline = outline + ' ' + 'u_' + user
                if nume > 0 and numu > 0:
                    outlines.append(outline)
                    # outfile.write(outline + '\n')
        # outfile.close()
        # print('generate random walk UEU done!')

        # 有重复的，需要去除重复的path
        # print(len(outlines))
        outlines = list(set(outlines))
        # print(len(outlines))
        with open(outfilename, 'w') as f:
            for line in outlines:
                f.write(line + '\n')
        print('generate random walk UEU done!')

        # 生成node_type_mapping file的代码比较慢
        node_type_mapping = []
        for line in outlines:
            line = line.split(' ')
            for w in line:
                if w not in node_type_mapping:
                    node_type_mapping.append(w)

        for i in range(len(node_type_mapping)):
            if node_type_mapping[i].split('_')[0] == 'u':
                node_type_mapping[i] = node_type_mapping[i] + 'u'
            elif node_type_mapping[i].split('_')[0] == 'e':
                node_type_mapping[i] = node_type_mapping[i] + 'e'
            else:
                continue

        with open(outfilename1, 'w') as f:
            for line in node_type_mapping:
                f.write(line + '\n')
        print('generate node_type_mapping UEU done!')

    def generate_random_ueleu(self, outfilename, numwalks, walklength):
        pass




dirpath = 'data/douban_shanghai'
dtt = divideTrainAndTest()
dtt.read_data(dirpath)
dtt.chooseTestUser()
dtt.divideUEPair2TrainAndTest()
dtt.statistic()
# dtt.save_uePair(dirpath)

# mprk = MetaPathRandomWalk()
# dirpath = 'data/douban_shanghai'
# mprk.read_data(dirpath)
# outfilename = os.path.join(dirpath, 'ueu_RandomWalk.txt')
# outfilename1 = os.path.join(dirpath, 'ueu_nodeTypeMapping.txt')
# numWalks = 10
# walkLength = 20
# mprk.generate_random_ueu(outfilename, outfilename1, numWalks, walkLength)

# node_type_mapping = []
# with open('./data/test/ueu_RandomWalk_test.txt', 'r') as outlines:
#     for line in outlines:
#         line = line.strip().split(' ')
#         for w in line:
#             if w not in node_type_mapping:
#                 node_type_mapping.append(w)
#     print('First is done!')
#
#     for i in range(len(node_type_mapping)):
#         if node_type_mapping[i].split('_')[0] == 'u':
#             node_type_mapping[i] = node_type_mapping[i] + ' ' + 'u'
#         elif node_type_mapping[i].split('_')[0] == 'e':
#             node_type_mapping[i] = node_type_mapping[i] + ' ' + 'e'
#         else:
#             continue
#     print('Second is done!')
#
# with open('./data/test/ueu_nodeTypeMapping_test.txt', 'w') as f:
#     for line in node_type_mapping:
#         f.write(line + '\n')
# print('generate node_type_mapping UEU done!')