#-*- coding:utf-8 _*-
"""
@author:charlesXu
@file: utils.py
@desc: 信息抽取程序主入口
@time: 2018/08/08
"""

import tensorflow as tf
import numpy as np
import os, argparse, time, random
import re

from elasticsearch import Elasticsearch
from datetime import datetime

# os.environ['CUDA_VISIBLE_DEVICES']='0'   # 设置只用一块显卡

import proprecess_money
from Enext_model import BiLSTM_CRF
from utils import str2bool, get_logger, get_entity, get_MON_entity
from data import read_corpus, read_dictionary, tag2label, random_embedding
from get_data import get_MONGO_data, del_MONGO_data




config=None
# 数据库操作
# db = pymysql.Connect("localhost", "root", "Aa123456", "zhizhuxia")
# print('Connect successful')
# cursor = db.cursor()
# redis = redis.Redis(host='127.0.0.1', port=6379)

# 连接ES
es = Elasticsearch(
     ['test.npacn.com:21180'],  #192.168.11.251
     http_auth=('admin', 'gXvBgE43&B$8'),
     #['192.168.11.211'],
     #port=9200,
    timeout= 30,
)
# 创建索引
# es.indices.create(index='zhizhixia_ner')
print('索引创建成功。')

text_list = []   # 创建一个tuple，用来装分句后的数据


## hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default='data', help='train data source')
parser.add_argument('--test_data', type=str, default='data', help='test data source')
parser.add_argument('--batch_size', type=int, default=256, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=20, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1545099085', help='model for test and demo')
args = parser.parse_args()


## get char embeddings
word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))
if args.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, args.embedding_dim)
else:
    embedding_path = 'pretrain_embedding.npy'
    embeddings = np.array(np.load(embedding_path), dtype='float32')


## read corpus and get training data
if args.mode != 'demo':
    train_path = os.path.join('.', args.train_data, 'train_data')   # 训练数据
    test_path = os.path.join('.', args.test_data, 'test_data')
    train_data = read_corpus(train_path)
    test_data = read_corpus(test_path);
    test_size = len(test_data)


## paths setting
paths = {}
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
output_path = os.path.join('.', args.train_data+"_save", timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))


## training Entity Extraction model
if args.mode == 'train':
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()

    ## hyperparameters-tuning, split train/dev
    # dev_data = train_data[:5000]; dev_size = len(dev_data)
    # train_data = train_data[5000:]; train_size = len(train_data)
    # print("train data: {0}\ndev data: {1}".format(train_size, dev_size))
    # model.train(train=train_data, dev=dev_data)

    ## train model on the whole training data
    print("train data: {}".format(len(train_data)))
    model.train(train=train_data, dev=test_data)  # use test_data as the dev_data to see overfitting phenomena

## testing model
elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    print("test data: {}".format(test_size))
    model.test(test_data)

## demo
elif args.mode == 'demo':

    # 这里指定了模型路径
    model_path = './data_save/1545099085/checkpoints'
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print('>>>>>>>>>>>',ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        # saver.restore(sess, tf.train.latest_checkpoint("Chatbot_Data/Info_Extraction_save"))
        # while(1):
            # print('Please input your sentence:')
            # demo_sent = input()

        # all_texts = get_MONGO_data()
        try:
            for one_text in get_MONGO_data():
                addr = one_text['addr']  # 归属地
                charge = one_text['charge'] # 犯罪原因
                judgementId = one_text['judgementId'] # 判决Id，唯一标示
                print(judgementId)
                keywords = one_text['keywords']      # 关键词
                court = one_text['court']           # 法院信息
                text = one_text['judge_text']  # 判决结果，是一个列表，继续循环
                proponents = one_text['proponents']   # 原告
                opponents = one_text['opponents']     # 被告

                if text:
                    demo_data = [(text, ['O'] * len(text))]
                    tag = model.demo_one(sess, demo_data)
                    PER, LOC, ORG= get_entity(tag, text)
                    MON = get_MON_entity(text)
                    print('PER: {}\nLOC: {}\nORG: {}\nMON: {}\n'.format(PER, LOC, ORG, MON))

                    # 将数据写入es
                    es.index(index='zhizhuxia_hunan', doc_type='ner_type',
                             body={'addr': addr,
                                   'charge': charge,
                                   'judgementId': judgementId,
                                   'keywords': keywords,
                                   'court': court,
                                   'judge_text': text,
                                   'PER': PER,
                                   'LOC': LOC,
                                   'ORG': ORG,
                                   'MON': MON,
                                   'proponents': proponents,
                                   'opponents': opponents,
                                   'timestamp': datetime.now()})

                    # 根据judgement_id删除数据
                    del_MONGO_data(judgementId)
                    print('Del succeed')
                   
                    # 调用关系抽取
        except Exception as e:
            print('Info Error is', e)
