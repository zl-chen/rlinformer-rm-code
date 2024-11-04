# %% [markdown]
# ## 1. 引入

# %%
import argparse
import datetime
import json
import os
import shutil
import sys
import pickle 
import time
# 加载torch也是需要时间的，算在里面。
start = time.time()
import pandas as pd
import torch
# from exp.exp_informer import Exp_Informer
from utils.visualization import *
from utils.initialize_random_seed import *
from utils.metrics import *
from utils.multi_lag_processor import *
from pyecharts.globals import CurrentConfig, OnlineHostType
import warnings



import os
import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

# np.set_printoptions(precision=2)
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)   #显示完整的列
pd.set_option('display.max_rows', None)  #显示完整的行


import matplotlib.pyplot as plt



# %% [markdown]
# ## 2. DataLoader

# %%
# 读取电池数据
def get_len(x):
    i = 0 
    for _ in x:
        i += 1
    return i

#battery_df =pd.read_csv("../dataset/processed_data/fade_rate_dict.csv")
battery_df =pd.read_csv("../dataset/processed_data/shu_old_fade_rate.csv")
#battery_df = pd.read_csv("../dataset/processed_data/nature_processed_dup.csv",index_col=0)

cols = list(battery_df.columns)

train_rate = 0.8
total_length = get_len(cols)
train_length = int(total_length * train_rate)

train_cols = random.sample(cols,train_length)
test_cols = [x for x in cols if x not in train_cols]

train_battery_df = battery_df[train_cols]
test_battery_df = battery_df[test_cols]

# %%
# 训练数据的时候用到,制作train、vali、test数据集

delta_data = train_battery_df
class Dataset_Custom(Dataset):
    # 在exp_informer.py中传值进来，覆盖掉这些默认参数
    # start，end是后来加的，用于描述该数据从几号索引取到几号索引（0-1600，1600为最长电池长度，不会超过他）
    def __init__(self, root_path, data_path, flag='train', size=None, 
                 features='S',timeenc=0,args=None):
        # size： [seq_len, label_len, pred_len]
        # 最大范围，写死了
        # self.start=0
        # self.end=1600
        # 分别为输入encoder的序列长度、输入decoder中原属数据长度，预测长度
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.root_path = root_path
        self.data_path = data_path
        self.flag=flag
        
        # print(features)
        # 读取最小值
        # print('aaa',self.root_path)
        # rawdataη = pd.read_excel(os.path.join(args.root_path,'Δη(train).xlsx'))

        rawdataη = delta_data

        #print(f"rawdatan = {os.path.join(args.root_path,'Δη(train).xlsx')}")



        #rawdataη = all_fade_rate_df
        rawdataη=rawdataη.iloc[:-1].values[:,:].astype(float)
        
        # rawdataMin=rawdataMin.values[self.start:self.end,:].astype(float)
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        # 0，表示train
        self.set_type = type_map[flag]
        
        # features为 S，表示单值预测
        self.features = features
       
        # 时间特征编码  args.embed, help='时间特征编码，选项：[timeF, fixed, learned]' ，默认为timeF
        #  这是注释   timeenc = 0 if args.embed!='timeF' else 1，默认为 1
        self.timeenc = timeenc
        # 时间特征编码的频率，就是进行特征工程的时候时间粒度选取多少，
        # '选项（options）:[s:secondly, t:minutely, h:hourly, d:daily, b:工作日（business days）, w:weekly, m:monthly], '
        
        
        self.args = args
        
        
        
        # 获取表格中所有列名（训练数据的）
        # self.dataTrain = pd.read_excel((os.path.join(self.root_path,'Δη(train).xlsx')))
        
        self.dataTrain = delta_data

        self.dataTrain=self.dataTrain.iloc[:-1]
        # self.scalerData 与 self.scalerI   为了把归一化的步骤传到外面
        self.encoderList, self.decoderList, self.scalerDataη, self.lenListSum = self.__getsamples(rawdataη)
        # print('len',self.encoderList.shape[0])


    
    
    def __getsamples(self, rawdataη):       
        lenList=[]
        lenListSum=0
        ηFlatten=[]
        # 归一化及复原(30列)
        for j,col in enumerate(self.dataTrain.columns):
            # 每列的长度
            len=(np.array(self.dataTrain.iloc[:,j].dropna())).shape[0]
            lenList.append(len)
            lenListSum=lenListSum+len
            for i in range(len):
                ηFlatten.append(rawdataη[i,j])
        # print('111',QDFlatten)
        # print('lenList',lenList[0])
        # 变成归一化接受的形式
        ηFlatten=np.array(ηFlatten).reshape(-1,1)
        
        # 归一化      
        self.scalerDataη = MinMaxScaler()
        self.scalerDataη = self.scalerDataη.fit(ηFlatten) 
        rawdataη = self.scalerDataη.transform(ηFlatten)
        # rawdataη = ηFlatten
        
        
            
        # 还原成原来的格式  
        ηNew=[]
        lenTemp=0
        # for j,col in enumerate(self.dataTrain.columns):
        # 执行训练数据中电池的个数次，即列数
        for i in range(self.dataTrain.shape[1]):
            # print(i)
            ηNewTemp=[]
          
            for j in range(lenList[i]):
                ηNewTemp.append(rawdataη[j+lenTemp][0])  
            lenTemp=lenTemp+lenList[i]
            ηNewTemp=np.array(ηNewTemp)
            
            ηNew.append(ηNewTemp)   
         
        ηNew=np.array(ηNew) 
        # print("MinNew",MinNew.size())
        # print("MinNew",MinNew)
            

        # XAll为43列合并后的，XPre为每一列的
        XAll=[]
        YAll=[]
        # 最后跑列
        for j,col in enumerate(self.dataTrain.columns):
            # XPre为每种充电方案的，XAll为42种充电方案汇总的
            sample_num=lenList[j] - self.seq_len - self.pred_len + 1
            # (sample-num,3,1,10)，第二个参数 1 表示channel为 1
            # X是encoder的输入,args.enc_in是encoder的输入维度，即几个特征
            XPre = torch.zeros((sample_num, self.seq_len,args.enc_in))
            # Y是decoder的输入，args.dec_in是decoder的输入维度，即几个特征
            YPre = torch.zeros((sample_num, self.label_len + self.pred_len, args.dec_in))


            # YPre = torch.zeros((sample_num, 1))
            
            # 200条原始数据的话，sample_num为190
            for i in range(sample_num):
                # encoder的输入开始
                s_begin = i
                # encoder的输入结束
                s_end = s_begin + self.seq_len
                # decoder的输入开始
                r_begin = s_end - self.label_len
                # decoder的输入结束
                r_end = r_begin + self.label_len + self.pred_len

                # 获取输入序列x
                # seq_x = self.data_x[s_begin:s_end]
                startX = i
                # end从10到200
                endX = i + self.seq_len
                # result=zip(QDNew[j][start:end], MinNew[j][start:end], VarNew[j][start:end])
                # j是第几列，start和end是起始以及终止的行数
                result_x=np.vstack((ηNew[j][s_begin:s_end].reshape((self.seq_len,1))))
                result_y=np.vstack((ηNew[j][r_begin:r_end].reshape((self.label_len+self.pred_len,1))))
                # print(result_x.shape)
                # 第一个参数 1 表示channel为 1 
                # XPre的shape为(sample_num,1,1,seq_len)
                XPre[i, :, :] = torch.from_numpy(np.array(list(result_x)))
                # YPre的shape为(sampe_num,1,1,label_len+pred_len)
                YPre[i, :, :] = torch.from_numpy(np.array(list(result_y)))
                
            XAll.append(XPre)
            YAll.append(YPre)
            
        # 一字排开，变成竖的
        
        # 把[100,10,1]分成[50,10,1],[30,10,1],[20,10,1]的函数
     
        # XAll.shape为（sample_num,seq_len,1）
        XAll=torch.cat(XAll,dim=0).reshape(-1,self.seq_len,args.enc_in).double()
        YAll=torch.cat(YAll,dim=0).reshape(-1,self.label_len+self.pred_len,args.dec_in).double()
        # YAll=torch.stack(YAll).reshape(-1,1,1).double()
        # YAll=torch.cat(YAll,dim=0).reshape(-1,1,1).double()
        # print('YAll.shape=',YAll.shape)
        # print('XAll',XAll.shape)
        # print('XAll',XAll[0])
        # print('XAll',XAll[1])
        # print("YAll",YAll[0])
        # print("YAll",YAll[1])
        # print("XAll854",XAll[854])
        # print("XAll855",XAll[855])
        # print('YAll',YAll.shape)
        # print("YAll853",YAll[853])
        # print("YAll854",YAll[854])
        # print("YAll855",YAll[855])
        # self.dataTrain.shape[1]是列数，此处为30
        sample_sum=lenListSum- self.dataTrain.shape[1]*(self.seq_len + self.pred_len - 1)
        train_num=int(0.8*sample_sum)
        test_num=int(0.1*sample_sum)
        val_num=sample_sum-train_num-test_num
        
        
         # 以同样方法打乱两个矩阵的函数
        def shuffle_two_matrix(a, b):
            # 以a的行数为基准，打乱a和b
            p = np.random.permutation(a.shape[0])
            return a[p], b[p]
        
        # 确保每次打乱得一样
        seed=args.seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] =str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic =True
        # 确保每次打乱的一样
        XAll,YAll=shuffle_two_matrix(XAll, YAll)
        
        
        
        if(self.flag=="train"):
            XAll=XAll[:train_num]
            YAll=YAll[:train_num]
        if(self.flag=="test"):
            XAll=XAll[train_num:train_num+test_num]
            YAll=YAll[train_num:train_num+test_num]
        if(self.flag=="val"):
            XAll=XAll[train_num+test_num:]
            YAll=YAll[train_num+test_num:]
        # print('ahh',XAll[0])
        # print('XAll',XAll.shape)
        # print('XAll',YAll.shape)
        # for i in range(30):
        #     print('XAll',XAll[351*i])
            # print('XAll',YAll[351])




        return (XAll, YAll, self.scalerDataη, lenListSum) 
    
   
    
    def __len__(self):
        # return self.lenListSum - 26*(self.seq_len + self.pred_len - 1)
        return self.encoderList.shape[0]
    
     # 外部使用【idx】来获取，idx的max值即上面的__len__
    def __getitem__(self, idx):
        seq_x=self.encoderList[idx, :, :]
        seq_y=self.decoderList[idx, :, :]
        # ？？？此处的seq_x_mark和seq_y_mark是假的，暂时不用，等要加上全局时间embedding的时候再加上
        # 获取带有掩码的输入序列x
        seq_x_mark = torch.zeros(1)
        # 获取带有掩码的输入序列x
        seq_y_mark = torch.zeros(1)
        # print('idx',idx)
        # if(idx>-1 and idx==0):
        #     print(idx)
        #     print('seq_x',seq_x)
        #     print('seq_y',seq_y)
        
        
        return seq_x, seq_y,seq_x_mark, seq_y_mark



# %% [markdown]
# ## 3. exp_Informer

# %%
# 数据加载器
import datetime
import sys

# 在自定义的data模块中
import pandas as pd

# from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
#
from exp.exp_basic import Exp_Basic
# 导入模型
from models.model import Informer, InformerStack

# 提前停止策略、修正学习率
from utils.tools import EarlyStopping, adjust_learning_rate
# 评价指标
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')

# 继承Exp_Basic类
class Exp_Informer(Exp_Basic):



    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)

    # 构造模型
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            # 可以先不改
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            # 如果self.args.model是informer，那么model_dict[self.args.model]就是Informer类
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    # 获取数据并进行处理，返回符合输入格式的数据
    def _get_data(self, flag):
        args = self.args
        data_dict = {
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            # 自定义的数据传到这边
            '{}'.format(self.args.data):Dataset_Custom,
            'custom':Dataset_Custom,
        }
        # 下面这个Data，此时是一个Dataset_Custom。
        # self.args.data：chicken（我做的此处是rose）;    Data是Dataset_Custom对象
        Data = data_dict[self.args.data]


        # 时间特征编码  embed',args.embed, help='时间特征编码，选项：[timeF, fixed, learned]' 
        timeenc = 0 if args.embed!='timeF' else 1

        # flag:设置任务类型
        # 根据flag设置训练设置和数据操作设置
        # 做测试的时候
        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = 1; freq=args.freq
        # 做预测的时候
        elif flag=='pred':
            # 如果是预测未来的任务
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            # 因为是预测任务，所以Data被赋值为Dataset_Pred对象
            Data = Dataset_Pred
       
        elif flag == 'val':
            shuffle_flag = False; drop_last = True; batch_size = 1; freq=args.freq
        # train的时候:打乱数据
        else:
            # 记得之后打乱
            shuffle_flag =True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        # 使用Dataset_Custom进行读取数据集，并转换为数组.:
        # 实例化Dataset_Custom对象
        # print('args.data_path:',args.data_path)
        # 下面这个Data，此时是一个Dataset_Custom。


        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            # informer原论文中，这三个分别为96，48，24，分别是输入encoder的序列长度、
            # （48+24）为输入decoder的序列长度，24为预测长度
            size=[args.seq_len, args.label_len, args.pred_len],
            # M、S、MS，表示多变量预测、单变量预测、多变量预测单变量
            features=args.features,
            # target=args.target,
            # inverse=args.inverse,
            timeenc=timeenc,
            # freq=freq,
            # scale=args.scale,
            # cols=args.cols,
            args=args
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        
        
        return data_set, data_loader

    # 选择模型优化器（这里是adam）
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # 选择损失标准(损失函数)
    def _select_criterion(self):
        criterion = nn.MSELoss()
        if self.args.loss == 'mse':
            criterion = nn.MSELoss()
        if self.args.loss == 'L1loss':
            criterion = nn.L1Loss()
        if self.args.loss == 'huberloss':
            criterion = nn.SmoothL1Loss()
        return criterion

    # 验证集的验证
    def vali(self,vali_data,vali_loader,criterion,args):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark,args)
            pred = pred[:, :, -1:] if args.features == 'MS' else pred

            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    # 训练集的训练
    def train(self,setting,info_dict,run_name_dir_ckp,run_ex_dir,args):

        # 做训练的时候这里面已经测试集评估功能 和 验证集的验证功能了,args.save_model_choos
        global scaler
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')
        # 存储模型的位置
        path = os.path.join(run_name_dir_ckp, setting)


        print(f"run_name = {run_name_dir_ckp}")
        print(f"setting = {setting}")

        # path = os.path.join(run_ex_dir, setting)#将模型和可视化文件存储在一起
        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()

        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True,save_model_choos=args.save_model_choos)

        model_optim = self._select_optimizer()
        # 损失函数,此处是mse
        criterion =  self._select_criterion()
        if self.args.use_amp:
            # autocast + GradScaler 可以达到自动混合精度训练的目的；
            # GradScaler是梯度
            scaler = torch.cuda.amp.GradScaler()
        # 训练的时候记录每个epoch产生的损失，包括训练集损失、验证集损失、测试集(评估集)损失
        all_epoch_train_loss = []
        all_epoch_vali_loss = []
        all_epoch_test_loss = []
        # 训练args.train_epochs个epoch，每一个epoch循环一遍整个数据集
        epoch_count = 0
        for epoch in range(self.args.train_epochs):
            epoch_count += 1
            iter_count = 0
            # 存储当前epoch下的每个迭代步的训练损失
            train_loss = []
            """
            模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()，在测试时添加model.eval()。
            
            其中model.train()是保证BN层用每一批数据的均值和方差，而model.eval()是保证BN用全部训练数据的均值和方差；
            而对于Dropout，model.train()是随机取一部分网络连接来训练更新参数，而model.eval()是利用到了所有网络连接。
            """
            self.model.train()
            epoch_time = time.time()
            # 在每个epoch里面迭代数据训练模型：遍历一遍数据
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                # 累计迭代次数
                iter_count += 1
                # 把模型的参数梯度设置为0:
                model_optim.zero_grad()
                # 训练集的预测值和真实值 : 这里的真实值是输入数据-滑动窗口，预测值是滑动川口里面的对应预测值。[批次,预测长度,1]

                #print(f"train_data = {train_data.shape}")



            
                
                pred, true = self._process_one_batch(train_data, batch_x, batch_y, batch_x_mark, batch_y_mark,args)
                # 对于多变量，把数组打平，【然后归一化】，然后再计算损失。
                pred = pred[:, :, -1:] if args.features == 'MS' else pred
                # print(type(pred),pred.shape)
                # print('predahh',pred.size)
                # print('true',true)
                # print(type(true),true.shape)
                # print(true)
                # print("-----------"*4)
                # sys.exit()
                # 计算损失
                # print(type(true),true.dtype)
                # print(type(pred),pred.dtype)
                """
                true:    <class 'torch.Tensor'> torch.float32
                pred:    <class 'torch.Tensor'> torch.float16
                """



                loss = criterion(pred.float(), true.float())
                # loss = criterion(pred.double(), true.double())
                # sys.exit()
                # 将每个迭代步的loss添加到train_loss列表
                train_loss.append(loss.item())
                # 每迭代一百个样本就打印一次
                # ？？？这边要改
                # if (i+1) % 100==0:
                if (i+1) % 120==0:
                    # 查看迭代100个样本所花费的时间，和这100个样本的训练损失值，还有当前所在epoch
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    # 查看处理速度
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                if self.args.use_amp:
                    # 达到自动混合精度训练的目的
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            # 打印遍历一遍整个训练集 所需要的时间，也就是此次epoch所需要的时间
            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            # 对训练集损失求均值
            train_loss = np.average(train_loss)
            # 验证集验证
            vali_loss = self.vali(vali_data, vali_loader, criterion,args)
            # 测试集进行评估模型，其实这里也是达到验证的作用
            test_loss = self.vali(test_data, test_loader, criterion,args)
            # 添加到列表中留存
            all_epoch_train_loss.append(float(round(train_loss,1)))
            all_epoch_vali_loss.append(float(round(vali_loss,1)))
            all_epoch_test_loss.append(float(round(test_loss,1)))
            # 完成每个epoch的训练就打印一次
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            # 判断是否提前停止，并且存最好的模型，保存模型 torch.save
            early_stopping(vali_loss, self.model, path,args.save_model_choos)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # 更新学习率
            adjust_learning_rate(model_optim, epoch+1, self.args)
            if(epoch%5==0):
                torch.save(self.model, "./model/batterySD/Δη/informer/单特征/informer_e" + str(epoch) + "_b" + str(args.batch_size) + "_dModel" + str(args.d_model) + "_dFF" + str(args.d_ff)+ "_s" + str(args.seq_len) + "_l" + str(args.label_len) + "_p" + str(args.pred_len)+ ".pkl")
        # 存储该次实验的更新迭代中的最优模型
        if args.save_model_choos==True:
            best_model_path = path+'/'+'checkpoint.pth'
            # 下面是加载模型，（这个模型最终在预测完之后要删除，因为占用内存大）
            self.model.load_state_dict(torch.load(best_model_path))
        # 实验记录
        info_dict["【训练】本次实验训练的train平均损失"] = round(float(np.mean(all_epoch_train_loss)),1)
        info_dict["【验证】本次实验训练的vali平均损失"]  = round(float(np.mean(all_epoch_vali_loss)),1)
        info_dict["【验证】本次实验训练的test平均损失"]  = round(float(np.mean(all_epoch_test_loss)),1)
        info_dict["----实际训练的epoch-------"] = epoch_count

        return self.model,info_dict,all_epoch_train_loss,all_epoch_vali_loss,all_epoch_test_loss,epoch_count

    # 测试集测试
    def test(self,setting,info_dict,run_ex_dir,args):
        test_data, test_loader = self._get_data(flag='test')#做测试的时候
        # 不启用 BatchNormalization 和 Dropout，因为不是训练模式
        self.model.eval()
        preds = []
        trues = []
        # batch_x是输入的一个批次的x数据，
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            # print(batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape)
            # print(batch_x, batch_y)
            # 返回的是数组,注意：loader里面已经把数据打乱了
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark,args)
            pred = pred[:, :, -1:] if args.features == 'MS' else pred
            # print(type(pred),pred.shape)
            # print(pred)
            # print(type(true),true.shape)
            # print(true)
            # print("-----------"*4)
            # sys.exit()
            # 把数组添加到列表
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            # if args.inverse == False:
            #     inverse_true = Standardization.inverse_transform(true)
            #     inverse_pred = Standardization.inverse_transform(pred)
        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        """
        test shape: (29, 32, 24, 1) (29, 32, 24, 1)
        test shape: (928, 24, 1) (928, 24, 1)
        """
        # result save
        folder_path = run_ex_dir+'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if args.features == 'M':
            # 取到最后一个值，因为那个才是要预测的。
            trues = trues[:,-1:,:]
            preds = preds[:,-1:,:]
            trues = trues.reshape(len(trues),trues.shape[-1])
            preds = preds.reshape(len(preds),preds.shape[-1])
            # print(trues)
            # sys.exit()
            trues = np.around(trues,decimals=1)
            preds = np.around(preds,decimals=1)
            trues = trues.tolist()
            preds = preds.tolist()
            preds = np.array(preds)
            trues = np.array(trues)

        if args.features != 'M':
            trues = trues[:, -1:, :]
            preds = preds[:, -1:, :]
            trues = trues.flatten()
            preds = preds.flatten()
            trues = np.around(trues, decimals=1)
            preds = np.around(preds, decimals=1)
            trues = trues.tolist()
            preds = preds.tolist()
            preds = [round(i, 1) for i in preds]
            trues = [round(i, 1) for i in trues]
            preds = np.array(preds)
            trues = np.array(trues)

        # 评估指标：测试集评估模型
        mae,rmse,smape,r2,ad_r2 = metric(preds, trues)
        mae, rmse, smape, r2, ad_r2 = round(float(mae),1),round(float(rmse),1),round(float(smape),1),round(float(r2),1),round(float(ad_r2),1)
        print('测试集评估结果：\t平均绝对误差 MAE:{}，均方根误差RMSE:{}，对称平均绝对百分比误差SMAPE:{}，决定系数R²：{}，校正R²:{} \n'.format(mae,rmse,smape,r2,ad_r2))
        # 存储评估指标
        info_dict["【评估】本次实验的test集平均绝对误差MAE"] = mae
        info_dict["【评估】本次实验的test集均方根误差RMSE"] = rmse
        info_dict["【评估】本次实验的test集对称平均绝对百分比误差SMAPE"] = smape
        info_dict["【评估】本次实验的test集决定系数R²"] = r2
        info_dict["【评估】本次实验的test集校正决定系数Ad_R²"] = ad_r2
        # 存储评估指标和向量
        np.save(folder_path+'metrics.npy', np.array([mae,rmse,smape,r2,ad_r2]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)
        if args.inverse == False:
            pass
        return info_dict,preds,trues

    # 预测未来
    def predict(self, setting,run_name_dir_ckp, run_ex_dir,args,load=False):
        # 从_get_data获取数据，【这句代码的返回结果搞不明白】
        pred_data, pred_loader = self._get_data(flag='pred')
        pred_date = pred_data.pred_date
        if args.freq[-1] == "t" or args.freq[-1] == 'h' or args.freq[-1] == 's':
            pred_date = [str(p) for p in pred_date[1:]]
        else:
            pred_date = [str(p).split(" ")[0] for p in pred_date[1:]]
        print("本次实验预测未来的时间范围：",pred_date)
        # 加载模型
        if load:
            path = os.path.join(run_name_dir_ckp ,setting)
            # path = os.path.join(run_ex_dir ,setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        # 清楚缓存
        self.model.eval()
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            # print(batch_x.shape,batch_y.shape,batch_x_mark.shape,batch_y_mark.shape)
            # torch.Size([1, 96, 1]) torch.Size([1, 48, 1]) torch.Size([1, 96, 3]) torch.Size([1, 72, 3])
            """
            [1, 96, 1]是输入的一个批次的X数据，可以认为是滑动窗口为96的X。
            [1, 48, 1]是输入的一个批次的Y数据，可以认为是滑动窗口为96的X的标签数据，48是inform解码器的开始令牌长度label_len，多步预测的展现。
            
            [1, 96, 3]是输入的X数据的Q、K、V向量的数组。
            [1, 72, 3]是输入的Y数据的Q、K、V向量的数组,其中，72=48+24，48是label_len，24是预测序列长度pred_len，也就是说24是被预测的，这里是作为已知输入的。
            """
            # print(batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape)
            # sys.exit()
            pred, true = self._process_one_batch(pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark,args)
            preds.append(pred.detach().cpu().numpy())


        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        preds = preds[:, :, -1:] if args.features == 'MS' else preds
        # print(preds)
        # print(type(preds),len(preds),preds.shape,preds)
        # sys.exit()
        # result save
        folder_path = run_ex_dir+'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if args.features == 'M':
            preds = preds[0]
            print("本次实验预测未来的结果：", preds)
            # 存储未来的预测结果到npy文件
            np.save(folder_path + 'real_prediction.npy', preds)
            assert len(preds) == len(pred_date)
            return preds, pred_date
        if args.features != 'M':
            # 这里要修改
            preds = preds.flatten().tolist()
            preds = [round(i, 1) for i in preds]
            print("本次实验预测未来的结果：",preds)
            # 存储未来的预测结果到npy文件
            np.save(folder_path+'real_prediction.npy', preds)
            assert len(preds) == len(pred_date)
            return preds, pred_date
        return preds,pred_date

    # 对一个batch进行的编码解码操作，就是训练模型
    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark,args):
        global dec_inp
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            # 返回一个形状为为size，size是一个list，代表了数组的shape,类型为torch.dtype，里面的每一个值都是0的tensor
            # batch_y.shape[0]是self.lbel_len + self.pred_len
            # batch_y.shape[-1]是特征数,单特征预测单特征的情况下，这里是1
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        # 在给定维度上对输入的张量序列seq 进行连接操作。
        """
        outputs = torch.cat(inputs, dim=0) → Tensor
        
        inputs : 待连接的张量序列，可以是任意相同Tensor类型的python 序列，可以是列表或者元组。
        dim : 选择的扩维, 必须在0到len(inputs[0])之间，沿着此维连接张量序列。
        """
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder（编码器-解码器）
        # 假如使用自动混合精度训练
        if self.args.use_amp:
            # pytorch 使用autocast半精度进行加速训练
            with torch.cuda.amp.autocast():
                # 假如在编码器中输出注意力
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        # 假如不使用自动混合精度训练
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        # 逆标准化输出数据
        # 暂时不管他
        # if self.args.inverse:
        #     outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        # 如果是MS。那么只留有一列输出
        # outputs = outputs[:, :, 1:] if args.features == 'MS' else outputs
        # 对y进行解码
        # 取出pred

        

        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        # 如果是M任务，那么进行打平再输出去计算梯度
        # output作为预测值，batch_y(取出pred部分，也就是长度40)作为真实值



        return outputs, batch_y


# %% [markdown]
# ## 4. 定义参数集合

# %%
def initialize_parameter():
    parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')
    parser.add_argument('--root_path', type=str, default='./data/batterySD/', help='（训练）数据文件的根路径（root path of the data file）')
    parser.add_argument('--target', type=str, default='price', help='S或MS任务中的目标特征列名（target feature in S or MS task）')
    # ？？？这边要大改，实际上这边先不管，因为我把与时间处理相关的内核代码删除了，外面不管传什么值都不影响
    parser.add_argument('--freq', type=str, default='w', help='时间特征编码的频率（freq for time features encoding）, '
                                                              '选项（options）:[s:secondly, t:minutely, h:hourly, d:daily, b:工作日（business days）, w:weekly, m:monthly], '
                                                              '你也可以使用更详细的频率，比如15分钟或3小时（you can also use more detailed freq like 15min or 3h）')

                                                              
    # 存储模型位置的地方
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='模型检查点的位置（location of model checkpoints）')
    # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]
    # ？？？，指的应该是channel的个数
    parser.add_argument('--enc_in', type=int, default=1, help='编码器输入大小（encoder input size）')
    parser.add_argument('--dec_in', type=int, default=1, help='解码器输入大小（decoder input size）')
    parser.add_argument('--c_out', type=int, default=1, help='输出尺寸（output size）')
    parser.add_argument('--d_model', type=int, default=16, help='模型维数（dimension of model）默认是512-------------------------模型维数')
    parser.add_argument('--n_heads', type=int, default=8, help='（num of heads）multi-head self-attention的head数')
    parser.add_argument('--e_layers', type=int, default=2, help='编码器层数（num of encoder layers）-------------------编码器层数')
    parser.add_argument('--d_layers', type=int, default=1, help='解码器层数（num of decoder layers）---------------------解码器层数')
    # 源代码也是设置的这些值，可以先不改
    parser.add_argument('--s_layers', type=str, default='3,2,1', help='堆栈编码器层数（num of stack encoder layers）---------------堆栈编码器层数')
    parser.add_argument('--d_ff', type=int, default=32, help='fcn维度（dimension of fcn），默认是2048--------------------FCN维度')
    """
    ？？？d_model和d_ff，之后可能要改
    预测未来短期时间1~3个月的时候，d_model和d_ff进行设置的小，如16、32或者16,16；
    预测未来短期时间4个月及以上的时候，d_model和d_ff进行设置的稍微大一点点，如16、64或者32,64；32,128。
    """
    # 源代码也是5，可以先不改
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--padding', type=int, default=0, help='padding type')
    # 即是否使用下采样，使用该参数表示不进行下采样
    parser.add_argument('--distil', action='store_false', help='是否在编码器中不使用知识蒸馏，使用此参数意味着不使用蒸馏'
                                                               '（whether to use distilling in encoder, using this argument means not using distilling）',
                        default=True)
    # prob是informer提出的一个创新点
    parser.add_argument('--attn', type=str, default='prob', help='用于编码器的注意力机制，选项：[prob, full]'
                                                                 '（attention used in encoder, options:[prob, full]）')

    # ？？？时间特征编码【未知】，这边先不管，因为我把与时间处理相关的内核代码删除了，外面不管传什么值都不影响
    parser.add_argument('--embed', type=str, default='timeF', help='时间特征编码，选项：[timeF, fixed, learned]'
                                                                   '（time features encoding, options:[timeF, fixed, learned]）')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true',default=True, help='是否在编码器中输出注意力'
                                                                        '（whether to output attention in ecoder）')
    # ？？？回头看
    parser.add_argument('--do_predict', action='store_true', default=True, help='是否预测看不见的未来数据'
                                                                                '（whether to predict unseen future data）')
    parser.add_argument('--mix', action='store_true', help='在生成解码器中使用混合注意力'
                                                            '（use mix attention in generative decoder）', default=True)
    # nargs=‘+’：表示参数可设置一个或多个
    parser.add_argument('--cols', type=str, nargs='+', help='将数据文件中的某些cols作为输入特性'
                                                            '（certain cols from the data files as the input features）')
    parser.add_argument('--num_workers', type=int, default=0, help='工作的数据加载器数量 data loader num workers')
    # ？？？回头看
    parser.add_argument('--train_epochs', type=int, default=60, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='训练输入数据的批大小 batch size of train input data--------------------批次大小，原文用的32')
    parser.add_argument('--patience', type=int, default=8, help='提前停止的连续轮数 early stopping patience')
    parser.add_argument('--des', type=str, default='forecasting', help='实验描述 exp description')

    parser.add_argument('--loss', type=str, default='mse', help='损失函数选项：loss function【mse、huberloss、L1loss】--------------------损失函数')

    parser.add_argument('--lradj', type=str, default='type1', help='校正的学习率adjust learning rate----------------------学习率更新算法')
    parser.add_argument('--use_amp', action='store_true', help='使用自动混合精度训练 use automatic mixed precision training--------',
                        default=True)
    parser.add_argument('--output', type=str, default='./output', help='输出路径')
    # 想要获得最终预测的话这里应该设置为True；否则将是获得一个标准化的预测。即进行了 逆标准化
    parser.add_argument('--inverse', action='store_true', help='逆标准化输出数据 inverse output data', default=True)
    parser.add_argument('--scale', action='store_true', help='是否进行标准化，默认是True', default=True)
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    # ？？？回头看，可能要改
    parser.add_argument('--itr', type=int, default=1, help='实验次数 experiments times----------------------------------多少次实验')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate-----------------------------初始学习率')
    # ？？？我设置成要保存模型了
    parser.add_argument('--save_model_choos', type=bool, default=True, help='是否保存模型，不保存的话不占用IO')
    parser.add_argument('--is_show_label', type=bool, default=True, help='是否显示图例数值')
    # seq_len其实就是n个滑动窗口的大小，pred_len就是一个滑动窗口的大小
    # 这个文件中用的是12个预测8个
    # ？？？要改的地方
    parser.add_argument('--seq_len', type=int, default=100,
                        help='Informer编码器的输入序列长度（input sequence length of Informer encoder）原始默认为96------------------------编码器输入序列长度seq_len')
    # label_len 和 pred_len 加起来是 decoder 的输入长度
    parser.add_argument('--label_len', type=int, default=50,
                        help='inform解码器的开始令牌长度（start token length of Informer decoder），原始默认为48-------------------------解码器的开始令牌起始位置label_len')
    # pred_len就是要预测的序列长度（要预测未来多少个时刻的数据），也就是Decoder中置零的那部分的长度
    parser.add_argument('--pred_len', type=int, default=50 ,help='预测序列长度（prediction sequence length）原始默认为24------------------预测序列长度pred_len')
    
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout，长序列预测用0.5，短期预测用0.05~0.2(一般是0.05)，如果shuffle_flag的训练部分为True，那么该值直接设置为0;模型参数多设置为0.5，要在0.5范围内；视情况而定。----')

    # 这两个应该用不到 
    parser.add_argument('--train_proportion', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--test_proportion', type=float, default=0.1, help='测试集比例')

    parser.add_argument('--seed', type=int, default=12345, help='random seed 随机数种子')
    parser.add_argument('--random_choos', type=bool, default=False, help='random seed 随机数种子，是否随机，为True一般用于多次实验')
    # 存在output文件夹下
    parser.add_argument('--sub_them', type=str, default='2变量多对一', help='单次运行的存储文件夹字后面的内容--------------------存储数据父文件夹名字')
    # parser.add_argument('--sub_them', type=str, default='月度', help='单次运行的存储文件夹的月字后面的内容--------------------存储数据父文件夹名字')
    parser.add_argument('--true_sheetname', type=str, default='Sheet1', help='真实值的月份名称,execl文件的sheetname--------------------------真实值的月份数值')
    # parser.add_argument('--true_price', type=str, default='7月第二第三周', help='真实值的月份名称,execl文件的sheetname--------------------------真实值的月份数值')
    # parser.add_argument('--true_price', type=str, default='1-6月', help='真实值的月份名称,execl文件的sheetname--------------------------真实值的月份数值')
    parser.add_argument('--model', type=str, required=False, default='informer',
                        help='model of experiment, options: [informer, informerstack]')
    # ？？？？算是很重要的参数了
    parser.add_argument('--data', type=str, required=False, default='batterySD', help='data them，取决了在data parse中寻找的是哪个数据文件的配置,很重要')
    # parser.add_argument('--data', type=str, required=False, default='chicken_MS',help='data them，取决了在data parse中寻找的是哪个数据文件的配置,很重要')

    # parser.add_argument('--true_file', type=str, required=False, default='./TrueValue/2020年真实值.xls', help='真实值数据的文件名')
    # parser.add_argument('--true_file', type=str, required=False, default='./TrueValue/周粒度实验的真实价格.xls', help='真实值数据的文件名')
    # parser.add_argument('--true_file', type=str, required=False, default='./TrueValue/月粒度实验的真实价格.xls', help='真实值数据的文件名')
    # parser.add_argument('--true_file', type=str, required=False, default='./TrueValue/日粒度实验的真实价格.xls', help='真实值数据的文件名')
    # parser.add_argument('--true_file', type=str, required=False, default='./TrueValue/日粒度与月粒度对比实验的真实价格.xls', help='真实值数据的文件名')
    parser.add_argument('--true_file', type=str, required=False, default='./TrueValue/周-预测的rose真实价格.xls', help='真实值数据的文件名')

    # parser.add_argument('--data_path', type=str, default='周粒度-多特征数据汇总.csv', help='data file')
    # 训练数据
    # 不用管，内部写死了
    parser.add_argument('--data_path', type=str, default='电池循环汇总(训练数据).xlsx', help='data file')

    # 不用管，内部写死
    parser.add_argument('--columns', type=list, required=False, default=["date",'price'], help='存储预测数据的时候的列名，多对多M')
    # parser.add_argument('--columns', type=list, required=False, default=["time", 'GZ_maize_prince','CD_maize_price','CD_SBM_price','ZJ_SBM_prince','price'], help='存储预测数据的时候的列名，多对一MS、一对一S任务')
    # parser.add_argument('--shuffle_flag_train', type=str, required=False, default=True, help='训练的时候是否打乱数据[未完成该定义]')
    # ？？？较重要的参数
    parser.add_argument('--features', type=str, default='S', help='预测任务选项（forecasting task, options）:[M, S, MS]; '
                                                                   'M:多变量预测多元（multivariate predict multivariate）, '
                                                                   'S:单变量预测单变量（univariate predict univariate）, '
                                                                   'MS:多变量预测单变量（multivariate predict univariate）')
    #----------------S任务下:下面的配置项不用修改,如果需要再进行修改-------------------
    parser.add_argument('--lag_sign', type=bool,required=False, default=False, help="是否进行滞后性处理，只需要进行一次即可。开启此选项进行一次处理后修改回为False，才有效。-------")
    parser.add_argument('--lag', type=int, default=0, help="滞后性处理的数值，代表滞后了多少，仅仅用于M或者MS模式----------")
    parser.add_argument('--original_multi_path', type=str, default='./Time_data/Uncleaned_data/价格-供求数据.xls',
                        help="供求价格的excel文件所在的路径")
    parser.add_argument('--output_multi_originalPath', type=str, default="./Time_data/Uncleaned_data/未进行滞后处理-价格-供求数据.csv",
                        help="生成供求价格的csv文件路径")
    parser.add_argument('--single_path', type=str, default="./Time_data/价格.csv", help="完整月均价数据的所在路径")
    parser.add_argument('--laged_multi_path', type=str, default="./data/Time_data/供求-价格.csv", help="经过滞后处理后的价格-供求数据")
    args = parser.parse_args(args=[])
    return args

"""
enc_in: informer的encoder的输入维度
dec_in: informer的decoder的输入维度
c_out: informer的decoder的输出维度
d_model: informer中self-attention的输入和输出向量维度
n_heads: multi-head self-attention的head数
e_layers: informer的encoder的层数
d_layers: informer的decoder的层数
d_ff: self-attention后面的FFN的中间向量表征维度
factor: probsparse attention中设置的因子系数
padding: decoder的输入中，作为占位的x_token是填0还是填1
distil: informer的encoder是否使用注意力蒸馏
attn: informer的encoder和decoder中使用的自注意力机制
embed: 输入数据的时序编码方式
activation: informer的encoder和decoder中的大部分激活函数
output_attention: 是否选择让informer的encoder输出attention以便进行分析

小数据集的预测可以先使用默认参数或适当减小d_model和d_ff的大小

"""


# %% [markdown]
# ## 5. 主函数

# %% [markdown]
# ## 5.1 前期准备

# %%
# 进行parser的变量初始化，获取实例。
args = initialize_parameter()

# 修改args的pred_len
args.pred_len = 50

# %%


# print("model：\t",args.model)
# 默认为false，暂时不用管
if args.lag_sign:
    lag_processor_main(args.original_multi_path, args.output_multi_originalPath, args.single_path, args.lag, args.laged_multi_path)
    print("已经处理完 滞后性数值进程---回退args.lag_sign参数为False并且建议定制好实验才可继续往下进行~")
    sys.exit()
# 判断GPU是否能够使用，并获取标识
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
# 判断是否使用多块GPU，默认不使用多块GPU
if False:
    # 获取显卡列表，type：str
    args.devices = args.devices.replace(' ', '')
    # 拆分显卡获取列表，type：list
    device_ids = args.devices.split(',')
    # 转换显卡id的数据类型
    args.device_ids = [int(id_) for id_ in device_ids]
    print("显卡设备id：", args.device_ids)
    # 获取第一块显卡
    args.gpu = args.device_ids[1]
    
    
# 初始化数据解析器，用于定义训练模式、预测模式、数据粒度的初始化选项。
"""
字典格式：{数据主题：{data：数据路径，'T':目标字段列名,'M'：，'S'：，'MS':}}

'M:多变量预测多元（multivariate predict multivariate）'，
'S:单变量预测单变量（univariate predict univariate）'，
'MS:多变量预测单变量（multivariate predict univariate）'。
"""
# 直接在data_parser中就可以设定args.enc_in, args.dec_in, args.c_out
data_parser = {
    
    'rose': {'data': 'data.csv', 'T': 'price', 'M': [2, 2, 2], 'S': [1, 1, 1], 'MS': [2, 2, 1]},
    'chicken_MS': {'data': '周粒度-多特征数据汇总.csv', 'T': 'price', 'M': [2, 2, 2], 'S': [1, 1, 1], 'MS': [5, 5, 1]},
    'Time_data': {'data': 'most_samll_test_1个变量.csv', 'T': 'X4', 'M': [3, 3, 3], 'S': [1, 1, 1], 'MS': [3, 3, 1]},
    # T不用管，我内部写死，实际上只看 'S': [1, 1, 1]，这个是单变量预测单变量
    'min': {'data': 'min2019.xlsx', 'T': 'price', 'MS': [2, 2, 2], 'S': [1, 1, 1], 'MS': [2, 2, 1]},
    # T不用管，文件地址也不用管，我内部写死，实际上只看 'S': [1, 1, 1]，这个是单变量预测单变量
    'batterySD': {'data': '电池', 'T': 'price', 'M': [2, 2, 2], 'S': [1, 1, 1], 'MS': [2, 2, 1]},
}
# 判断在parser中定义的数据主题是否在解析器中
if args.data in data_parser.keys():  
    # 根据args里面定义的数据主题，获取对应的初始化数据解析器info信息，type：dict
    # 此处data_info就是获取到的 data_parse中的 rose的数据
    data_info = data_parser[args.data]
    # 获取该数据主题的数据文件的路径（相对路径），父目录在上面定义过了
    args.data_path = data_info['data']
    # 从数据解析器中获取 S或 MS任务中的目标特征列名。
    # 此处target没有用，内部写死了
    args.target = data_info['T']
    # 从数据解析器中 根据变量features的初始化信息 获取 编码器输入大小，解码器输入大小，输出尺寸
    # args.features取值为S、M、MS，即单变量以及多变量
    # rose是S，所以取的是1，1，1，分别描述了encoder输入特征种类数、decoder输入特征种类数以及模型输出特征种类数，那是不是对于rose来说，M与MS字段没有作用
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]
# 堆栈编码器层数，type：list，可以先不管
args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ', '').split(',')]
# 时间特征编码的频率，就是进行特征工程的时候时间粒度选取多少
# ？？？可能要删，不用管，内部写死了
args.detail_freq = args.freq
args.freq = args.freq[-1:]
# print('Args in experiment:')
# print(args)
now_time = datetime.datetime.now().strftime('%mM_%dD %HH:%Mm:%Ss').replace(" ", "_").replace(":", "_")
# 获取模型实例
Exp = Exp_Informer

# info_dict存储的是模型信息
info_dict = dict()

# sys.exit()
# 构建单次运行的存储路径：informer_e50_b1024_dModel32_dFF128_s80_l40_p_40_min
run_name_dir_old = args.model + "_e" + str(args.train_epochs) + "_b" + str(args.batch_size) + "_dModel" + str(args.d_model) + "_dFF" + str(args.d_ff)+ "_s" + str(args.seq_len) + "_l" + str(args.label_len) + "_p" + str(args.pred_len)+ "_" + args.data
# 右侧的args.output表示output文件夹 
# output\rose_1变量一对一_w
args.output = os.path.join(args.output,args.data+"_" + args.sub_them)


# 输出的文件夹位置：output\min_1变量一对一\informer_e50_b1024_dModel32_dFF128_s80_l40_p_40_min
run_name_dir = os.path.join(args.output, run_name_dir_old)
if not os.path.exists(run_name_dir):
    os.makedirs(run_name_dir)
# 单次运行的n个实验的模型存储的路径：需要判断是否存在，训练的时候已经判断了
# ./checkpoints/batterySD
run_name_dir_ckp_main = os.path.join(args.checkpoints, args.data)
# './checkpoints/batterySD\\TwoFeatures(Δη,QD)\\informer_e50_b32_dModel32_dFF128_s100_l50_p50_batterySD'   
run_name_dir_ckp = os.path.join(run_name_dir_ckp_main,'Δη/informer/singleFeature' ,run_name_dir_old)

# 要进行多少次实验，一次实验就是完成一个模型的训练-测试-预测 过程。默认2次，rose用了5次
for ii in range(args.itr):
    print("-------------.....第{}次实验.....------------".format(ii+1))
    # 存到output文件夹下了
    run_ex_dir = os.path.join(run_name_dir, "第_{}_次实验记录".format(ii + 1))
    if args.random_choos == True:
        pass
    else:
        # 固定随机性
        # setup_seed(args.seed)
        seed=args.seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] =str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic =True
    if not os.path.exists(run_ex_dir):
        os.makedirs(run_ex_dir)
    # 添加实验info，存到 json文件中
    info_dict["实验序号"] = ii+1
    info_dict["model"] = args.model
    info_dict["data_them"] = args.data
    info_dict["编码器的输入序列长度 seq_len【滑动窗口大小】"] = args.seq_len
    info_dict["解码器的开始解码令牌起始位置 label_len"] = args.label_len
    info_dict["预测未来序列长度 pred_len"] = args.pred_len
    info_dict["时间特征编码的频率【数据粒度】freq"] = args.freq
    info_dict["dorpout"] = args.dropout
    info_dict["批次大小 batch_size"] = args.batch_size
    info_dict["提前停止的连续轮数 patience"] = args.patience
    info_dict["随机种子seed"] = args.seed
    info_dict["损失函数loss"] = args.loss
    info_dict["是否随机实验 random_choos"] = args.random_choos
    info_dict["滞后数值 lag"] = args.lag
    info_dict["编码器输入大小 enc_in"] = args.enc_in
    info_dict["解码器输入大小 dec_in"] = args.dec_in
    info_dict["输出尺寸 c_out"] = args.c_out
    info_dict["模型维数 d_model"] = args.d_model
    info_dict["多头部注意力机制的头部个数 n_heads"] = args.n_heads
    info_dict["编码器层数 e_layers"] = args.e_layers
    info_dict["解码器层数 d_layers"] = args.d_layers
    info_dict["堆栈编码器层数 s_layers"] = str(args.s_layers)
    info_dict["self-attention后面的FFN的中间向量表征维度 d_ff"] = args.d_ff
    info_dict["probsparse attn factor"] = args.factor
    info_dict["是否在编码器中不使用知识蒸馏 distil"] = args.distil
    info_dict["编码器的注意力机制 attn"] = args.attn
    info_dict["填充的值 padding"] = args.padding
    info_dict["时间特征编码 embed"] = args.embed
    info_dict["激活函数 activation"] = args.activation
    info_dict["是否在编码器中输出注意力 output_attention"] = args.output_attention
    info_dict["是否预测看不见的未来数据 do_predict"] = args.do_predict
    info_dict["在生成解码器中使用混合注意力 mix"] = args.mix
    info_dict["实验次数 itr"] = args.itr
    info_dict["校正的学习率 lradj"] = args.lradj
    info_dict["使用自动混合精度训练 use_amp"] = args.use_amp
    info_dict["逆标准化输出数据 inverse"] = args.inverse
    info_dict["优化器初始学习率 learning_rate"] = args.learning_rate

    
    # 实验设置记录要点，方便打印，同时也作为文件名字传入参数，setting record of experiments
    # args.model = 'informer'，args.data = 'min'，args.features = 'ETTh1'，最后一个不用管，内部写死了
    setting = '{}_{}_{}_{}'.format(ii + 1, args.model, args.data, args.features)
    # 设置实验，将数据参数和模型变量传入实例
    exp = Exp(args)  # set experiments

   



# %% [markdown]
# ## 5.2 训练模型

# %%
train_flag = True
plot_flag = True


if type(len) == int:
    del len

def get_len(x):
    i = 0 
    for _ in x:
        i += 1
    return i

battery_df =pd.read_csv("../dataset/processed_data/shu_old_fade_rate.csv")
#battery_df = pd.read_csv("../dataset/processed_data/nature_processed_dup.csv",index_col=0)

cols = list(battery_df.columns)



with open("./args/train_args.pkl","rb") as f:
    (train_all_cols,test_cols) = pickle.load(f)
    
print(f"all: {len(train_all_cols)}")


with open("./args/bucket_list.pickle","rb") as handler: 
    bucket_list =  pickle.load(handler)

# %%
temp_train_all_cols = train_all_cols

# %%
import sys  

remove_idx = int(sys.argv[1]) 

if remove_idx >= 0:
    setting = f"model_remove{remove_idx}"
else:
    setting = "model_all"

if remove_idx >= 0:
    #train_all_cols = temp_train_all_cols[0:remove_idx] + temp_train_all_cols[remove_idx+1:]
    train_all_cols =  bucket_list[remove_idx] 
    
    print(f"has removed {len(temp_train_all_cols)} : {len(train_all_cols) }")
    
else:
    train_all_cols = temp_train_all_cols

# %%
# 训练

base_path = "./checkpoints/basemodel"

test_battery_df = battery_df[test_cols]

if type(len) == int:
    del len

run_name_dir_ckp = f"{base_path}/"


train_battery_df = battery_df[train_all_cols]



if train_flag:

    if type(len) == int:
        del len

    delta_data = train_battery_df

    print('>>>>>>>start training :  {}  >>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    print('d_model=', args.d_model)
    print('d_ff=', args.d_ff)
    print('batch=', args.batch_size)




    model, info_dict, all_epoch_train_loss, all_epoch_vali_loss, all_epoch_test_loss, epoch_count = exp.train(
        setting, info_dict, run_name_dir_ckp, run_ex_dir,args)

if plot_flag == True:

    #  1. d_model=32，d_ff=128，batch=64

    desp = "pred_len:50"

    delta_data = test_battery_df

    from sklearn.metrics import mean_squared_error # 均方误差
    import matplotlib.pyplot as plt

    print('d_model=', args.d_model)
    print('d_ff=', args.d_ff)
    print('batch=', args.batch_size)

    seed=args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] =str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic =True


    # 为了不经过训练，导入模型
    dataset = Dataset_Custom(
                root_path=args.root_path,
                data_path=args.data_path,

                flag='test',

                size=[args.seq_len, args.label_len, args.pred_len],

                features=args.features,

                timeenc=0,

                args=args
            )
    scalerDataη1=dataset.scalerDataη
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if not args.use_multi_gpu else self.args.devices
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')
    model_dict = {
                'informer':Informer,
                'informerstack':InformerStack,
            }
    e_layers = args.e_layers if args.model=='informer' else args.s_layers
    model = model_dict[args.model](
        args.enc_in,
        args.dec_in, 
        args.c_out, 
        args.seq_len, 
        args.label_len,
        args.pred_len, 
        args.factor,
        args.d_model, 
        args.n_heads, 
        e_layers, # args.e_layers,
        args.d_layers, 
        args.d_ff,
        args.dropout, 
        args.attn,
        args.embed,
        args.freq,
        args.activation,
        args.output_attention,
        args.distil,
        args.mix,
        device
    ).float()





    with  torch.no_grad():
        path = os.path.join(run_name_dir_ckp ,setting)
        best_model_path = path+'/'+'checkpoint.pth'

        print(f"best_model_path = {best_model_path}")


        
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        model.to('cpu')


        dataAll = delta_data
    
        dataAll=dataAll.iloc[:-1]
        lenList=[]
        lenListSum=0
        colListAll=[]

        # 每列的长度
        for j,col in enumerate(dataAll.columns):

            len=(np.array(dataAll.iloc[:,j].dropna())).shape[0]
            lenList.append(len)
            lenListSum=lenListSum+len
            colListAll.append(col)


        print('lenList',lenList)    

        metrics_dict = {}

        for i in range(dataAll.shape[1]):

            if(i>-1):
                lenList[i]


                rawdataNewη1 = delta_data
                rawdataNewη1=rawdataNewη1.iloc[:-1].values[:,i].reshape(-1, 1)

                rawdataNewη1 = scalerDataη1.transform(rawdataNewη1)
            
    
                result1=np.vstack((rawdataNewη1))
                    


                res_list1_encoder1 = torch.from_numpy(result1[:args.seq_len])
                res_list1_decoder1 = torch.from_numpy(result1[args.seq_len-args.label_len:args.seq_len])


                start = 0

                resItemList1=[]


            

                while(start<(lenList[i]-args.seq_len)//args.pred_len):
                    window_encoder= torch.tensor(res_list1_encoder1[start*args.pred_len: start*args.pred_len+args.seq_len])

                    window_decoder= torch.tensor(res_list1_encoder1[start*args.pred_len+args.seq_len- args.label_len: start*args.pred_len+args.seq_len])

                    seq_x_mark = torch.zeros(1)

                    seq_y_mark = torch.zeros(1)
                    
                    
                    if args.use_gpu:
                        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if not args.use_multi_gpu else self.args.devices
                        device = torch.device('cuda:{}'.format(args.gpu))

                    else:
                        device = torch.device('cpu')
                    global dec_inp
                    batch_x = window_encoder.float()
                    batch_y = window_decoder.float()

                    batch_x_mark = seq_x_mark.float(            )
                    batch_y_mark = seq_y_mark.float()
                    batch_x=batch_x.unsqueeze(0)
                    batch_y=batch_y.unsqueeze(0)
                    batch_x_mark=batch_x_mark.unsqueeze(0)
                    batch_y_mark=batch_y_mark.unsqueeze(0)
                    # decoder input
                    if args.padding==0:

                        dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float()
                    elif args.padding==1:
                        dec_inp = torch.ones([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float()
    
                    dec_inp = torch.cat([batch_y[:,:args.label_len,:], dec_inp], dim=1).float()

                    if args.use_amp:

                        with torch.cuda.amp.autocast():

                            if args.output_attention:
                                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    else:
                        if args.output_attention:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)


                    f_dim = -1 if args.features=='MS' else 0

                    batch_y = batch_y[:,-args.pred_len:,f_dim:]


                    res_list1_encoder1=torch.cat([res_list1_encoder1,outputs.squeeze(0)],dim=0)
                    
                

                    start = start + 1
                    
                
                res_listnew=scalerDataη1.inverse_transform(res_list1_encoder1[:,-1].reshape(-1,1)).ravel()  
                res_listnew=res_listnew[0:lenList[i]]


                testIndex = i

                rawdata =   dataAll.iloc[:,i].dropna()


                x = np.arange(1,lenList[i]+1)

                y = rawdata[:]

                y2 = np.array(res_listnew[:]).tolist()
                y2_length = np.array(y2).shape[0]

                x = x[:y2_length]
                y = y[:y2_length]

                print(f"test index {testIndex}")
                print('MSE：',mean_squared_error(y,y2))

                pd.DataFrame.from_dict({"ture":y,"predict":y2}).to_csv(f"{path}/{colListAll[i]}.csv")


                from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,r2_score

                metrics_row = []
                metrics_row.append( np.sqrt(mean_squared_error(y,y2)))
                metrics_row.append(mean_absolute_error(y,y2))
                metrics_row.append(mean_absolute_percentage_error(y,y2))
                metrics_row.append(r2_score(y,y2))
                metrics_dict[colListAll[i]] = metrics_row


        metrics_df = pd.DataFrame.from_dict(metrics_dict,orient="index",columns=["rmse","mae","mape","r2"])
        metrics_df.to_csv(f"{path}/metrics.csv")


if plot_flag == True:

    #  1. d_model=32，d_ff=128，batch=64

    desp = "pred_len:50"

    delta_data = battery_df[temp_train_all_cols]

    from sklearn.metrics import mean_squared_error # 均方误差
    import matplotlib.pyplot as plt

    print('d_model=', args.d_model)
    print('d_ff=', args.d_ff)
    print('batch=', args.batch_size)

    seed=args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] =str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic =True


    # 为了不经过训练，导入模型
    dataset = Dataset_Custom(
                root_path=args.root_path,
                data_path=args.data_path,

                flag='test',

                size=[args.seq_len, args.label_len, args.pred_len],

                features=args.features,

                timeenc=0,

                args=args
            )
    scalerDataη1=dataset.scalerDataη
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if not args.use_multi_gpu else self.args.devices
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')
    model_dict = {
                'informer':Informer,
                'informerstack':InformerStack,
            }
    e_layers = args.e_layers if args.model=='informer' else args.s_layers
    model = model_dict[args.model](
        args.enc_in,
        args.dec_in, 
        args.c_out, 
        args.seq_len, 
        args.label_len,
        args.pred_len, 
        args.factor,
        args.d_model, 
        args.n_heads, 
        e_layers, # args.e_layers,
        args.d_layers, 
        args.d_ff,
        args.dropout, 
        args.attn,
        args.embed,
        args.freq,
        args.activation,
        args.output_attention,
        args.distil,
        args.mix,
        device
    ).float()





    with  torch.no_grad():
        path = os.path.join(run_name_dir_ckp ,setting)
        best_model_path = path+'/'+'checkpoint.pth'

        print(f"best_model_path = {best_model_path}")


        
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        model.to('cpu')


        dataAll = delta_data
    
        dataAll=dataAll.iloc[:-1]
        lenList=[]
        lenListSum=0
        colListAll=[]

        # 每列的长度
        for j,col in enumerate(dataAll.columns):

            len=(np.array(dataAll.iloc[:,j].dropna())).shape[0]
            lenList.append(len)
            lenListSum=lenListSum+len
            colListAll.append(col)


        print('lenList',lenList)    

        metrics_dict = {}

        for i in range(dataAll.shape[1]):

            if(i>-1):
                lenList[i]


                rawdataNewη1 = delta_data
                rawdataNewη1=rawdataNewη1.iloc[:-1].values[:,i].reshape(-1, 1)

                rawdataNewη1 = scalerDataη1.transform(rawdataNewη1)
            
    
                result1=np.vstack((rawdataNewη1))
                    


                res_list1_encoder1 = torch.from_numpy(result1[:args.seq_len])
                res_list1_decoder1 = torch.from_numpy(result1[args.seq_len-args.label_len:args.seq_len])


                start = 0

                resItemList1=[]


            

                while(start<(lenList[i]-args.seq_len)//args.pred_len):
                    window_encoder= torch.tensor(res_list1_encoder1[start*args.pred_len: start*args.pred_len+args.seq_len])

                    window_decoder= torch.tensor(res_list1_encoder1[start*args.pred_len+args.seq_len- args.label_len: start*args.pred_len+args.seq_len])

                    seq_x_mark = torch.zeros(1)

                    seq_y_mark = torch.zeros(1)
                    
                    
                    if args.use_gpu:
                        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if not args.use_multi_gpu else self.args.devices
                        device = torch.device('cuda:{}'.format(args.gpu))

                    else:
                        device = torch.device('cpu')
                    global dec_inp2
                    batch_x = window_encoder.float()
                    batch_y = window_decoder.float()

                    batch_x_mark = seq_x_mark.float(            )
                    batch_y_mark = seq_y_mark.float()
                    batch_x=batch_x.unsqueeze(0)
                    batch_y=batch_y.unsqueeze(0)
                    batch_x_mark=batch_x_mark.unsqueeze(0)
                    batch_y_mark=batch_y_mark.unsqueeze(0)
                    # decoder input
                    if args.padding==0:

                        dec_inp2 = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float()
                    elif args.padding==1:
                        dec_inp2 = torch.ones([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float()
    
                    dec_inp2 = torch.cat([batch_y[:,:args.label_len,:], dec_inp2], dim=1).float()

                    if args.use_amp:

                        with torch.cuda.amp.autocast():

                            if args.output_attention:
                                outputs = model(batch_x, batch_x_mark, dec_inp2, batch_y_mark)[0]
                            else:
                                outputs = model(batch_x, batch_x_mark, dec_inp2, batch_y_mark)

                    else:
                        if args.output_attention:
                            outputs = model(batch_x, batch_x_mark, dec_inp2, batch_y_mark)[0]
                        else:
                            outputs = model(batch_x, batch_x_mark, dec_inp2, batch_y_mark)


                    f_dim = -1 if args.features=='MS' else 0

                    batch_y = batch_y[:,-args.pred_len:,f_dim:]


                    res_list1_encoder1=torch.cat([res_list1_encoder1,outputs.squeeze(0)],dim=0)
                    
                

                    start = start + 1
                    
                
                res_listnew=scalerDataη1.inverse_transform(res_list1_encoder1[:,-1].reshape(-1,1)).ravel()  
                res_listnew=res_listnew[0:lenList[i]]


                testIndex = i

                rawdata =   dataAll.iloc[:,i].dropna()


                x = np.arange(1,lenList[i]+1)

                y = rawdata[:]

                y2 = np.array(res_listnew[:]).tolist()
                y2_length = np.array(y2).shape[0]

                x = x[:y2_length]
                y = y[:y2_length]

                print(f"test index {testIndex}")
                print('MSE：',mean_squared_error(y,y2))

                pd.DataFrame.from_dict({"ture":y,"predict":y2}).to_excel(f"{path}/{colListAll[i]}.xlsx")


                from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,r2_score

                metrics_row = []
                metrics_row.append( np.sqrt(mean_squared_error(y,y2)))
                metrics_row.append(mean_absolute_error(y,y2))
                metrics_row.append(mean_absolute_percentage_error(y,y2))
                metrics_row.append(r2_score(y,y2))
                metrics_dict[colListAll[i]] = metrics_row


        metrics_df = pd.DataFrame.from_dict(metrics_dict,orient="index",columns=["rmse","mae","mape","r2"])
        metrics_df.to_csv(f"{path}/metrics.csv")



# %%
# metrics

all_metrics = pd.read_csv(f"{base_path}/{setting}/metrics.csv",index_col=0)

metrics_value = ["rmse","mae","mape","r2"]

for v in metrics_value:
    
    all_metrics[v].rename(f"all_{v}").to_csv(f"{base_path}/{setting}/{v}_metrics.csv")


