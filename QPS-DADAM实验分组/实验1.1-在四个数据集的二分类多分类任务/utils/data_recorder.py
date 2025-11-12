import numpy as np
from abc import ABC, abstractmethod
import math
import torch

def disable_method(method):
    def wrapper(*args,**kwargs):
        print("父类方法已被禁用")
    return wrapper

class LogisticsRegressionRecorder:
    def __init__(self):
        self.loss_dict = {}
        self.f1_dict = {}
        self.accu_dict = {}
        self.recall_dict = {}
        self.lossopt_dict={}
        self.regret_dict={}
        self.sum_loss_dict={}   # 用于求取regret
    
    def init_alg_model(self,alg_label):
        """ 
            初始化模型，向字典中对应alg_lable中初始化为空表
        """
        self.lossopt_dict[alg_label] = []
        self.loss_dict[alg_label] = []
        self.f1_dict[alg_label] = []
        self.accu_dict[alg_label] = []
        self.recall_dict[alg_label] = []
        self.regret_dict[alg_label]=[]
        self.sum_loss_dict[alg_label]=0

    # 将对应alg_label的list更新为一个新list
    
    def update_lossopt_ls(self,alg_label, ls:list):
        self.lossopt_dict[alg_label] = ls

    def update_loss_ls(self,alg_label, ls:list):
        self.loss_dict[alg_label] = ls
    
    def update_f1_ls(self,alg_label, ls:list):
        self.f1_dict[alg_label] = ls

    def update_accu_ls(self,alg_label, ls:list):
        self.accu_dict[alg_label] = ls

    def update_recall_ls(self,alg_label, ls:list):
        self.recall_dict[alg_label] = ls
    
    def update_regret_ls(self,alg_label, ls:list):
        self.regret_dict[alg_label] = ls

    # 向对应alg_label的list中添加一个值
    def append_loss(self,alg_label,number: float):
        self.loss_dict[alg_label].append(number)
    
    def append_f1(self,alg_label,number: float):
        self.f1_dict[alg_label].append(number)
    
    def append_accu(self,alg_label,number: float):
        self.accu_dict[alg_label].append(number)
    
    def append_lossopt(self,alg_label,number: float):
        self.lossopt_dict[alg_label].append(number)
    
    def append_recall(self,alg_label,number: float):
        self.recall_dict[alg_label].append(number)
    
    def append_regret(self,alg_label,number: float):
        self.regret_dict[alg_label].append(number)
    
    def get_all_data(self):
        '''
            导出所有的数据为一个字典
        '''
        return {
            "lossopt":self.lossopt_dict,
            "loss":self.loss_dict,
            "f1":self.f1_dict,
            "accu":self.accu_dict,
            "recall":self.recall_dict,
            "regret":self.regret_dict
        }

    def get_all_data_for_writer_scalars(self):
        '''
            将数据转化为可用于summary writer
        '''
        def transform_dict_to_list(src):
            return [dict(zip(src.keys(), values)) for values in zip(*src.values())]
  
        loss_ls=transform_dict_to_list(self.loss_dict)
        lossopt_ls=transform_dict_to_list(self.lossopt_dict)
        f1_ls=transform_dict_to_list(self.f1_dict)
        accu_ls=transform_dict_to_list(self.accu_dict)
        recall_ls=transform_dict_to_list(self.recall_dict)
        regret_ls=transform_dict_to_list(self.regret_dict)

        return {
            "lossopt":lossopt_ls,
            "loss":loss_ls,
            "f1":f1_ls,
            "accu":accu_ls,
            "recall":recall_ls,
            "regret":regret_ls
        }      

    def get_data_by_alg_label(self,alg_label):
        '''
            导出对应alg_lable的数据，输出格式为dict
        '''
        return {
            "lossopt":self.lossopt_dict[alg_label],
            "loss":self.loss_dict[alg_label],
            "f1":self.f1_dict[alg_label],
            "accu":self.accu_dict[alg_label],
            "recall":self.recall_dict[alg_label],
            "regret":self.regret_dict_dict[alg_label],
        }
      
    def get_data_by_measure(self,measure):
        '''
            导出对应alg_lable的数据，输出格式为dict
        '''
        if(measure=="lossopt"):
            return self.lossopt_dict
        elif(measure=="loss"):
            return self.loss_dict
        elif(measure=="f1"):
            return self.f1_dict
        elif(measure=="accu"):
            return self.accu_dict
        elif(measure=="recall"):
            return self.recall_dict
        elif(measure=="regret"):
            return self.regret_dict
        else:
            return {}
        
    def get_data_by_measure_and_alg_label(self,measure,alg_label):
        '''
            导出对应alg_lable且measure的数据，输出格式为list
        '''
        return self.get_data_by_measure(measure)[alg_label]

    @abstractmethod
    def auto_cal_and_append(self):
        pass


class LRTrainingRecorder(LogisticsRegressionRecorder):
    def __init__(self,is_binary=True):
        super().__init__()
        self.is_binary=is_binary    # 用于标识是否为二分类问题还是多分类问题

    def auto_cal_and_append(self,alg_label,w,w_opt,loss_fn:callable,get_scores:callable,predict:callable,x_train,y_train,x_test,y_test):
        '''
            自动计算并插入
        '''
        super().auto_cal_and_append()
        if self.is_binary:
            # 如果是二分类问题，则get scores为binary
            y_test=y_test.squeeze()
            average="binary"
        else:
            # 如果是多分类问题，则get scores传入，并且计算最大下标
            y_test=np.argmax(y_test, axis=0)
            average="macro"

        # 计算损失
        loss = loss_fn(x_train, y_train, w)
        if w_opt is not None:
            loss_opt = loss_fn(x_train, y_train, w_opt)
        else:
            loss_opt=0

        # 计算初始的准确率等参数
        predicted = predict(w, x_test).squeeze()
        accuracy, f1, recall = get_scores(y_test, predicted, average)

        # 计算累计损失和reget
        self.sum_loss_dict[alg_label]=self.sum_loss_dict[alg_label]+loss
        # 求取当前时间
        t=(len(self.loss_dict[alg_label]))
        regret=self.sum_loss_dict[alg_label]/(t+1)

        # 插入
        self.append_loss(alg_label,loss)
        self.append_lossopt(alg_label,loss_opt)
        self.append_accu(alg_label,accuracy)
        self.append_f1(alg_label,f1)
        self.append_recall(alg_label,recall)
        self.append_regret(alg_label,regret)


class DCTrainingRecorder(LogisticsRegressionRecorder):
    def __init__(self):
        super().__init__()
        self.relative_loss_dict={}   # 使用gtadam中的相对损失

    def init_alg_model(self,alg_label):
        super().init_alg_model(alg_label)
        self.relative_loss_dict[alg_label]=[]

    def update_relative_loss_ls(self,alg_label, ls:list):
        self.relative_loss_dict[alg_label] = ls

    def append_relative_loss(self,alg_label,number: float):
        self.relative_loss_dict[alg_label].append(number)

    def get_all_data(self):
        '''
            导出所有的数据为一个字典
        '''
        res=super().get_all_data()
        res["relative_loss"]=self.relative_loss_dict
        return res

    def get_all_data_for_writer_scalars(self):
        '''
            将数据转化为可用于summary writer
        '''
        def transform_dict_to_list(src):
            return [dict(zip(src.keys(), values)) for values in zip(*src.values())]
        res=super().get_all_data_for_writer_scalars()
        relative_loss_ls=transform_dict_to_list(self.relative_loss_dict)
        res["relative_loss"]=relative_loss_ls
        return res

    def get_data_by_alg_label(self,alg_label):
        '''
            导出对应alg_lable的数据，输出格式为dict
        '''
        res=super().get_data_by_alg_label(alg_label)
        res["relative_loss"]=self.relative_loss_dict[alg_label]
        return res
      
    def get_data_by_measure(self,measure):
        '''
            导出对应alg_lable的数据，输出格式为dict
        '''
        res=super().get_data_by_measure(measure)
        if measure=="relative_loss":
            return self.relative_loss_dict
        return res
            

    def get_data_by_measure_and_alg_label(self,measure,alg_label):
        '''
            导出对应alg_lable且measure的数据，输出格式为list
        '''
        return self.get_data_by_measure(measure)[alg_label]

    def auto_cal_and_append(self,alg_label,w,w_opt,loss_fn:callable,get_scores:callable,predict:callable,x_train,y_train,transform_func:callable,t):
        '''
            自动计算并插入
        '''
        x_train=transform_func(x_train,t)
        y_train=transform_func(y_train,t)

        # 计算损失
        loss = loss_fn(x_train, y_train, w)
        loss_opt = loss_fn(x_train, y_train, w_opt)

        # 计算初始的准确率等参数
        predicted = predict(w, x_train).squeeze()
        accuracy, f1, recall = get_scores(y_train.squeeze(), predicted, "binary")

        # 计算累计损失和reget
        self.sum_loss_dict[alg_label]=self.sum_loss_dict[alg_label]+loss
 
        regret=self.sum_loss_dict[alg_label]/(t+1)
        
        # 相对损失
        relative_loss=(loss-loss_opt)/(loss_opt)
        
        # 插入
        self.append_loss(alg_label,loss)
        self.append_lossopt(alg_label,loss_opt)
        self.append_accu(alg_label,accuracy)
        self.append_f1(alg_label,f1)
        self.append_recall(alg_label,recall)
        self.append_regret(alg_label,regret)
        self.append_relative_loss(alg_label,relative_loss)


class DCTrainingRecorderForNon(DCTrainingRecorder):
    def __init__(self):
        super().__init__()
        self.prefix_sum_loss_dict = {}
        self.sum_regret_dict={}   # 用于存放local regret从1到T累加和
    
    def init_alg_model(self, alg_label):
        super().init_alg_model(alg_label)
        self.prefix_sum_loss_dict[alg_label] = []
        self.sum_regret_dict[alg_label]=0
    
    def append_prefix_sum_loss(self, alg_label, number: float):
        self.prefix_sum_loss_dict[alg_label].append(number)

    @disable_method
    def auto_cal_and_append(self,alg_label,w,w_opt,loss_fn:callable,get_scores:callable,predict:callable,x_train,y_train,transform_func:callable,t):
        super().auto_cal_and_append(alg_label,w,w_opt,loss_fn,get_scores,predict,x_train,y_train,transform_func,t)

    # 非凸 local regret
    def auto_cal_and_append(self,alg_label,w,w_opt,loss_fn:callable,get_scores:callable,predict:callable,x_train,y_train,transform_func:callable,t,window: int):

        '''
            自动计算并插入
        '''
        x_train=transform_func(x_train,t)
        y_train=transform_func(y_train,t)

        # 计算损失
        loss = loss_fn(x_train, y_train, w)
        loss_opt = loss_fn(x_train, y_train, w_opt)

        # 计算初始的准确率等参数
        predicted = predict(w, x_train).squeeze()
        accuracy, f1, recall = get_scores(y_train.squeeze(), predicted, "binary")

        # 计算累计损失和reget
        self.sum_loss_dict[alg_label]=self.sum_loss_dict[alg_label]+loss
        
        if t-window>=0:
            pre_sum_loss=self.prefix_sum_loss_dict[alg_label][t-window] 
            local_regret=(self.sum_loss_dict[alg_label]-pre_sum_loss)/window
        else:
            pre_sum_loss=0
            local_regret=(self.sum_loss_dict[alg_label]-pre_sum_loss)/max(t,1)

        self.sum_regret_dict[alg_label]=self.sum_regret_dict[alg_label]+local_regret
        regret=self.sum_regret_dict[alg_label]/max(t,1)
        
        # 相对损失
        relative_loss=(loss-loss_opt)/(loss_opt)
        
        # 插入
        self.append_loss(alg_label,loss)
        self.append_lossopt(alg_label,loss_opt)
        self.append_accu(alg_label,accuracy)
        self.append_f1(alg_label,f1)
        self.append_recall(alg_label,recall)
        self.append_regret(alg_label,regret)
        self.append_relative_loss(alg_label,relative_loss)
        if isinstance(self.sum_loss_dict[alg_label], torch.Tensor):
            self.append_prefix_sum_loss(alg_label,self.sum_loss_dict[alg_label].clone())
        else:
            self.append_prefix_sum_loss(alg_label,self.sum_loss_dict[alg_label].copy())


class LinearRegressionTrainingRecorder:
    def __init__(self):
        self.loss_dict = {}
        self.lossopt_dict={}
        self.regret_dict={}
        self.sum_loss_dict={}   # 用于求取regret
        self.mean_w_dict = {}
        self.all_w_dict={}
    
    def init_alg_model(self,alg_label):
        """ 
            初始化模型，向字典中对应alg_lable中初始化为空表
        """
        self.lossopt_dict[alg_label] = []
        self.loss_dict[alg_label] = []
        self.regret_dict[alg_label]=[]
        self.sum_loss_dict[alg_label]=0
        self.mean_w_dict[alg_label]=[]
        self.all_w_dict[alg_label]=[]


    # 将对应alg_label的list更新为一个新list
    
    def update_lossopt_ls(self,alg_label, ls:list):
        self.lossopt_dict[alg_label] = ls

    def update_loss_ls(self,alg_label, ls:list):
        self.loss_dict[alg_label] = ls
    
    def update_allw_ls(self,alg_label, ls:list):
        self.all_w_dict[alg_label] = ls
    
    def update_meanw_ls(self,alg_label, ls:list):
        self.mean_w_dict_w_dict[alg_label] = ls
    
    def update_regret_ls(self,alg_label, ls:list):
        self.regret_dict[alg_label] = ls

    # 向对应alg_label的list中添加一个值
    def append_loss(self,alg_label,number: float):
        self.loss_dict[alg_label].append(number)
    
    def append_meanw(self,alg_label,number: float):
        self.mean_w_dict[alg_label].append(number)
    
    def append_allw(self,alg_label,allw_ls):
        self.all_w_dict[alg_label].append(allw_ls)
    
    def append_lossopt(self,alg_label,number: float):
        self.lossopt_dict[alg_label].append(number)

    def append_regret(self,alg_label,number:float):
        self.regret_dict[alg_label].append(number)

    def get_all_data(self):
        '''
            导出所有的数据为一个字典
        '''
        return {
            "lossopt":self.lossopt_dict,
            "loss":self.loss_dict,
            "regret":self.regret_dict,
            "meanw":self.mean_w_dict,
            "allw":self.all_w_dict
        }

    def get_all_data_for_writer_scalars(self):
        '''
            将数据转化为可用于summary writer
        '''
        def transform_dict_to_list(src):
            return [dict(zip(src.keys(), values)) for values in zip(*src.values())]
  
        loss_ls=transform_dict_to_list(self.loss_dict)
        lossopt_ls=transform_dict_to_list(self.lossopt_dict)
        regret_ls=transform_dict_to_list(self.regret_dict)

        return {
            "lossopt":lossopt_ls,
            "loss":loss_ls,
            "regret":regret_ls
        }

    def get_data_by_alg_label(self,alg_label):
        '''
            导出对应alg_lable的数据，输出格式为dict
        '''
        return {
            "lossopt":self.lossopt_dict[alg_label],
            "loss":self.loss_dict[alg_label],
            "regret":self.regret_dict_dict[alg_label],
            "meanw":self.mean_w_dict[alg_label],
            "allw":self.all_w_dict[alg_label]
        }
      
    def get_data_by_measure(self,measure):
        '''
            导出对应alg_lable的数据，输出格式为dict
        '''
        if(measure=="lossopt"):
            return self.lossopt_dict
        elif(measure=="loss"):
            return self.loss_dict
        elif(measure=="meanw"):
            return self.mean_w_dict
        elif(measure=="allw"):
            return self.all_w_dict
        elif(measure=="regret"):
            return self.regret_dict
        else:
            return {}
        
    def get_data_by_measure_and_alg_label(self,measure,alg_label):
        '''
            导出对应alg_lable且measure的数据，输出格式为list
        '''
        return self.get_data_by_measure(measure)[alg_label]

    @abstractmethod
    def auto_cal_and_append(self):
        pass

class SLTrainingRecorder(LinearRegressionTrainingRecorder):
    def __init__(self):
        super().__init__()

    def auto_cal_and_append(self,alg_label,allw,loss_fn:callable,x_train,y_train,transform_func: callable,t,agents_pos_ls):
        '''
            自动计算并插入
        '''
        super().auto_cal_and_append()
        # 导出所需数据
        x_train=transform_func(x_train,t)
        y_train=transform_func(y_train,t)
        meanw=np.mean(allw, axis=0)
        # 计算损失
        loss = loss_fn( x_train, y_train,meanw , agents_pos_ls)
        loss_opt = loss_fn(x_train, y_train, y_train, agents_pos_ls)

        # 计算累计损失和reget
        self.sum_loss_dict[alg_label]=self.sum_loss_dict[alg_label]+loss

        regret=self.sum_loss_dict[alg_label]/(t+1)
        
        # 插入
        self.append_loss(alg_label,loss)
        self.append_lossopt(alg_label,loss_opt)
        self.append_regret(alg_label,regret)
        self.append_allw(alg_label,allw.copy())
        self.append_meanw(alg_label,meanw)


class TTTrainingRecorder(LinearRegressionTrainingRecorder):
    def __init__(self):
        super().__init__()

    def auto_cal_and_append(self,alg_label,allw,loss_fn:callable,x_train,y_train,transform_func: callable,t,C):
        '''
            自动计算并插入
        '''
        super().auto_cal_and_append()
        # 导出所需数据
        x_train=transform_func(x_train,t,"x")
        y_train=transform_func(y_train,t,"y")
        C_ls=transform_func(C,t,"C")
        meanw=np.mean(allw, axis=0)
        try:
            loss = loss_fn(meanw, y_train,  C_ls)
        except:
            loss=np.nan

        loss_opt = loss_fn(x_train, y_train,  C_ls)

        # 计算累计损失和reget
        self.sum_loss_dict[alg_label]=self.sum_loss_dict[alg_label]+loss

        regret=self.sum_loss_dict[alg_label]/(t+1)
        
        # 插入
        self.append_loss(alg_label,loss)
        self.append_lossopt(alg_label,loss_opt)
        self.append_regret(alg_label,regret)
        self.append_allw(alg_label,allw.copy())
        self.append_meanw(alg_label,meanw)


class SSPTrainingRecorder(LinearRegressionTrainingRecorder):
    def __init__(self):
        super().__init__()

    def auto_cal_and_append(self,alg_label,allw,loss_fn:callable,x_train,y_train,transform_func: callable,t,P):
        '''
            自动计算并插入
        '''
        super().auto_cal_and_append()
        # 导出所需数据
        x_train=transform_func(x_train,t)
        y_train=transform_func(y_train,t)
        P_data=transform_func(P,t)
        meanw=np.mean(allw, axis=0).squeeze(axis=1)

        loss = loss_fn(x_train, meanw, P_data)
        loss_opt = loss_fn(x_train, y_train, P_data)

        # 计算累计损失和reget
        self.sum_loss_dict[alg_label]=self.sum_loss_dict[alg_label]+loss
        regret=self.sum_loss_dict[alg_label]/(t+1)
        
        # 插入
        self.append_loss(alg_label,loss)
        self.append_lossopt(alg_label,loss_opt)
        self.append_regret(alg_label,regret)
        self.append_allw(alg_label,allw.copy())
        self.append_meanw(alg_label,meanw)

class SSPTrainingRecorderForNon(LinearRegressionTrainingRecorder):
    def __init__(self):
        super().__init__()
        self.prefix_sum_loss_dict = {}
        self.sum_regret_dict={}   # 用于存放local regret从1到T累加和
    
    def init_alg_model(self, alg_label):
        super().init_alg_model(alg_label)
        self.prefix_sum_loss_dict[alg_label] = []
        self.sum_regret_dict[alg_label]=0
    
    def append_prefix_sum_loss(self, alg_label, number: float):
        self.prefix_sum_loss_dict[alg_label].append(number)


    # 非凸 local regret
    def auto_cal_and_append(self,alg_label,allw,loss_fn:callable,x_train,y_train,transform_func: callable,t,P,window: int):
        '''
            自动计算并插入
        '''
        super().auto_cal_and_append()
        # 导出所需数据
        x_train=transform_func(x_train,t)
        y_train=transform_func(y_train,t)
        P_data=transform_func(P,t)
        meanw=np.mean(allw, axis=0).squeeze(axis=1)

        loss = loss_fn(x_train, meanw, P_data)
        loss_opt = loss_fn(x_train, y_train, P_data)

        # 计算累计损失和reget
        self.sum_loss_dict[alg_label]=self.sum_loss_dict[alg_label]+loss
        
        if t-window>=0:
            pre_sum_loss=self.prefix_sum_loss_dict[alg_label][t-window] 
            local_regret=(self.sum_loss_dict[alg_label]-pre_sum_loss)/window
        else:
            pre_sum_loss=0
            local_regret=(self.sum_loss_dict[alg_label]-pre_sum_loss)/max(t,1)

        self.sum_regret_dict[alg_label]=self.sum_regret_dict[alg_label]+local_regret
        regret=self.sum_regret_dict[alg_label]/max(t,1)
        
        # 插入
        self.append_loss(alg_label,loss)
        self.append_lossopt(alg_label,loss_opt)
        self.append_regret(alg_label,regret)
        self.append_allw(alg_label,allw.copy())
        self.append_meanw(alg_label,meanw)
        self.append_prefix_sum_loss(alg_label,self.sum_loss_dict[alg_label].copy())

# 用于图像去模糊的单一算法的多个图像（非凸）
class IDTrainingRecorderForOne(LinearRegressionTrainingRecorder):
    def __init__(self):
        super().__init__()
        self.psnr_dict={}
        self.ssim_dict={}
    
    def init_alg_model(self,alg_label):
        """ 
            初始化模型，向字典中对应alg_lable中初始化为空表
        """
        super().init_alg_model(alg_label)
        self.psnr_dict[alg_label]=[]
        self.ssim_dict[alg_label]=[]
    
    
    def update_psnr_ls(self,alg_label, ls:list):
        self.psnr_dict[alg_label] = ls

    def update_ssim_ls(self,alg_label, ls:list):
        self.ssim_dict[alg_label] = ls


    def append_psnr(self,alg_label,number:float):
        self.psnr_dict[alg_label].append(number)

    def append_ssim(self,alg_label,number:float):
        self.ssim_dict[alg_label].append(number)

    def get_all_data(self):
        '''
            导出所有的数据为一个字典
        '''
        return {
            "lossopt":self.lossopt_dict,
            "loss":self.loss_dict,
            "regret":self.regret_dict,
            "meanw":self.mean_w_dict,
            "allw":self.all_w_dict,
            "psnr":self.psnr_dict,
            "ssim":self.ssim_dict
        }

    def get_all_data_for_writer_scalars(self):
        '''
            将数据转化为可用于summary writer
        '''
        def transform_dict_to_list(src):
            return [dict(zip(src.keys(), values)) for values in zip(*src.values())]
  
        loss_ls=transform_dict_to_list(self.loss_dict)
        lossopt_ls=transform_dict_to_list(self.lossopt_dict)
        regret_ls=transform_dict_to_list(self.regret_dict)
        psnr_ls=transform_dict_to_list(self.psnr_dict)
        ssim_ls=transform_dict_to_list(self.ssim_dict)

        return {
            "lossopt":lossopt_ls,
            "loss":loss_ls,
            "regret":regret_ls,
            "psnr":psnr_ls,
            "ssim":ssim_ls
        }

    def get_data_by_alg_label(self,alg_label):
        '''
            导出对应alg_lable的数据，输出格式为dict
        '''
        return {
            "lossopt":self.lossopt_dict[alg_label],
            "loss":self.loss_dict[alg_label],
            "regret":self.regret_dict[alg_label],
            "meanw":self.mean_w_dict[alg_label],
            "allw":self.all_w_dict[alg_label],
            "psnr":self.psnr_dict[alg_label],
            "ssim":self.ssim_dict[alg_label]
        }
      
    def get_data_by_measure(self,measure):
        '''
            导出对应alg_lable的数据，输出格式为dict
        '''
        if(measure=="lossopt"):
            return self.lossopt_dict
        elif(measure=="loss"):
            return self.loss_dict
        elif(measure=="meanw"):
            return self.mean_w_dict
        elif(measure=="allw"):
            return self.all_w_dict
        elif(measure=="regret"):
            return self.regret_dict
        elif(measure=="psnr"):
            return self.psnr_dict
        elif(measure=="ssim"):
            return self.ssim_dict
        else:
            return {}
        
    def get_data_by_measure_and_alg_label(self,measure,alg_label):
        '''
            导出对应alg_lable且measure的数据，输出格式为list
        '''
        return self.get_data_by_measure(measure)[alg_label]

    
    def auto_cal_and_append(self,alg_label,original,allw,motion_blur_kernel_ls,gaussian_noise_ls,loss_fn:callable,calculate_psnr:callable,calculate_ssim:callable):
        '''
            自动计算并插入
        '''
        super().auto_cal_and_append()
        meanw=np.mean(allw, axis=0)

        # 导出所需数据
        # loss_ls=[]
        # for tmp_w, motion_kernel, gaussian_noise in zip(allw, motion_blur_kernel_ls, gaussian_noise_ls):
        #     loss_ls.append(loss_fn(original, tmp_w, motion_kernel, gaussian_noise))
        # avg_loss=np.mean(loss_ls)
        # avg_loss_opt=avg_loss.copy()
        avg_loss=0
        for motion_kernel, gaussian_noise in zip(motion_blur_kernel_ls, gaussian_noise_ls):
            avg_loss+=loss_fn(original, meanw, motion_kernel, gaussian_noise)
        avg_loss_opt=avg_loss.copy()
        psnr=calculate_psnr(original,meanw)
        ssim=calculate_ssim(original,meanw)

        # 插入
        self.append_loss(alg_label,avg_loss)
        self.append_lossopt(alg_label,avg_loss_opt)
        self.append_regret(alg_label,0)
        self.append_psnr(alg_label,psnr)
        self.append_ssim(alg_label,ssim)
        # self.append_allw(alg_label,allw.copy())
        # self.append_meanw(alg_label,meanw)
        
# 多机器跟踪问题 multi-robot target tracking
class MRTTTrainingRecorder(LinearRegressionTrainingRecorder):
    def __init__(self):
        super().__init__()

    def auto_cal_and_append(self,alg_label,allw,loss_fn:callable,t):
        '''
            自动计算并插入
        '''
        super().auto_cal_and_append()
        
        meanw=np.mean(allw, axis=0)
        loss=0
        # 计算损失
        for i in range(allw.shape[0]):
            loss += loss_fn(allw[i],t,i)

        loss_opt =loss

        # 计算累计损失和reget
        self.sum_loss_dict[alg_label]=self.sum_loss_dict[alg_label]+loss

        regret=self.sum_loss_dict[alg_label]/(t+1)
        
        # 插入
        self.append_loss(alg_label,loss)
        self.append_lossopt(alg_label,loss_opt)
        self.append_regret(alg_label,regret)
        self.append_allw(alg_label,allw.copy())
        self.append_meanw(alg_label,meanw)


# 分布式压缩感知问题 Compressed Sensing
class CSTrainingRecorder(LinearRegressionTrainingRecorder):
    t=0
    def __init__(self):
        super().__init__()
        self.re_dict={}

    def init_alg_model(self,alg_label):
        """ 
            初始化模型，向字典中对应alg_lable中初始化为空表
        """
        super().init_alg_model(alg_label)
        self.re_dict[alg_label]=[]
    
    
    def update_re_ls(self,alg_label, ls:list):
        self.re_dict[alg_label] = ls

    def append_re(self,alg_label,number:float):
        self.re_dict[alg_label].append(number)

    def get_all_data(self):
        '''
            导出所有的数据为一个字典
        '''
        return {
            "lossopt":self.lossopt_dict,
            "loss":self.loss_dict,
            "regret":self.regret_dict,
            "meanw":self.mean_w_dict,
            "allw":self.all_w_dict,
            "re":self.re_dict,
            "ssim":self.ssim_dict
        }

    def get_all_data_for_writer_scalars(self):
        '''
            将数据转化为可用于summary writer
        '''
        def transform_dict_to_list(src):
            return [dict(zip(src.keys(), values)) for values in zip(*src.values())]
  
        loss_ls=transform_dict_to_list(self.loss_dict)
        lossopt_ls=transform_dict_to_list(self.lossopt_dict)
        regret_ls=transform_dict_to_list(self.regret_dict)
        re_ls=transform_dict_to_list(self.re_dict)

        return {
            "lossopt":lossopt_ls,
            "loss":loss_ls,
            "regret":regret_ls,
            "re":re_ls,
        }

    def get_data_by_alg_label(self,alg_label):
        '''
            导出对应alg_lable的数据，输出格式为dict
        '''
        return {
            "lossopt":self.lossopt_dict[alg_label],
            "loss":self.loss_dict[alg_label],
            "regret":self.regret_dict_dict[alg_label],
            "meanw":self.mean_w_dict[alg_label],
            "allw":self.all_w_dict[alg_label],
            "re":self.re_dict[alg_label],
        }
      
    def get_data_by_measure(self,measure):
        '''
            导出对应alg_lable的数据，输出格式为dict
        '''
        if(measure=="lossopt"):
            return self.lossopt_dict
        elif(measure=="loss"):
            return self.loss_dict
        elif(measure=="meanw"):
            return self.mean_w_dict
        elif(measure=="allw"):
            return self.all_w_dict
        elif(measure=="regret"):
            return self.regret_dict
        elif(measure=="relative_error"):
            return self.re_dict
        else:
            return {}
        
    def get_data_by_measure_and_alg_label(self,measure,alg_label):
        '''
            导出对应alg_lable且measure的数据，输出格式为list
        '''
        return self.get_data_by_measure(measure)[alg_label]
    

    def auto_cal_and_append(self,alg_label,allw,loss_fn:callable,x_true,relative_error_fn:callable,indices=None):
        '''
            自动计算并插入
        '''
        super().auto_cal_and_append()
        
        if indices is None:
            indices=list(range(allw.shape[0]))
        nums=len(indices)
        loss=0
        loss_opt=0
        # 计算损失
        for i in range(nums):
            index=indices[i]
            loss += loss_fn(allw[index],index)
            loss_opt +=loss_fn(x_true,i)

        loss=loss/nums
        loss_opt=loss_opt/nums
        # 计算累计损失和reget
        self.sum_loss_dict[alg_label]=self.sum_loss_dict[alg_label]+loss

        regret=self.sum_loss_dict[alg_label]/(self.t+1)

        tmp_w_node=np.array([allw[ii] for ii in indices])
        meanw=np.mean(tmp_w_node, axis=0)

        relative_error=relative_error_fn(tmp_w_node)
        # 插入
        self.append_loss(alg_label,loss)
        self.append_lossopt(alg_label,loss_opt)
        self.append_regret(alg_label,regret)
        self.append_allw(alg_label,tmp_w_node.copy())
        self.append_meanw(alg_label,meanw)
        self.append_re(alg_label,relative_error)