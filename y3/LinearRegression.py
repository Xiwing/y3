import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train = pd.read_csv("train_.csv",encoding='gbk') # 导入train的数据
test = pd.read_csv("test_.csv",encoding='gbk') # 导入test的数据
Y_train = train.outcome # y值
X_train = train[['0','1','2','3','4','5','6','7','8','9','10','11','12']] # x值
alpha = 0.01 # 学习率
X_test = test[['0','1','2','3','4','5','6','7','8','9','10','11','12']]


class LinearRegression:
    def __init__(self,train,test): # 预处理，label是y的真实值，data是x的值
        num_features = self.X_train.shape[1] # 数组的列数，有多少个x
        self.theta = np.zeros((num_features,1)) # theta是一个矩阵，θ的个数与x一一对应
        print(num_features)
    def train(self,alpha,num_iterations = 500): # 训练函数，alpha是学习率α，num_iterations是迭代次数
         cost_history= self.gradient_descent(alpha,num_iterations)
         return self.theta,cost_history

    def gradient_descent(self, alpha, num_iterations): # 执行梯度下降，参数更新和梯度计算
        cost_history = [] # 建立一个列表记录损失值
        for i in range (num_iterations):# 一共的迭代次数，因为进行迭代，所以要建立函数进行迭代
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data,self.labels)) # 把每次得到的损失值放在列表末尾，用append；要写一个函数计算损失值
        return cost_history

    def gradient_step(self,alpha): # 梯度下降的步骤，用到的是train，只执行一次
        num_examples = self.data.shape[0] # 样本个数，有多少列
        prediction = LinearRegression.hypothesis(self.data,self.theta)#计算预测值，所以有需要创建一个函数,要得到预测值就要知道当前的x（数据）和当前的θ
        delta = prediction - self.labels # 得到残差Δ
        theta =self.theta
        theta = theta - alpha*(1/num_examples)*(np.dot(delta.T,self.data)).T#注意Δ要进行转置，因为是矩阵乘法;第二次转置是因为一开始得到的θ是列矩阵，转置后变为行矩阵
        self.theta = theta # 更新θ

    def cost_function(self,data,labels):
        num_exemples = data.shape[0]
        delta = LinearRegression.hypothesis(self.data,self.theta)-labels
        cost = (1/2)*np.dot(delta.T,delta)
        return cost[0][0]

    @staticmethod # 可直接进行调用
    def hypothesis(self,data,theta): # 计算预测值
        predictions =np.dot(self,data)
        return predictions

