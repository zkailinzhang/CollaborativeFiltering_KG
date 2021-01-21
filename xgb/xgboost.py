


# 0 1:1 9:1 19:1 21:1 24:1 34:1 36:1 39:1 42:1 53:1 56:1 65:1 69:1 77:1 86:1 88:1 92:1 95:1 102:1 106:1 117:1 122:1
# 1 3:1 9:1 19:1 21:1 30:1 34:1 36:1 40:1 41:1 53:1 58:1 65:1 69:1 77:1 86:1 88:1 92:1 95:1 102:1 106:1 118:1 124:1
# 0 1:1 9:1 20:1 21:1 24:1 34:1 36:1 39:1 41:1 53:1 56:1 65:1 69:1 77:1 86:1 88:1 92:1 95:1 102:1 106:1 117:1 122:1
# 0 3:1 9:1 19:1 21:1 24:1 34:1 36:1 39:1 51:1 53:1 56:1 65:1 69:1 77:1 86:1 88:1 92:1 95:1 102:1 106:1 116:1 122:1
# 0 4:1 7:1 11:1 22:1 29:1 34:1 36:1 40:1 41:1 53:1 58:1 65:1 69:1 77:1 86:1 88:1 92:1 95:1 102:1 105:1 119:1 124:1
# 0 3:1 10:1 20:1 21:1 23:1 34:1 37:1 40:1 42:1 54:1 55:1 65:1 69:1 77:1 86:1 88:1 92:1 95:1 102:1 106:1 118:1 126:1
# 1 3:1 9:1 11:1 21:1 30:1 34:1 36:1 40:1 51:1 53:1 58:1 65:1 69:1 77:1 86:1 88:1 92:1 95:1 102:1 106:1 117:1 124:1

"""上面是libsvm的数据存储格式， 也是一种常用的格式，存储的稀疏数据。 
第一列是label. a:b a表示index， b表示在该index下的数值， 这就类似于one-hot"""
# https://github.com/dmlc/xgboost/blob/master/demo/data/agaricus.txt.train
# https://github.com/dmlc/xgboost/blob/master/demo/data/agaricus.txt.test


import numpy as np
import scipy.sparse    # 稀疏矩阵的处理
import pickle
import xgboost as xgb

# libsvm format data 的读入方式， 直接用xgb的DMatrix
dtrain = xgb.DMatrix('./xgbdata/agaricus.txt.train')
dtest = xgb.DMatrix('./xgbdata/agaricus.txt.test')


"""paramet setting"""
param = {
    'max_depth': 2,
    'eta': 1, 
    'silent': 1,
    'objective': 'binary:logistic'
}
watch_list = [(dtest, 'eval'), (dtrain, 'train')]  # 这个是观测的时候在什么上面的结果  观测集
num_round = 5
model = xgb.train(params=param, dtrain=dtrain, num_boost_round=num_round, evals=watch_list)

'''
num_boost_round: 树的个数
'booster':'gbtree', 这个指定基分类器
'objective': 'multi:softmax', 多分类的问题， 这个是优化目标，必须得有，因为xgboost里面有求一阶导数和二阶导数，其实就是这个。
'num_class':10, 类别数，与 multisoftmax 并用
'gamma':损失下降多少才进行分裂， 控制叶子节点的个数
'max_depth':12, 构建树的深度，越大越容易过拟合
'lambda':2, 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
'subsample':0.7, 随机采样训练样本
'colsample_bytree':0.7, 生成树时进行的列采样
'min_child_weight':3, 孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束
'silent':0 ,设置成1则没有运行信息输出，最好是设置为0.
'eta': 0.007, 如同学习率
'seed':1000,
'nthread':7, cpu 线程数


'''
#2
"""预测"""
pred = model.predict(dtest)    # 这里面表示的是正样本的概率是多少

from sklearn.metrics import accuracy_score
labels = dtrain.get_label()
predict_label = [round(values) for values in pred]
accuracy_score(labels, predict_label)   # 0.993

#3保存
"""两种方式： 第一种， pickle的序列化和反序列化"""
pickle.dump(model, open('./model/xgb1.pkl', 'wb'))
model1 = pickle.load(open('./model/xgb1.pkl', 'rb'))
model1.predict(dtest)

"""第二种模型的存储与导入方式 - joblib"""
import joblib
joblib.dump(model, './model/xgb.pkl')
model2 = joblib.load('./model/xgb.pkl')
model2.predict(dtest)


#4 交叉验证
# 这是模型本身的参数
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
num_round = 5   # 这个是和训练相关的参数

xgb.cv(param, dtrain, num_round, nfold=5, metrics={'error'}, seed=3)


#5 样本权重

# 这个函数是说在训练之前，先做一个预处理，计算一下正负样本的个数，然后加一个权重,解决样本不平衡的问题
def preproc(dtrain, dtest, param): 
    labels = dtrain.get_label()
    ratio = float(np.sum(labels==0)) / np.sum(labels==1)
    param['scale_pos_ratio'] = ratio
    return (dtrain, dtest, param)

# 下面我们在做交叉验证， 指明fpreproc这个参数就可以调整样本权重
xgb.cv(param, dtrain, num_round, nfold=5, metrics={'auc'}, seed=3, fpreproc=preproc)


#6 自定义loss
# 自定义目标函数（log似然损失），这个是逻辑回归的似然损失。 交叉验证
# 注意： 需要提供一阶和二阶导数

def logregobj(pred, dtrain):
    labels = dtrain.get_label()
    pred = 1.0 / (1+np.exp(-pred))    # sigmoid函数
    grad = pred - labels
    hess = pred * (1-pred)
    return grad, hess     # 返回一阶导数和二阶导数

def evalerror(pred, dtrain):
    labels = dtrain.get_label()
    return 'error', float(sum(labels!=(pred>0.0)))/len(labels)

param = {'max_depth':2, 'eta':1, 'silent':1}

# 自定义目标函数训练
model = xgb.train(param, dtrain, num_round, watch_list, logregobj, evalerror)

# 交叉验证
xgb.cv(param, dtrain, num_round, nfold=5, seed=3, obj=logregobj, feval=evalerror)

#7 用前n颗预测
# 前1棵
pred1 = model.predict(dtest, ntree_limit=1)
evalerror(pred1, dtest)

#8 画出特征重要度
from xgboost import plot_importance
plot_importance(model, max_num_features=10)


#9 sklearn GridSearchCV调参
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

model = XGBClassifier()
learning_rate = [0.0001, 0.001, 0.1, 0.2, 0.3]
param_grid = dict(learning_rate=learning_rate)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(x_train, y_train)

print("best: %f using %s" %(grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']

for mean, param in zip(means, params):
    print("%f  with： %r" % (mean, param))