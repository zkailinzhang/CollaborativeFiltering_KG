# -*- coding: utf-8 -*-
import math
import random
import csv

from py2neo import Graph

graph =  Graph("http://172.25.2.114:7474/browser/",username='',password='') 



rst = graph.run('MATCH (a)-[r:watch]->(b) RETURN a,b LIMIT 25') 


def splitData():
    train,test = {},{}

    seed = 10
    random.seed(seed)

    for l in rst: 
        user = l['a'].get('user_id') 
        item = l['b'].get('program') 

        if random.randint(0,8)==1:
            train.setdefault(user,{}) 
            train[user][item]=1.0
        else:
            test.setdefault(user,{}) 
            test[user][item]=1.0
    
    return train,test

def precisionAndRecall(train,test,W,K,N):

    hit=0;pre=0;rec=0
    for user in train.keys():
        tu = test.get(user,{})
        rank=recommender(user,train,W,K,N)
        
        for item,pui in rank.items():
            if item in tu:
                hit+=1
        pre+=N
        rec+=len(tu)
    return hit/(pre*1.0),hit/(rec*1.0)


def Coverage(train,test,W,K=3,N=10):
    train = train
    recommend_items = set() 
    all_items = set()  
    for user,items in train.items():
        for i in items.keys():
            all_items.add(i)

        rank = recommender(user,train,W,K,N)

        for i,_ in rank.items():
            recommend_items.add(i)

    return len(recommend_items) / (len(all_items) * 1.0)



def Popularity(train,test,W,K=3,N=10):
    train = train 

    item_popularity = dict() 
    for user,items in train.items():
        for i in items.keys():
            item_popularity.setdefault(i,0)
            item_popularity[i] += 1

    ret,n = 0,0        
    for user in train.keys():
        rank = recommender(user,train,W,K=K,N=N)   

        for item,_ in rank.items():
            ret += math.log(1 + item_popularity[item])
            n += 1
    ret /= n * 1.0
    return ret


def itemSimilarity(train,method='IUF'):

    C=dict() 
    N=dict()
    for u,items in train.items():
        for i  in items:
            N[i] = N.get(i,0)+1
            for j in items:
                if i==j:
                    continue
                C.setdefault(i,{})          
                if method=='IUF':               
                    C[i][j]=C[i].get(j,0)+1/math.log(1+len(items)*1.0)
                else:
                    C[i][j]=C[i].get(j,0)+1
    
    W=dict()
    for i,related_items in C.items():
        for j,cij in related_items.items():
            W.setdefault(i,{})          
            W[i][j]=cij/math.sqrt(N[i]*N[j])   
    return W


def recommender(user,train,W,K,N):
    rank=dict()     
    interacted_items = train[user]
    for i,pi in interacted_items.items():
        for j,wj in sorted(W[i].items(),key=lambda c:c[1], reverse=True)[0:K]: 
            if j in interacted_items:
                continue  
            rank[j]=rank.get(j,0)+pi*wj
    return dict(sorted(rank.items(),key = lambda c :c[1],reverse = True)[0:N])



def testItemCF(filename):
    
    train,test=splitData
    W=itemSimilarity(train,method='IUF')

    result = open('result_ibcf.data','w')
    print(u'不同K值下推荐算法的各项指标(精度、召回率、覆盖率、流行度)\n')
    print('K\t\tprecision\trecall\t\tCoverage\tPopularity')
    for k in [5,10,20,40,80,160]:
        pre,rec=precisionAndRecall(train,test,W,k,10)
        cov = Coverage(train,test,W,k,10)
        pop = Popularity(train,test,W,k,10)
        print("%3d\t\t%.2f%%\t\t%.2f%%\t\t%.2f%%\t\t%.6f" % (k,pre * 100,rec * 100,cov * 100,pop))
        result.write(str(k)+' '+str('%2.2f' % (pre * 100))+' '+str('%2.2f' % (rec * 100))+' '+str('%2.2f' % (cov * 100))+' '+str('%2.6f' % pop)+'\n')



def testRecommend():
    """
    测试推荐
    """
    train,test=splitData()
    W= itemSimilarity(train,method='IUF')
    user = '019953051993183'  
    rank=recommender(user,train,W,3,10)  #test
    print(rank)

    
if __name__=='__main__':

    testRecommend()

    testItemCF()


