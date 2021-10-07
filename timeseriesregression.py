import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression



            
X = pd.read_csv("新竹_2019.csv", encoding = "ISO-8859-1", engine='python')#,na_values=" ?"
#X.set_index("日期" , inplace=True)
X= X.drop([0])

X.head()
#X.iloc[1:,3:].isin(['#'])
#X['0'].str.contains(pat='#|*|x|A', case=True, flags=0, na=nan, regex=True)
for i in range(24):
    booling = X[str(i)].str.contains('A|#|x|\*')
    X[str(i)][booling]=None

nrmask = X.iloc[0:,3:].isin(['NR'])#
X.iloc[0:,3:][nrmask] = 0 

X.isnull()

   
for i in range(0,X.shape[0]):#len(X)
    for j in range(3,X.shape[1]):#->這樣印
        if X.iloc[i][j] == None:#
            if j>3 and j<=X.shape[1]-2:#
                tmp = j-1
                tmp2 = j+1
                while X.iloc[i][tmp]== None:
                    if tmp > 3:
                        tmp=tmp-1
                    else:
                        tmp = -2
                        break
                while X.iloc[i][tmp2]== None:    
                    if tmp2 <X.shape[1]-1:
                        tmp2 = tmp2+1
                    else:
                        tmp2 = tmp
                        break
                if tmp == -2:
                    if tmp2!=-2:
                        tmp = tmp2
                        X.iloc[i][j] = (float(X.iloc[i][tmp]) + float(X.iloc[i][tmp2]))/2    
                    else:
                        X.iloc[i][j]=0
                else:
                    X.iloc[i][j] = (float(X.iloc[i][tmp]) + float(X.iloc[i][tmp2]))/2                
                    #if 'NO' in X.iloc[i][tmp]:
                        #print(X.iloc[i][tmp])
            elif j >= X.shape[1]-1:
                tmp = j-1
                while X.iloc[i][tmp]==None:
                    tmp = tmp-1
                tmp2 = 3
                tmp3 = i +1
                while X.iloc[tmp3][tmp2]==None:
                    if tmp2>=3 and tmp2<=X.shape[1]-2:
                        tmp2 = tmp2+1
                    else:
                        tmp3 = tmp3+1
                        tmp2 = 3
                    if tmp3==X.shape[0]-1 and tmp2==X.shape[1]-1:
                        tmp3=-2
                        break
                    
                #if tmp3<X.shape[0]-2 and X.iloc[tmp3][tmp2]==None:
                #    tmp3=tmp3+1
                if tmp3 != -2:
                    X.iloc[i][j] = (float(X.iloc[i][tmp]) + float(X.iloc[tmp3][tmp2]))/2  
                else:
                    X.iloc[i][j] = X.iloc[i][tmp] 
            else: #elif j<0:
                tmp = j+1
                while X.iloc[i][tmp]==None:
                    if tmp<X.shape[1]-2:
                        tmp = tmp+1
                    else:
                        tmp = -2
                        break
                tmp2 = i-1
                tmp3 = X.shape[1]-1
                while X.iloc[tmp2][tmp3]==None:
                    if tmp3>4 and tmp3<=X.shape[1]-1:
                        tmp3 = tmp3-1
                    else:
                        tmp2 = tmp2-1
                        tmp3 = X.shape[1]-1 
                        break
                    if tmp2 == X.shape[0]-1 and tmp3 == X.shape[1]-1:
                        tmp3 = -2
                        break
                if tmp != -2 and tmp3!=-2:
                    X.iloc[i][j] = (float(X.iloc[i][tmp]) + float(X.iloc[tmp2][tmp3]))/2
                elif  tmp != -2:                                                                             
                    X.iloc[i][j] = X.iloc[i][tmp]
                else:
                    X.iloc[i][j] = 0
               
X.isnull().sum() > 0
# 根据上一步结果，筛选需要填充的列
X.columns[X.isnull().sum() > 0]

train = X.iloc[4824:5922,:]#十、十一月
test = X.iloc[5922:,:]#十二月

#train= train.drop([2],axis=1)
#train= train.drop([1],axis=1)
train= train.iloc[:,3:]
test= test.iloc[:,3:]
#train = train.values

#print(train[0])
#b=np.reshape(train,(18,61*24))
#D = train.set_index('日期      ')
s1 = train.iloc[0:18,0:1]   
#result = s1.values
#b=np.reshape(s1.values,(1,18))
traindata=pd.DataFrame(np.reshape(s1.values,(18,1)))

for i in range (61):#0~60
    for j in range (24):
        s = train.iloc[18*i:18*(i+1),j:j+1]
        s1 = pd.DataFrame(np.reshape(s.values,(18,1)))
        if i!=0 or j!=0:
            traindata = pd.concat([traindata,s1],axis=1)
            
#traindata.columns = ['0', '1', '2', '3', '4','5','6','7','8','8','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
            
s2 = test.iloc[0:18,0:1]   
#result = s1.values
#b=np.reshape(s1.values,(1,18))
testdata=pd.DataFrame(np.reshape(s2.values,(18,1)))
            
for i in range (31):#0~60
    for j in range (24):
        s = test.iloc[18*i:18*(i+1),j:j+1]
        s2 = pd.DataFrame(np.reshape(s.values,(18,1)))
        if i!=0 or j!=0:
            testdata = pd.concat([testdata,s2],axis=1)

        #pd.concat([df1,df2,df3],axis=0)
#data.drop([0],axis=1)
#print(train[0])
# Instantiate model with 1000 decision trees

rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
# Train the model on training data
trainx = traindata.iloc[9:10,0:6]
trainy = traindata.iloc[9:10,6:7]
for i in range (1,traindata.shape[1]-6):
    tx = traindata.iloc[9:10,0+i:6+i]
    ty = traindata.iloc[9:10,6+i:7+i]
    trainx = pd.concat([trainx,tx],axis=0)
    trainy = pd.concat([trainy,ty],axis=0)

testx = testdata.iloc[9:10,0:6]
testy = testdata.iloc[9:10,6:7]
for i in range (1,testdata.shape[1]-6):
    tx = testdata.iloc[9:10,0+i:6+i]
    ty = testdata.iloc[9:10,6+i:7+i]
    testx = pd.concat([testx,tx],axis=0)
    testy = pd.concat([testy,ty],axis=0)

    
rf.fit(trainx, trainy)

y_pred = rf.predict(testx)
print("Random Forest 以PM2.5資料預測後一小時PM2.5 MAE:",mean_absolute_error(testy, y_pred))

lgmodel = LinearRegression(fit_intercept=True)
lgmodel.fit(trainx.values, trainy.values)
y_pred = lgmodel.predict(testx.values)
print("Linear Regression 以PM2.5資料預測後一小時PM2.5 MAE:",mean_absolute_error(testy, y_pred))


trainx = traindata.iloc[9:10,0:6]
trainy = traindata.iloc[9:10,11:12]
for i in range (1,traindata.shape[1]-11):
    tx = traindata.iloc[9:10,0+i:6+i]
    ty = traindata.iloc[9:10,11+i:12+i]
    trainx = pd.concat([trainx,tx],axis=0)
    trainy = pd.concat([trainy,ty],axis=0)

testx = testdata.iloc[9:10,0:6]
testy = testdata.iloc[9:10,11:12]
for i in range (1,testdata.shape[1]-11):
    tx = testdata.iloc[9:10,0+i:6+i]
    ty = testdata.iloc[9:10,11+i:12+i]
    testx = pd.concat([testx,tx],axis=0)
    testy = pd.concat([testy,ty],axis=0)

rf1 = RandomForestRegressor(n_estimators = 100, random_state = 42)
# Train the model on training data    
rf1.fit(trainx, trainy)

y_pred = rf1.predict(testx)
print("Random Forest以PM2.5資料預測後六小時PM2.5 MAE:",mean_absolute_error(testy, y_pred))

lgmodel1 = LinearRegression(fit_intercept=True)
lgmodel1.fit(trainx.values, trainy.values)
y_pred = lgmodel.predict(testx.values)
print("Linear Regression 以PM2.5資料預測後六小時PM2.5 MAE:",mean_absolute_error(testy, y_pred))








rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
# Train the model on training data
trainx = traindata.iloc[:,0:6]
trainx = pd.DataFrame(np.reshape(trainx.values,(1,108)))
trainy = traindata.iloc[9:10,6:7]
for i in range (1,traindata.shape[1]-6):
    txd = traindata.iloc[:,0+i:6+i]
    tx = pd.DataFrame(np.reshape(txd.values,(1,108)))
    ty = traindata.iloc[9:10,6+i:7+i]
    trainx = pd.concat([trainx,tx],axis=0)
    trainy = pd.concat([trainy,ty],axis=0)

testx = testdata.iloc[:,0:6]
testx = pd.DataFrame(np.reshape(testx.values,(1,108)))
testy = testdata.iloc[9:10,6:7]
for i in range (1,testdata.shape[1]-6):
    txd = testdata.iloc[:,0+i:6+i]
    tx = pd.DataFrame(np.reshape(txd.values,(1,108)))
    ty = testdata.iloc[9:10,6+i:7+i]
    testx = pd.concat([testx,tx],axis=0)
    testy = pd.concat([testy,ty],axis=0)

    
rf.fit(trainx, trainy)

y_pred = rf.predict(testx)
print("Random Forest 以所有空氣資料預測後一小時PM2.5 MAE:",mean_absolute_error(testy, y_pred))

lgmodel = LinearRegression(fit_intercept=True)
lgmodel.fit(trainx.values, trainy.values)
y_pred = lgmodel.predict(testx.values)
print("Linear Regression 以所有空氣資料預測後一小時PM2.5 MAE:",mean_absolute_error(testy, y_pred))





rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
# Train the model on training data
trainx = traindata.iloc[:,0:6]
trainx = pd.DataFrame(np.reshape(trainx.values,(1,108)))
trainy = traindata.iloc[9:10,11:12]
for i in range (1,traindata.shape[1]-11):
    txd = traindata.iloc[:,0+i:6+i]
    tx = pd.DataFrame(np.reshape(txd.values,(1,108)))
    ty = traindata.iloc[9:10,11+i:12+i]
    trainx = pd.concat([trainx,tx],axis=0)
    trainy = pd.concat([trainy,ty],axis=0)

testx = testdata.iloc[:,0:6]
testx = pd.DataFrame(np.reshape(testx.values,(1,108)))
testy = testdata.iloc[9:10,11:12]
for i in range (1,testdata.shape[1]-11):
    txd = testdata.iloc[:,0+i:6+i]
    tx = pd.DataFrame(np.reshape(txd.values,(1,108)))
    ty = testdata.iloc[9:10,11+i:12+i]
    testx = pd.concat([testx,tx],axis=0)
    testy = pd.concat([testy,ty],axis=0)

    
rf.fit(trainx, trainy)

y_pred = rf.predict(testx)
print("Random Forest 以所有空氣資料預測後六小時PM2.5 MAE:",mean_absolute_error(testy, y_pred))

lgmodel = LinearRegression(fit_intercept=True)
lgmodel.fit(trainx.values, trainy.values)
y_pred = lgmodel.predict(testx.values)
print("Linear Regression 以所有空氣資料預測後六小時PM2.5 MAE:",mean_absolute_error(testy, y_pred))

