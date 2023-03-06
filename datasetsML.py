import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.linear_model import SGDClassifier

biggerdataset=pd.read_excel(r"C:\Users\48667\Desktop\dane.xlsx")
danewsadowe=biggerdataset
frame=pd.DataFrame(danewsadowe)
#print(frame)
#print(frame.columns.tolist())

dane={
    "wartosć1" :frame['wart1'],
    "wartość2" :frame['wart2'],
    "wartość3" :frame['wart3'],
    "wartość4" :frame['wart4'],
    "etykieta" :frame['etykieta']
}


final_frame=pd.DataFrame(dane)
#print(final_frame)

X=biggerdataset.iloc[0:,:4]
#print(X)
Y=biggerdataset.iloc[:,4:]
#print(Y)

Xtrain,Xtest,Ytrain,Ytest= train_test_split(X.values,Y.values.ravel(),train_size=0.95)

# model=KNeighborsClassifier(n_neighbors=3)
# model.fit(Xtrain,Ytrain)
# Ypredict=model.predict(Xtest)
# print(f"Metodą KNN osiągnięto dokładność : {accuracy_score(Ytest,Ypredict)} ")

print(Ytest)

model1=DecisionTreeClassifier()
model1.fit(Xtrain,Ytrain)
Ypredict1=model1.predict(Xtest)
print(f"Metodą klasyfikacji drzewa osiągnięto dokładność : {accuracy_score(Ytest,Ypredict1)}")
print(Ypredict1)




# model2= LogisticRegression(max_iter=1000)
# model2.fit(Xtrain,Ytrain)
# Ypredict2=model2.predict(Xtest)
# #print(Ypredict2)
# print(f"Metodą Regresji logistycznej osiagnieto dokładnosć : {accuracy_score(Ytest,Ypredict2)}" )



# model3=LinearRegression()
# model3.fit(Xtrain,Ytrain)
# Ypredict3=model3.predict(Xtest)

# #print(Ypredict3)

model4=RandomForestClassifier()
model4.fit(Xtrain,Ytrain)
Ypredict4=model4.predict(Xtest)
print(f"Metodą Clasyfikacji przypadkowego drzewa osiagnieto dokładnosć : {accuracy_score(Ytest,Ypredict4)}")

print(Ypredict4)


model5=BaggingClassifier()
model5.fit(Xtrain,Ytrain)
Ypredict5=model5.predict(Xtest)
print(f"Metodą Bagging  osiagnieto dokładnosć :{accuracy_score(Ytest,Ypredict5)} ")
print(Ypredict5)

# model6=SGDClassifier()
# model6.fit(Xtrain,Ytrain)
# Ypredict6=model6.predict(Xtest)
# print(f"Metodą SGD osiagnieto dokładnosć :{accuracy_score(Ytest,Ypredict6)} ")
# #print(Ypredict5)







#print("************")
#print(Ytest)



rows, cols = (1, 4)
tabela = [[0]*cols]*rows
w=int(len(tabela[0])-1)


def Repeat(x):
    _size = len(x)
    repeated = []
    for i in range(_size):
        k = i + 1
        for j in range(k, _size):
            if x[i] == x[j] and x[i] not in repeated:
                repeated.append(x[i])
    return len(repeated)




i=0
for i in range(4):
    if i <4:
        w=float(input(f"Podaj długość boku {i}: "))
        tabela[0][i]=w
        #print(tabela)
        i+=1
    if i==4: 
            print(model1.predict(tabela))
            print(model5.predict(tabela))
            print(model4.predict(tabela))
            figura1=model1.predict(tabela)
            figura5=model5.predict(tabela)
            figura4=model4.predict(tabela)
            if figura5==0 and figura1==0 or figura4==0:
                print("Podane długości boków odpowiadają figurze KOŁO")
            elif figura5 ==1 and figura1==1 or figura4==1:
                print("Podane długości boków odpowiadają figurze KWADRAT")
            elif figura5==2 and figura1==2 or figura4==2:
                print("Podane długości boków odpowiadają figurze TRÓJKĄT RÓWNOOBOCZNY")
            elif figura5==3 and figura1==3 or figura4==3:
                print("Podane długości boków odpowiadają figurze TRÓJKĄT RÓWNORAMIENNEGO")
            elif figura5==4 and figura1==4 or figura4==4:
                print("Podane długości boków odpowiadają figurze TRÓJKĄT PROSTOKĄTNY")
            elif figura5==5 and figura1==5 or figura4==5:
                 print("Podane długości boków odpowiadają figurze TRAPEZ RÓWNORAMIENNY")
            elif figura5==6 and figura1==6 or figura4==6:
                print("Podane długości boków odpowiadają figurze TRAPEZA PROSTOKĄTNEGO")
            elif figura5==7 and figura1==7 or figura4==7:
                print("Podane długości boków odpowiadają figurze PROSTOKĄT")
                 
