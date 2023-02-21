from sklearn import *
import pandas as pd
import numpy as np
import xlsxwriter
data=datasets.load_wine()
target=(data['target'])
#print(target)

values= data["data"]
#print(values)

frame=pd.DataFrame(values)
#print(frame)

#print (f"Wartość maksymalna to {frame.max()}, wartość minimalna to {frame.min()}")

#frame1=pd.DataFrame(values[0:2])## wybór wierszy
#frame2=pd.DataFrame(values[:2])## wybór wierszy

b={
    "v1":frame[0],
    "v2":frame[1],
    "v3":frame[2],
    "v4":frame[3],
    "v5":frame[4],
    "v6":frame[5],
    "v7":frame[6],
    "v8":frame[7],
    "v9":frame[8],
    "v10":frame[9],
    "v11":frame[10],
    "v12":frame[10],
}
df_b=pd.DataFrame(b)
#print(df_b)

c={"target":target}
df_c=pd.DataFrame(c)
#print(df_c)

bnb=np.array(df_b)
#print(bnb)
bnc=np.array(df_c)
#print(bnc)

bnbf=np.append(bnb,bnc,axis=1)
#print(bnbf)

bnbf=pd.DataFrame(bnbf)
#print(bnbf[0:3])

zbior_win= pd.DataFrame({
    "col1":bnbf[0],
    "col2":bnbf[1],
    "col3":bnbf[2],
    "col4":bnbf[3],
    "col5":bnbf[4],
    "col6":bnbf[5],
    "col7":bnbf[6],
    "col8":bnbf[7],
    "col9":bnbf[8],
    "col10":bnbf[9],
    "col11":bnbf[10],
    "col12":bnbf[11],
    "etykieta":bnbf[12]

})
#print(zbior_win[0:3])

#zbior_win.to_excel("zbior_danych_wino.xlsx")

w=pd.read_excel("zbior_danych_wino.xlsx")
print(w)
