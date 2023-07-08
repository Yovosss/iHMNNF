import  codecs
import csv
import numpy as np
from collections import Counter
import pandas as pd

data1 = {"address":['有','天津','北京','北京','北京',None,None,None],
"preference_level": ['哈哈','3.5','51','51','10','1','1', '1']
}

df = pd.DataFrame(data1)

# print(df.info())
print(df)
print("-------------------")
df.set_index(['address'], drop=False, inplace=True)
# print(df.loc[1:2,'address'])

print(df)
print("-------------------")

print(df.iloc[0:1,:])
