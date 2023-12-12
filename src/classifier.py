import pandas as pd
import numpy as np
dict_trans = {
    0:'E',
    1:'D',
    2:'C',
    3:'B',
    4:'A'
}

def add_pop_class(data):
    target = 'popularity'
    data.sort_values(target,inplace=True)
    result = np.array_split(data,5)


    for i,data in enumerate(result):
        data['popularit_class'] = dict_trans[i]
        if i > 0:
            result[0] = pd.concat([result[0],result[i]])

    return result[0]

    
        
    