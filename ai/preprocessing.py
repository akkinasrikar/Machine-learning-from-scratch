import pandas as pd
class LabelEncoder:
    def fit_transform(self,data):
        duplicate_list=list(data)
        unique=list(set(data))
        l=len(unique)
        numbers=list(range(0,l))
        for i in range(len(data)):
            ind=unique.index(duplicate_list[i])
            duplicate_list[i]=numbers[ind]
        data=pd.DataFrame(duplicate_list)
        return data