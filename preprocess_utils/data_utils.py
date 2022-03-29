import pandas as pd

def load_data(file_name, sheet_name = 'sheet_name', sep = "\t", encoding = 'cp949'):
    """
    Load data into pandas dataframe.
    Data type: csv, xlsx, txt
    Parameter:
      - sheet: sheet name in xlsx file.
      - sep: columns separation index in txt file.
      - encoding: encoding type of file.
    """
    if file_name.endswith('.csv'):
        data = pd.read_csv(file_name, encoding = encoding)
    elif file_name.endswith('.xlsx'):
        if sheet_name == 'sheet_name':
            data = pd.read_excel(file_name)
        else:
            data = pd.read_excel(file_name, sheet_name = sheet_name)    
    elif file_name.endswith('.txt'):
        data = pd.read_csv(file_name, sep = "\t", engine='python', encoding = encoding)
    else:
        raise Exception("File extension deviates from [csv, xlsx, txt].")
    
    return data
    
def one_hot_encoder(data, numeric_variable, categorical_variable, target_variable):
    data[categorical_variable] = data[categorical_variable].astype('category')
    numeric_data = data.loc[:, numeric_variable]
    dummies_data = pd.get_dummies(data[categorical_variable])
    one_hot_data = pd.concat([numeric_data,dummies_data], axis = 1)
    one_hot_data[target_variable] = data[target_variable]
    return one_hot_data

class NA_Cheak(object):
    """
    Data NA Check and handling Module
    handling method: 
        fill_mean: Fill NA as mean of column.
        fill_value: Fill NA  as specific value.
        fill_forward: Fill NA as forward value.
        fill_backward: Fill NA as backward value.
        eliminate: Eliminate row which has a NA.

    """
    def __init__(self, data):
        self.data = data

    def NA_count(self):
        print(self.data.isna().sum())
    
    def handling_NA(self, columns, method = 'eliminate', fill_value = 0):
        if method == 'fill_mean':
            self.data.loc[:, columns] = self.data.loc[:, columns].fillna(self.data.loc[:, columns].mean())
        elif method == 'fill_value':
            self.data.loc[:, columns] = self.data.loc[:, columns].fillna(fill_value)
        elif method == 'fill_forward':
            self.data.loc[:, columns] = self.data.loc[:, columns].fillna(method='ffill')
        elif method == 'fill_backward':
            self.data.loc[:, columns] = self.data.loc[:, columns].fillna(method='bfill')
        elif method == 'eliminate':
            self.data = self.data[columns].dropna()
        else:
            raise Exception("method deviates from [fill_mean, fill_value, fill_forward, fill_backward, eliminate].")
