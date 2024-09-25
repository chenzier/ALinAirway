import pickle

"""
加载/读取 pkl文件
"""
def save_obj(obj, name ):
    if name[-3:] != 'pkl':
        temp=name+'.pkl'
    else:
        temp=name
    with open(temp , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    if name[-3:] != 'pkl':
        temp=name+'.pkl'
    else:
        temp=name
    # print(temp)
    with open(temp, 'rb') as f:
        return pickle.load(f)

