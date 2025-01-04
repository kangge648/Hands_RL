import numpy as np

K = 10
probs = np.random.uniform(size=K)
estimates = np.array([1] * K)

# 假设有一个列表
my_list = ['a', 'b', 'c']
 
# 使用 enumerate() 遍历列表，同时获取索引和值
for index, value in enumerate(my_list):
    print(index, value)