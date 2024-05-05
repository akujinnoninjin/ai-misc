# numbers = [5, 2, 3, 5, 6]
# numbers2 = numbers

# numbers2.sort()

# print(numbers)

# my_data = {'name':'Frank', 'age':'26'}
# # for item in my_data:
# #     print(item)

# for item in my_data.keys():
#     item += '2525254'
#     print(item)
from icecream import ic
from functools import reduce 

# def nested_set(dic, keys, value):
#     for key in keys[:-1]:
#         dic = dic.setdefault(key, {})
#     dic[keys[-1]] = value
    
# data = { 'name': [{'foo':'bar', 'abc':'def'}, {'baz':'qux'}]}
# maplist = ('name', 1, 'foo')

# ic(data)

# ic(reduce(lambda a,b: a.__getitem__(b), maplist, data))

# value = None
# # ic(nested_set(data, maplist, value ))
# ic(value)

# from pathlib import Path
# file_path = Path('.')
# file_patterns = ['model.*', 'model-*-of-*.*', '*.gguf']
# file_extensions = ['.txt',]

# test = [f for f in file_path.iterdir() if f.suffix.lower() in file_extensions]
# ic(test)
# test = [f for f in test if any(f.match(fp) for fp in file_patterns)]
# ic(test)

dict = {1: "test", 5: "test", 234: "test", 2: "test", 44: "test"}
ic(max(dict))