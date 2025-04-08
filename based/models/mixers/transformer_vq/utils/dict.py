import torch
from typing import List, Dict


def recursive_apply_dict(d, func):
    if isinstance(d, dict):
        return {k: recursive_apply_dict(v, func) for k, v in d.items()}
    
    elif isinstance(d, torch.Tensor):
        return func(d)
    
    else:
        return d
    
def average_nested_dicts(dict_list: List[Dict]):
    """
    递归地对嵌套字典的值进行平均。
    """
    if not dict_list:
        raise ValueError("dict_list is empty.")
    if not isinstance(dict_list, list):
        raise ValueError("dict_list must be a list.")
    
    keys = dict_list[0].keys()

    result = {}
    for key in keys:
        # 如果值是字典，则递归处理
        if isinstance(dict_list[0][key], dict):
            result[key] = average_nested_dicts([d[key] for d in dict_list])
        else:
            # 否则直接计算平均值
            result[key] = sum(d[key] for d in dict_list) / len(dict_list)
    
    return result