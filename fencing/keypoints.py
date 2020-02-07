import json
import numpy as np

def load_key_point(path, ):
    with open(path, 'r') as fh:
        df_kp = json.load(fh)
    res = [np.array(el) for el in df_kp]
    for i, el in enumerate(res):
        if el.ndim != 3:
            res[i] = np.zeros([2, 25, 3])
    return [el for el in res]
