# 패키지 업로드
import torch
import numpy as np
import pandas as pd

# 함수 생성
def mytranspose(x):
    if isinstance(x, np.ndarray):
        return x.T
    elif isinstance(x, pd.DataFrame):
        return x.transpose()
    elif isinstance(x, torch.Tensor):
        return x.clone() if x.ndim == 1 else x.t().clone()
    else:
        raise TypeError("지원하지 않는 타입입니다.")