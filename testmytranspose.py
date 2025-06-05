# 패키지 업로드
import torch
import numpy as np
import pandas as pd
from mytranspose import mytranspose

    # 테스트 데이터 준비
D = np.array([1, 2, 3, 4])
E = np.array(['red', 'white', 'red', np.nan])  # 문자열과 NaN 포함
F = np.array([True, True, True, False])
np_array = np.array([[1, 2], [3, 4]])

test_dict = {
    0: np.array([[1, 2, 3],
                 [4, 5, 6]]),
    1: np.array([[1, 2],
                 [3, 4],
                 [5, 6],
                 [7, 8],
                 [9, 10]]),             # 5×2
    2: np.empty((0, 0)),                # 빈 행렬
    3: np.array([[1, 2]]),              # 1×2
    4: np.array([[1], [2]]),            # 2×1
    5: np.array([1, 2, np.nan, 3]),     # 1-D 벡터, NaN 포함
    6: np.array([np.nan]),              # 1-D 벡터 (NaN 하나)
    7: np.array([]),                    # 빈 벡터
    8: pd.DataFrame({"d": D, "e": E, "f": F}),
    "tensor": torch.tensor(np_array)
}

# 출력 함수
def show_before_after(label, obj):
    print(f"{'='*8}  {label}  {'='*8}")
    print("Before:")
    print(obj, "\n")
    print("After (mytranspose):")
    print(mytranspose(obj), "\n")

# 반복 실행
numeric_keys   = sorted(k for k in test_dict if isinstance(k, int))
non_numeric_keys = [k for k in test_dict if not isinstance(k, int)]

for key in numeric_keys + non_numeric_keys:
    show_before_after(key, test_dict[key])

print("hello world!")

print("my name is v2")
