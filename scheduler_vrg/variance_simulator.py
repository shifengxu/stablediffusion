import math
import torch
from torch import Tensor
from utils import log_info

class VarianceSimulator:
    """
    Variance simulator. Use binary search
    """
    def __init__(self, x_arr, y_arr):
        log_info(f"VarianceSimulator::__init__()...")
        self.ts_cnt = len(y_arr)
        self.x_arr = x_arr
        self.y_arr = y_arr  # input x, output y.
        log_info(f"  ts_cnt    : {self.ts_cnt}")
        log_info(f"  x_arr     : {len(self.x_arr)}")
        log_info(f"  y_arr     : {len(self.y_arr)}")
        arr_to_str = lambda arr: '[' + ', '.join(["{:.8f}".format(f) for f in arr]) + ']'
        log_info(f"  x_arr[:5] : {arr_to_str(self.x_arr[:5])}")
        log_info(f"  x_arr[-5:]: {arr_to_str(self.x_arr[-5:])}")
        log_info(f"  y_arr[:5] : {arr_to_str(self.y_arr[:5])}")
        log_info(f"  y_arr[-5:]: {arr_to_str(self.y_arr[-5:])}")
        log_info(f"VarianceSimulator::__init__()...Done")

    def __call__(self, aacum: Tensor, include_index=False):
        """Use binary search"""
        # define left bound index and right bound index
        lbi = torch.zeros_like(aacum, dtype=torch.long)
        rbi = torch.ones_like(aacum, dtype=torch.long)
        rbi *= (self.ts_cnt - 1)
        iter_cnt = math.ceil(math.log(self.ts_cnt, 2))
        for _ in range(iter_cnt):
            mdi = torch.floor(torch.div(lbi + rbi,  2))  # middle index
            mdi = mdi.long()
            flag0 = aacum <= self.x_arr[mdi]
            flag1 = ~flag0
            lbi[flag0] = mdi[flag0]
            rbi[flag1] = mdi[flag1]
        # for
        # after iteration, lbi will be the target index
        res = self.y_arr[lbi]

        # But the input aacum value may have difference with x_arr. So here
        # we handle the difference and make the result smooth
        # Firstly, find the right-hand index: re-use variable "rbi"
        rbi = torch.ones_like(aacum, dtype=torch.long)
        rbi *= (self.ts_cnt - 1)
        rbi = torch.minimum(torch.add(lbi, 1), rbi)

        # make the result smooth
        flag = torch.lt(lbi, rbi)
        lb_arr = self.x_arr[lbi]
        rb_arr = self.x_arr[rbi]
        portion = (lb_arr[flag] - aacum[flag]) / (lb_arr[flag] - rb_arr[flag])
        res[flag] = res[flag] * (1 - portion) + self.y_arr[rbi][flag] * portion

        # a2s = lambda x: ', '.join([f"{i:3d}" for i in x[:5]])  # arr to str
        # log_info(f"lbi[:5]  : {a2s(lbi[:5])}")
        # log_info(f"lbi[-5:] : {a2s(lbi[-5:])}")
        if include_index:
            return res, lbi
        return res

    def to(self, device):
        self.x_arr = self.x_arr.to(device)
        self.y_arr = self.y_arr.to(device)

# class

def test():
    """ unit test"""
    a2s = lambda x: ', '.join([f"{i:.4f}" for i in x])  # arr to str
    var_arr = torch.tensor([5, 4, 3, 2, 1], dtype=torch.float64)
    vs = VarianceSimulator('cosine', var_arr)
    in_arr = torch.tensor([0.999844, 0.898566, 0.647377, 0.340756, 0.094031])
    out_arr = vs(in_arr)
    print('input : ', a2s(in_arr))
    print('output: ', a2s(out_arr))

    in_arr = torch.tensor([0.9999, 0.94, 0.71, 0.49, 0.19, 0.06, 0.01, 0.001])
    out_arr = vs(in_arr)
    print('input : ', a2s(in_arr))
    print('output: ', a2s(out_arr))

if __name__ == '__main__':
    test()
