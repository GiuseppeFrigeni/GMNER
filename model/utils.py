
import numpy as np
from prettytable import PrettyTable
import os
import torch

def get_max_len_max_len_a(data_bundle, max_len=10):
    """
    当给定max_len=10的时候计算一个最佳的max_len_a

    :param data_bundle:
    :param max_len:
    :return:
    """
    max_len_a = -1
    for name, ds in data_bundle.iter_datasets():
        if name=='train':continue
        src_seq_len = np.array(ds.get_field('src_seq_len').content)
        tgt_seq_len = np.array(ds.get_field('tgt_seq_len').content)
        _len_a = round(max(np.maximum(tgt_seq_len - max_len+2, 0)/src_seq_len), 1)

        if _len_a>max_len_a:
            max_len_a = _len_a

    return max_len, max_len_a

def _is_iterable(value):
    # 检查是否是iterable的, duck typing
    try:
        iter(value)
        return True
    except BaseException as e:
        return False
    
def pretty_table_printer(dataset_or_ins) -> PrettyTable:
    r"""
    用于在 **fastNLP** 中展示数据的函数::

        >>> ins = Instance(field_1=[1, 1, 1], field_2=[2, 2, 2], field_3=["a", "b", "c"])
        +-----------+-----------+-----------------+
        |  field_1  |  field_2  |     field_3     |
        +-----------+-----------+-----------------+
        | [1, 1, 1] | [2, 2, 2] | ['a', 'b', 'c'] |
        +-----------+-----------+-----------------+

    :param dataset_or_ins: 要展示的 :class:`~fastNLP.core.DataSet` 或者 :class:`~fastNLP.core.Instance` 实例；
    :return: 根据命令行大小进行自动截断的数据表格；
    """
    x = PrettyTable()
    try:
        sz = os.get_terminal_size()
        column = sz.columns
        row = sz.lines
    except OSError:
        column = 144
        row = 11

    if type(dataset_or_ins).__name__ == "DataSet":
        x.field_names = list(dataset_or_ins.field_arrays.keys())
        c_size = len(x.field_names)
        for ins in dataset_or_ins:
            x.add_row([sub_column(ins[k], column, c_size, k) for k in x.field_names])
            row -= 1
            if row < 0:
                x.add_row(["..." for _ in range(c_size)])
                break
    elif type(dataset_or_ins).__name__ == "Instance":
        x.field_names = list(dataset_or_ins.fields.keys())
        c_size = len(x.field_names)
        x.add_row([sub_column(dataset_or_ins[k], column, c_size, k) for k in x.field_names])

    else:
        raise Exception("only accept  DataSet and Instance")
    x.align = "l"

    return x

def sub_column(string: str, c: int, c_size: int, title: str) -> str:
    r"""
    对传入的字符串进行截断，方便在命令行中显示。

    :param string: 要被截断的字符串；
    :param c: 命令行列数；
    :param c_size: :class:`~fastNLP.core.Instance` 或 :class:`~fastNLP.core.DataSet` 的 ``field`` 数目；
    :param title: 列名；
    :return: 对一个过长的列进行截断的结果；
    """
    avg = max(int(c / c_size / 2), len(title))
    string = str(string)
    res = ""
    counter = 0
    for char in string:
        if ord(char) > 255:
            counter += 2
        else:
            counter += 1
        res += char
        if counter > avg:
            res = res + "..."
            break
    return res

def _move_dict_value_to_device(*args, device: torch.device):
    """

    move data to model's device, element in *args should be dict. This is a inplace change.
    :param device: torch.device
    :param args:
    :return:
    """
    if not isinstance(device, torch.device):
        raise TypeError(f"device must be `torch.device`, got `{type(device)}`")

    for arg in args:
        if isinstance(arg, dict):
            for key, value in arg.items():
                if isinstance(value, torch.Tensor):
                    arg[key] = value.to(device)
        else:
            raise TypeError("Only support `dict` type right now.")