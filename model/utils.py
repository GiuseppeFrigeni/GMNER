
import numpy as np
from prettytable import PrettyTable
import os
import torch
from typing import List, Dict, Union
import inspect
from collections import Counter, namedtuple

_CheckRes = namedtuple('_CheckRes', ['missing', 'unused', 'duplicated', 'required', 'all_needed',
                                     'varargs'])


def _check_arg_dict_list(func, args):
    if isinstance(args, dict):
        arg_dict_list = [args]
    else:
        arg_dict_list = args
    assert callable(func) and isinstance(arg_dict_list, (list, tuple))
    assert len(arg_dict_list) > 0 and isinstance(arg_dict_list[0], dict)
    spect = inspect.getfullargspec(func)
    all_args = set([arg for arg in spect.args if arg != 'self'])
    defaults = []
    if spect.defaults is not None:
        defaults = [arg for arg in spect.defaults]
    start_idx = len(spect.args) - len(defaults)
    default_args = set(spect.args[start_idx:])
    require_args = all_args - default_args
    input_arg_count = Counter()
    for arg_dict in arg_dict_list:
        input_arg_count.update(arg_dict.keys())
    duplicated = [name for name, val in input_arg_count.items() if val > 1]
    input_args = set(input_arg_count.keys())
    missing = list(require_args - input_args)
    unused = list(input_args - all_args)
    varargs = [] if not spect.varargs else [spect.varargs]
    return _CheckRes(missing=missing,
                     unused=unused,
                     duplicated=duplicated,
                     required=list(require_args),
                     all_needed=list(all_args),
                     varargs=varargs)

class _CheckError(Exception):
    r"""

    _CheckError. Used in losses.LossBase, metrics.MetricBase.
    """

    def __init__(self, check_res: _CheckRes, func_signature: str):
        errs = [f'Problems occurred when calling `{func_signature}`']

        if check_res.varargs:
            errs.append(f"\tvarargs: {check_res.varargs}(Does not support pass positional arguments, please delete it)")
        if check_res.missing:
            errs.append(f"\tmissing param: {check_res.missing}")
        if check_res.duplicated:
            errs.append(f"\tduplicated param: {check_res.duplicated}")
        if check_res.unused:
            errs.append(f"\tunused param: {check_res.unused}")

        Exception.__init__(self, '\n'.join(errs))

        self.check_res = check_res
        self.func_signature = func_signature

def _build_args(func, **kwargs):
    r"""
    根据func的初始化参数，从kwargs中选择func需要的参数

    :param func: callable
    :param kwargs: 参数
    :return:dict. func中用到的参数
    """
    spect = inspect.getfullargspec(func)
    if spect.varkw is not None:
        return kwargs
    needed_args = set(spect.args)
    defaults = []
    if spect.defaults is not None:
        defaults = [arg for arg in spect.defaults]
    start_idx = len(spect.args) - len(defaults)
    output = {name: default for name, default in zip(spect.args[start_idx:], defaults)}
    output.update({name: val for name, val in kwargs.items() if name in needed_args})
    return output


def iob2(tags: List[str]) -> List[str]:
    r"""
    检查数据是否是合法的IOB数据，如果是IOB1会被自动转换为IOB2。两种格式的区别见
    https://datascience.stackexchange.com/questions/37824/difference-between-iob-and-iob2-format

    :param tags: 需要转换的tags
    """
    for i, tag in enumerate(tags):
        if tag == "O":
            continue
        split = tag.split("-")
        if len(split) != 2 or split[0] not in ["I", "B"]:
            raise TypeError("The encoding schema is not a valid IOB type.")
        if split[0] == "B":
            continue
        elif i == 0 or tags[i - 1] == "O":  # conversion IOB1 to IOB2
            tags[i] = "B" + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = "B" + tag[1:]
    return tags

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
        
    
def seq_len_to_mask(seq_len, max_len=None):
    r"""

    将一个表示sequence length的一维数组转换为二维的mask，不包含的位置为0。
    转变 1-d seq_len到2-d mask.

    .. code-block::
    
        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.size())
        torch.Size([14, 15])
        >>> seq_len = np.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.shape)
        (14, 15)
        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len, max_len=100)
        >>>print(mask.size())
        torch.Size([14, 100])

    :param np.ndarray,torch.LongTensor seq_len: shape将是(B,)
    :param int max_len: 将长度pad到这个长度。默认(None)使用的是seq_len中最长的长度。但在nn.DataParallel的场景下可能不同卡的seq_len会有
        区别，所以需要传入一个max_len使得mask的长度是pad到该长度。
    :return: np.ndarray, torch.Tensor 。shape将是(B, max_length)， 元素类似为bool或torch.uint8
    """
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        max_len = int(max_len) if max_len else int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask

r"""
.. todo::
    doc
"""

__all__ = [
    "cached_path",
    "get_filepath",
    "get_cache_path",
    "split_filename_suffix",
    "get_from_cache",
]

import os
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests import HTTPError
from tqdm import tqdm

from ._logger import logger

PRETRAINED_BERT_MODEL_DIR = {
    'en': 'bert-base-cased.zip',
    'en-large-cased-wwm': 'bert-large-cased-wwm.zip',
    'en-large-uncased-wwm': 'bert-large-uncased-wwm.zip',

    'en-large-uncased': 'bert-large-uncased.zip',
    'en-large-cased': 'bert-large-cased.zip',

    'en-base-uncased': 'bert-base-uncased.zip',
    'en-base-cased': 'bert-base-cased.zip',

    'en-base-cased-mrpc': 'bert-base-cased-finetuned-mrpc.zip',

    'en-distilbert-base-uncased': 'distilbert-base-uncased.zip',

    'multi-base-cased': 'bert-base-multilingual-cased.zip',
    'multi-base-uncased': 'bert-base-multilingual-uncased.zip',

    'cn': 'bert-chinese-wwm.zip',
    'cn-base': 'bert-base-chinese.zip',
    'cn-wwm': 'bert-chinese-wwm.zip',
    'cn-wwm-ext': "bert-chinese-wwm-ext.zip"
}

PRETRAINED_GPT2_MODEL_DIR = {
    'en': 'gpt2.zip',
    'en-medium': 'gpt2-medium.zip',
    'en-large': 'gpt2-large.zip',
    'en-xl': 'gpt2-xl.zip'
}

PRETRAINED_ROBERTA_MODEL_DIR = {
    'en': 'roberta-base.zip',
    'en-large': 'roberta-large.zip'
}

PRETRAINED_ELMO_MODEL_DIR = {
    'en': 'elmo_en_Medium.zip',
    'en-small': "elmo_en_Small.zip",
    'en-original-5.5b': 'elmo_en_Original_5.5B.zip',
    'en-original': 'elmo_en_Original.zip',
    'en-medium': 'elmo_en_Medium.zip'
}

PRETRAIN_STATIC_FILES = {
    'en': 'glove.840B.300d.zip',

    'en-glove-6b-50d': 'glove.6B.50d.zip',
    'en-glove-6b-100d': 'glove.6B.100d.zip',
    'en-glove-6b-200d': 'glove.6B.200d.zip',
    'en-glove-6b-300d': 'glove.6B.300d.zip',
    'en-glove-42b-300d': 'glove.42B.300d.zip',
    'en-glove-840b-300d': 'glove.840B.300d.zip',
    'en-glove-twitter-27b-25d': 'glove.twitter.27B.25d.zip',
    'en-glove-twitter-27b-50d': 'glove.twitter.27B.50d.zip',
    'en-glove-twitter-27b-100d': 'glove.twitter.27B.100d.zip',
    'en-glove-twitter-27b-200d': 'glove.twitter.27B.200d.zip',

    'en-word2vec-300d': "GoogleNews-vectors-negative300.txt.gz",

    'en-fasttext-wiki': "wiki-news-300d-1M.vec.zip",
    'en-fasttext-crawl': "crawl-300d-2M.vec.zip",

    'cn': "tencent_cn.zip",
    'cn-tencent': "tencent_cn.zip",
    'cn-fasttext': "cc.zh.300.vec.gz",
    'cn-sgns-literature-word': 'sgns.literature.word.txt.zip',
    'cn-char-fastnlp-100d': "cn_char_fastnlp_100d.zip",
    'cn-bi-fastnlp-100d': "cn_bi_fastnlp_100d.zip",
    "cn-tri-fastnlp-100d": "cn_tri_fastnlp_100d.zip"
}

DATASET_DIR = {
    # Classification, English
    'aclImdb': "imdb.zip",
    "yelp-review-full": "yelp_review_full.tar.gz",
    "yelp-review-polarity": "yelp_review_polarity.tar.gz",
    "sst-2": "SST-2.zip",
    "sst": "SST.zip",

    # Classification, Chinese
    "chn-senti-corp": "chn_senti_corp.zip",
    "weibo-senti-100k": "WeiboSenti100k.zip",
    "thuc-news": "THUCNews.zip",

    # Matching, English
    "mnli": "MNLI.zip",
    "snli": "SNLI.zip",
    "qnli": "QNLI.zip",
    "rte": "RTE.zip",

    # Matching, Chinese
    "cn-xnli": "XNLI.zip",

    # Sequence Labeling, Chinese
    "msra-ner": "MSRA_NER.zip",
    "peopledaily": "peopledaily.zip",
    "weibo-ner": "weibo_NER.zip",

    # Chinese Word Segmentation
    "cws-pku": 'cws_pku.zip',
    "cws-cityu": "cws_cityu.zip",
    "cws-as": 'cws_as.zip',
    "cws-msra": 'cws_msra.zip',

    # Summarization, English
    "ext-cnndm": "ext-cnndm.zip",

    # Question & answer, Chinese
    "cmrc2018": "cmrc2018.zip"

}

PRETRAIN_MAP = {'elmo': PRETRAINED_ELMO_MODEL_DIR,
                "bert": PRETRAINED_BERT_MODEL_DIR,
                "static": PRETRAIN_STATIC_FILES,
                'gpt2': PRETRAINED_GPT2_MODEL_DIR,
                'roberta': PRETRAINED_ROBERTA_MODEL_DIR}

#  用于扩展fastNLP的下载
FASTNLP_EXTEND_DATASET_URL = 'fastnlp_dataset_url.txt'
FASTNLP_EXTEND_EMBEDDING_URL = {'elmo': 'fastnlp_elmo_url.txt',
                                'bert':'fastnlp_bert_url.txt',
                                'static': 'fastnlp_static_url.txt',
                                'gpt2': 'fastnlp_gpt2_url.txt',
                                'roberta': 'fastnlp_roberta_url.txt'
                                }


def cached_path(url_or_filename: str, cache_dir: str = None, name=None) -> Path:
    r"""
    给定一个url，尝试通过url中的解析出来的文件名字filename到{cache_dir}/{name}/{filename}下寻找这个文件，
    
    1. 如果cache_dir=None, 则cache_dir=~/.fastNLP/; 否则cache_dir=cache_dir
    2. 如果name=None, 则没有中间的{name}这一层结构；否者中间结构就为{name}

    如果有该文件，就直接返回路径
    
    如果没有该文件，则尝试用传入的url下载

    或者文件名(可以是具体的文件名，也可以是文件夹)，先在cache_dir下寻找该文件是否存在，如果不存在则去下载, 并
    将文件放入到cache_dir中.

    :param str url_or_filename: 文件的下载url或者文件名称。
    :param str cache_dir: 文件的缓存文件夹。如果为None，将使用"~/.fastNLP"这个默认路径
    :param str name: 中间一层的名称。如embedding, dataset
    :return:
    """
    if cache_dir is None:
        data_cache = Path(get_cache_path())
    else:
        data_cache = cache_dir

    if name:
        data_cache = os.path.join(data_cache, name)

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ("http", "https"):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, Path(data_cache))
    elif parsed.scheme == "" and Path(os.path.join(data_cache, url_or_filename)).exists():
        # File, and it exists.
        return Path(os.path.join(data_cache, url_or_filename))
    elif parsed.scheme == "":
        # File, but it doesn't exist.
        raise FileNotFoundError("file {} not found in {}.".format(url_or_filename, data_cache))
    else:
        # Something unknown
        raise ValueError(
            "unable to parse {} as a URL or as a local path".format(url_or_filename)
        )


def get_filepath(filepath):
    r"""
    如果filepath为文件夹，
    
        如果内含多个文件, 返回filepath
        
        如果只有一个文件, 返回filepath + filename

    如果filepath为文件
        
        返回filepath

    :param str filepath: 路径
    :return:
    """
    if os.path.isdir(filepath):
        files = os.listdir(filepath)
        if len(files) == 1:
            return os.path.join(filepath, files[0])
        else:
            return filepath
    elif os.path.isfile(filepath):
        return filepath
    else:
        raise FileNotFoundError(f"{filepath} is not a valid file or directory.")


def get_cache_path():
    r"""
    获取fastNLP默认cache的存放路径, 如果将FASTNLP_CACHE_PATH设置在了环境变量中，将使用环境变量的值，使得不用每个用户都去下载。

    :return str:  存放路径
    """
    if 'FASTNLP_CACHE_DIR' in os.environ:
        fastnlp_cache_dir = os.environ.get('FASTNLP_CACHE_DIR')
        if os.path.isdir(fastnlp_cache_dir):
            return fastnlp_cache_dir
        else:
            raise NotADirectoryError(f"{os.environ['FASTNLP_CACHE_DIR']} is not a directory.")
    fastnlp_cache_dir = os.path.expanduser(os.path.join("~", ".fastNLP"))
    return fastnlp_cache_dir


def _get_base_url(name):
    r"""
    根据name返回下载的url地址。

    :param str name: 支持dataset和embedding两种
    :return:
    """
    # 返回的URL结尾必须是/
    environ_name = "FASTNLP_{}_URL".format(name.upper())

    if environ_name in os.environ:
        url = os.environ[environ_name]
        if url.endswith('/'):
            return url
        else:
            return url + '/'
    else:
        URLS = {
            'embedding': "http://212.129.155.247/embedding/",
            "dataset": "http://212.129.155.247/dataset/"
        }
        if name.lower() not in URLS:
            raise KeyError(f"{name} is not recognized.")
        return URLS[name.lower()]


def _get_embedding_url(embed_type, name):
    r"""
    给定embedding类似和名称，返回下载url

    :param str embed_type: 支持static, bert, elmo。即embedding的类型
    :param str name: embedding的名称, 例如en, cn, based等
    :return: str, 下载的url地址
    """
    #  从扩展中寻找下载的url
    _filename = FASTNLP_EXTEND_EMBEDDING_URL.get(embed_type, None)
    if _filename:
        url = _read_extend_url_file(_filename, name)
        if url:
            return url
    embed_map = PRETRAIN_MAP.get(embed_type, None)
    if embed_map:
        filename = embed_map.get(name, None)
        if filename:
            url = _get_base_url('embedding') + filename
            return url
        raise KeyError("There is no {}. Only supports {}.".format(name, list(embed_map.keys())))
    else:
        raise KeyError(f"There is no {embed_type}. Only supports bert, elmo, static, gpt2, roberta")

def _read_extend_url_file(filename, name)->str:
    r"""
    filename中的内容使用制表符隔开，第一列是名称，第二列是下载的url地址

    :param str filename: 在默认的路径下寻找file这个文件
    :param str name: 需要寻找的资源的名称
    :return: str,None
    """
    cache_dir = get_cache_path()
    filepath = os.path.join(cache_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        if name == parts[0]:
                            return parts[1]
    return None

def _get_dataset_url(name):
    r"""
    给定dataset的名称，返回下载url

    :param str name: 给定dataset的名称，比如imdb, sst-2等
    :return: str
    """
    #  从扩展中寻找下载的url
    url = _read_extend_url_file(FASTNLP_EXTEND_DATASET_URL, name)
    if url:
        return url

    filename = DATASET_DIR.get(name, None)
    if filename:
        url = _get_base_url('dataset') + filename
        return url
    else:
        raise KeyError(f"There is no {name}.")


def split_filename_suffix(filepath):
    r"""
    给定filepath 返回对应的name和suffix. 如果后缀是多个点，仅支持.tar.gz类型
    
    :param filepath: 文件路径
    :return: filename, suffix
    """
    filename = os.path.basename(filepath)
    if filename.endswith('.tar.gz'):
        return filename[:-7], '.tar.gz'
    return os.path.splitext(filename)


def get_from_cache(url: str, cache_dir: Path = None) -> Path:
    r"""
    尝试在cache_dir中寻找url定义的资源; 如果没有找到; 则从url下载并将结果放在cache_dir下，缓存的名称由url的结果推断而来。会将下载的
    文件解压，将解压后的文件全部放在cache_dir文件夹中。

    如果从url中下载的资源解压后有多个文件，则返回目录的路径; 如果只有一个资源文件，则返回具体的路径。
    
    :param url: 资源的 url
    :param cache_dir: cache 目录
    :return: 路径
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = re.sub(r".+/", "", url)
    dir_name, suffix = split_filename_suffix(filename)

    # 寻找与它名字匹配的内容, 而不关心后缀
    match_dir_name = match_file(dir_name, cache_dir)
    if match_dir_name:
        dir_name = match_dir_name
    cache_path = cache_dir / dir_name

    # get cache path to put the file
    if cache_path.exists():
        return get_filepath(cache_path)

    # make HEAD request to check ETag TODO ETag可以用来判断资源是否已经更新了，之后需要加上
    # response = requests.head(url, headers={"User-Agent": "fastNLP"})
    # if response.status_code != 200:
    #     raise IOError(
    #         f"HEAD request failed for url {url} with status code {response.status_code}."
    #     )

    # add ETag to filename if it exists
    # etag = response.headers.get("ETag")

    if not cache_path.exists():
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        # GET file object
        req = requests.get(url, stream=True, headers={"User-Agent": "fastNLP"})
        if req.status_code == 200:
            success = False
            fd, temp_filename = tempfile.mkstemp()
            uncompress_temp_dir = None
            try:
                content_length = req.headers.get("Content-Length")
                total = int(content_length) if content_length is not None else None
                progress = tqdm(unit="B", total=total, unit_scale=1)
                logger.info("%s not found in cache, downloading to %s" % (url, temp_filename))

                with open(temp_filename, "wb") as temp_file:
                    for chunk in req.iter_content(chunk_size=1024 * 16):
                        if chunk:  # filter out keep-alive new chunks
                            progress.update(len(chunk))
                            temp_file.write(chunk)
                progress.close()
                logger.info(f"Finish download from {url}")

                # 开始解压
                if suffix in ('.zip', '.tar.gz', '.gz'):
                    uncompress_temp_dir = tempfile.mkdtemp()
                    logger.debug(f"Start to uncompress file to {uncompress_temp_dir}")
                    if suffix == '.zip':
                        unzip_file(Path(temp_filename), Path(uncompress_temp_dir))
                    elif suffix == '.gz':
                        ungzip_file(temp_filename, uncompress_temp_dir, dir_name)
                    else:
                        untar_gz_file(Path(temp_filename), Path(uncompress_temp_dir))
                    filenames = os.listdir(uncompress_temp_dir)
                    if len(filenames) == 1:
                        if os.path.isdir(os.path.join(uncompress_temp_dir, filenames[0])):
                            uncompress_temp_dir = os.path.join(uncompress_temp_dir, filenames[0])

                    cache_path.mkdir(parents=True, exist_ok=True)
                    logger.debug("Finish un-compressing file.")
                else:
                    uncompress_temp_dir = temp_filename
                    cache_path = str(cache_path) + suffix

                # 复制到指定的位置
                logger.info(f"Copy file to {cache_path}")
                if os.path.isdir(uncompress_temp_dir):
                    for filename in os.listdir(uncompress_temp_dir):
                        if os.path.isdir(os.path.join(uncompress_temp_dir, filename)):
                            shutil.copytree(os.path.join(uncompress_temp_dir, filename), cache_path / filename)
                        else:
                            shutil.copyfile(os.path.join(uncompress_temp_dir, filename), cache_path / filename)
                else:
                    shutil.copyfile(uncompress_temp_dir, cache_path)
                success = True
            except Exception as e:
                logger.error(e)
                raise e
            finally:
                if not success:
                    if cache_path.exists():
                        if cache_path.is_file():
                            os.remove(cache_path)
                        else:
                            shutil.rmtree(cache_path)
                os.close(fd)
                os.remove(temp_filename)
                if uncompress_temp_dir is None:
                    pass
                elif os.path.isdir(uncompress_temp_dir):
                    shutil.rmtree(uncompress_temp_dir)
                elif os.path.isfile(uncompress_temp_dir):
                    os.remove(uncompress_temp_dir)
            return get_filepath(cache_path)
        else:
            raise HTTPError(f"Status code:{req.status_code}. Fail to download from {url}.")


def unzip_file(file: Path, to: Path):
    # unpack and write out in CoNLL column-like format
    from zipfile import ZipFile

    with ZipFile(file, "r") as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall(to)


def untar_gz_file(file: Path, to: Path):
    import tarfile

    with tarfile.open(file, 'r:gz') as tar:
        tar.extractall(to)


def ungzip_file(file: str, to: str, filename:str):
    import gzip

    g_file = gzip.GzipFile(file)
    with open(os.path.join(to, filename), 'wb+') as f:
        f.write(g_file.read())
    g_file.close()


def match_file(dir_name: str, cache_dir: Path) -> str:
    r"""
    匹配的原则是: 在cache_dir下的文件与dir_name完全一致, 或除了后缀以外和dir_name完全一致。
    如果找到了两个匹配的结果将报错. 如果找到了则返回匹配的文件的名称; 没有找到返回空字符串

    :param dir_name: 需要匹配的名称
    :param cache_dir: 在该目录下找匹配dir_name是否存在
    :return str: 做为匹配结果的字符串
    """
    files = os.listdir(cache_dir)
    matched_filenames = []
    for file_name in files:
        if re.match(dir_name + '$', file_name) or re.match(dir_name + '\\..*', file_name):
            matched_filenames.append(file_name)
    if len(matched_filenames) == 0:
        return ''
    elif len(matched_filenames) == 1:
        return matched_filenames[-1]
    else:
        raise RuntimeError(f"Duplicate matched files:{matched_filenames}, this should be caused by a bug.")


def _get_bert_dir(model_dir_or_name: str = 'en-base-uncased'):
    if model_dir_or_name.lower() in PRETRAINED_BERT_MODEL_DIR:
        model_url = _get_embedding_url('bert', model_dir_or_name.lower())
        model_dir = cached_path(model_url, name='embedding')
        # 检查是否存在
    elif os.path.isdir(os.path.abspath(os.path.expanduser(model_dir_or_name))):
        model_dir = os.path.abspath(os.path.expanduser(model_dir_or_name))
    else:
        logger.error(f"Cannot recognize BERT dir or name ``{model_dir_or_name}``.")
        raise ValueError(f"Cannot recognize BERT dir or name ``{model_dir_or_name}``.")
    return str(model_dir)


def _get_gpt2_dir(model_dir_or_name: str = 'en'):
    if model_dir_or_name.lower() in PRETRAINED_GPT2_MODEL_DIR:
        model_url = _get_embedding_url('gpt2', model_dir_or_name.lower())
        model_dir = cached_path(model_url, name='embedding')
        # 检查是否存在
    elif os.path.isdir(os.path.abspath(os.path.expanduser(model_dir_or_name))):
        model_dir = os.path.abspath(os.path.expanduser(model_dir_or_name))
    else:
        logger.error(f"Cannot recognize GPT2 dir or name ``{model_dir_or_name}``.")
        raise ValueError(f"Cannot recognize GPT2 dir or name ``{model_dir_or_name}``.")
    return str(model_dir)


def _get_roberta_dir(model_dir_or_name: str = 'en'):
    if model_dir_or_name.lower() in PRETRAINED_ROBERTA_MODEL_DIR:
        model_url = _get_embedding_url('roberta', model_dir_or_name.lower())
        model_dir = cached_path(model_url, name='embedding')
        # 检查是否存在
    elif os.path.isdir(os.path.abspath(os.path.expanduser(model_dir_or_name))):
        model_dir = os.path.abspath(os.path.expanduser(model_dir_or_name))
    else:
        logger.error(f"Cannot recognize RoBERTa dir or name ``{model_dir_or_name}``.")
        raise ValueError(f"Cannot recognize RoBERTa dir or name ``{model_dir_or_name}``.")
    return str(model_dir)


def _get_file_name_base_on_postfix(dir_path, postfix):
    r"""
    在dir_path中寻找后缀为postfix的文件.
    :param dir_path: str, 文件夹
    :param postfix: 形如".bin", ".json"等
    :return: str，文件的路径
    """
    files = list(filter(lambda filename: filename.endswith(postfix), os.listdir(os.path.join(dir_path))))
    if len(files) == 0:
        raise FileNotFoundError(f"There is no file endswith {postfix} file in {dir_path}")
    elif len(files) > 1:
        raise FileExistsError(f"There are multiple *{postfix} files in {dir_path}")
    return os.path.join(dir_path, files[0])

def check_loader_paths(paths: Union[str, Dict[str, str]]) -> Dict[str, str]:
    r"""
    检查传入dataloader的文件的合法性。如果为合法路径，将返回至少包含'train'这个key的dict。类似于下面的结果::

        {
            'train': '/some/path/to/', # 一定包含，建词表应该在这上面建立，剩下的其它文件应该只需要处理并index。
            'test': 'xxx' # 可能有，也可能没有
            ...
        }

    如果paths为不合法的，将直接进行raise相应的错误. 如果paths内不包含train也会报错。

    :param str paths: 路径. 可以为一个文件路径(则认为该文件就是train的文件); 可以为一个文件目录，将在该目录下寻找包含train(文件名
        中包含train这个字段), test, dev这三个字段的文件或文件夹; 可以为一个dict, 则key是用户自定义的某个文件的名称，value是这个文件的路径。
    :return:
    """
    if isinstance(paths, (str, Path)):
        paths = os.path.abspath(os.path.expanduser(paths))
        if os.path.isfile(paths):
            return {'train': paths}
        elif os.path.isdir(paths):
            filenames = os.listdir(paths)
            filenames.sort()
            files = {}
            for filename in filenames:
                path_pair = None
                if 'train' in filename:
                    path_pair = ('train', filename)
                if 'dev' in filename:
                    if path_pair:
                        raise Exception(
                            "Directory:{} in {} contains both `{}` and `dev`.".format(filename, paths, path_pair[0]))
                    path_pair = ('dev', filename)
                if 'test' in filename:
                    if path_pair:
                        raise Exception(
                            "Directory:{} in {} contains both `{}` and `test`.".format(filename, paths, path_pair[0]))
                    path_pair = ('test', filename)
                if path_pair:
                    if path_pair[0] in files:
                        raise FileExistsError(f"Two files contain `{path_pair[0]}` were found, please specify the "
                                              f"filepath for `{path_pair[0]}`.")
                    files[path_pair[0]] = os.path.join(paths, path_pair[1])
            if 'train' not in files:
                raise KeyError(f"There is no train file in {paths}.")
            return files
        else:
            raise FileNotFoundError(f"{paths} is not a valid file path.")
    
    elif isinstance(paths, dict):
        if paths:
            if 'train' not in paths:
                raise KeyError("You have to include `train` in your dict.")
            for key, value in paths.items():
                if isinstance(key, str) and isinstance(value, str):
                    value = os.path.abspath(os.path.expanduser(value))
                    if not os.path.exists(value):
                        raise TypeError(f"{value} is not a valid path.")
                    paths[key] = value
                else:
                    raise TypeError("All keys and values in paths should be str.")
            return paths
        else:
            raise ValueError("Empty paths is not allowed.")
    else:
        raise TypeError(f"paths only supports str and dict. not {type(paths)}.")
    

class Option(dict):
    r"""a dict can treat keys as attributes"""

    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, key, value):
        if key.startswith('__') and key.endswith('__'):
            raise AttributeError(key)
        self.__setitem__(key, value)

    def __delattr__(self, item):
        try:
            self.pop(item)
        except KeyError:
            raise AttributeError(item)

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)

def _get_func_signature(func):
    r"""

    Given a function or method, return its signature.
    For example:
    
    1 function::
    
        def func(a, b='a', *args):
            xxxx
        get_func_signature(func) # 'func(a, b='a', *args)'
        
    2 method::
    
        class Demo:
            def __init__(self):
                xxx
            def forward(self, a, b='a', **args)
        demo = Demo()
        get_func_signature(demo.forward) # 'Demo.forward(self, a, b='a', **args)'
        
    :param func: a function or a method
    :return: str or None
    """
    if inspect.ismethod(func):
        class_name = func.__self__.__class__.__name__
        signature = inspect.signature(func)
        signature_str = str(signature)
        if len(signature_str) > 2:
            _self = '(self, '
        else:
            _self = '(self'
        signature_str = class_name + '.' + func.__name__ + _self + signature_str[1:]
        return signature_str
    elif inspect.isfunction(func):
        signature = inspect.signature(func)
        signature_str = str(signature)
        signature_str = func.__name__ + signature_str
        return signature_str