r"""undocumented"""

__all__ = [
    "Loader"
]

from typing import Union, Dict

from .data_bundle import DataBundle
from .utils import _get_dataset_url, get_cache_path, cached_path
from .utils import check_loader_paths
from .dataset import DataSet
from .instance import Instance
from ._logger import logger

def _read_conll(path, encoding='utf-8',sep=None, indexes=None, dropna=True):
    r"""
    Construct a generator to read conll items.

    :param path: file path
    :param encoding: file's encoding, default: utf-8
    :param sep: seperator
    :param indexes: conll object's column indexes that needed, if None, all columns are needed. default: None
    :param dropna: weather to ignore and drop invalid data,
            :if False, raise ValueError when reading invalid data. default: True
    :return: generator, every time yield (line number, conll item)
    """

    def parse_conll(sample):
        sample = list(map(list, zip(*sample)))
        sample = [sample[i] for i in indexes]
        for f in sample:
            if len(f) <= 0:
                raise ValueError('empty field')
        return sample

    with open(path, 'r', encoding=encoding) as f:
        sample = []
        start = next(f).strip()
        if start != '':
            sample.append(start.split(sep)) if sep else sample.append(start.split())
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if line == '':
                if len(sample):
                    try:
                        res = parse_conll(sample)
                        sample = []
                        yield line_idx, res
                    except Exception as e:
                        if dropna:
                            logger.warning('Invalid instance which ends at line: {} has been dropped.'.format(line_idx))
                            sample = []
                            continue
                        raise ValueError('Invalid instance which ends at line: {}'.format(line_idx))
            elif line.startswith('#'):
                continue
            else:
                sample.append(line.split(sep)) if sep else sample.append(line.split())
        if len(sample) > 0:
            try:
                res = parse_conll(sample)
                yield line_idx, res
            except Exception as e:
                if dropna:
                    return
                logger.error('invalid instance ends at line: {}'.format(line_idx))
                raise e


class Loader:
    r"""
    各种数据 Loader 的基类，提供了 API 的参考.
    Loader支持以下的三个函数

    - download() 函数：自动将该数据集下载到缓存地址，默认缓存地址为~/.fastNLP/datasets/。由于版权等原因，不是所有的Loader都实现了该方法。该方法会返回下载后文件所处的缓存地址。
    - _load() 函数：从一个数据文件中读取数据，返回一个 :class:`~fastNLP.DataSet` 。返回的DataSet的内容可以通过每个Loader的文档判断出。
    - load() 函数：将文件分别读取为DataSet，然后将多个DataSet放入到一个DataBundle中并返回
    
    """
    
    def __init__(self):
        pass
    
    def _load(self, path: str) -> DataSet:
        r"""
        给定一个路径，返回读取的DataSet。

        :param str path: 路径
        :return: DataSet
        """
        raise NotImplementedError
    
    def load(self, paths: Union[str, Dict[str, str]] = None) -> DataBundle:
        r"""
        从指定一个或多个路径中的文件中读取数据，返回 :class:`~fastNLP.io.DataBundle` 。

        :param Union[str, Dict[str, str]] paths: 支持以下的几种输入方式：

            0.如果为None，则先查看本地是否有缓存，如果没有则自动下载并缓存。

            1.传入一个目录, 该目录下名称包含train的被认为是train，包含test的被认为是test，包含dev的被认为是dev，如果检测到多个文件名包含'train'、 'dev'、 'test'则会报错::

                data_bundle = xxxLoader().load('/path/to/dir')  # 返回的DataBundle中datasets根据目录下是否检测到train
                #  dev、 test等有所变化，可以通过以下的方式取出DataSet
                tr_data = data_bundle.get_dataset('train')
                te_data = data_bundle.get_dataset('test')  # 如果目录下有文件包含test这个字段

            2.传入一个dict，比如train，dev，test不在同一个目录下，或者名称中不包含train, dev, test::

                paths = {'train':"/path/to/tr.conll", 'dev':"/to/validate.conll", "test":"/to/te.conll"}
                data_bundle = xxxLoader().load(paths)  # 返回的DataBundle中的dataset中包含"train", "dev", "test"
                dev_data = data_bundle.get_dataset('dev')

            3.传入文件路径::

                data_bundle = xxxLoader().load("/path/to/a/train.conll") # 返回DataBundle对象, datasets中仅包含'train'
                tr_data = data_bundle.get_dataset('train')  # 取出DataSet

        :return: 返回的 :class:`~fastNLP.io.DataBundle`
        """
        if paths is None:
            paths = self.download()
        paths = check_loader_paths(paths)
        datasets = {name: self._load(path) for name, path in paths.items()}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle
    
    def download(self) -> str:
        r"""
        自动下载该数据集

        :return: 下载后解压目录
        """
        raise NotImplementedError(f"{self.__class__} cannot download data automatically.")
    
    @staticmethod
    def _get_dataset_path(dataset_name):
        r"""
        传入dataset的名称，获取读取数据的目录。如果数据不存在，会尝试自动下载并缓存（如果支持的话）

        :param str dataset_name: 数据集的名称
        :return: str, 数据集的目录地址。直接到该目录下读取相应的数据即可。
        """
        
        default_cache_path = get_cache_path()
        url = _get_dataset_url(dataset_name)
        output_dir = cached_path(url_or_filename=url, cache_dir=default_cache_path, name='dataset')
        
        return output_dir

class ConllLoader(Loader):
    r"""
    ConllLoader支持读取的数据格式: 以空行隔开两个sample，除了分割行，每一行用空格或者制表符隔开不同的元素。如下例所示:

    Example::

        # 文件中的内容
        Nadim NNP B-NP B-PER
        Ladki NNP I-NP I-PER

        AL-AIN NNP B-NP B-LOC
        United NNP B-NP B-LOC
        Arab NNP I-NP I-LOC
        Emirates NNPS I-NP I-LOC
        1996-12-06 CD I-NP O
        ...

        # 如果用以下的参数读取，返回的DataSet将包含raw_words和pos两个field, 这两个field的值分别取自于第0列与第1列
        dataset = ConllLoader(headers=['raw_words', 'pos'], indexes=[0, 1])._load('/path/to/train.conll')
        # 如果用以下的参数读取，返回的DataSet将包含raw_words和ner两个field, 这两个field的值分别取自于第0列与第2列
        dataset = ConllLoader(headers=['raw_words', 'ner'], indexes=[0, 3])._load('/path/to/train.conll')
        # 如果用以下的参数读取，返回的DataSet将包含raw_words, pos和ner三个field
        dataset = ConllLoader(headers=['raw_words', 'pos', 'ner'], indexes=[0, 1, 3])._load('/path/to/train.conll')

    ConllLoader返回的DataSet的field由传入的headers确定。

    数据中以"-DOCSTART-"开头的行将被忽略，因为该符号在conll 2003中被用为文档分割符。

    """
    
    def __init__(self, headers, sep=None, indexes=None, dropna=True):
        r"""
        
        :param list headers: 每一列数据的名称，需为List or Tuple  of str。``header`` 与 ``indexes`` 一一对应
        :param list sep: 指定分隔符，默认为制表符
        :param list indexes: 需要保留的数据列下标，从0开始。若为 ``None`` ，则所有列都保留。Default: ``None``
        :param bool dropna: 是否忽略非法数据，若 ``False`` ，遇到非法数据时抛出 ``ValueError`` 。Default: ``True``
        """
        super(ConllLoader, self).__init__()
        if not isinstance(headers, (list, tuple)):
            raise TypeError(
                'invalid headers: {}, should be list of strings'.format(headers))
        self.headers = headers
        self.dropna = dropna
        self.sep=sep
        if indexes is None:
            self.indexes = list(range(len(self.headers)))
        else:
            if len(indexes) != len(headers):
                raise ValueError
            self.indexes = indexes
    
    def _load(self, path):
        r"""
        传入的一个文件路径，将该文件读入DataSet中，field由ConllLoader初始化时指定的headers决定。

        :param str path: 文件的路径
        :return: DataSet
        """
        ds = DataSet()
        for idx, data in _read_conll(path,sep=self.sep, indexes=self.indexes, dropna=self.dropna):
            ins = {h: data[i] for i, h in enumerate(self.headers)}
            ds.append(Instance(**ins))
        return ds


class Conll2003Loader(ConllLoader):
    r"""
    用于读取conll2003任务的数据。数据的内容应该类似与以下的内容, 第一列为raw_words, 第二列为pos, 第三列为chunking，第四列为ner。

    Example::

        Nadim NNP B-NP B-PER
        Ladki NNP I-NP I-PER

        AL-AIN NNP B-NP B-LOC
        United NNP B-NP B-LOC
        Arab NNP I-NP I-LOC
        Emirates NNPS I-NP I-LOC
        1996-12-06 CD I-NP O
        ...

    返回的DataSet的内容为

    .. csv-table:: 下面是Conll2003Loader加载后数据具备的结构。
       :header: "raw_words", "pos", "chunk", "ner"

       "[Nadim, Ladki]", "[NNP, NNP]", "[B-NP, I-NP]", "[B-PER, I-PER]"
       "[AL-AIN, United, Arab, ...]", "[NNP, NNP, NNP, ...]", "[B-NP, B-NP, I-NP, ...]", "[B-LOC, B-LOC, I-LOC, ...]"
       "[...]", "[...]", "[...]", "[...]"

    """
    
    def __init__(self):
        headers = [
            'raw_words', 'pos', 'chunk', 'ner',
        ]
        super(Conll2003Loader, self).__init__(headers=headers)
    
    def _load(self, path):
        r"""
        传入的一个文件路径，将该文件读入DataSet中，field由ConllLoader初始化时指定的headers决定。

        :param str path: 文件的路径
        :return: DataSet
        """
        ds = DataSet()
        for idx, data in _read_conll(path, indexes=self.indexes, dropna=self.dropna):
            doc_start = False
            for i, h in enumerate(self.headers):
                field = data[i]
                if str(field[0]).startswith('-DOCSTART-'):
                    doc_start = True
                    break
            if doc_start:
                continue
            ins = {h: data[i] for i, h in enumerate(self.headers)}
            ds.append(Instance(**ins))
        return ds
    
    def download(self, output_dir=None):
        raise RuntimeError("conll2003 cannot be downloaded automatically.")