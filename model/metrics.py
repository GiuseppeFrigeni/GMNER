
#from fastNLP import Metric
#from fastNLP.core.metrics import _compute_f_pre_rec
import inspect
from abc import abstractmethod
from collections import defaultdict


from .utils import _CheckError
from .utils import _CheckRes
from .utils import _build_args
from .utils import _check_arg_dict_list
from .utils import _get_func_signature


class MetricBase(object):
    r"""
    所有metrics的基类,所有的传入到Trainer, Tester的Metric需要继承自该对象，需要覆盖写入evaluate(), get_metric()方法。
    
        evaluate(xxx)中传入的是一个batch的数据。
        
        get_metric(xxx)当所有数据处理完毕，调用该方法得到最终的metric值
        
    以分类问题中，Accuracy计算为例
    假设model的forward返回dict中包含 `pred` 这个key, 并且该key需要用于Accuracy::
    
        class Model(nn.Module):
            def __init__(xxx):
                # do something
            def forward(self, xxx):
                # do something
                return {'pred': pred, 'other_keys':xxx} # pred's shape: batch_size x num_classes
                
    假设dataset中 `label` 这个field是需要预测的值，并且该field被设置为了target
    对应的AccMetric可以按如下的定义, version1, 只使用这一次::
    
        class AccMetric(MetricBase):
            def __init__(self):
                super().__init__()
    
                # 根据你的情况自定义指标
                self.corr_num = 0
                self.total = 0
    
            def evaluate(self, label, pred): # 这里的名称需要和dataset中target field与model返回的key是一样的，不然找不到对应的value
                # dev或test时，每个batch结束会调用一次该方法，需要实现如何根据每个batch累加metric
                self.total += label.size(0)
                self.corr_num += label.eq(pred).sum().item()
    
            def get_metric(self, reset=True): # 在这里定义如何计算metric
                acc = self.corr_num/self.total
                if reset: # 是否清零以便重新计算
                    self.corr_num = 0
                    self.total = 0
                return {'acc': acc} # 需要返回一个dict，key为该metric的名称，该名称会显示到Trainer的progress bar中


    version2，如果需要复用Metric，比如下一次使用AccMetric时，dataset中目标field不叫label而叫y，或者model的输出不是pred::
    
        class AccMetric(MetricBase):
            def __init__(self, label=None, pred=None):
                # 假设在另一场景使用时，目标field叫y，model给出的key为pred_y。则只需要在初始化AccMetric时，
                #   acc_metric = AccMetric(label='y', pred='pred_y')即可。
                # 当初始化为acc_metric = AccMetric()，即label=None, pred=None, fastNLP会直接使用'label', 'pred'作为key去索取对
                #   应的的值
                super().__init__()
                self._init_param_map(label=label, pred=pred) # 该方法会注册label和pred. 仅需要注册evaluate()方法会用到的参数名即可
                # 如果没有注册该则效果与version1就是一样的
    
                # 根据你的情况自定义指标
                self.corr_num = 0
                self.total = 0
    
            def evaluate(self, label, pred): # 这里的参数名称需要和self._init_param_map()注册时一致。
                # dev或test时，每个batch结束会调用一次该方法，需要实现如何根据每个batch累加metric
                self.total += label.size(0)
                self.corr_num += label.eq(pred).sum().item()
    
            def get_metric(self, reset=True): # 在这里定义如何计算metric
                acc = self.corr_num/self.total
                if reset: # 是否清零以便重新计算
                    self.corr_num = 0
                    self.total = 0
                return {'acc': acc} # 需要返回一个dict，key为该metric的名称，该名称会显示到Trainer的progress bar中


    ``MetricBase`` 将会在输入的字典 ``pred_dict`` 和 ``target_dict`` 中进行检查.
    ``pred_dict`` 是模型当中 ``forward()`` 函数或者 ``predict()`` 函数的返回值.
    ``target_dict`` 是DataSet当中的ground truth, 判定ground truth的条件是field的 ``is_target`` 被设置为True.

    ``MetricBase`` 会进行以下的类型检测:

    1. self.evaluate当中是否有varargs, 这是不支持的.
    2. self.evaluate当中所需要的参数是否既不在 ``pred_dict`` 也不在 ``target_dict`` .
    3. self.evaluate当中所需要的参数是否既在 ``pred_dict`` 也在 ``target_dict`` .

    除此以外，在参数被传入self.evaluate以前，这个函数会检测 ``pred_dict`` 和 ``target_dict`` 当中没有被用到的参数
    如果kwargs是self.evaluate的参数，则不会检测


    self.evaluate将计算一个批次(batch)的评价指标，并累计。 没有返回值
    self.get_metric将统计当前的评价指标并返回评价结果, 返回值需要是一个dict, key是指标名称，value是指标的值

    """

    def __init__(self):
        self._param_map = {}  # key is param in function, value is input param.
        self._checked = False
        self._metric_name = self.__class__.__name__

    @property
    def param_map(self):
        if len(self._param_map) == 0:  # 如果为空说明还没有初始化
            func_spect = inspect.getfullargspec(self.evaluate)
            func_args = [arg for arg in func_spect.args if arg != 'self']
            for arg in func_args:
                self._param_map[arg] = arg
        return self._param_map

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_metric(self, reset=True):
        raise NotImplemented

    def set_metric_name(self, name: str):
        r"""
        设置metric的名称，默认是Metric的class name.

        :param str name:
        :return: self
        """
        self._metric_name = name
        return self

    def get_metric_name(self):
        r"""
        返回metric的名称
        
        :return:
        """
        return self._metric_name

    def _init_param_map(self, key_map=None, **kwargs):
        r"""检查key_map和其他参数map，并将这些映射关系添加到self._param_map

        :param dict key_map: 表示key的映射关系
        :param kwargs: key word args里面的每一个的键-值对都会被构造成映射关系
        :return: None
        """
        value_counter = defaultdict(set)
        if key_map is not None:
            if not isinstance(key_map, dict):
                raise TypeError("key_map must be `dict`, got {}.".format(type(key_map)))
            for key, value in key_map.items():
                if value is None:
                    self._param_map[key] = key
                    continue
                if not isinstance(key, str):
                    raise TypeError(f"key in key_map must be `str`, not `{type(key)}`.")
                if not isinstance(value, str):
                    raise TypeError(f"value in key_map must be `str`, not `{type(value)}`.")
                self._param_map[key] = value
                value_counter[value].add(key)
        for key, value in kwargs.items():
            if value is None:
                self._param_map[key] = key
                continue
            if not isinstance(value, str):
                raise TypeError(f"in {key}={value}, value must be `str`, not `{type(value)}`.")
            self._param_map[key] = value
            value_counter[value].add(key)
        for value, key_set in value_counter.items():
            if len(key_set) > 1:
                raise ValueError(f"Several parameters:{key_set} are provided with one output {value}.")

        # check consistence between signature and _param_map
        func_spect = inspect.getfullargspec(self.evaluate)
        func_args = [arg for arg in func_spect.args if arg != 'self']
        for func_param, input_param in self._param_map.items():
            if func_param not in func_args:
                raise NameError(
                    f"Parameter `{func_param}` is not in {_get_func_signature(self.evaluate)}. Please check the "
                    f"initialization parameters, or change its signature.")

    def __call__(self, pred_dict, target_dict):
        r"""
        这个方法会调用self.evaluate 方法.
        在调用之前，会进行以下检测:
            1. self.evaluate当中是否有varargs, 这是不支持的.
            2. self.evaluate当中所需要的参数是否既不在``pred_dict``也不在``target_dict``.
            3. self.evaluate当中所需要的参数是否既在``pred_dict``也在``target_dict``.

            除此以外，在参数被传入self.evaluate以前，这个函数会检测``pred_dict``和``target_dict``当中没有被用到的参数
            如果kwargs是self.evaluate的参数，则不会检测
        :param pred_dict: 模型的forward函数或者predict函数返回的dict
        :param target_dict: DataSet.batch_y里的键-值对所组成的dict(即is_target=True的fields的内容)
        :return:
        """

        if not self._checked:
            if not callable(self.evaluate):
                raise TypeError(f"{self.__class__.__name__}.evaluate has to be callable, not {type(self.evaluate)}.")
            # 1. check consistence between signature and _param_map
            func_spect = inspect.getfullargspec(self.evaluate)
            func_args = set([arg for arg in func_spect.args if arg != 'self'])
            for func_arg, input_arg in self._param_map.items():
                if func_arg not in func_args:
                    raise NameError(f"`{func_arg}` not in {_get_func_signature(self.evaluate)}.")

            # 2. only part of the _param_map are passed, left are not
            for arg in func_args:
                if arg not in self._param_map:
                    self._param_map[arg] = arg  # This param does not need mapping.
            self._evaluate_args = func_args
            self._reverse_param_map = {input_arg: func_arg for func_arg, input_arg in self._param_map.items()}

        # need to wrap inputs in dict.
        mapped_pred_dict = {}
        mapped_target_dict = {}
        for input_arg, mapped_arg in self._reverse_param_map.items():
            if input_arg in pred_dict:
                mapped_pred_dict[mapped_arg] = pred_dict[input_arg]
            if input_arg in target_dict:
                mapped_target_dict[mapped_arg] = target_dict[input_arg]

        # missing
        if not self._checked:
            duplicated = []
            for input_arg, mapped_arg in self._reverse_param_map.items():
                if input_arg in pred_dict and input_arg in target_dict:
                    duplicated.append(input_arg)
            check_res = _check_arg_dict_list(self.evaluate, [mapped_pred_dict, mapped_target_dict])
            # only check missing.
            # replace missing.
            missing = check_res.missing
            replaced_missing = list(missing)
            for idx, func_arg in enumerate(missing):
                # Don't delete `` in this information, nor add ``
                replaced_missing[idx] = f"{self._param_map[func_arg]}" + f"(assign to `{func_arg}` " \
                                                                         f"in `{self.__class__.__name__}`)"

            check_res = _CheckRes(missing=replaced_missing,
                                  unused=check_res.unused,
                                  duplicated=duplicated,
                                  required=check_res.required,
                                  all_needed=check_res.all_needed,
                                  varargs=check_res.varargs)

            if check_res.missing or check_res.duplicated:
                raise _CheckError(check_res=check_res,
                                  func_signature=_get_func_signature(self.evaluate))
            self._checked = True
        refined_args = _build_args(self.evaluate, **mapped_pred_dict, **mapped_target_dict)

        self.evaluate(**refined_args)

        return


def _compute_f_pre_rec(beta_square, tp, fn, fp):
    r"""

    :param tp: int, true positive
    :param fn: int, false negative
    :param fp: int, false positive
    :return: (f, pre, rec)
    """
    pre = tp / (fp + tp + 1e-13)
    rec = tp / (fn + tp + 1e-13)
    f = (1 + beta_square) * pre * rec / (beta_square * pre + rec + 1e-13)

    return f, pre, rec

class Seq2SeqSpanMetric(MetricBase):
    def __init__(self, eos_token_id, num_labels, region_num,target_type='bpe',print_mode = False):
        super(Seq2SeqSpanMetric, self).__init__()
        self.eos_token_id = eos_token_id
        self.num_labels = num_labels
        self.word_start_index = num_labels+2  # +2是由于有前面有两个特殊符号，sos和eos
        self.region_num = region_num

        self.fp = 0
        self.tp = 0
        self.fn = 0
        self.em = 0
        self.total = 0
        self.uc = 0
        self.nc = 0
        self.tc = 0
        self.sc = 0
        self.target_type = target_type  # 如果是span的话，必须是偶数的span，否则是非法的
        self.print_mode = print_mode

    def evaluate(self, target_span, pred, tgt_tokens, region_pred,region_label,cover_flag,predict_mode = False):
       
        region_pred = region_pred[:,1:,:].tolist()
        bbox_num = region_label.size(-1) -1  ## -1维度的最后一个item 0/1 表示 是否有region

        self.total += pred.size(0)
        pred_eos_index = pred.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()
        target_eos_index = tgt_tokens.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()

        pred = pred[:, 1:]  # 去掉</s>
        tgt_tokens = tgt_tokens[:, 1:]
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1) # bsz
        pred_seq_len = (pred_seq_len - 2).tolist()
        target_seq_len = target_eos_index.flip(dims=[1]).eq(target_eos_index[:, -1:]).sum(dim=1) # bsz
        target_seq_len = (target_seq_len-2).tolist()
        # pred_spans = []
        batch_pred_pairs =[]
        batch_target_pairs =[]
        for i, (ts, ps) in enumerate(zip(target_span, pred.tolist())):
            if not isinstance(ts,list):  ####!!! 有的过来是array 有的过来是list
                ts= ts.tolist()
            em = 0
            ps = ps[:pred_seq_len[i]]
            if pred_seq_len[i]==target_seq_len[i]:
                em = int(tgt_tokens[i, :target_seq_len[i]].eq(pred[i, :target_seq_len[i]]).sum().item()==target_seq_len[i])
            self.em += em
            all_pairs = {}
            cur_pair = []
            if len(ps):
                k = 0
                while k < len(ps)-2:
                    if ps[k]<self.word_start_index: # 是类别预测
                        if len(cur_pair) > 0:  # 之前有index 预测，且为升序，则添加pair
                            if all([cur_pair[i]<cur_pair[i+1] for i in range(len(cur_pair)-1)]):
                                
                                if ps[k] == 2:
                                    all_pairs[tuple(cur_pair)] = [region_pred[i][k],[ps[k+1]]]  ## 相关
                                elif ps[k] == 3:
                                    all_pairs[tuple(cur_pair)] = [[bbox_num],[ps[k+1]]]   ## 不相关
                                else:
                                    print("region relation error!")
                        cur_pair = []
                        k = k+2
                    else: # 记录当前 pair 的index 预测
                        cur_pair.append(ps[k])
                        k= k+1
                if len(cur_pair) > 0:
                    if all([cur_pair[i]<cur_pair[i+1] for i in range(len(cur_pair)-1)]):
                        
                        if ps[k] == 2:
                            all_pairs[tuple(cur_pair)] = [region_pred[i][k],[ps[k+1]]]  ## 相关
                        elif ps[k] == 3:
                            all_pairs[tuple(cur_pair)] = [[bbox_num],[ps[k+1]]]   ## 不相关
                        else:
                            print("region relation error!")

           
            all_ts = {}
           
            for e in range(len(ts)):  ## i -> sample ,e -> entity
               
                if cover_flag[i][e] == 0: ## not cover
                    true_region =[bbox_num+1]
                elif cover_flag[i][e] == 2: ## 不相关
                    if region_label[i][e][-1] == 1 : ## no region
                        true_region = [bbox_num]
                    else:
                        import pdb;pdb.set_trace()
                elif cover_flag[i][e] == 1:  ## 相关
                    if region_label[i][e][-1] == 0 :
                        true_region = region_label[i][e].nonzero().squeeze(1).tolist()
                    else:
                        import pdb;pdb.set_trace()
                
                text_span = ts[e][:-2]
                entity_type = ts[e][-1]
               
                all_ts[tuple(text_span)] = [true_region,[entity_type]]

            
         

            tp,fp,fn,uc, nc, tc, sc = _compute_tp_fn_fp(all_pairs, all_ts,self.region_num)
            if self.print_mode:
                print("all_pairs: "+str(all_pairs))
                print("all_ts: "+str(all_ts))
                print('tp: %d fp: %d  fn: %d'%(tp,fp,fn))
            
            
            
            self.tp += tp
            self.fp += fp
            self.fn += fn
            self.uc += uc
            self.nc += nc
            self.tc += tc
            self.sc += sc
            
            batch_pred_pairs.append(all_pairs)
            batch_target_pairs.append(all_ts)
        
        if predict_mode:
            return batch_pred_pairs,batch_target_pairs
            

    def get_metric(self, reset=True):
        res = {}
        f, pre, rec = _compute_f_pre_rec(1, self.tp, self.fn, self.fp)
        res['f'] = round(f*100, 2)
        res['rec'] = round(rec*100, 2)
        res['pre'] = round(pre*100, 2)
        res['em'] = round(self.em/self.total, 4)
        res['uc'] =round(self.uc)
        res['nc'] =round(self.nc)
        res['tc'] =round(self.tc)
        res['sc'] =round(self.sc)
        if reset:
            self.total = 0
            self.fp = 0
            self.tp = 0
            self.fn = 0
            self.em = 0
            self.uc =0
            self.nc =0
            self.tc =0
            self.sc =0
        return res


def _compute_tp_fn_fp(ps, ts,region_num):
    
    supports = len(ts)
    pred_sum = len(ps)
    correct_num = 0
    useful_correct = 0
    noregion_correct = 0
    span_correct = 0
    type_correct = 0
    for k,v in ps.items():
        span = k
        region_pred, entity_type = v
        if span in ts:
            r,e = ts[span]
            if set(e) == set(entity_type) and len(set(region_pred) & set(r)) != 0:
               
                if region_num not in set(r):
                    useful_correct +=1 
                else:
                    noregion_correct +=1
                correct_num += 1
            if set(e) == set(entity_type):
                type_correct +=1
            span_correct +=1 
    
    tp = correct_num
    fp = pred_sum - correct_num
    fn = supports - correct_num
    return tp,fp,fn,useful_correct,noregion_correct,type_correct,span_correct




