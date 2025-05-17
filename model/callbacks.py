#from fastNLP.callbacks import Callback
#from fastNLP import DataSet, Tester
from .dataset import DataSet
from .tester import Tester

import fitlog
from copy import deepcopy
from typing import Dict, Optional


class Callback:
    r"""
    实际使用的 callback 类，不管是 **fastNLP** 默认提供的一些 callback 实例，还是用户自己定制的 callback 类，都应该继承该基类；
    callback 调用时机顺序大概如下::

        Trainer.__init__():
            on_after_trainer_initialized(trainer, driver)
        Trainer.run():
            if num_eval_sanity_batch>0:
                on_sanity_check_begin(trainer)  # 如果设置了num_eval_sanity_batch
                on_sanity_check_end(trainer, sanity_check_res)
            try:
                on_train_begin(trainer)
                while cur_epoch_idx < n_epochs:
                    on_train_epoch_begin(trainer)
                    while batch_idx_in_epoch<=num_batches_per_epoch:
                        on_fetch_data_begin(trainer)
                        batch = next(dataloader)
                        on_fetch_data_end(trainer)
                        on_train_batch_begin(trainer, batch, indices)
                        on_before_backward(trainer, outputs)  # 其中 outputs 是经过 output_mapping（如果设置了） 后的，否则即为 model 的输出。
                        on_after_backward(trainer)
                        on_before_zero_grad(trainer, optimizers)  # 实际调用受到 accumulation_steps 影响
                        on_after_zero_grad(trainer, optimizers)  # 实际调用受到 accumulation_steps 影响
                        on_before_optimizers_step(trainer, optimizers)  # 实际调用受到 accumulation_steps 影响
                        on_after_optimizers_step(trainer, optimizers)  # 实际调用受到 accumulation_steps 影响
                        on_train_batch_end(trainer)
                    on_train_epoch_end(trainer)
            except BaseException:
                self.on_exception(trainer, exception)
            finally:
                on_train_end(trainer)

    其它 callback 例如 **on_evaluate_begin(trainer)** / **on_evaluate_end(trainer, results)** / **on_save_model(trainer)** / 
    **on_load_model(trainer)** / **on_save_checkpoint(trainer)** / **on_load_checkpoint(trainer)** 将根据需要在 :meth:`Trainer.run <fastNLP.core.controllers.Trainer.run>` 
    中特定的时间调用。
    """

    def on_after_trainer_initialized(self, trainer, driver):
        r"""
        在 ``Trainer`` 初始化后会被触发；

        :param trainer: :class:`~fastNLP.core.controllers.Trainer` 实例；
        :param driver: :class:`~fastNLP.core.controllers.Trainer` 中的 ``driver`` 实例；
        """
        pass

    def on_sanity_check_begin(self, trainer):
        r"""
        在 '预跑'检测 开始前会被触发；

        :param trainer: :class:`~fastNLP.core.controllers.Trainer` 实例；
        """
        pass

    def on_sanity_check_end(self, trainer, sanity_check_res):
        r"""
        在 '预跑'检测 开始后会被触发；

        :param trainer: :class:`~fastNLP.core.controllers.Trainer` 实例；
        :param sanity_check_res: 预跑得到的评测结果，关于对于 **预跑** 的解释，请见 :meth:`~fastNLP.core.controllers.trainer.Trainer.run`；
        """
        pass

    def on_train_begin(self, trainer):
        r"""
        在训练开始前会被触发；

        :param trainer: :class:`~fastNLP.core.controllers.Trainer` 实例；
        """
        pass

    def on_train_end(self, trainer):
        r"""
        在训练完成后会被触发；

        :param trainer: :class:`~fastNLP.core.controllers.Trainer` 实例；
        """
        pass

    def on_train_epoch_begin(self, trainer):
        r"""
        在训练过程中的每一个 epoch 开始前会被触发；

        :param trainer: :class:`~fastNLP.core.controllers.Trainer` 实例；
        """
        pass

    def on_train_epoch_end(self, trainer):
        r"""
        在训练过程中的每一个 epoch 完成后会被触发；此时 trainer.cur_epoch_idx 已经完成加 1 操作。

        :param trainer: :class:`~fastNLP.core.controllers.Trainer` 实例；
        """
        pass

    def on_fetch_data_begin(self, trainer):
        r"""
        在训练过程中准备取出下一个 batch 的数据时触发

        :param trainer: :class:`~fastNLP.core.controllers.Trainer` 实例；
        """
        pass

    def on_fetch_data_end(self, trainer):
        r"""
        在训练过程中拿到当前的 batch 数据后会被触发；

        :param trainer: :class:`~fastNLP.core.controllers.Trainer` 实例；
        """
        pass

    def on_train_batch_begin(self, trainer, batch, indices):
        r"""
        在取得数据，执行完 ``input_mapping`` (如果 :class:`~fastNLP.core.controllers.Trainer` 传有该参数），并且移动 ``batch`` 中的张量到了指定设备之后会被触发。
        其中 ``batch`` 中的数据格式要么是 ``Dataloader`` 返回的每个 ``batch`` 的格式；要么是 ``input_mapping`` 之后的内容。
        如果 ``batch`` 是 ``dict`` 类型，直接增删其中的 key 或 修改其中的 value 会影响到输入模型的中的 ``batch`` 数据。

        :param trainer: :class:`~fastNLP.core.controllers.Trainer` 实例；
        :param batch: batch 的数据，已经经过 ``input_mapping`` (如果有) 以及移动到指定设备 。
        :param list[int] indices: 当前的 ``batch`` 是数据集中的哪些数据。仅在 ``DataLoader`` 支持得到当前 ``batch index`` 的时候有值，
            其它时候为 ``None`` 。
        """
        pass

    def on_train_batch_end(self, trainer):
        r"""
        完成一个 batch 的训练（forward）、梯度回传（backward）、梯度更新（step）、梯度置零、batch_idx_in_epoch 与
        global_forward_batches 累计加1操作之后会被触发。其中梯度更新、梯度置零操作会考虑 **accumulation_steps** ，所以不一定在当前 batch 会
        执行。

        :param trainer: :class:`~fastNLP.core.controllers.Trainer` 实例；
        """
        pass

    def on_exception(self, trainer, exception):
        r"""
        在训练过程遇到异常时调用。

        :param trainer: :class:`~fastNLP.core.controllers.Trainer` 实例；
        :param exception: 遭遇的异常；
        """
        pass

    def on_save_model(self, trainer):
        r"""
        当调用 :meth:`Trainer.save_model() <fastNLP.core.controllers.Trainer.save_model>` 时调用，此刻模型还未保存。

        :param trainer: :class:`~fastNLP.core.controllers.Trainer` 实例；
        """
        pass

    def on_load_model(self, trainer):
        r"""
        当调用 :meth:`Trainer.load_model() <fastNLP.core.controllers.Trainer.load_model>` 加载模型时调用，此刻模型还未加载。

        :param trainer: :class:`~fastNLP.core.controllers.Trainer` 实例；
        """
        pass

    def on_save_checkpoint(self, trainer) -> Dict:
        r"""
        当 Trainer 将要保存 checkpoint 的时候触发 (即调用 :meth:`Trainer.save_checkpoint() <fastNLP.core.controllers.Trainer.save_checkpoint>`
        函数时)，该函数用于保存当前 callback 在恢复时需要的相关数据。

        :param trainer: :class:`~fastNLP.core.controllers.Trainer` 实例；
        """
        pass

    def on_load_checkpoint(self, trainer, states: Optional[Dict]):
        r"""
        当 Trainer 要恢复 checkpoint 的时候触发（即调用 :meth:`Trainer.load_checkpoint() <fastNLP.core.controllers.Trainer.load_checkpoint>`
        函数时, 此刻 Trainer 与 Driver 已经加载好自身的状态）， 参数 states 为 Callback 在调用 :meth:`on_save_checkpoint` 的返回值。

        :param trainer: :class:`~fastNLP.core.controllers.Trainer` 实例；
        :param states:
        """
        pass

    def on_before_backward(self, trainer, outputs):
        r"""
        在 backward 前执行。

        :param trainer: :class:`~fastNLP.core.controllers.Trainer` 实例；
        :param outputs: ``model`` 的返回内容。如果有 ``output_mapping``，则 ``outputs`` 中的内容为已经执行了 ``output_mapping`` 后的结果。
        """
        pass

    def on_after_backward(self, trainer):
        r"""
        在 ``backward`` 后执行。在多卡场景下，由于 ``accumulation_steps`` 的影响，仅在需要真正 ``update`` 参数那次梯度回传才会触发梯度同步，
        因此在多卡且使用 ``accumulation_steps`` 时，可能存在某些 step 各卡上梯度不一致的问题。

        :param trainer: :class:`~fastNLP.core.controllers.Trainer` 实例；
        """
        pass

    def on_before_optimizers_step(self, trainer, optimizers):
        r"""
        在进行 optimizer 优化进行前调用。该接口不一定每次前向计算都会触发，实际调用会受到 ``accumulation_steps`` 的影响。

        :param trainer: :class:`~fastNLP.core.controllers.Trainer` 实例；
        :param optimizers: 优化器，内容为在 :class:`~fastNLP.core.controllers.Trainer` 初始化时传入的值。
        """
        pass

    def on_after_optimizers_step(self, trainer, optimizers):
        r"""
        在进行 optimizer 优化进行后调用。该接口不一定每次前向计算都会触发，实际调用会受到 ``accumulation_steps`` 的影响。

        :param trainer: :class:`~fastNLP.core.controllers.Trainer` 实例；
        :param optimizers: 优化器，内容为在 :class:`~fastNLP.core.controllers.Trainer` 初始化时传入的值。
        """
        pass

    def on_before_zero_grad(self, trainer, optimizers):
        r"""
        在进行模型梯度置零前调用。该接口不一定每次前向计算都会触发，实际调用会受到 ``accumulation_steps`` 的影响。

        :param trainer: :class:`~fastNLP.core.controllers.Trainer` 实例；
        :param optimizers: 优化器，内容为在 :class:`~fastNLP.core.controllers.Trainer` 初始化时传入的值。
        """
        pass

    def on_after_zero_grad(self, trainer, optimizers):
        r"""
        在进行模型梯度置零后调用。该接口不一定每次前向计算都会触发，实际调用会受到 ``accumulation_steps`` 的影响。

        :param trainer: :class:`~fastNLP.core.controllers.Trainer` 实例；
        :param optimizers: 优化器，内容为在 :class:`~fastNLP.core.controllers.Trainer` 初始化时传入的值。
        """
        pass

    def on_evaluate_begin(self, trainer):
        r"""
        在将要进行 ``evaluate`` 时调用。如果是设置的以 step 数量或自定义地决定 evaluate 的频率，该接口是在 :meth:`on_train_batch_end` 之后
        进行调用。如果是以 epoch 数量决定调用时机，该接口是在 :meth:`on_train_epoch_end` 之后调用。

        :param trainer: :class:`~fastNLP.core.controllers.Trainer` 实例；
        """
        pass

    def on_evaluate_end(self, trainer, results):
        r"""
        结束 evaluate 时调用，并把 evaluate 的结果传入。

        :param trainer: :class:`~fastNLP.core.controllers.Trainer` 实例；
        :param results: :class:`~fastNLP.core.controllers.Trainer` 内置的 ``Evaluator`` 评测的结果，通常是个 ``dict``；
        """
        pass

    @property
    def callback_name(self):
        r"""
        ``callback`` 的名称，我们会使用该名称从 ``checkpoint`` 中读取的相应的 ``state`` 并传递给 :meth:`on_load_checkpoint` 函数。

        :return: 用于区分该 ``callback`` 实例的名称；
        """
        return self.__class__.__name__

    @property
    def need_reproducible_sampler(self) -> bool:
        r"""
        当前 callback 是否需要能够复现的 sampler 。一般用于 checkpoint 类的 callback 。
        """
        return False

class FitlogCallback(Callback):
    r"""
    该callback可将loss和progress写入到fitlog中; 如果Trainer有dev的数据，将自动把dev的结果写入到log中; 同时还支持传入
    一个(或多个)test数据集进行测试(只有在trainer具有dev时才能使用)，每次在dev上evaluate之后会在这些数据集上验证一下。
    并将验证结果写入到fitlog中。这些数据集的结果是根据dev上最好的结果报道的，即如果dev在第3个epoch取得了最佳，则
    fitlog中记录的关于这些数据集的结果就是来自第三个epoch的结果。
    """

    def __init__(self, data=None, tester=None, log_loss_every=0, verbose=1, log_exception=True,
                 raise_threshold=0, better_dev_eval=True, eval_begin_epoch=-1):
        r"""

        :param ~fastNLP.DataSet,Dict[~fastNLP.DataSet] data: 传入DataSet对象，会使用多个Trainer中的metric对数据进行验证。如果需要
            传入多个DataSet请通过dict的方式传入，dict的key将作为对应dataset的name传递给fitlog。data的结果的名称以'data'开头。
        :param ~fastNLP.Tester,Dict[~fastNLP.Tester] tester: Tester对象，将在on_valid_end时调用。tester的结果的名称以'tester'开头
        :param int log_loss_every: 多少个step记录一次loss(记录的是这几个batch的loss平均值)，如果数据集较大建议将该值设置得
            大一些，不然会导致log文件巨大。默认为0, 即不要记录loss。
        :param int verbose: 是否在终端打印evaluation的结果，0不打印。
        :param bool log_exception: fitlog是否记录发生的exception信息
        :param float raise_threshold: 如果metric值低于这个就会raise exception
        :param bool better_dev_eval: 仅当dev取得更好的结果才做evaluate
        """
        super().__init__()
        self.datasets = {}
        self.testers = {}
        self._log_exception = log_exception
        self.raise_threshold = raise_threshold
        self.eval_begin_epoch = eval_begin_epoch

        assert isinstance(log_loss_every, int) and log_loss_every>=0
        if tester is not None:
            if isinstance(tester, dict):
                for name, test in tester.items():
                    if not isinstance(test, Tester):
                        raise TypeError(f"{name} in tester is not a valid fastNLP.Tester.")
                    self.testers['tester-' + name] = test
            if isinstance(tester, Tester):
                self.testers['tester-test'] = tester
            for tester in self.testers.values():
                setattr(tester, 'verbose', 0)

        if isinstance(data, dict):
            for key, value in data.items():
                assert isinstance(value, DataSet), f"Only DataSet object is allowed, not {type(value)}."
            for key, value in data.items():
                self.datasets['data-' + key] = value
        elif isinstance(data, DataSet):
            self.datasets['data-test'] = data
        elif data is not None:
            raise TypeError("data receives dict[DataSet] or DataSet object.")

        self.verbose = verbose
        self._log_loss_every = log_loss_every
        self._avg_loss = 0
        self.best_test_metric_sofar = 0
        self.best_test_sofar = None
        self.best_test_epoch = 0
        self.best_dev_test = None
        self.best_dev_epoch = 0
        self.better_dev_eval = better_dev_eval

    def on_train_begin(self):
        if (len(self.datasets) > 0 or len(self.testers) > 0) and self.trainer.dev_data is None:
            raise RuntimeError("Trainer has no dev data, you cannot pass extra data to do evaluation.")

        if len(self.datasets) > 0:
            for key, data in self.datasets.items():
                tester = Tester(data=data, model=self.model,
                                batch_size=self.trainer.kwargs.get('dev_batch_size', self.trainer.batch_size),
                                metrics=self.trainer.metrics,
                                verbose=0,
                                use_tqdm=self.trainer.kwargs.get('test_use_tqdm', self.trainer.use_tqdm),
                                sampler=self.trainer.kwargs.get('test_sampler', None))
                self.testers[key] = tester
        fitlog.add_progress(total_steps=self.n_steps)

        if self.trainer.save_path is not None:
            model_name = "best_" + "_".join([self.model.__class__.__name__, self.trainer.metric_key, self.trainer.start_time])
            fitlog.add_other(name='model_name', value=model_name)

    def on_epoch_begin(self):
        if self.eval_begin_epoch>0 and self.epoch>self.eval_begin_epoch:
            self.trainer.validate_every = -1

    def on_backward_begin(self, loss):
        if self._log_loss_every >0:
            self._avg_loss += loss.item()
            if self.step %self._log_loss_every==0:
                fitlog.add_loss(self._avg_loss /self._log_loss_every *self.update_every, name='loss', step=self.step, epoch=self.epoch)
                self._avg_loss = 0

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
        if better_result:
            eval_result = deepcopy(eval_result)
            eval_result['step'] = self.step
            eval_result['epoch'] = self.epoch
            fitlog.add_best_metric(eval_result)
        fitlog.add_metric(eval_result, step=self.step, epoch=self.epoch)
        indicator, indicator_val = _check_eval_results(eval_result, metric_key=metric_key)
        if indicator_val < self.raise_threshold:
            print("indicator_val: "+str(indicator_val))
            raise RuntimeError("The program has been running off.")

        if len(self.testers) > 0:
            do_eval = True
            if self.better_dev_eval:
                if not better_result:
                    do_eval = False
            if do_eval:
                for idx, (key, tester) in enumerate(self.testers.items()):
                    try:
                        eval_result = tester.test()
                        if self.verbose != 0:
                            self.pbar.write("FitlogCallback evaluation on {}:".format(key))
                            self.pbar.write(tester._format_eval_results(eval_result))
                        fitlog.add_metric(eval_result, name=key, step=self.step, epoch=self.epoch)
                        if idx == 0:
                            indicator, indicator_val = _check_eval_results(eval_result, metric_key=self.trainer.metric_key)
                            if indicator_val>self.best_test_metric_sofar:
                                self.best_test_metric_sofar = indicator_val
                                self.best_test_epoch = self.epoch
                                self.best_test_sofar = eval_result

                        if better_result:
                            self.best_dev_test = eval_result
                            self.best_dev_epoch = self.epoch
                            fitlog.add_best_metric(eval_result, name=key)
                    except Exception as e:
                        self.pbar.write("Exception happens when evaluate on DataSet named `{}`.".format(key))
                        raise e

    def on_train_end(self):
        if self.best_test_sofar:
            line1 = "Best test performance(may not correspond to the best dev performance):{} achieved at Epoch:{}.".format(
                self.best_test_sofar, self.best_test_epoch)
            self.logger.info(line1)
            fitlog.add_to_line(line1)
        if self.best_dev_test:
            line2 = "Best test performance(correspond to the best dev performance):{} achieved at Epoch:{}.".format(
                self.best_dev_test, self.best_dev_epoch)
            self.logger.info(line2)
            fitlog.add_to_line(line2)
        fitlog.finish()

    def on_exception(self, exception):
        fitlog.finish(status=1)
        if self._log_exception:
            fitlog.add_other(repr(exception), name='except_info')


def _check_eval_results(metrics, metric_key=None):
    # metrics: tester返回的结果
    # metric_key: 一个用来做筛选的指标，来自Trainer的初始化
    if isinstance(metrics, tuple):
        loss, metrics = metrics

    if isinstance(metrics, dict):
        metric_dict = list(metrics.values())[0]  # 取第一个metric

        if metric_key is None:
            indicator_val, indicator = list(metric_dict.values())[0], list(metric_dict.keys())[0]
        else:
            # metric_key is set
            if metric_key not in metric_dict:
                raise RuntimeError(f"metric key {metric_key} not found in {metric_dict}")
            indicator_val = metric_dict[metric_key]
            indicator = metric_key
    else:
        raise RuntimeError("Invalid metrics type. Expect {}, got {}".format((tuple, dict), type(metrics)))
    return indicator, indicator_val


from fastNLP import WarmupCallback as FWarmupCallback
import math
class WarmupCallback(FWarmupCallback):

    def __init__(self, warmup=0.1, schedule='constant'):
        """

        :param int,float warmup: 如果warmup为int，则在该step之前，learning rate根据schedule的策略变化; 如果warmup为float，
            如0.1, 则前10%的step是按照schedule策略调整learning rate。
        :param str schedule: 以哪种方式调整。
            linear: 前warmup的step上升到指定的learning rate(从Trainer中的optimizer处获取的), 后warmup的step下降到0；
            constant前warmup的step上升到指定learning rate，后面的step保持learning rate.
        """
        super().__init__()
        self.warmup = max(warmup, 0.)

        self.initial_lrs = []  # 存放param_group的learning rate
        if schedule == 'constant':
            self.get_lr = self._get_constant_lr
        elif schedule == 'linear':
            self.get_lr = self._get_linear_lr
        elif schedule == 'inverse_square':
            self.get_lr = self._get_inverse_square_lr
        else:
            raise RuntimeError("Only support 'linear', 'constant'.")

    def _get_inverse_square_lr(self, progress):
        if progress<self.warmup:
            return progress/self.warmup
        return max((math.sqrt(progress) - 1.) / (math.sqrt(self.warmup) - 1.), 0.)


class OutputIndiceCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_batch_begin(self, batch_x, batch_y, indices):
        self.indices = indices

    def on_exception(self, exception):
        print(self.indices)