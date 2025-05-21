import torch
import torch.ao.quantization
from .modeling_bart_multi_concat import BartEncoder, BartDecoder, BartModel
from transformers import BartTokenizer
from .utils import seq_len_to_mask

#from fastNLP import seq_len_to_mask
#from fastNLP.modules import Seq2SeqEncoder, Seq2SeqDecoder, State
#from fastNLP.models import Seq2SeqModel

#Imports for State
from typing import Union, List, Tuple

import torch.nn.functional as F
from torch import nn


class State:

    """
    每个 ``Decoder`` 都有对应的 :class:`State` 对象用来承载 ``encoder`` 的输出以及当前时刻之前的 ``decode`` 状态。

    :param encoder_output: 如果不为 ``None`` ，内部元素需要为 :class:`torch.Tensor`，默认其中第一维是 ``batch_size``
        维度
    :param encoder_mask: 如果部位 ``None``，内部元素需要为 :class:`torch.Tensor`，默认其中第一维是 ``batch_size``
        维度
    :param kwargs:
    """
    def __init__(self, encoder_output: Union[torch.Tensor, List, Tuple]=None, 
                encoder_mask: Union[torch.Tensor, List, Tuple]=None, **kwargs):
        self.encoder_output = encoder_output
        self.encoder_mask = encoder_mask
        self._decode_length = 0

    @property
    def num_samples(self):
        """
        返回的 State 中包含的是多少个 sample 的 encoder 状态，主要用于 Generate 的时候确定 batch_size 的大小。
        """
        if self.encoder_output is not None:
            return self.encoder_output.size(0)
        else:
            return None

    @property
    def decode_length(self):
        """
        当前 Decode 到哪个 token 了，decoder 只会从 decode_length 之后的 token 开始 decode, 为 **0** 说明还没开始 decode。
        """
        return self._decode_length

    @decode_length.setter
    def decode_length(self, value):
        self._decode_length = value

    def _reorder_state(self, state: Union[torch.Tensor, list, tuple], indices: torch.LongTensor, dim: int = 0):
        if isinstance(state, torch.Tensor):
            state = state.index_select(index=indices, dim=dim)
        elif isinstance(state, list):
            for i in range(len(state)):
                assert state[i] is not None
                state[i] = self._reorder_state(state[i], indices, dim)
        elif isinstance(state, tuple):
            tmp_list = []
            for i in range(len(state)):
                assert state[i] is not None
                tmp_list.append(self._reorder_state(state[i], indices, dim))
            state = tuple(tmp_list)
        else:
            raise TypeError(f"Cannot reorder data of type:{type(state)}")

        return state

    def reorder_state(self, indices: torch.LongTensor):
        if self.encoder_mask is not None:
            self.encoder_mask = self._reorder_state(self.encoder_mask, indices)
        if self.encoder_output is not None:
            self.encoder_output = self._reorder_state(self.encoder_output, indices)

class Seq2SeqDecoder(nn.Module):

    """
    **Sequence-to-Sequence Decoder** 的基类。一定需要实现 :meth:`forward` 和 :meth:`decode` 函数，剩下的函数根据需要实现。每个 ``Seq2SeqDecoder`` 都应该有相应的
    :class:`~fastNLP.modules.torch.decoder.State` 对象用来承载该 ``Decoder`` 所需要的 ``Encoder`` 输出、``Decoder`` 需要记录的历史信（例如 :class:`~fastNLP.modules.torch.encoder.LSTM`
    的 hidden 信息）。
    """
    def __init__(self):
        super().__init__()

    def forward(self, tokens: "torch.LongTensor", state: State, **kwargs):
        """

        :param tokens: ``[batch_size, max_len]``
        :param state: ``state`` 包含了 ``encoder`` 的输出以及 ``decode`` 之前的内容
        :return: 返回值可以为 ``[batch_size, max_len, vocab_size]`` 的张量，也可以是一个 :class:`list`，但是第一个元素必须是词的预测分布
        """
        raise NotImplemented

    def reorder_states(self, indices: torch.LongTensor, states):
        """
        根据 ``indices`` 重新排列 ``states`` 中的状态，在 ``beam search`` 进行生成时，会用到该函数。

        :param indices:
        :param states:
        """
        assert isinstance(states, State), f"`states` should be of type State instead of {type(states)}"
        states.reorder_state(indices)

    def init_state(self, encoder_output: Union[torch.Tensor, list, tuple], encoder_mask: Union[torch.Tensor, list, tuple]):
        """
        初始化一个 :class:`~fastNLP.modules.torch.decoder.State` 对象，用来记录 ``encoder`` 的输出以及 ``decode`` 已经完成的部分。

        :param encoder_output: 如果不为 ``None`` ，内部元素需要为 :class:`torch.Tensor`，默认其中第一维是 batch_size
            维度
        :param encoder_mask: 如果不为 ``None``，内部元素需要为 :class:`torch.Tensor`，默认其中第一维是 batch_size
            维度
        :return: 一个 :class:`~fastNLP.modules.torch.decoder.State` 对象，记录了 ``encoder`` 的输出
        """
        state = State(encoder_output, encoder_mask)
        return state

    def decode(self, tokens: torch.LongTensor, state) -> torch.FloatTensor:
        """
        根据 ``states`` 中的内容，以及 ``tokens`` 中的内容进行之后的生成。

        :param tokens: ``[batch_size, max_len]``，截止到上一个时刻所有的 token 输出。
        :param state: 记录了 ``encoder`` 输出与 ``decoder`` 过去状态
        :return: `下一个时刻的分布，形状为 ``[batch_size, vocab_size]``
        """
        outputs = self(state=state, tokens=tokens)
        if isinstance(outputs, torch.Tensor):
            return outputs[:, -1]
        else:
            raise RuntimeError("Unrecognized output from the `forward()` function. Please override the `decode()` function.")
        

class Seq2SeqEncoder(nn.Module):
    """
    所有 **Sequence2Sequence Encoder** 的基类。需要实现 :meth:`forward` 函数

    """
    def __init__(self):
        super().__init__()

    def forward(self, tokens: torch.LongTensor, seq_len: torch.LongTensor):
        """

        :param tokens: ``[batch_size, max_len]]``，encoder 的输入
        :param seq_len: ``[batch_size,]``
        :return:
        """
        raise NotImplementedError
    

class Seq2SeqModel(nn.Module):
    """
    可以用于在 :class:`~fastNLP.core.controllers.Trainer` 中训练的 **Seq2Seq模型** 。正常情况下，继承了该函数之后，只需要
    实现 classmethod ``build_model`` 即可。如果需要使用该模型进行生成，需要把该模型输入到 :class:`~fastNLP.models.torch.SequenceGeneratorModel`
    中。在本模型中， :meth:`forward` 会把 encoder 后的结果传入到 decoder 中，并将 decoder 的输出 output 出来。

    :param encoder: :class:`~fastNLP.modules.torch.encoder.Seq2SeqEncoder` 对象，需要实现对应的 :meth:`forward` 函数，接受两个参数，第一个为
        ``[batch_size, max_len]`` 的 source tokens, 第二个为 ``[batch_size,]`` 的 source 的长度；需要返回两个 tensor： 
        
            - ``encoder_outputs`` : ``[batch_size, max_len, hidden_size]``
            - ``encoder_mask`` :  ``[batch_size, max_len]``，为 **0** 的地方为 pad。
        如果encoder的输出或者输入有变化，可以重载本模型的 :meth:`prepare_state` 函数或者 :meth:`forward` 函数。
    :param decoder: :class:`~fastNLP.modules.torch.decoder.Seq2SeqEncoder` 对象，需要实现 :meth:`init_state` 函数，需要接受两个参数，分别为
        上述的 ``encoder_outputs`` 和 ``encoder_mask``。若decoder需要更多输入，请重载当前模型的 :meth:`prepare_state` 或 :meth:`forward` 函数。
    """
    def __init__(self, encoder: Seq2SeqEncoder, decoder: Seq2SeqDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_tokens: "torch.LongTensor", tgt_tokens: "torch.LongTensor",
                src_seq_len: "torch.LongTensor"=None, tgt_seq_len: "torch.LongTensor"=None):
        """
        :param src_tokens: source 的 token，形状为 ``[batch_size, max_len]``
        :param tgt_tokens: target 的 token，形状为 ``[batch_size, max_len]``
        :param src_seq_len: source的长度，形状为 ``[batch_size,]``
        :param tgt_seq_len: target的长度，形状为 ``[batch_size,]``
        :return: 字典 ``{'pred': torch.Tensor}``, 其中 ``pred`` 的形状为 ``[batch_size, max_len, vocab_size]``
        """
        state = self.prepare_state(src_tokens, src_seq_len)
        decoder_output = self.decoder(tgt_tokens, state)
        if isinstance(decoder_output, torch.Tensor):
            return {'pred': decoder_output}
        elif isinstance(decoder_output, (tuple, list)):
            return {'pred': decoder_output[0]}
        else:
            raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")

    def train_step(self, src_tokens: "torch.LongTensor", tgt_tokens: "torch.LongTensor",
                    src_seq_len: "torch.LongTensor"=None, tgt_seq_len: "torch.LongTensor"=None):
        """
        :param src_tokens: source 的 token，形状为 ``[batch_size, max_len]``
        :param tgt_tokens: target 的 token，形状为 ``[batch_size, max_len]``
        :param src_seq_len: source的长度，形状为 ``[batch_size,]``
        :param tgt_seq_len: target的长度，形状为 ``[batch_size,]``
        :return: 字典 ``{'loss': torch.Tensor}``
        """
        res = self(src_tokens, tgt_tokens, src_seq_len, tgt_seq_len)
        pred = res['pred']
        if tgt_seq_len is not None:
            mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1))
            tgt_tokens = tgt_tokens.masked_fill(mask.eq(0), -100)
        loss = F.cross_entropy(pred[:, :-1].transpose(1, 2), tgt_tokens[:, 1:])
        return {'loss': loss}

    def prepare_state(self, src_tokens: "torch.LongTensor", src_seq_len: "torch.LongTensor"=None):
        """
        调用 encoder 获取 state，会把 encoder 的 ``encoder_output``, ``encoder_mask`` 直接传入到 :meth:`decoder.init_state` 中初始化一个 state

        :param src_tokens: source 的 token，形状为 ``[batch_size, max_len]``
        :param src_seq_len: source的长度，形状为 ``[batch_size,]``
        :return: decode 初始化的 state
        """
        encoder_output, encoder_mask = self.encoder(src_tokens, src_seq_len)
        state = self.decoder.init_state(encoder_output, encoder_mask)
        return state

    @classmethod
    def build_model(cls, *args, **kwargs):
        """
        需要实现本方法来进行 :class:`Seq2SeqModel` 的初始化

        :return:
        """
        raise NotImplementedError("A `Seq2SeqModel` must implement its own classmethod `build_model()`.")


def mask_image(image_feature):
    mask = image_feature.sum(dim=-1).gt(0)
    return mask

class FBartEncoder(Seq2SeqEncoder):
    def __init__(self, encoder):
        super().__init__()
        assert isinstance(encoder, BartEncoder)
        self.bart_encoder = encoder

        self.dequant_for_mask_image = torch.ao.quantization.DeQuantStub()

    def forward(self, src_tokens, image_feature, src_seq_len, text_only=False):
        mask = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
        if not text_only:
            image_feature_fp32_for_mask = self.dequant_for_mask_image(image_feature)
            image_mask = mask_image(image_feature_fp32_for_mask)
        else:
            image_mask = None    

        img_feat_, dict_encoder_output = self.bart_encoder(input_ids=src_tokens, image_feature=image_feature, attention_mask=mask, image_mask = image_mask, return_dict=True,
                                 output_hidden_states=True, text_only=text_only)  # last_hidden_state: tensor(bsz, max_len, 768),  hidden_states: tuple((baz, max_len, 768)),  attentions
        encoder_outputs = dict_encoder_output.last_hidden_state
        hidden_states = dict_encoder_output.hidden_states
        
        if text_only:
            return img_feat_, encoder_outputs, mask, hidden_states
        
        multi_modal_mask = torch.cat((image_mask,mask),dim=-1)
        return img_feat_, encoder_outputs, multi_modal_mask, hidden_states


class FBartDecoder(Seq2SeqDecoder):
    def __init__(self, decoder, pad_token_id, label_ids, use_encoder_mlp=True):
        super().__init__()
        assert isinstance(decoder, BartDecoder)
        self.decoder = decoder
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        causal_mask = causal_mask.triu(diagonal=1)
        self.register_buffer('causal_masks', causal_mask.float())
        self.pad_token_id = pad_token_id
        self.label_start_id = min(label_ids)
        self.label_end_id = max(label_ids)+1
        mapping = torch.LongTensor([0, 2]+label_ids)
        self.register_buffer('mapping', mapping)
        self.src_start_index = len(mapping)  # 加上一个
        hidden_size = decoder.embed_tokens.weight.size(1)
        if use_encoder_mlp:
            self.encoder_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                             nn.Dropout(0.3),
                                             nn.ReLU(),
                                             nn.Linear(hidden_size, hidden_size))

    def forward(self, tokens, state, text_only=False):
        # bsz, max_len = tokens.size()
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask

        first = state.first

        # tokens之后的0全是padding，因为1是eos, 在pipe中规定的
        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        # 把输入做一下映射
        mapping_token_mask = tokens.lt(self.src_start_index)  # 为1的地方应该从mapping中取index
        
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]

        src_tokens_index = tokens - self.src_start_index # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)

        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)

        if self.training:
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(self.pad_token_id)  # decoder需要让pad位置为1
            dict_decoder_output = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True)
        else:
            past_key_values = state.past_key_values
            dict_decoder_output = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True)
        hidden_state = dict_decoder_output.last_hidden_state  # bsz x max_len x hidden_size

        ## 比CaGFBartDecoder 少一个dropout
        if not self.training:
            state.past_key_values = dict_decoder_output.past_key_values

        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_tokens.size(-1)),
                                       fill_value=-1e24)

        # 首先计算的是
        eos_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[2:3])  # bsz x max_len x 1
        tag_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id])  # bsz x max_len x num_class

        # bsz x max_word_len x hidden_size
        src_outputs = state.encoder_output

        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len, 为1的地方是padding
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)

        mask = mask.unsqueeze(1).__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        # 与CaGFBartDecoder不同，此处仅涉及 encoder输出和decoder输出，而CaGFBartDecoder将输入和encoder的输出平均
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores

        return logits

    def decode(self,img_feat_,  tokens, state, text_only=False):
        # return self(tokens, state)[:, -1]
        return self(img_feat_, tokens, state, text_only=text_only)


class CaGFBartDecoder(FBartDecoder):
    def __init__(self, decoder, pad_token_id, label_ids, avg_feature=True, use_encoder_mlp=False,box_num = 36):
        super().__init__(decoder, pad_token_id, label_ids, use_encoder_mlp=use_encoder_mlp)
        self.avg_feature = avg_feature  # 如果是avg_feature就是先把token embed和对应的encoder output平均，
        self.dropout_layer = nn.Dropout(0.3)
        self.box_num = box_num
        hidden_size = decoder.embed_tokens.weight.size(1)
        self.region_select = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                             nn.Dropout(0.3),
                                             nn.ReLU(),
                                             nn.Linear(hidden_size,self.box_num))

        self.dequant_before_mm = torch.ao.quantization.DeQuantStub()


    def forward(self, img_feat_, tokens, state, text_only=False):  
        
        bsz, max_len = tokens.size()
        encoder_outputs = state.encoder_output  
        encoder_pad_mask = state.encoder_mask 
        
        first = state.first  # 原始sentence内部是index，之外padding部分是0
        target = tokens
        
        ## tokens to tokenize-id
        # mask target
        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)  
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:]) 
        # 把输入(即target)做一下映射，变成【encoder】的embed的id
        mapping_token_mask = tokens.lt(self.src_start_index)  
        # 映射类别
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)  
        tag_mapped_tokens = self.mapping[mapped_tokens]    # 映射特殊字符
        # 映射token
        src_tokens_index = tokens - self.src_start_index  # 还原index，因为tokens的index从标签类别之后开始
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)  # 不是token的部分置零 
        src_tokens = state.src_tokens  
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)  # sentence部分正常取，pad部分取“0”  
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1) 
        # 两个映射组合
        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)  
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)  
        #### 以上在准备decoder的input_ids, 将 index, 类别 表示的 taget 转换为 encoder 能读懂的 tokenize id 
        
        if self.training:
            tokens = tokens[:, :-1]  
            decoder_pad_mask = tokens.eq(self.pad_token_id)  # decoder需要让pad位置为1   
            
            dict_decoder_output = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,  # (bsz, max_len, 768)
                                encoder_padding_mask=encoder_pad_mask,  
                                decoder_padding_mask=decoder_pad_mask,  # (bsz, max_target-1)
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)], 
                                return_dict=True)  # BaseModelOutputWithPast类，包括last_hidden_state, past_key_values, hidden_states, attentions
        else:
            past_key_values = state.past_key_values
            #try:
            dict_decoder_output = self.decoder(input_ids=tokens,
                                    encoder_hidden_states=encoder_outputs,
                                    encoder_padding_mask=encoder_pad_mask,
                                    decoder_padding_mask=None,
                                    decoder_causal_mask=None,
                                    past_key_values=past_key_values,
                                    use_cache=True,
                                    return_dict=True)
            #except:
                #import pdb;pdb.set_trace()
        hidden_state = dict_decoder_output.last_hidden_state  # bsz x target_len x hidden_size
        hidden_state = self.dropout_layer(hidden_state)
        if not self.training:
            state.past_key_values = dict_decoder_output.past_key_values

        logits_shape = (hidden_state.size(0), hidden_state.size(1), self.src_start_index + src_tokens.size(-1))
        logits = hidden_state.new_full(logits_shape, fill_value=-1e24, dtype=torch.float32, device=hidden_state.device )   # (nsz, max_target， 54 + max_len)
        
        
        # --- Score Calculations ---
        embed_tokens_weight_for_eos = self.decoder.embed_tokens.weight[2:3]
        embed_tokens_weight_for_tags = self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id]


        hidden_state = self.dequant_before_mm(hidden_state)
        eos_scores = F.linear(hidden_state, self.dropout_layer(embed_tokens_weight_for_eos))
        tag_scores = F.linear(hidden_state, self.dropout_layer(embed_tokens_weight_for_tags))

         # Initialize features derived from encoder and raw image input
        src_img_outputs = None  # Encoded image features part
        input_img_embed = None  # Processed raw image features (img_feat_)

        if text_only:
            # In text_only mode, the entire encoder_output is considered text features.
            # img_feat_ is expected to be None, and we don't process it.
            src_outputs = state.encoder_output
            # src_img_outputs remains None
            # input_img_embed remains None
        else:
            # In multimodal mode (not text_only)
            # Split the encoder_output from the multimodal encoder
            src_outputs = state.encoder_output[:, self.box_num:, :]       # Text part
            src_img_outputs = state.encoder_output[:, :self.box_num, :]   # Image part

            # Process raw projected image features (img_feat_) if they were provided
            if img_feat_ is not None:
                input_img_embed = self.dropout_layer(img_feat_)
            # else: input_img_embed remains None if img_feat_ was None (e.g. optional raw features)

            # Apply MLP to the encoded image features part if MLP exists and features are present
            if hasattr(self, 'encoder_mlp') and self.encoder_mlp is not None and src_img_outputs is not None:
                src_img_outputs = self.encoder_mlp(src_img_outputs)
        
        # Apply MLP to the encoded text features part if MLP exists and features are present
        # This applies to src_outputs whether it came from text_only or multimodal path.
        if hasattr(self, 'encoder_mlp') and self.encoder_mlp is not None and src_outputs is not None:
            src_outputs = self.encoder_mlp(src_outputs)
        
        
        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len 
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1) # (bsz, max_len, 768)  # 取sentence内的encoder_output
        else:
            mask = torch.zeros_like(src_tokens, dtype=torch.bool, device=src_tokens.device)

        mask = mask.unsqueeze(1)

        input_embed = self.dropout_layer(self.decoder.embed_tokens(src_tokens))  # bsz x max_word_len x hidden_size



        if self.avg_feature:  # 先把feature合并一下
            src_outputs = (src_outputs + input_embed)/2
            if src_img_outputs is not None and input_img_embed is not None:
                src_img_outputs = (src_img_outputs + input_img_embed) /2

        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)
        if src_img_outputs is not None:
            img_scores = torch.einsum('blh,bnh->bln', hidden_state, src_img_outputs)
        else:
            img_scores = None

        if not self.avg_feature:
            gen_scores = torch.einsum('blh,bnh->bln', hidden_state, input_embed)  
            word_scores = (gen_scores + word_scores)/2
            if input_img_embed is not None and img_scores is not None:
                gen_img_scores = torch.einsum('blh,bnh->bln', hidden_state, input_img_embed)  
                img_scores = (gen_img_scores + img_scores)/2
        
        mask = mask.__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))  # 2 是结束符
        word_scores = word_scores.masked_fill(mask, -1e32)  

        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores   # logits: (bsz, target, 类别数 + max_len)
       
        if self.training:
            region_ind = target[:,:-1].eq(2)   ## bsz, max_len
            if img_scores is not None:
                img_logits = img_scores[region_ind]  ## ??, box_num
            else:
                img_logits = None
            return logits, img_logits   ## logits:(bsz, target_len, n_class+max_len)  region_pred:(bsz, ??, max_box+1 )
        else:  
            logits = logits[:,-1,:] ## logits:(bsz, n_class+max_len)
            if img_scores is not None:
                img_logits = img_scores[:,-1,:]
            else:
                img_logits = None
            return logits, img_logits





class BartSeq2SeqModel(Seq2SeqModel):
    @classmethod  
    def build_model(cls, bart_model, tokenizer, label_ids, decoder_type=None,
                    use_encoder_mlp=False,box_num = 36):
        model = BartModel.from_pretrained(bart_model)
        num_tokens, _ = model.encoder.embed_tokens.weight.shape

        # --- MODIFICATION START ---
        # Attempt to get the list of "unique_no_split_tokens" from the tokenizer's
        # standard attributes for added tokens.
        _list_of_added_token_strings = []
        if hasattr(tokenizer, 'added_tokens_encoder') and tokenizer.added_tokens_encoder:
            _list_of_added_token_strings = list(tokenizer.added_tokens_encoder.keys())
        # Fallback for some tokenizer versions/types if added_tokens_encoder is not populated but get_added_vocab is
        elif hasattr(tokenizer, 'get_added_vocab') and tokenizer.get_added_vocab():
            _list_of_added_token_strings = list(tokenizer.get_added_vocab().keys())
        
        if not _list_of_added_token_strings:
            print("Warning: No added tokens found via tokenizer.added_tokens_encoder or get_added_vocab(). "
                  "If 'unique_no_split_tokens' were expected, their embeddings won't be specially initialized.")
        # --- MODIFICATION END ---

        # Resize token embeddings to accommodate the original tokens + new tokens
        # The `model.resize_token_embeddings` method handles the actual embedding matrix resizing.
        model.resize_token_embeddings(len(_list_of_added_token_strings) + num_tokens)

        encoder = model.encoder
        decoder = model.decoder

        # 将类别（eg: "<<person>>"）添加到decoder原本词表之前，embed使用“类别名”的embed
        _tokenizer = BartTokenizer.from_pretrained(bart_model)   
        for token in _list_of_added_token_strings:
            if token[:2] == '<<':  # 特殊字符
                index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
                if len(index)>1:
                    raise RuntimeError(f"{token} wrong split")   
                else:
                    index = index[0]
                assert index>=num_tokens, (index, num_tokens, token) 
                indexes = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(token[2:-2])) 
                embed = model.encoder.embed_tokens.weight.data[indexes[0]]  
                for i in indexes[1:]:
                    embed += model.decoder.embed_tokens.weight.data[i]
                embed /= len(indexes)   
                model.decoder.embed_tokens.weight.data[index] = embed  

        encoder = FBartEncoder(encoder)
        if decoder_type is None:
            decoder = FBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids, box_num =box_num)  # label_ids是"<<actor>>"在_tokenizer中的id，在原词表之后
        elif decoder_type == 'avg_score':
            decoder = CaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids,
                                      avg_feature=False, use_encoder_mlp=use_encoder_mlp,box_num =box_num)
        elif decoder_type == 'avg_feature':
            decoder = CaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids,
                                      avg_feature=True, use_encoder_mlp=use_encoder_mlp,box_num =box_num)
        else:
            raise RuntimeError("Unsupported feature.")

        return cls(encoder=encoder, decoder=decoder)

    def prepare_state(self, src_tokens, image_feature, src_seq_len=None, first=None, tgt_seq_len=None, text_only=False):
        if text_only:
            image_feature = None
        img_feat_, encoder_outputs, encoder_mask, hidden_states = self.encoder(src_tokens,image_feature, src_seq_len, text_only=text_only)
        src_embed_outputs = hidden_states[0]
        state = BartState(encoder_outputs, encoder_mask, src_tokens, first, src_embed_outputs)
        # BartState 包括: src_tokens, first, src_embed_outputs
        return img_feat_, state

    def forward(self, src_tokens,image_feature, tgt_tokens, src_seq_len, tgt_seq_len, first, text_only=False):
        """

        :param torch.LongTensor src_tokens: source的token
        :param torch.LongTensor tgt_tokens: target的token
        :param torch.LongTensor first: 显示每个, bsz x max_word_len
        :param torch.LongTensor src_seq_len: src的长度
        :param torch.LongTensor tgt_seq_len: target的长度，默认用不上
        :return: {'pred': torch.Tensor}, 其中pred的shape为bsz x max_len x vocab_size
        """
        
        img_feat_, state = self.prepare_state(src_tokens, image_feature,src_seq_len, first, tgt_seq_len, text_only=text_only)
        decoder_output, region_pred = self.decoder(img_feat_, tgt_tokens, state, text_only=text_only)  # (bsz, max_target, 95) # 95, 每个预测的token上分 max_len+类别数 类
        if isinstance(decoder_output, torch.Tensor):
            return {'pred': decoder_output,'region_pred':region_pred}
        else:
            raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")



class BartState(State):
    def __init__(self, encoder_output, encoder_mask, src_tokens, first, src_embed_outputs):
        super().__init__(encoder_output, encoder_mask)
        self.past_key_values = None
        self.src_tokens = src_tokens
        self.first = first
        self.src_embed_outputs = src_embed_outputs

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.src_tokens = self._reorder_state(self.src_tokens, indices)
        if self.first is not None:
            self.first = self._reorder_state(self.first, indices)
        self.src_embed_outputs = self._reorder_state(self.src_embed_outputs, indices)
        if self.past_key_values is not None:
            new = []
            for layer in self.past_key_values:
                new_layer = {}
                for key1 in list(layer.keys()):
                    new_layer_ = {}
                    for key2 in list(layer[key1].keys()):
                        if layer[key1][key2] is not None:
                            layer[key1][key2] = self._reorder_state(layer[key1][key2], indices)
                            # print(key1, key2, layer[key1][key2].shape)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new