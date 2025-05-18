import sys
sys.path.append('../')
import os
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
    # os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import warnings
                                                 
warnings.filterwarnings('ignore')

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence # Make sure this is imported
from itertools import chain



from model.data_pipe import BartNERPipe
from model.bart_multi_concat import BartSeq2SeqModel  
from model.generater_multi_concat import SequenceGeneratorModel     
from model.metrics import Seq2SeqSpanMetric 
from model.losses import get_loss

#import fitlog
import datetime
#from fastNLP import Trainer

from torch import optim
#from fastNLP import  SequentialSampler, BucketSampler,GradientClipCallback, cache_results, EarlyStopCallback
from model.sampler import SequentialSampler, BucketSampler

#from model.callbacks import WarmupCallback
#from fastNLP.core.samplers import SortedSampler
#from fastNLP.core.sampler import  ConstTokenNumSampler
#from model.callbacks import FitlogCallback

from model.dataset import DataSet
from model.batch import DataSetIter
from tqdm import tqdm, trange
from model.utils import _move_dict_value_to_device
#from fastNLP.core.utils import _move_dict_value_to_device
#import random

#fitlog.debug()

# Enable anomaly detection to find the operation that causes NaNs
#torch.autograd.set_detect_anomaly(True)

#fitlog.set_log_dir('logs')





import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--bart_name', default='facebook/bart-large', type=str)
parser.add_argument('--datapath', default='./Twitter_GMNER/txt/', type=str)
parser.add_argument('--image_feature_path',default='./data/Twitter_GMNER_vinvl', type=str)
parser.add_argument('--image_annotation_path',default='./Twitter_GMNER/xml/', type=str)
parser.add_argument('--region_loss_ratio',default='1.0', type=float)
parser.add_argument('--box_num',default='16', type=int)
parser.add_argument('--normalize',default=False, action = "store_true")
parser.add_argument('--use_kl',default=False,action ="store_true")
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--n_epochs', default=30, type=int)
parser.add_argument('--max_len', default=30, type=int)
parser.add_argument('--batch_size',default=16,type=int)
parser.add_argument('--seed',default=42,type=int)
parser.add_argument("--save_model",default=0,type=int)
parser.add_argument("--save_path",default='save_models/best',type=str)
parser.add_argument("--log",default='./logs',type=str)
args= parser.parse_args()


dataset_name = 'twitter-ner'
args.length_penalty = 1



args.target_type = 'word'
args.schedule = 'linear'
args.decoder_type = 'avg_feature'
args.num_beams = 1   
args.use_encoder_mlp = 1
args.warmup_ratio = 0.01
eval_start_epoch = 0


if 'twitter' in dataset_name:  
    max_len, max_len_a = args.max_len, 0.6
else:
    print("Error dataset_name!")


if isinstance(args.decoder_type, str) and args.decoder_type.lower() == 'none':
    args.decoder_type = None
demo = False

def get_data():

    pipe = BartNERPipe(image_feature_path=args.image_feature_path, 
                       image_annotation_path=args.image_annotation_path,
                       max_bbox =args.box_num,
                       normalize=args.normalize,
                       tokenizer=args.bart_name, 
                       target_type=args.target_type)
    if dataset_name == 'twitter-ner': 
        paths ={
            'train': os.path.join(args.datapath,'train.txt'),
            'dev': os.path.join(args.datapath,'dev.txt'),
            'test': os.path.join(args.datapath,'test.txt') }
        data_bundle = pipe.process_from_file(paths, demo=demo)
        
    return data_bundle, pipe.tokenizer, pipe.mapping2id

data_bundle, tokenizer, mapping2id = get_data()

print(f'max_len_a:{max_len_a}, max_len:{max_len}')

print("The number of tokens in tokenizer ", tokenizer.vocab_size)  

bos_token_id = 0
eos_token_id = 1
label_ids = list(mapping2id.values())


model = BartSeq2SeqModel.build_model(args.bart_name, tokenizer, label_ids=label_ids, decoder_type=args.decoder_type,
                                     use_encoder_mlp=args.use_encoder_mlp,box_num = args.box_num)


vocab_size = len(tokenizer)

model = SequenceGeneratorModel(model, bos_token_id=bos_token_id,
                               eos_token_id=eos_token_id, 
                               max_length=max_len, max_len_a=max_len_a,num_beams=args.num_beams, do_sample=False,
                               repetition_penalty=1, length_penalty=args.length_penalty, pad_token_id=eos_token_id,
                               restricter=None, top_k = 1
                               )

## parameter scale
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total/1e6))
##

import torch
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

#torch.manual_seed(args.seed)


parameters =[]
params = {'lr':args.lr}
params['params'] = [param for name, param in model.named_parameters() ]
parameters.append(params)

optimizer = optim.AdamW(parameters)


metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids), region_num =args.box_num, target_type=args.target_type,print_mode = False )

train_dataset = data_bundle.get_dataset('train')
eval_dataset = data_bundle.get_dataset('dev')
test_dataset = data_bundle.get_dataset('test')
print(train_dataset[:3])

device = torch.device(device)
model.to(device)


def create_collate_fn(tensor_fields_map, padding_values_map):
    """
    Creates a custom collate function for a batch of Instance objects.
    - Fields specified as keys in `tensor_fields_map` will be converted to tensors.
    - If these tensorized fields are sequences of varying lengths, they will be padded
      using values from `padding_values_map`.
    - Other fields will be returned as lists.

    Args:
        tensor_fields_map (dict): A dictionary where keys are field names
                                  from the Instance objects that should be
                                  converted to tensors.
                                  Example: {'input_ids': True, 'labels': True}
        padding_values_map (dict): A dictionary mapping field names to their
                                   respective padding values. This is used for
                                   fields that are tensorized and found to be
                                   variable-length sequences.
                                   Example: {'input_ids': 0, 'labels': -100}
    Returns:
        function: The collate function.
    """
    if not isinstance(padding_values_map, dict):
        raise ValueError("padding_values_map must be a dictionary.")

    def collate_fn(batch):
        if not batch:
            return {}

        first_instance_fields = batch[0].fields # Assuming consistent field names
        collated_batch = {}

        for field_name in first_instance_fields.keys():
            values_for_field = [instance[field_name] for instance in batch]

            if field_name in tensor_fields_map:
                try:
                    # Convert all items for this field to tensors
                    tensor_items = [torch.as_tensor(item) for item in values_for_field]

                    # Check if items need padding or can be simply stacked
                    if not tensor_items: # Should not happen if batch is not empty
                        collated_batch[field_name] = torch.empty(0)
                        continue

                    first_item_ndim = tensor_items[0].ndim
                    
                    if first_item_ndim == 0: # All items are scalar tensors
                        collated_batch[field_name] = torch.stack(tensor_items)
                    else: # Items are non-scalar (sequences or multi-dimensional)
                        # Check if all tensors have the exact same shape
                        is_same_shape = True
                        first_shape = tensor_items[0].shape
                        for t in tensor_items[1:]:
                            if t.shape != first_shape:
                                is_same_shape = False
                                break
                        
                        if is_same_shape:
                            # All tensors have the same shape, stack them
                            collated_batch[field_name] = torch.stack(tensor_items)
                        else:
                            # Shapes differ, requires padding.
                            # This typically applies to sequences (1D tensors of varying lengths)
                            # or lists of multi-D tensors where the first dimension varies (e.g. region_label)
                            padding_value = padding_values_map.get(field_name)
                            if padding_value is None:
                                print(f"Warning: No padding value specified in padding_values_map for variable-length field '{field_name}'. Defaulting to 0.")
                                padding_value = 0
                            
                            collated_batch[field_name] = pad_sequence(
                                tensor_items, batch_first=True, padding_value=float(padding_value) # pad_sequence expects float for padding_value
                            )
                except Exception as e:
                    print(f"Error collating field '{field_name}': {e}. Returning as list.")
                    collated_batch[field_name] = values_for_field # Fallback to list
            else:
                # Field not in tensor_fields_map, return as a list
                collated_batch[field_name] = values_for_field
        
        return collated_batch

    return collate_fn


fields_tensors = ['src_tokens', 'image_feature', 'tgt_tokens', 'src_seq_len', 'tgt_seq_len', 'first', 'region_label']

padding_values = {
    'src_tokens': tokenizer.pad_token_id,
    'tgt_tokens': tokenizer.pad_token_id, # Or eos_token_id if your model specifically uses that for padding targets
    'first': 0,                          # Assuming 0 is a safe padding for this field
    'region_label': -100,                # Assuming -100 for ignored labels, adjust if necessary
    # 'image_feature' is pre-padded in BartNERPipe, so it won't hit the pad_sequence path.
    # 'src_seq_len', 'tgt_seq_len' are scalars, they will be stacked.
}

collate_fn = create_collate_fn(
    tensor_fields_map=fields_tensors,
    padding_values_map=padding_values
)



def Training(args, train_idx, train_data, model, device, optimizer):
    
    #train_sampler = BucketSampler(seq_len_field_name='src_seq_len',batch_size=args.batch_size)   # 带Bucket的 Random Sampler. 可以随机地取出长度相似的元素
    #train_data_iterator = DataSetIter(train_data, batch_size=args.batch_size) #sampler=train_sampler)
    train_data_iterator = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    

    train_loss = 0.
    train_region_loss = 0.

    # for batch_x, batch_y in tqdm(train_data_iterator, total=len(train_data_iterator)):
    for batch_idx, batch_x in enumerate(tqdm(train_data_iterator, desc=f"Epoch {train_idx} Training")):
        _move_dict_value_to_device(batch_x, device=device)
        src_tokens = batch_x['src_tokens']
        image_feature = batch_x['image_feature']

        tgt_tokens = batch_x['tgt_tokens']
        src_seq_len = batch_x['src_seq_len']
        tgt_seq_len = batch_x['tgt_seq_len']
        first = batch_x['first']
        region_label = batch_x['region_label']



        results = model(src_tokens,image_feature, tgt_tokens, src_seq_len=src_seq_len, tgt_seq_len=tgt_seq_len, first=first)
        pred_raw, region_pred_raw = results['pred'],results['region_pred']       

         

        #pred = torch.clamp(pred_raw, min=-30, max=30)
        pred = pred_raw
        if region_pred_raw is not None:
            #region_pred = torch.clamp(region_pred_raw, min=-30, max=30)
            region_pred = region_pred_raw
        else:
            region_pred = None


        loss, region_loss = get_loss(tgt_tokens, tgt_seq_len, pred, region_pred,region_label,use_kl=args.use_kl)
        

        train_loss += loss.item() if not torch.isnan(loss) else 0 # Avoid propagating NaN to sum
        train_region_loss += region_loss.item() if not torch.isnan(region_loss) else 0
        
        


        all_loss = loss + args.region_loss_ratio * region_loss

    

        optimizer.zero_grad() # Ensure grads are clear before backward
        all_loss.backward()

    

        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step() # This step would corrupt weights if grads were NaN



    print("train_loss: %f"%(train_loss))
    print("train_region_loss: %f"%(train_region_loss))
    return train_loss, train_region_loss


def Inference(args,eval_data, model, device, metric):
    #data_iterator = DataSetIter(eval_data, batch_size=args.batch_size * 2, sampler=SequentialSampler())
    # for batch_x, batch_y in tqdm(data_iterator, total=len(data_iterator)):
    data_iterator = torch.utils.data.DataLoader(
        dataset=eval_data, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn)
    
    for batch_x in tqdm(data_iterator,desc=f"Inference"):

        _move_dict_value_to_device(batch_x, device=device)
        src_tokens = batch_x['src_tokens']
        image_feature = batch_x['image_feature']
        tgt_tokens = batch_x['tgt_tokens']
        src_seq_len = batch_x['src_seq_len']
        tgt_seq_len = batch_x['tgt_seq_len']
        first = batch_x['first']
        region_label = batch_x['region_label']
        target_span = batch_x['target_span']
        cover_flag = batch_x['cover_flag']

        results = model.predict(src_tokens,image_feature, src_seq_len=src_seq_len, first=first)
        
        pred,region_pred = results['pred'],results['region_pred']   ## logits:(bsz,tgt_len,class+max_len)  region_logits:(??,8)
        
        metric.evaluate(target_span, pred, tgt_tokens, region_pred,region_label,cover_flag)
    res = metric.get_metric()  ## {'f': 20.0, 'rec': 16.39, 'pre': 25.64, 'em': 0.125, 'uc': 0}
    return res




def Predict(args,eval_data, model, device, metric,tokenizer,ids2label):
    #data_iterator = DataSetIter(eval_data, batch_size=args.batch_size * 2, sampler=SequentialSampler())
    # for batch_x, batch_y in tqdm(data_iterator, total=len(data_iterator)):
    data_iterator = torch.utils.data.DataLoader(
        dataset=eval_data, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn)

    with open (args.pred_output_file,'w') as fw:
        for batch_x in tqdm(data_iterator,desc=f"Predict"):
            _move_dict_value_to_device(batch_x, device=device)
            src_tokens = batch_x['src_tokens']
            image_feature = batch_x['image_feature']
            tgt_tokens = batch_x['tgt_tokens']
            src_seq_len = batch_x['src_seq_len']
            tgt_seq_len = batch_x['tgt_seq_len']
            first = batch_x['first']
            region_label = batch_x['region_label']
            target_span = batch_x['target_span']
            cover_flag = batch_x['cover_flag']

            results = model.predict(src_tokens,image_feature, src_seq_len=src_seq_len, first=first)
            
            pred,region_pred = results['pred'],results['region_pred']   ## logits:(bsz,tgt_len,class+max_len)  region_logits:(??,8)
            
            pred_pairs, target_pairs = metric.evaluate(target_span, pred, tgt_tokens, region_pred,region_label,cover_flag,predict_mode=True)
            
            raw_words = batch_x['raw_words']
            word_start_index = 8 ## 2 + 2 +4
            assert len(pred_pairs) == len(target_pairs)
            for i in range(len(pred_pairs)):
                cur_src_token = src_tokens[i].cpu().numpy().tolist()
                fw.write(' '.join(raw_words[i])+'\n')
                fw.write('Pred: ')
                for k,v in pred_pairs[i].items():
                    entity_span_ind_list =[]
                    for kk in k:
                        entity_span_ind_list.append(cur_src_token[kk-word_start_index])
                    entity_span = tokenizer.decode(entity_span_ind_list)
                    
                    region_pred, entity_type_ind = v
                    entity_type = ids2label[entity_type_ind[0]]
                    
                    fw.write('('+entity_span+' , '+ str(region_pred)+' , '+entity_type+' ) ')
                fw.write('\n')
                fw.write(' GT : ')
                for k,v in target_pairs[i].items():
                    entity_span_ind_list =[]
                    for kk in k:
                        entity_span_ind_list.append(cur_src_token[kk-word_start_index])
                    entity_span = tokenizer.decode(entity_span_ind_list)
        
                    region_pred, entity_type_ind = v
                    entity_type = ids2label[entity_type_ind[0]]

                    fw.write('('+entity_span+' , '+ str(region_pred)+' , '+entity_type+' ) ')
                fw.write('\n\n')
        res = metric.get_metric()  
        fw.write(str(res))
    return res



max_dev_f = 0.
max_test_f = 0.
best_dev = {}
best_test = {}
best_dev_corresponding_test = {}

for train_idx in range(args.n_epochs):
    print("-"*12+"Epoch: "+str(train_idx)+"-"*12)

    model.train()
    train_loss, train_region_loss = Training(args,train_idx=train_idx,train_data=train_dataset, model=model, device=device,
                                                optimizer=optimizer)
    

    model.eval()
    dev_res = Inference(args,eval_data=eval_dataset, model=model, device=device, metric = metric)
    dev_f = dev_res['f']
    print("dev: "+str(dev_res))

   
    test_res = Inference(args,eval_data=test_dataset, model=model, device=device, metric = metric)
    
    
    test_f = test_res['f']
    print("test: "+str(test_res))

    train_res = Inference(args,eval_data=train_dataset, model=model, device=device, metric = metric)
    train_f = train_res['f']
    print("train: "+str(train_res))



    if dev_f >= max_dev_f:
        max_dev_f = dev_f 
        if args.save_model:
            model_to_save = model.module if hasattr(model, 'module') else model  
            torch.save(model_to_save.state_dict(), args.save_path)
        best_dev = dev_res
        best_dev['epoch'] = train_idx
        best_dev_corresponding_test = test_res
        best_dev_corresponding_test['epoch'] = train_idx
        
   
    if test_f >= max_test_f:
        max_test_f = test_f 
        best_test = test_res
        best_test['epoch'] = train_idx

print("                   best_dev: "+str(best_dev))
print("best_dev_corresponding_test: "+str(best_dev_corresponding_test))
print("                  best_test: "+str(best_test))

if args.save_path and args.save_model:
    print("-"*12+'Predict'+'-'*12)
    ids2label = {2+i:l for i,l in enumerate(mapping2id.keys())}

    model_path = args.save_path.rsplit('/')
    args.pred_output_file = '/'.join(model_path[:-1])+'/pred_'+model_path[-1]+'.txt'

    model.load_state_dict(torch.load(args.save_path))
    model.to(device)

    print(test_dataset[:3])
    test_dataset.set_target('raw_words', 'raw_target')


    model.eval()
    test_res = Predict(args,eval_data=test_dataset, model=model, device=device, metric = metric,tokenizer=tokenizer,ids2label=ids2label)
    test_f = test_res['f']
    print("test: "+str(test_res))