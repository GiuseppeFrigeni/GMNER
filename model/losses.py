
# from fastNLP import LossBase
import torch.nn.functional as F
#from fastNLP import seq_len_to_mask
from .utils import seq_len_to_mask
import torch


# class Seq2SeqLoss(LossBase):
#     def __init__(self):
#         super().__init__()

def get_loss(tgt_tokens, tgt_seq_len, pred, region_pred,region_label,use_kl = True):
    
    tgt_seq_len = tgt_seq_len - 1  ### 不算开始符号
    mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1) - 1).eq(0)
    tgt_tokens = tgt_tokens[:, 1:].masked_fill(mask, -100)   ### 处理之后没有开始0，[  57,   58,   59,   60,    2,    1, -100, -100]

    if torch.isnan(pred).any() or torch.isinf(pred).any():
        print("LOSS_FN_DEBUG: 'pred' (input to cross_entropy) contains NaN/Inf.")
    if torch.isnan(tgt_tokens).any(): # Should not happen with -100 fill
        print("LOSS_FN_DEBUG: 'tgt_tokens' (input to cross_entropy) contains NaN.")

    loss = F.cross_entropy(target=tgt_tokens, input=pred.transpose(1, 2))

   
    
    region_mask = region_label[:,:,:-1].sum(dim=-1).gt(0)   ## only for related 

    if region_pred is not None and region_mask.sum()!=0:
        # --- START TEMPORARY MODIFICATION FOR DEBUGGING ---
        #print("LOSS_FN_DEBUG: SKIPPING KLDiv region_loss, setting to 0.0 for testing.")
        region_loss = torch.tensor(0.0, device=loss.device, requires_grad=True) 
        # --- END TEMPORARY MODIFICATION FOR DEBUGGING ---
        """
        if use_kl:
            bbox_num = region_pred.size(-1)
           
            target_for_kl = region_label[region_mask][:,:-1]
            target_for_kl = target_for_kl.to(region_pred.dtype).detach()
            
            print(f"LOSS_FN_DEBUG: target_for_kl (dtype: {target_for_kl.dtype}) min: {target_for_kl.min()}, max: {target_for_kl.max()}, has_zeros: {(target_for_kl == 0).any()}, sum_along_last_dim_example: {target_for_kl[0].sum() if target_for_kl.numel() > 0 else 'N/A'}")
            log_softmax_output = F.log_softmax(region_pred, dim=-1) # Calculate once for print and use
            print(f"LOSS_FN_DEBUG: input_to_kl (log_softmax output, dtype: {log_softmax_output.dtype}) min: {log_softmax_output.min()}, max: {log_softmax_output.max()}")
        
            if not torch.allclose(target_for_kl.sum(dim=-1), torch.ones_like(target_for_kl.sum(dim=-1)), atol=1e-5): # adjust atol for float32
                print(f"LOSS_FN_DEBUG: WARNING! target_for_kl does not sum to 1 for all samples after dtype conversion.")
                # print(target_for_kl.sum(dim=-1))
            # --- Manual Gradient Check ---
            softmax_rp = F.softmax(region_pred, dim=-1)
            manual_grad_approx = softmax_rp - target_for_kl # This is grad w.r.t. logits
        
            if torch.isnan(manual_grad_approx).any() or torch.isinf(manual_grad_approx).any():
                print("LOSS_FN_DEBUG: manual_grad_approx (softmax(logits) - target) CONTAINS NaN/Inf!")
                print(f"  softmax_rp min: {softmax_rp.min()}, max: {softmax_rp.max()}, hasNaN: {torch.isnan(softmax_rp).any()}")
                print(f"  target_for_kl min: {target_for_kl.min()}, max: {target_for_kl.max()}, hasNaN: {torch.isnan(target_for_kl).any()}")
            else:
                print("LOSS_FN_DEBUG: manual_grad_approx (softmax(logits) - target) is FINITE.")
            # --- End Manual Gradient Check ---
            # At the start of get_loss
            print(f"LOSS_FN_DEBUG: region_label.requires_grad: {region_label.requires_grad}")
            # Inside use_kl branch
            print(f"LOSS_FN_DEBUG: region_pred.requires_grad: {region_pred.requires_grad}")
            print(f"LOSS_FN_DEBUG: target_for_kl.requires_grad: {target_for_kl.requires_grad}")

            region_loss = F.kl_div(input=log_softmax_output, target=target_for_kl, reduction='batchmean')

        ## BCE
        else:
           
            region_label = region_label[region_mask][:,:-1]
            pos_tag = region_label.new_full(region_label.size(),fill_value = 1.)
            neg_tag = region_label.new_full(region_label.size(),fill_value = 0.)
            BCE_target = torch.where(region_label > 0,pos_tag,neg_tag)

            if torch.isnan(region_pred).any() or torch.isinf(region_pred).any():
                print("LOSS_FN_DEBUG: 'region_pred' (input to BCE) contains NaN/Inf.")
            if torch.isnan(BCE_target).any() or torch.isinf(BCE_target).any(): # BCE_target should be 0s and 1s
                print("LOSS_FN_DEBUG: 'BCE_target' (input to BCE) contains NaN/Inf.")

            bbox_num = region_pred.size(-1)
            sample_weight = region_pred.new_full((bbox_num,),fill_value=1.)
            region_loss = F.binary_cross_entropy_with_logits(region_pred, target=BCE_target ,weight =  sample_weight)
            """
    else:
        region_loss = torch.tensor(0.,requires_grad=True).to(loss.device)
    
    return loss , region_loss

