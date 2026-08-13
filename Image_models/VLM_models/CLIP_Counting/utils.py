import numpy as np
import pandas as pd
import torch

import clip_count_utils

def run_on_my_data(model,processor,target_data,target,ref,normalize,device,factor,sample_size,num_classes=4,linear_shift=True,start_with_target_with_num=False,normalize_before_scoring=False,auto_factor=False):
    ref_aug_sentences=[f"{word} {clip_count_utils.object_name(ref)}" for word in clip_count_utils.NUMBER_WORDS[:num_classes]]
    target_aug_sentences=[f"{word} {clip_count_utils.object_name(target)}" for word in clip_count_utils.NUMBER_WORDS[:num_classes]]

    ref_diff,ref_prompt_single=clip_count_utils.get_ref_difference(
        ref_aug_sentences=ref_aug_sentences,
        ref_object=[clip_count_utils.object_name(ref)],
        model=model,
        processor=processor,
        device=device,
        normalize=normalize
    )
    target_text_sample={}
    target_text_sample["target_obj_embeds"],target_text_sample["target_obj_aug_embeds"]=clip_count_utils.get_target_rep(
        target_object=[clip_count_utils.object_name(target)],
        target_aug_sentences=target_aug_sentences,
        model=model,
        processor=processor,
        device=device,
        normalize=normalize
    )

    # assert (auto_factor and factor is None) or (not auto_factor and factor is not None), "only one in auto_factor and factor can be none"
    # # if auto_factor:
    # if start_with_target_with_num:
    #     factor_norm = target_text_sample["target_obj_aug_embeds"].norm(p=2,dim=1,keepdim=True)/ref_prompt_single.norm(p=2,dim=1,keepdim=True)
    # else:
    #     factor_norm = target_text_sample["target_obj_embeds"].norm(p=2,dim=1,keepdim=True)/ref_prompt_single.norm(p=2,dim=1,keepdim=True)

    # print(ref_diff.size(),ref_prompt_single.size())
    ref_diff_projection = (torch.mm(ref_diff, ref_prompt_single.t()) / torch.mm(ref_prompt_single, ref_prompt_single.t()).squeeze()) * ref_prompt_single
    # ref_diff_aligned_A = (torch.mm(ref_diff-ref_diff_projection, target_text_sample["target_obj_embeds"].t()) / torch.mm(target_text_sample["target_obj_embeds"], target_text_sample["target_obj_embeds"].t()).squeeze()) * target_text_sample["target_obj_embeds"]
    ref_diff_aligned_B = (torch.sum(ref_diff-ref_diff_projection * target_text_sample["target_obj_aug_embeds"], dim=1, keepdim=True)/torch.sum(target_text_sample["target_obj_aug_embeds"] * target_text_sample["target_obj_aug_embeds"], dim=1, keepdim=True))*target_text_sample["target_obj_aug_embeds"]
    # tar_diff = target_text_sample["target_obj_aug_embeds"]-target_text_sample["target_obj_embeds"]
    # tar_diff_projection = (torch.mm(tar_diff, target_text_sample["target_obj_embeds"].t()) / torch.mm(target_text_sample["target_obj_embeds"], target_text_sample["target_obj_embeds"].t()).squeeze()) * target_text_sample["target_obj_embeds"]
    # tar_diff_aligned = (torch.mm(tar_diff-tar_diff_projection, target_text_sample["target_obj_embeds"].t()) / torch.mm(target_text_sample["target_obj_embeds"], target_text_sample["target_obj_embeds"].t()).squeeze()) * target_text_sample["target_obj_embeds"]

    # # print(in_group_variance(tar_diff),in_group_variance(tar_diff_projection),in_group_variance(tar_diff-tar_diff_projection))
    # factor = tar_diff_projection.norm(p=2,dim=1).mean() / tar_diff_projection.norm(p=2,dim=1)
    # # print("====")
    # # print(ref_diff.norm(p=2,dim=1))
    # print(ref_diff_projection.size(),ref_diff.size(),ref_prompt_single.size())
    ref_diff = ref_diff - ref_diff_projection - ref_diff_aligned_B #+ (1-factor) * (tar_diff - tar_diff_projection - tar_diff_aligned)
    # print(tar_diff_projection.size(),factor[:,None].size())

    # ref_diff = tar_diff - tar_diff_projection #- tar_diff_aligned
    # print(tar_diff_aligned.norm(p=2,dim=1))
    # ref_diff_aligned = (torch.sum(ref_diff * target_text_sample["target_obj_aug_embeds"], dim=1, keepdim=True)/torch.sum(target_text_sample["target_obj_aug_embeds"] * target_text_sample["target_obj_aug_embeds"], dim=1, keepdim=True))*target_text_sample["target_obj_aug_embeds"]
    # tar_diff = target_text_sample["target_obj_aug_embeds"] - target_text_sample["target_obj_embeds"]
    # ref_diff_aligned = (torch.sum(ref_diff *tar_diff , dim=1, keepdim=True)/torch.sum(tar_diff * tar_diff, dim=1, keepdim=True))*tar_diff

    # ref_diff_aligned_A = (torch.mm(ref_diff, target_text_sample["target_obj_embeds"].t()) / torch.mm(target_text_sample["target_obj_embeds"], target_text_sample["target_obj_embeds"].t()).squeeze()) * target_text_sample["target_obj_embeds"]

    # ref_diff_aligned_B = (torch.sum(ref_diff * target_text_sample["target_obj_aug_embeds"], dim=1, keepdim=True)/torch.sum(target_text_sample["target_obj_aug_embeds"] * target_text_sample["target_obj_aug_embeds"], dim=1, keepdim=True))*target_text_sample["target_obj_aug_embeds"]
    # #
    # ref_diff_aligned = ref_diff_aligned_A + ref_diff_aligned_B
    # print(ref_diff_aligned.norm(p=2,dim=1))
    # ref_diff = ref_diff - ref_diff_aligned
    # print(ref_diff.norm(p=2,dim=1))
    # print(ref_diff.size(),target_text_sample["target_obj_embeds"].size(),target_text_sample["target_obj_aug_embeds"].size())


    # numerator = -torch.mm(target_text_sample["target_obj_embeds"], target_text_sample["target_obj_aug_embeds"].t()).squeeze()
    # denominator = torch.mm(target_text_sample["target_obj_embeds"], ref_diff.t()).squeeze()
    # coef = numerator / denominator

    # print(coef.size(),ref_diff.size())

    merged_text_embeds = clip_count_utils.apply_reff_diff(target_text_sample["target_obj_aug_embeds"],target_text_sample["target_obj_aug_embeds"],ref_diff,1,linear_shift,start_with_target_with_num)

    # if normalize_before_scoring:
    merged_text_embeds=merged_text_embeds/merged_text_embeds.norm(p=2,dim=-1,keepdim=True)

    flat_predictions=[]
    flat_labels=[]
    for number in range(2,num_classes+2):
        if sample_size is None:
            selected_data = target_data[number]
        else:
            selected_data = target_data[number][:sample_size]
        predictions=[]
        for sample in selected_data:
            pixel_values=processor(text=None, images=sample["img"], return_tensors="pt", padding=True)["pixel_values"] # torch.Size([1, 3, 224, 224])
            image_embeds = clip_count_utils.get_image_embeds(
                model=model,
                pixel_values=pixel_values.to(device),
                device=device
            )
            _,logits_per_image= clip_count_utils.get_logits(model,merged_text_embeds,image_embeds)
            probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label prob
            predictions.append(torch.argmax(probs).item()+2)
        flat_predictions+=predictions
        flat_labels+=[number]*len(predictions)

    acc=np.mean(np.array(flat_predictions)==np.array(flat_labels)).round(4)*100

    return flat_predictions,flat_labels,acc

def get_file_name(task,model_name,ref,data_name="",num_classes=""):
    model_part = model_name.split("/")[-1]
    return f"{task}_{model_part}_{ref}_{data_name}_{num_classes}.csv"
