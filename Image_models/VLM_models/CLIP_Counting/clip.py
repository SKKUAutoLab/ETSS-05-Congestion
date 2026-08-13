import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from clip_count_utils import run_on_my_data_img_retrievel
from utils import get_file_name, run_on_my_data

MODEL_NAMES = {
    "clip_base_32": "openai/clip-vit-base-patch32",
    "clip_base_16": "openai/clip-vit-base-patch16",
    "clip_large_14": "openai/clip-vit-large-patch14",
}

# model_name= "openai/clip-vit-base-patch32" # "openai/clip-vit-large-patch14" #"openai/clip-vit-base-patch32" "openai/clip-vit-base-patch16"
# model_name="openai/clip-vit-large-patch14"
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# device="cuda"
# model = CLIPModel.from_pretrained(model_name).to(device)
# model.requires_grad=False
# processor = CLIPProcessor.from_pretrained(model_name)


def image_retrievel(model_name,ref_obj,sample_size,augmented_data,local_directory,device="cpu"):
    model_name = MODEL_NAMES.get(model_name, model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.requires_grad=False
    processor = CLIPProcessor.from_pretrained(model_name)

    normalize=False
    # sample_size = len(augmented_data['dogs'][2])
    num_classes = 4
    linear_shift=True
    task = 'image_retrievel'
    start_with_target_with_num = True


    all_probs_by_factors = []
    all_mean_probs_by_factors = []
    all_probs_by_target = []
    for factor in [0,1]: # output original results and results after applying our method
        # all_probs_by_target = []
        all_mean_probs_by_target = []
        for target in augmented_data.keys():
            all_probs_per_class=run_on_my_data_img_retrievel(
                model=model,
                processor=processor,
                target_data= augmented_data[target],
                target=target,
                ref=ref_obj,
                normalize=normalize,
                device=device,
                factor=factor,
                sample_size=sample_size,
                num_classes=num_classes,
                linear_shift=linear_shift,
                start_with_target_with_num=start_with_target_with_num)
            mean_prob = np.mean([all_probs_per_class[i][i] for i in range(len(all_probs_per_class))])
            all_mean_probs_by_target.append(mean_prob)
            # all_probs_by_target.append(all_probs_per_class)
        # all_probs_by_factors.append(all_probs_by_target)
        all_mean_probs_by_factors.append(all_mean_probs_by_target)

    # pb_pd = pd.DataFrame(all_probs_by_factors,columns=list(augmented_data.keys()))
    # pb_pd.index = factors_list[1:]
    # pb_pd.to_csv(f"csv/final/{fn}")

    mean_pb_pd = pd.DataFrame(all_mean_probs_by_factors,columns=list(augmented_data.keys()))
    mean_pb_pd.index = [[ele]*len(all_mean_probs_by_target) for ele in [0,1]]
    mean_pb_pd["average"] = np.array(all_mean_probs_by_factors).mean(axis=1)
    mean_pb_pd.to_csv(os.path.join(local_directory,get_file_name(task,model_name,ref_obj,data_name="custom_data",num_classes=num_classes)))


def img_clf(model_name,ref_obj,sample_size,augmented_data,local_directory,device="cpu"):
    model_name = MODEL_NAMES.get(model_name, model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.requires_grad=False
    processor = CLIPProcessor.from_pretrained(model_name)
    normalize=False
    num_classes = 4
    linear_shift=True
    factors_list = [0,0.2]
    task = 'img_clf'
    start_with_target_with_num = True

    acc_by_factors=[]
    for factor in factors_list:
        acc_by_target=[]
        for target in augmented_data.keys():
            _,_,acc=run_on_my_data(
                model=model,
                processor=processor,
                target_data= augmented_data[target],
                target=target,
                ref=ref_obj,
                normalize=normalize,
                device=device,
                factor=factor,
                sample_size=sample_size,
                num_classes=num_classes,
                linear_shift=linear_shift,
                start_with_target_with_num=start_with_target_with_num)
            acc_by_target.append(acc)
        acc_by_factors.append(acc_by_target)
    acc_pd = pd.DataFrame(np.array(acc_by_factors),columns=list(augmented_data.keys()))
    acc_pd.index = factors_list
    acc_pd["average"] = np.array(acc_by_factors).mean(axis=1)
    acc_pd.to_csv(os.path.join(local_directory,get_file_name(task,model_name,ref_obj,data_name="",num_classes="")))


def _sta_bin_prompts(ref_obj, thresholds):
    ref_obj = ref_obj or "crowd"
    if ref_obj.lower() in {"crowd", "crowds"}:
        subject = "people in a crowd"
    else:
        subject = ref_obj

    rounded = [int(round(value)) for value in thresholds]
    return [
        f"a photo with fewer than {rounded[0]} {subject}",
        f"a photo with between {rounded[0]} and {rounded[1]} {subject}",
        f"a photo with between {rounded[1]} and {rounded[2]} {subject}",
        f"a photo with more than {rounded[2]} {subject}",
    ]


def img_clf_sta(model_name, ref_obj, crowd_data, local_directory, device="cpu", sample_size=None):
    model_name = MODEL_NAMES.get(model_name, model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    model.requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(model_name)

    samples = crowd_data["samples"]
    if sample_size is not None:
        samples = samples[:sample_size]
    thresholds = crowd_data["thresholds"]
    prompts = _sta_bin_prompts(ref_obj, thresholds)

    text_inputs = processor(text=prompts, images=None, return_tensors="pt", padding=True)
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    with torch.no_grad():
        text_embeds = model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    rows = []
    correct = 0
    for sample in samples:
        with Image.open(sample["image_path"]) as image:
            image_inputs = processor(text=None, images=image.convert("RGB"), return_tensors="pt")
        image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
        with torch.no_grad():
            image_embeds = model.get_image_features(**image_inputs)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            logits = torch.matmul(image_embeds, text_embeds.t()) * model.logit_scale.exp()
            probs = logits.softmax(dim=1).squeeze(0).detach().cpu().numpy()

        pred = int(np.argmax(probs))
        label = int(sample["label"])
        correct += int(pred == label)
        row = {
            "image_path": sample["image_path"],
            "count": sample["count"],
            "label": label,
            "prediction": pred,
            "correct": pred == label,
        }
        for idx, prompt in enumerate(prompts):
            row[f"class_{idx}_prompt"] = prompt
            row[f"class_{idx}_prob"] = float(probs[idx])
        rows.append(row)

    acc = round((correct / len(samples)) * 100, 4) if samples else 0.0
    results = pd.DataFrame(rows)
    results_path = os.path.join(
        local_directory,
        get_file_name(
            "sta_img_clf",
            model_name,
            ref_obj,
            data_name=crowd_data["split"],
            num_classes=len(prompts),
        ),
    )
    results.to_csv(results_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "dataset_root": crowd_data["root"],
                "split": crowd_data["split"],
                "num_samples": len(samples),
                "accuracy": acc,
                "thresholds": thresholds,
                "prompts": prompts,
                "results_path": results_path,
            }
        ]
    )
    summary_path = os.path.join(
        local_directory,
        get_file_name(
            "sta_img_clf_summary",
            model_name,
            ref_obj,
            data_name=crowd_data["split"],
            num_classes=len(prompts),
        ),
    )
    summary.to_csv(summary_path, index=False)
    print(f"STA classification accuracy: {acc}% ({correct}/{len(samples)})")
    print(f"Saved results to {results_path}")
    return acc
