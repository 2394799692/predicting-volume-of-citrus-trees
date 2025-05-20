import argparse
import os
import re
import time

import numpy as np
import torch
from PIL import Image
import cv2
from matplotlib import pyplot as plt

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything
from segment_anything import sam_model_registry, SamPredictor
from segment_anything_hq import sam_model_registry as sam_hq_model_registry


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]
    boxes_filt = boxes_filt[filt_mask]

    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def show_mask(mask, ax, random_color=False):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color else np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list, name, original_image_size):
    value = 0
    mask_img = torch.zeros(mask_list.shape[-2:], dtype=torch.uint8)
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    mask_img = mask_img.numpy()

    mask_img_resized = cv2.resize(mask_img, (original_image_size[0], original_image_size[1]), interpolation=cv2.INTER_NEAREST)

    # Normalize mask to range [0, 255] for visualization
    mask_img_resized = cv2.normalize(mask_img_resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    mask_img_resized = mask_img_resized.astype(np.uint8)

    output_path = os.path.join(output_dir, name)
    cv2.imwrite(output_path, mask_img_resized)



def drawContours(path='outputs/mask.jpg', out='outputs/output_contours.png'):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result[:, :] = (255, 255, 255)
    cv2.drawContours(result, contours, -1, (0, 0, 0), thickness=5)
    flag = cv2.imwrite(out, result)
    if not flag:
        print("Error: cv2.imread() failed to read the image.")


def remove_small_points(img_pth):
    from skimage.measure import label, regionprops
    img_gray = cv2.imread(img_pth, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img_gray, 127, 255, type=cv2.THRESH_BINARY)
    img_label, num = label(thresh, connectivity=1, background=0, return_num=True)
    props = regionprops(img_label)
    resMatrix = np.zeros(img_label.shape).astype(np.uint8)
    threshold_area = 0
    for i in range(len(props)):
        if props[i].area > threshold_area:
            threshold_area = props[i].area
    for i in range(len(props)):
        if props[i].area >= threshold_area:
            tmp = (img_label == i + 1).astype(np.uint8)
            resMatrix += tmp
    resMatrix *= 255
    return resMatrix


def is_image_by_extension(file_path):
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.ico'}
    filename, file_extension = os.path.splitext(file_path)
    return file_extension.lower() in allowed_extensions


def remove_chinese_and_rename(file_path):
    non_chinese_filename = re.sub(r'(?<!\.)[\u4e00-\u9fff]+', '', os.path.basename(file_path))
    if non_chinese_filename:
        new_filepath = os.path.join(os.path.dirname(file_path), non_chinese_filename)
        os.rename(file_path, new_filepath)
        return new_filepath
    return None


def extract_and_save_contour_region(original_image_path, contour_image_path, output_path):
    original_image = cv2.imread(original_image_path)
    contour_image = cv2.imread(contour_image_path)
    contour_image = cv2.resize(contour_image, (original_image.shape[1], original_image.shape[0]))
    contour_gray = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(contour_gray, 127, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    white_background = np.ones_like(original_image) * 255
    region_of_interest = cv2.bitwise_and(original_image, original_image, mask=mask)
    white_background[mask == 255] = region_of_interest[mask == 255]
    cv2.imwrite(output_path, white_background)


def model_test(file_path):
    model = load_model(config_file, grounded_checkpoint, device=device)
    for filename in os.listdir(file_path):
        time_start = time.time()
        image_path = os.path.join(file_path, filename)
        if not os.path.isfile(image_path):
            continue
        if not is_image_by_extension(filename):
            print(filename + '不是图片!')
            continue

        print(image_path)
        file_dir, file_extension = os.path.split(image_path)
        file_name, file_ext = os.path.splitext(os.path.basename(image_path))
        non_chinese_filename = re.sub(r'[\u4e00-\u9fff]', '', file_name)
        if non_chinese_filename:
            new_filename = non_chinese_filename + file_ext
            new_filepath = os.path.join(file_dir, new_filename)
            os.rename(image_path, new_filepath)
            filename = new_filename
            image_path = new_filepath
            print(f'文件已重命名为 "{new_filepath}"')
        else:
            print('文件名中没有中文字符或重命名失败。')

        image_pil, image = load_image(image_path)
        original_image_size = (image_pil.width, image_pil.height)

        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, device=device
        )
        if len(boxes_filt) == 0:
            print(filename + ' boxes为空----')
            continue

        if use_sam_hq:
            predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
        else:
            predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(device),
            multimask_output=False,
        )
        if len(masks) == 0:
            print(filename + ' masks为空----')
            continue

        min_area = float('inf')
        min_box_index = -1
        for i, box in enumerate(boxes_filt):
            width = box[2] - box[0]
            height = box[3] - box[1]
            area = width * height
            if area < min_area:
                min_area = area
                min_box_index = i

        boxes_filt = boxes_filt[min_box_index: min_box_index + 1]
        masks = masks[min_box_index: min_box_index + 1]

        # Create a copy of the image to draw the output
        output_image = image.copy()
        for mask in masks:
            mask_np = mask.cpu().numpy()[0]
            # Create an RGBA mask image
            mask_rgba = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
            mask_rgba[mask_np == 1, :3] = [0, 255, 0]  # Green color for the mask
            mask_rgba[mask_np == 1, 3] = 153  # Alpha channel (153/255 = 0.6)

            # Overlay the mask on the output image with transparency
            output_image = cv2.addWeighted(output_image, 1.0, mask_rgba[:, :, :3], 0.6, 0)

        for box in boxes_filt:
            x0, y0, x1, y1 = box.int().numpy()
            cv2.rectangle(output_image, (x0, y0), (x1, y1), (0, 0, 255), 2)  # Red color for the box

        cv2.imwrite(os.path.join(temp, filename), output_image)

        save_mask_data(pic1, masks, boxes_filt, pred_phrases, name=filename, original_image_size=original_image_size)

        res_img = remove_small_points(os.path.join(pic1, filename))
        cv2.imwrite('outputs/temp.png', res_img)

        contour_output_path = os.path.join(pic2, filename)
        drawContours('outputs/temp.png', contour_output_path)
        os.remove('outputs/temp.png')

        extracted_region_output_path = os.path.join(pic2, "extracted_" + filename)
        extract_and_save_contour_region(image_path, contour_output_path, extracted_region_output_path)

        time_end = time.time()
        time_sum = time_end - time_start
        print('------------------------------------------------')
        print('共用时：' + time_sum.__str__() + 's')
        print('------------------------------------------------')


if __name__ == "__main__":
    # 定义参数字典
    params = {
        'config': 'data/GroundingDINO_SwinT_OGC.py', #模型配置文件路径
        'grounded_checkpoint': 'data/groundingdino_swint_ogc.pth', #基础检测模型路径
        'sam_hq_checkpoint': 'data/sam_hq_vit_b.pth',  #分割模型路径，选择不同的分割模型需要修改下面的模型类型
        'input_files': 'inputs/image',  #分割图片文件夹路径
        'box_threshold': 0.3, #模型检测框阈值参数
        'text_threshold': 0.25, #模型文本对应阈值参数
        'text_prompt': 'single largest area tree in the middle.', #文本描述词
        'device': 'cuda' #使用cuda
    }

    # 直接使用参数字典中的值
    config_file = params['config']
    grounded_checkpoint = params['grounded_checkpoint']
    sam_version = 'vit_b'  # 使用的分割模型类型，vit_h为最优分割模型，vit_b为轻量级模型
    sam_checkpoint = None
    sam_hq_checkpoint = params['sam_hq_checkpoint']
    use_sam_hq = True
    file_path = params['input_files']
    text_prompt = params['text_prompt']
    box_threshold = params['box_threshold']
    text_threshold = params['text_threshold']
    device = params['device']

    pic1 = os.path.join('outputs', 'pic1')
    pic2 = os.path.join('outputs', 'pic2')
    temp = os.path.join('outputs', 'temp')

    # 例如，创建输出目录
    os.makedirs(pic1, exist_ok=True)
    os.makedirs(pic2, exist_ok=True)
    os.makedirs(temp, exist_ok=True)

    # 然后调用你的模型测试函数
    model_test(file_path)
