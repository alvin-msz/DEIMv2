import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig

# 添加类别映射
label_map = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorbike', 5: 'aeroplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'trafficlight',
    11: 'firehydrant', 12: 'streetsign', 13: 'stopsign', 14: 'parkingmeter',
    15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse',
    20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra',
    25: 'giraffe', 26: 'hat', 27: 'backpack', 28: 'umbrella', 29: 'shoe',
    30: 'eyeglasses', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sportsball', 38: 'kite', 39: 'baseballbat',
    40: 'baseballglove', 41: 'skateboard', 42: 'surfboard', 43: 'tennisracket',
    44: 'bottle', 45: 'plate', 46: 'wineglass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hotdog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'sofa',
    64: 'pottedplant', 65: 'bed', 66: 'mirror', 67: 'diningtable', 68: 'window',
    69: 'desk', 70: 'toilet', 71: 'door', 72: 'tv', 73: 'laptop',
    74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cellphone', 78: 'microwave',
    79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 83: 'blender',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddybear',
    89: 'hairdrier', 90: 'toothbrush', 91: 'hairbrush'
}

def extract_features_and_heatmap(model, images, layer_name='backbone'):
    """提取特征并生成热力图"""
    features = {}
    
    def hook_fn(module, input, output):
        features[layer_name] = output
        print(f"Feature type: {type(output)}")
        if isinstance(output, (list, tuple)):
            print(f"Feature list length: {len(output)}")
            for i, feat in enumerate(output):
                if isinstance(feat, torch.Tensor):
                    print(f"  Feature {i} shape: {feat.shape}")
        elif isinstance(output, torch.Tensor):
            print(f"Feature tensor shape: {output.shape}")
    
    # 注册hook到指定层
    if hasattr(model.model, 'backbone'):
        hook = model.model.backbone.register_forward_hook(hook_fn)
    else:
        # 如果结构不同，可能需要调整
        hook = list(model.model.children())[-2].register_forward_hook(hook_fn)
    
    # 前向传播
    with torch.no_grad():
        outputs = model.model(images)
    
    hook.remove()
    
    # 生成热力图
    if layer_name in features:
        feature_maps = features[layer_name]
        
        # 处理不同类型的特征输出
        if isinstance(feature_maps, (list, tuple)):
            # 如果是列表，取最后一个特征图
            feature_maps = feature_maps[-1]
        
        if isinstance(feature_maps, torch.Tensor):
            # 确保是4D张量 [B, C, H, W]
            if len(feature_maps.shape) == 4:
                heatmap = torch.mean(feature_maps, dim=1, keepdim=True)  # [B, 1, H, W]
            else:
                print(f"Unexpected feature map shape: {feature_maps.shape}")
                return outputs, None
        else:
            print(f"Unexpected feature map type: {type(feature_maps)}")
            return outputs, None
            
        return outputs, heatmap
    
    return outputs, None

def save_heatmap(heatmap, original_image, save_path):
    """保存特征热力图"""
    # 转换为numpy并归一化
    heatmap_np = heatmap.squeeze().cpu().numpy()
    heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min())
    
    # 调整大小到原图尺寸
    original_size = original_image.size
    heatmap_resized = cv2.resize(heatmap_np, original_size)
    
    # 应用颜色映射
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    
    # 与原图叠加
    original_np = np.array(original_image) / 255.0
    overlay = 0.6 * original_np + 0.4 * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    
    # 保存
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_resized, cmap='jet')
    plt.title('Feature Heatmap')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def draw_detections(image, labels, boxes, scores, thrh=0.45):
    """绘制检测框"""
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    
    # 处理不同的张量格式
    if isinstance(labels, torch.Tensor):
        if len(labels.shape) > 1:
            labels = labels.squeeze()
        if len(boxes.shape) > 2:
            boxes = boxes.squeeze()
        if len(scores.shape) > 1:
            scores = scores.squeeze()
    
    # 过滤低置信度检测
    mask = scores > thrh
    filtered_labels = labels[mask]
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    
    for j, (label, box, score) in enumerate(zip(filtered_labels, filtered_boxes, filtered_scores)):
        draw.rectangle(list(box), outline='red', width=2)
        # 使用类别名称而不是数字ID
        category_id = label.item()
        category_name = label_map.get(category_id, f'class_{category_id}')
        text = f"{category_name} @ {round(score.item(), 2)}"
        draw.text((box[0], box[1]), text=text, fill='blue', font=font)
    
    return image

def process_single_image(model, device, image_path, output_dir, size=(640, 640), vit_backbone=False):
    """处理单张图像"""
    # 加载图像
    original_image = Image.open(image_path).convert('RGB')
    w, h = original_image.size
    orig_size = torch.tensor([[w, h]]).to(device)
    
    # 预处理
    transforms = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
                if vit_backbone else T.Lambda(lambda x: x)
    ])
    im_data = transforms(original_image).unsqueeze(0).to(device)
    
    # 提取特征和检测
    model_outputs, heatmap = extract_features_and_heatmap(model, im_data)
    detection_outputs = model.postprocessor(model_outputs, orig_size)
    
    # 处理检测输出 - 根据postprocessor的输出格式调整
    if isinstance(detection_outputs, (list, tuple)) and len(detection_outputs) == 3:
        labels, boxes, scores = detection_outputs
    elif isinstance(detection_outputs, list) and len(detection_outputs) == 1:
        # 如果是单个字典格式
        result = detection_outputs[0]
        labels = result['labels']
        boxes = result['boxes'] 
        scores = result['scores']
    else:
        print(f"Unexpected detection output format: {type(detection_outputs)}")
        print(f"Detection output: {detection_outputs}")
        return
    
    # 保存热力图
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    if heatmap is not None:
        heatmap_path = os.path.join(output_dir, f"{base_name}_heatmap.jpg")
        save_heatmap(heatmap, original_image, heatmap_path)
        print(f"Feature heatmap saved: {heatmap_path}")
    
    # 保存检测结果
    detection_image = draw_detections(original_image.copy(), labels, boxes, scores)
    detection_path = os.path.join(output_dir, f"{base_name}_detection.jpg")
    detection_image.save(detection_path)
    print(f"Detection result saved: {detection_path}")

def main(args):
    """主函数"""
    cfg = YAMLConfig(args.config, resume=args.resume)
    
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')
    
    cfg.model.load_state_dict(state)
    
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.eval()
            self.postprocessor = cfg.postprocessor.eval()
        
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    device = torch.device(args.device)
    model = Model().to(device)
    img_size = cfg.yaml_cfg["eval_spatial_size"]
    vit_backbone = cfg.yaml_cfg.get('DINOv3STAs', False)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理输入
    if os.path.isfile(args.input):
        # 单张图像
        process_single_image(model, device, args.input, args.output_dir, img_size, vit_backbone)
    elif os.path.isdir(args.input):
        # 批量处理
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in os.listdir(args.input) 
                      if os.path.splitext(f)[1].lower() in image_extensions]
        
        print(f"Found {len(image_files)} images to process...")
        for img_file in image_files:
            img_path = os.path.join(args.input, img_file)
            process_single_image(model, device, img_path, args.output_dir, img_size, vit_backbone)
    else:
        print(f"Invalid input path: {args.input}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('--input', type=str, required=True, help='Input image file or directory')
    parser.add_argument('--output-dir', type=str, default='./outputs/heatmap', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    main(args)
