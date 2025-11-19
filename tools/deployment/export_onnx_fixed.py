"""
修复TracerWarning的ONNX导出脚本
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import torch
import torch.nn as nn
from engine.core import YAMLConfig

def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)
    
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
        cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()
    model.eval()
    
    # 减少内存使用
    img_size = cfg.yaml_cfg["eval_spatial_size"]
    data = torch.rand(1, 3, *img_size)  # 使用batch_size=1
    size = torch.tensor([[img_size[0], img_size[1]]])
    
    # 预热模型
    with torch.no_grad():
        _ = model(data, size)
    
    output_file = args.resume.replace('.pth', '.onnx') if args.resume else 'model.onnx'
    
    # 设置环境变量减少内存使用
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # 动态轴设置
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            'images': {0: 'batch_size'},
            'orig_target_sizes': {0: 'batch_size'}
        }
    
    torch.onnx.export(
        model,
        (data, size),
        output_file,
        input_names=['images', 'orig_target_sizes'],
        output_names=['labels', 'boxes', 'scores'],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        verbose=False,
        do_constant_folding=True,
        # 添加这些参数来处理TracerWarning
        training=torch.onnx.TrainingMode.EVAL,
        strip_doc_string=True,
    )
    
    print(f'Export completed: {output_file}')
    
    # 验证导出的模型
    if args.check:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print('ONNX model validation passed!')
    
    # 简化模型
    if args.simplify:
        import onnx
        import onnxsim
        print('Simplifying ONNX model...')
        
        # 设置输入形状用于简化
        input_shapes = None
        if args.dynamic:
            input_shapes = {'images': data.shape, 'orig_target_sizes': size.shape}
        else:
            input_shapes = {'images': [1, 3, *img_size], 'orig_target_sizes': [1, 2]}
        
        try:
            onnx_model_simplify, check = onnxsim.simplify(
                output_file, 
                test_input_shapes=input_shapes
            )
            onnx.save(onnx_model_simplify, output_file)
            print(f'Simplify onnx model {check}...')
        except Exception as e:
            print(f'Simplification failed: {e}')
            print('Model exported without simplification.')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True, type=str)
    parser.add_argument('--resume', '-r', required=True, type=str)
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    parser.add_argument('--check', action='store_true', help='Check exported ONNX model')
    parser.add_argument('--simplify', action='store_true', help='Simplify ONNX model using onnxsim')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic batch size')
    args = parser.parse_args()
    main(args)
