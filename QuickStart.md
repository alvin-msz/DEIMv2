## Quick Start Command
### train 
```shell
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/deimv2/deimv2_xxx.yml --use-amp --seed=0 -u train_dataloader.num_workers=0 val_dataloader.num_workers=0
```
### val
```shell
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/deimv2/deimv2_xxx.yml --test-only -r outputs/xxx/best_stg2.pth -u val_dataloader.num_workers=0
```
### export pth to onnx
```shell
python tools/deployment/export_onnx_fixed.py --check -c configs/deimv2/deimv2_xxx.yml -r outputs/xxx/best_stg2.pth --simplify --opset 16 --dynamic
```
### inference img demo
```shell
python tools/inference/torch_inf.py -c configs/deimv2/deimv2_xxx.yml -r outputs/xxx/best_stg2.pth --input xxx.jpg --device cuda:0
```
```shell
python tools/inference/onnx_inf.py --onnx outputs/deimv2_hgnetv2_n_xxx/best_stg2.onnx --input xxx.jpg --model-size n
```