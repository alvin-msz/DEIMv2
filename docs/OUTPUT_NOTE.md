**Q**: 训练结束以后，权重文件中有：**`best_stg1.pth`** 和 **`best_stg2.pth`**，请问训练最好的一轮应该是哪个权重？<br>
**A**: 根据代码分析，**`best_stg2.pth` 是最终的最佳权重**，应该使用这个文件。

## 📋 两个权重文件的区别

````python path=engine/solver/det_solver.py mode=EXCERPT
if epoch >= self.train_dataloader.collate_fn.stop_epoch:
    dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
else:
    dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')
````

### **Stage 1 (best_stg1.pth)**
- **时间段**：第0轮到 `stop_epoch` 轮（在您的配置中是第148轮）
- **特点**：使用**强数据增强**阶段的最佳权重
  - Mosaic、MixUp、CopyBlend等增强技术
  - 多尺度训练
  - 更激进的数据变换

### **Stage 2 (best_stg2.pth)**  
- **时间段**：第148轮之后到训练结束（第160轮）
- **特点**：使用**无/弱数据增强**阶段的最佳权重
  - 关闭或减弱数据增强
  - 固定输入尺寸
  - 更接近推理时的数据分布

## 🎯 为什么选择 best_stg2.pth？

1. **更接近推理环境**：Stage 2的训练条件更接近实际推理时的数据分布
2. **精细调优结果**：在Stage 1基础上进行的精细调优
3. **更稳定的性能**：没有强数据增强的干扰，性能更稳定

## 📊 从您的配置看

````yaml path=configs/deimv2/deimv2_hgnetv2_n_person.yml mode=EXCERPT
epoches: 160
train_dataloader: 
  collate_fn:
    stop_epoch: 148    # Stage 1结束，Stage 2开始
````

- **Stage 1**：第0-147轮（148轮），使用强增强
- **Stage 2**：第148-159轮（12轮），使用弱/无增强
- **最佳权重**：`best_stg2.pth`（第148轮后的最佳性能）

## 🔄 训练过程中的权重切换

代码中还有一个有趣的机制：

````python path=engine/solver/det_solver.py mode=EXCERPT
if epoch == self.train_dataloader.collate_fn.stop_epoch:
    self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
    self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')
````

在第148轮时，模型会：
1. 加载 `best_stg1.pth` 作为起点
2. 重置EMA参数
3. 开始Stage 2的精细调优

## 💡 使用建议

### **推理时使用**
```bash
# 使用最佳权重进行推理
python tools/inference/torch_inf.py -c your_config.yml -r outputs/your_model/best_stg2.pth
```

### **模型转换时使用**
```bash
# 转换为部署格式
python tools/reference/convert_weight.py outputs/your_model/best_stg2.pth
```

### **继续训练时使用**
```bash
# 如果要继续训练，也应该从best_stg2.pth开始
python train.py -c your_config.yml --resume outputs/your_model/best_stg2.pth
```

## ⚠️ 特殊情况

如果训练在Stage 1阶段就停止了（比如第100轮），那么只会有 `best_stg1.pth`，这时就使用这个文件。

**总结**：优先使用 `best_stg2.pth`，如果没有这个文件，再使用 `best_stg1.pth`。
