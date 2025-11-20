#!/usr/bin/env python3
import json
import time
import os
import sys
import argparse
import numpy as np
from collections import defaultdict, deque
from datetime import datetime

# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class TrainingMonitor:
    def __init__(self, log_file, max_points=1000, update_interval=2, save_plots=True):
        self.log_file = log_file
        self.max_points = max_points
        self.update_interval = update_interval
        self.save_plots = save_plots
        
        # 存储数据
        self.data = defaultdict(lambda: deque(maxlen=max_points))
        self.epochs = deque(maxlen=max_points)
        self.timestamps = deque(maxlen=max_points)
        
        # 记录文件位置
        self.file_position = 0
        
        # 输出目录
        self.output_dir = os.path.dirname(log_file)
        self.plot_path = os.path.join(self.output_dir, 'training_monitor.png')
        
        # 设置图形
        self.setup_plots()
        
    def setup_plots(self):
        """设置matplotlib图形"""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle('Real-time Training Monitor', fontsize=16, color='white')
        
        # 创建子图
        gs = self.fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 主要损失
        self.ax_main = self.fig.add_subplot(gs[0, :])
        self.ax_main.set_title('Main Losses', color='white')
        self.ax_main.set_xlabel('Epoch')
        self.ax_main.set_ylabel('Loss')
        self.ax_main.grid(True, alpha=0.3)
        
        # 分类损失
        self.ax_cls = self.fig.add_subplot(gs[1, 0])
        self.ax_cls.set_title('Classification Losses', color='white')
        self.ax_cls.set_xlabel('Epoch')
        self.ax_cls.set_ylabel('Loss')
        self.ax_cls.grid(True, alpha=0.3)
        
        # 回归损失
        self.ax_reg = self.fig.add_subplot(gs[1, 1])
        self.ax_reg.set_title('Regression Losses', color='white')
        self.ax_reg.set_xlabel('Epoch')
        self.ax_reg.set_ylabel('Loss')
        self.ax_reg.grid(True, alpha=0.3)
        
        # 学习率和mAP
        self.ax_lr = self.fig.add_subplot(gs[2, 0])
        self.ax_lr.set_title('Learning Rate', color='white')
        self.ax_lr.set_xlabel('Epoch')
        self.ax_lr.set_ylabel('Learning Rate')
        self.ax_lr.grid(True, alpha=0.3)
        
        self.ax_map = self.fig.add_subplot(gs[2, 1])
        self.ax_map.set_title('mAP', color='white')
        self.ax_map.set_xlabel('Epoch')
        self.ax_map.set_ylabel('mAP')
        self.ax_map.grid(True, alpha=0.3)
        
        # 颜色映射
        self.colors = plt.cm.Set3(np.linspace(0, 1, 20))
        
    def read_new_lines(self):
        """读取日志文件中的新行"""
        new_data = []
        try:
            with open(self.log_file, 'r') as f:
                f.seek(self.file_position)
                new_lines = f.readlines()
                self.file_position = f.tell()
                
                for line in new_lines:
                    line = line.strip()
                    if line and line.startswith('{') and line.endswith('}'):
                        try:
                            data = json.loads(line)
                            new_data.append(data)
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            print(f"Log file {self.log_file} not found")
        except Exception as e:
            print(f"Error reading log file: {e}")
            
        return new_data
    
    def update_data(self, new_entries):
        """更新数据"""
        for entry in new_entries:
            if 'epoch' in entry:
                epoch = entry['epoch']
                self.epochs.append(epoch)
                self.timestamps.append(datetime.now())
                
                # 更新所有损失数据
                for key, value in entry.items():
                    if isinstance(value, (int, float)) and key != 'epoch':
                        self.data[key].append(value)
                    elif isinstance(value, list) and key == 'test_coco_eval_bbox':
                        self.data[key].append(value)  # 特殊处理mAP数据
    
    def update_plots(self):
        """更新图形"""
        # 读取新数据
        new_entries = self.read_new_lines()
        if new_entries:
            self.update_data(new_entries)
        
        if not self.epochs:
            return
            
        epochs = list(self.epochs)
        
        # 清除所有子图
        for ax in [self.ax_main, self.ax_cls, self.ax_reg, self.ax_lr, self.ax_map]:
            ax.clear()
            ax.grid(True, alpha=0.3)
        
        # 重新设置标题和标签
        self.ax_main.set_title('Main Losses', color='white')
        self.ax_cls.set_title('Classification Losses', color='white')
        self.ax_reg.set_title('Regression Losses', color='white')
        self.ax_lr.set_title('Learning Rate', color='white')
        self.ax_map.set_title('mAP', color='white')
        
        for ax in [self.ax_main, self.ax_cls, self.ax_reg, self.ax_lr, self.ax_map]:
            ax.set_xlabel('Epoch')
        
        self.ax_main.set_ylabel('Loss')
        self.ax_cls.set_ylabel('Loss')
        self.ax_reg.set_ylabel('Loss')
        self.ax_lr.set_ylabel('Learning Rate')
        self.ax_map.set_ylabel('mAP@0.5')
        
        # 绘制主要损失
        main_losses = ['train_loss']
        color_idx = 0
        for loss_name in main_losses:
            if loss_name in self.data and len(self.data[loss_name]) > 0:
                values = list(self.data[loss_name])
                if len(values) == len(epochs):
                    self.ax_main.plot(epochs, values, 
                                    label=loss_name.replace('train_', ''), 
                                    color=self.colors[color_idx % len(self.colors)],
                                    linewidth=2)
                    color_idx += 1
        
        self.ax_main.legend()
        
        # 绘制分类损失
        cls_losses = [k for k in self.data.keys() if 'mal' in k or 'fgl' in k]
        color_idx = 0
        for loss_name in cls_losses[:6]:  # 限制显示数量
            if len(self.data[loss_name]) > 0:
                values = list(self.data[loss_name])
                if len(values) == len(epochs):
                    self.ax_cls.plot(epochs, values, 
                                   label=loss_name.replace('train_loss_', ''), 
                                   color=self.colors[color_idx % len(self.colors)],
                                   linewidth=1.5)
                    color_idx += 1
        
        if cls_losses:
            self.ax_cls.legend(fontsize=8)
        
        # 绘制回归损失
        reg_losses = [k for k in self.data.keys() if 'bbox' in k or 'giou' in k]
        color_idx = 0
        for loss_name in reg_losses[:6]:  # 限制显示数量
            if len(self.data[loss_name]) > 0:
                values = list(self.data[loss_name])
                if len(values) == len(epochs):
                    self.ax_reg.plot(epochs, values, 
                                   label=loss_name.replace('train_loss_', ''), 
                                   color=self.colors[color_idx % len(self.colors)],
                                   linewidth=1.5)
                    color_idx += 1
        
        if reg_losses:
            self.ax_reg.legend(fontsize=8)
        
        # 绘制学习率
        if 'train_lr' in self.data and len(self.data['train_lr']) > 0:
            values = list(self.data['train_lr'])
            if len(values) == len(epochs):
                self.ax_lr.plot(epochs, values, color='orange', linewidth=2)
                self.ax_lr.set_yscale('log')
        
        # 绘制mAP
        map_keys = [k for k in self.data.keys() if 'test_coco_eval_bbox' in k]
        if map_keys and len(self.data[map_keys[0]]) > 0:
            # test_coco_eval_bbox是一个列表，取第一个值作为mAP
            map_values = []
            for bbox_list in self.data[map_keys[0]]:
                if isinstance(bbox_list, list) and len(bbox_list) > 0:
                    map_values.append(bbox_list[1])  # mAP@0.5
                else:
                    map_values.append(0.0)
            
            if len(map_values) == len(epochs):
                self.ax_map.plot(epochs, map_values, color='green', linewidth=2)
        
        # 添加当前状态信息
        if epochs:
            current_epoch = epochs[-1]
            current_loss = list(self.data['train_loss'])[-1] if 'train_loss' in self.data else 0
            current_lr = list(self.data['train_lr'])[-1] if 'train_lr' in self.data else 0
            
            status_text = f"Epoch: {current_epoch} | Loss: {current_loss:.4f} | LR: {current_lr:.2e}"
            self.fig.suptitle(f'Real-time Training Monitor - {status_text}', 
                            fontsize=14, color='white')
        
        # 保存图片
        if self.save_plots:
            plt.tight_layout()
            plt.savefig(self.plot_path, dpi=100, bbox_inches='tight')
            print(f"Plot saved to: {self.plot_path}")
    
    def start_monitoring(self):
        """开始监控"""
        print(f"Starting real-time monitoring of {self.log_file}")
        print(f"Plots will be saved to: {self.plot_path}")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                self.update_plots()
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            # 最后保存一次
            self.update_plots()
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Real-time training monitor')
    parser.add_argument('--log-file', '-l', 
                       default='outputs/deimv2_hgnetv2_n_persion_yolo_label_tuning/log.txt',
                       help='Path to log file')
    parser.add_argument('--max-points', '-m', type=int, default=1000,
                       help='Maximum number of points to display')
    parser.add_argument('--update-interval', '-u', type=int, default=5,
                       help='Update interval in seconds')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"Log file {args.log_file} does not exist!")
        return
    
    monitor = TrainingMonitor(args.log_file, args.max_points, args.update_interval)
    monitor.start_monitoring()

if __name__ == '__main__':
    main()
