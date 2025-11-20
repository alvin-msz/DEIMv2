#!/usr/bin/env python3
import json
import time
import os
import argparse
from collections import defaultdict, deque
from datetime import datetime

class TerminalMonitor:
    def __init__(self, log_file, max_points=100):
        self.log_file = log_file
        self.max_points = max_points
        self.data = defaultdict(lambda: deque(maxlen=max_points))
        self.file_position = 0
        
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
        except Exception as e:
            print(f"Error reading log file: {e}")
            
        return new_data
    
    def update_data(self, new_entries):
        """更新数据"""
        for entry in new_entries:
            if 'epoch' in entry:
                for key, value in entry.items():
                    if isinstance(value, (int, float)):
                        self.data[key].append(value)
    
    def display_status(self):
        """在终端显示状态"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 80)
        print("🚀 REAL-TIME TRAINING MONITOR")
        print("=" * 80)
        print(f"📁 Log file: {self.log_file}")
        print(f"⏰ Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)
        
        if not self.data:
            print("⏳ Waiting for training data...")
            return
        
        # 显示当前状态
        current_data = {}
        for key, values in self.data.items():
            if values:
                current_data[key] = values[-1]
        
        if 'epoch' in current_data:
            print(f"📊 Current Epoch: {current_data['epoch']}")
        
        if 'train_loss' in current_data:
            print(f"📉 Training Loss: {current_data['train_loss']:.6f}")
            
        if 'train_lr' in current_data:
            print(f"📈 Learning Rate: {current_data['train_lr']:.2e}")
        
        # 显示mAP
        map_keys = [k for k in current_data.keys() if 'test_coco_eval_bbox' in k]
        if map_keys:
            bbox_data = current_data[map_keys[0]]
            if isinstance(bbox_data, list) and len(bbox_data) > 0:
                print(f"🎯 mAP@0.5:0.95: {bbox_data[0]:.4f}")
        
        print("-" * 80)
        
        # 显示损失趋势
        if 'train_loss' in self.data and len(self.data['train_loss']) >= 2:
            losses = list(self.data['train_loss'])
            recent_losses = losses[-10:]  # 最近10个epoch
            
            print("📈 Recent Loss Trend (last 10 epochs):")
            trend_str = ""
            for i, loss in enumerate(recent_losses):
                if i > 0:
                    if loss < recent_losses[i-1]:
                        trend_str += "📉"
                    elif loss > recent_losses[i-1]:
                        trend_str += "📈"
                    else:
                        trend_str += "➡️"
                trend_str += f" {loss:.4f} "
            print(trend_str)
        
        print("-" * 80)
        print("Press Ctrl+C to stop monitoring")
    
    def start_monitoring(self, update_interval=5):
        """开始监控"""
        print(f"Starting terminal monitoring of {self.log_file}")
        
        try:
            while True:
                new_entries = self.read_new_lines()
                if new_entries:
                    self.update_data(new_entries)
                
                self.display_status()
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\n\n✅ Monitoring stopped by user")

def main():
    parser = argparse.ArgumentParser(description='Terminal training monitor')
    parser.add_argument('--log-file', '-l', 
                       default='outputs/deimv2_hgnetv2_n_persion_yolo_label_tuning/log.txt',
                       help='Path to log file')
    parser.add_argument('--update-interval', '-u', type=int, default=5,
                       help='Update interval in seconds')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"❌ Log file {args.log_file} does not exist!")
        return
    
    monitor = TerminalMonitor(args.log_file)
    monitor.start_monitoring(args.update_interval)

if __name__ == '__main__':
    main()