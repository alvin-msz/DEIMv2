#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

class COCOEvalAnalyzer:
    def __init__(self, eval_path, class_names=None):
        self.eval_data = torch.load(eval_path, map_location='cpu')
        self.class_names = class_names or [f'Class_{i}' for i in range(self.eval_data['precision'].shape[2])]
        
    def plot_pr_curves(self, save_path='pr_curves.png'):
        """绘制PR曲线"""
        precision = self.eval_data['precision']  # 使用self.eval_data
        recall = self.eval_data['recall']
        
        # 使用IoU=0.5的结果
        iou_idx = 0  # IoU=0.5
        area_idx = 0  # all areas
        max_det_idx = 2  # maxDets=100
        
        plt.figure(figsize=(12, 8))
        
        # 为每个类别绘制PR曲线
        num_classes = precision.shape[2]
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
        
        for class_idx in range(num_classes):
            # 获取该类别的precision和recall
            prec = precision[iou_idx, :, class_idx, area_idx, max_det_idx]
            
            # 过滤有效值
            valid_mask = prec > -1
            prec_valid = prec[valid_mask]
            
            # 生成recall点
            recall_points = np.linspace(0, 1, len(prec_valid))
            
            # 计算AP
            ap = np.mean(prec_valid[prec_valid > -1]) if np.any(prec_valid > -1) else 0
            
            class_name = self.class_names[class_idx] if self.class_names else f'Class {class_idx}'
            plt.plot(recall_points, prec_valid, 
                    color=colors[class_idx], 
                    label=f'{class_name} (AP={ap:.3f})',
                    linewidth=2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves (IoU=0.5)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 使用close()而不是show()
        
    def plot_map_analysis(self, save_path='map_analysis.png'):
        """分析不同IoU阈值和面积下的mAP"""
        precision = self.eval_data['precision']  # 修复：使用self.eval_data
        
        # 计算不同条件下的AP
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        area_names = ['All', 'Small', 'Medium', 'Large']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for area_idx, area_name in enumerate(area_names):
            ax = axes[area_idx]
            
            # 计算每个IoU阈值下的mAP
            map_scores = []
            for iou_idx in range(len(iou_thresholds)):
                prec = precision[iou_idx, :, :, area_idx, 2]  # maxDets=100
                # 计算所有类别的平均AP
                valid_prec = prec[prec > -1]
                map_score = np.mean(valid_prec) if len(valid_prec) > 0 else 0
                map_scores.append(map_score)
            
            ax.plot(iou_thresholds, map_scores, 'o-', linewidth=2, markersize=6)
            ax.set_xlabel('IoU Threshold')
            ax.set_ylabel('mAP')
            ax.set_title(f'mAP vs IoU Threshold ({area_name} Objects)')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_detection_summary(self, save_path='detection_summary.png'):
        """生成检测结果总结图"""
        precision = self.eval_data['precision']  # 修复：使用self.eval_data和添加self参数
        
        # 计算关键指标
        num_classes = precision.shape[2]
        
        # AP@0.5, AP@0.75, AP@0.5:0.95
        ap50_idx = 0   # IoU=0.5
        ap75_idx = 5   # IoU=0.75
        
        metrics = {
            'AP@0.5': [],
            'AP@0.75': [],
            'AP@0.5:0.95': []
        }
        
        for class_idx in range(num_classes):
            # AP@0.5
            prec_50 = precision[ap50_idx, :, class_idx, 0, 2]
            ap50 = np.mean(prec_50[prec_50 > -1]) if np.any(prec_50 > -1) else 0
            metrics['AP@0.5'].append(ap50)
            
            # AP@0.75
            prec_75 = precision[ap75_idx, :, class_idx, 0, 2]
            ap75 = np.mean(prec_75[prec_75 > -1]) if np.any(prec_75 > -1) else 0
            metrics['AP@0.75'].append(ap75)
            
            # AP@0.5:0.95 (平均所有IoU阈值)
            all_prec = precision[:, :, class_idx, 0, 2]
            ap_all = np.mean(all_prec[all_prec > -1]) if np.any(all_prec > -1) else 0
            metrics['AP@0.5:0.95'].append(ap_all)
        
        # 绘制条形图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(num_classes)
        width = 0.25
        
        bars1 = ax.bar(x - width, metrics['AP@0.5'], width, label='AP@0.5', alpha=0.8)
        bars2 = ax.bar(x, metrics['AP@0.75'], width, label='AP@0.75', alpha=0.8)
        bars3 = ax.bar(x + width, metrics['AP@0.5:0.95'], width, label='AP@0.5:0.95', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Average Precision')
        ax.set_title('Detection Performance by Class')
        ax.set_xticks(x)
        ax.set_xticklabels([self.class_names[i] if self.class_names else f'C{i}' for i in range(num_classes)])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_text_report(self, save_path='evaluation_report.txt'):
        """生成文本评估报告"""
        precision = self.eval_data['precision']
        
        with open(save_path, 'w') as f:
            f.write("=== COCO Evaluation Report ===\n\n")
            
            # 基本信息
            f.write(f"Number of classes: {precision.shape[2]}\n")
            f.write(f"IoU thresholds: 0.5:0.05:0.95 ({precision.shape[0]} thresholds)\n")
            f.write(f"Recall thresholds: {precision.shape[1]} points\n\n")
            
            # 计算各类别的AP
            f.write("Per-class Average Precision:\n")
            f.write("-" * 50 + "\n")
            
            for class_idx in range(precision.shape[2]):
                # AP@0.5
                prec_50 = precision[0, :, class_idx, 0, 2]
                ap50 = np.mean(prec_50[prec_50 > -1]) if np.any(prec_50 > -1) else 0
                
                # AP@0.75
                prec_75 = precision[5, :, class_idx, 0, 2]
                ap75 = np.mean(prec_75[prec_75 > -1]) if np.any(prec_75 > -1) else 0
                
                # AP@0.5:0.95
                all_prec = precision[:, :, class_idx, 0, 2]
                ap_all = np.mean(all_prec[all_prec > -1]) if np.any(all_prec > -1) else 0
                
                class_name = self.class_names[class_idx] if self.class_names else f'Class_{class_idx}'
                f.write(f"{class_name:15} | AP@0.5: {ap50:.3f} | AP@0.75: {ap75:.3f} | AP@0.5:0.95: {ap_all:.3f}\n")
            
            # 总体mAP
            f.write("\n" + "=" * 50 + "\n")
            f.write("Overall Performance:\n")
            
            # 计算总体mAP
            all_ap50 = []
            all_ap75 = []
            all_ap_all = []
            
            for class_idx in range(precision.shape[2]):
                prec_50 = precision[0, :, class_idx, 0, 2]
                ap50 = np.mean(prec_50[prec_50 > -1]) if np.any(prec_50 > -1) else 0
                all_ap50.append(ap50)
                
                prec_75 = precision[5, :, class_idx, 0, 2]
                ap75 = np.mean(prec_75[prec_75 > -1]) if np.any(prec_75 > -1) else 0
                all_ap75.append(ap75)
                
                all_prec = precision[:, :, class_idx, 0, 2]
                ap_all = np.mean(all_prec[all_prec > -1]) if np.any(all_prec > -1) else 0
                all_ap_all.append(ap_all)
            
            f.write(f"mAP@0.5     : {np.mean(all_ap50):.3f}\n")
            f.write(f"mAP@0.75    : {np.mean(all_ap75):.3f}\n")
            f.write(f"mAP@0.5:0.95: {np.mean(all_ap_all):.3f}\n")
        
    def generate_report(self, output_dir='eval_analysis'):
        """生成完整的评估报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("🔍 Generating evaluation analysis...")
        
        # 1. PR曲线
        # self.plot_pr_curves(os.path.join(output_dir, 'pr_curves.png'))
        
        # 2. mAP分析
        # self.plot_map_analysis(os.path.join(output_dir, 'map_analysis.png'))
        
        # 3. 类别性能对比
        # self.plot_detection_summary(os.path.join(output_dir, 'class_performance.png'))
        
        # 4. 生成文本报告
        self.generate_text_report(os.path.join(output_dir, 'evaluation_report.txt'))
        
        print(f"✅ Analysis complete! Results saved in {output_dir}/")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_path', type=str, required=True, 
                       help='Path to eval.pth file')
    parser.add_argument('--class_names', type=str, nargs='+', 
                       help='Class names')
    parser.add_argument('--output_dir', type=str, default='outputs/eval',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    analyzer = COCOEvalAnalyzer(args.eval_path, args.class_names)
    analyzer.generate_report(args.output_dir)

if __name__ == '__main__':
    main()
