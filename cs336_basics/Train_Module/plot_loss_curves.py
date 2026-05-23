import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def extract_params_from_name(name):
    """
    从文件夹/文件名中提取参数信息
    
    格式示例: d512_l4_h16_ctx256_bs128_lr3e-04_seed42
    """
    params = {}
    
    # 提取 d_model
    d_model_match = re.search(r'd(\d+)', name)
    if d_model_match:
        params['d_model'] = int(d_model_match.group(1))
    
    # 提取 num_layers
    layers_match = re.search(r'l(\d+)', name)
    if layers_match:
        params['num_layers'] = int(layers_match.group(1))
    
    # 提取 num_heads
    heads_match = re.search(r'h(\d+)', name)
    if heads_match:
        params['num_heads'] = int(heads_match.group(1))
    
    # 提取 context_length
    ctx_match = re.search(r'ctx(\d+)', name)
    if ctx_match:
        params['context_length'] = int(ctx_match.group(1))
    
    # 提取 batch_size
    bs_match = re.search(r'bs(\d+)', name)
    if bs_match:
        params['batch_size'] = int(bs_match.group(1))
    
    # 提取 learning_rate
    lr_match = re.search(r'lr([\de.-]+)', name)
    if lr_match:
        params['lr'] = float(lr_match.group(1))
    
    # 提取 seed
    seed_match = re.search(r'seed(\d+)', name)
    if seed_match:
        params['seed'] = int(seed_match.group(1))
    
    return params

def load_loss_curves(checkpoints_dir):
    """
    加载所有checkpoints文件夹中的loss曲线数据
    """
    all_curves = []
    
    for subdir in os.listdir(checkpoints_dir):
        subdir_path = os.path.join(checkpoints_dir, subdir)
        
        if not os.path.isdir(subdir_path):
            continue
        
        # 查找loss_curve CSV文件
        csv_files = [f for f in os.listdir(subdir_path) if f.startswith('loss_curve') and f.endswith('.csv')]
        
        if not csv_files:
            print(f"Warning: No loss_curve CSV found in {subdir}")
            continue
        
        csv_path = os.path.join(subdir_path, csv_files[0])
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_path)
            
            # 提取参数信息
            params = extract_params_from_name(subdir)
            
            # 生成图例标签 - 包含所有重要参数
            label_parts = []
            if 'lr' in params:
                label_parts.append(f"lr={params['lr']:.1e}")
            if 'batch_size' in params:
                label_parts.append(f"bs={params['batch_size']}")
            if 'context_length' in params:
                label_parts.append(f"ctx={params['context_length']}")
            if 'd_model' in params:
                label_parts.append(f"d={params['d_model']}")
            if 'num_layers' in params:
                label_parts.append(f"l={params['num_layers']}")
            if 'num_heads' in params:
                label_parts.append(f"h={params['num_heads']}")
            
            label = ', '.join(label_parts)
            
            all_curves.append({
                'df': df,
                'label': label,
                'params': params,
                'name': subdir
            })
            
            print(f"Loaded: {subdir} -> {label}")
            
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
    
    return all_curves

def plot_loss_curves(all_curves):
    """
    绘制对比loss曲线
    """
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    for curve in all_curves:
        df = curve['df']
        label = curve['label']
        
        # 绘制原始loss曲线
        plt.plot(df['iteration'], df['loss'], label=label, linewidth=1.5)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 设置y轴范围（根据数据自动调整）
    all_losses = [df['loss'].min() for curve in all_curves for df in [curve['df']]]
    min_loss = min(all_losses) * 0.9
    max_loss = 10  # 大致上限
    
    plt.ylim(bottom=min_loss, top=max_loss)
    
    plt.show()

def plot_smoothed_loss_curves(all_curves, window_size=50):
    """
    绘制平滑后的loss曲线（使用移动平均）
    """
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    for curve in all_curves:
        df = curve['df']
        label = curve['label']
        
        # 计算移动平均
        smoothed_loss = df['loss'].rolling(window=window_size, center=True).mean()
        
        plt.plot(df['iteration'], smoothed_loss, label=label, linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Smoothed Loss', fontsize=12)
    plt.title(f'Training Loss Comparison (Smoothed, window={window_size})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 设置y轴范围
    all_losses = [df['loss'].rolling(window=window_size, center=True).mean().dropna().min() 
                  for curve in all_curves for df in [curve['df']]]
    min_loss = min(all_losses) * 0.9
    plt.ylim(bottom=min_loss, top=10)
    
    plt.show()

def print_params_summary(all_curves):
    """
    打印所有配置的参数摘要
    """
    print("\n=== 参数配置对比 ===")
    print(f"{'配置名称':<40} | {'学习率':<10} | {'批次':<6} | {'上下文':<6} | {'d_model':<6} | {'层数':<4} | {'头数':<4}")
    print("-" * 100)
    
    for curve in all_curves:
        params = curve['params']
        name = curve['name']
        print(f"{name:<40} | {params.get('lr', 'N/A'):<10} | {params.get('batch_size', 'N/A'):<6} | {params.get('context_length', 'N/A'):<6} | {params.get('d_model', 'N/A'):<6} | {params.get('num_layers', 'N/A'):<4} | {params.get('num_heads', 'N/A'):<4}")

if __name__ == "__main__":
    # checkpoints文件夹路径
    checkpoints_dir = "./checkpoints"
    
    # 加载所有loss曲线数据
    all_curves = load_loss_curves(checkpoints_dir)
    
    if not all_curves:
        print("No loss curves found!")
        exit()
    
    # 打印参数摘要
    print_params_summary(all_curves)
    
    # 绘制原始loss曲线
    plot_loss_curves(all_curves)
    
    # 绘制平滑后的loss曲线
    plot_smoothed_loss_curves(all_curves, window_size=50)