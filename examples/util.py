import torch
import os

def save_per_example_lrp_res(res_map, model):
    for name, param in model.named_parameters():
        # 检查参数是否有梯度
        #print(name,  param.grad is not None)
        if param.grad is not None:
            if len(param.grad.shape)>1:
                res_map[name].append(param.grad.clone().detach().sum(dim=-1).squeeze(-1))
            else:
                res_map[name].append(param.grad.clone().detach())

    return res_map
    
def save_grad_info(model, save_path='weight_gradients.json'):
    # 获取权重名称和对应的梯度
    weight_gradients = {}
    weight_gradients_we = {}

    # 遍历模型的所有命名参数
    for name, param in model.named_parameters():
        # 检查参数是否有梯度
        print(name,  param.grad is not None)
        if param.grad is not None:
            # 存储权重名称和对应的梯度
            weight_gradients[name] = param.grad.clone().detach()
            weight_gradients_we[name] = param.clone().detach()
            print(f"Parameter: {name}")
            print(f"  Gradient shape: {param.grad.shape}")
            print(f"  Gradient norm: {torch.norm(param.grad):.6f}")
            print(f"  Gradient stats - min: {param.grad.min():.6f}, max: {param.grad.max():.6f}, mean: {param.grad.mean():.6f}")
            print("-" * 80)

    # 可选：按梯度范数排序
    print("\n=== 按梯度范数排序 ===")
    sorted_gradients = sorted(weight_gradients.items(), key=lambda x: torch.norm(x[1]), reverse=True)

    for name, grad in sorted_gradients:  # 显示前个梯度最大的权重
        print(f"{name}: norm = {torch.norm(grad):.6f}, grad shape:{ grad.shape} weight shape:{weight_gradients_we[name].shape}")

    # 可选：保存梯度信息到文件
    import json
    import numpy as np

    # 将梯度信息转换为可序列化的格式
    gradient_info = {}
    for name, grad in weight_gradients.items():
        gradient_info[name] = {
            'shape': list(grad.shape),
            'norm': float(torch.norm(grad.float()).cpu().numpy()),
            'mean': float(grad.float().mean().cpu().numpy()),
            'std': float(grad.float().std().cpu().numpy()),
            'min': float(grad.float().min().cpu().numpy()),
            'max': float(grad.float().max().cpu().numpy())
        }

    # 保存到JSON文件
    with open(save_path, 'w') as f:
        json.dump(gradient_info, f, indent=2)

    print(f"\n梯度信息已保存到 weight_gradients.json，共 {len(weight_gradients)} 个参数")

