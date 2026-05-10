# SDXL LoRA 微调

## 使用方法

按顺序运行即可：

```bash
# 第一步：训练
python train_lora.py

# 第二步：生成图片
python generate_lora.py

# 第三步：评估并对比基线
python evaluate_lora.py
```

---

## 说明

- `train_lora.py` → 训练 2000 steps，LoRA 权重保存到 `sdxl_lora_coco/`
- `generate_lora.py` → 用训练好的 LoRA 生成 100 张图片
- `evaluate_lora.py` → 计算 FID 和 CLIP Score，自动与基线对比
