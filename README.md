# 实验2：基于图像-文本联合表示的鸟类多模态分类（完整代码）

对应手册《实验2：鸟类多模态分类实验手册》，把手册里所有 `请补全....` 的地方补齐，
并覆盖完整路线：**单模态 → 双模态 → 跨模态对齐 → 多模态融合 → 消融与错误分析**。

```
bird_multimodal/
├── requirements.txt              依赖清单
├── configs/default.yaml          统一配置（换融合方式只改 fusion.name 一个字段）
├── bird_multimodal/
│   ├── config.py                 配置读取（yaml/json + 点号路径 + --set 覆盖）
│   ├── utils.py                  随机种子 / 指标 / 训练与评估循环 / 落盘
│   ├── data_utils.py             【阶段1】原始数据 → 6:2:2 → annotations.json
│   ├── datasets.py               Image / Text / Multimodal 三个 Dataset
│   ├── image_encoder.py          【阶段3】ResNet → [512] / [49,512]（03_image_encoder(1).py）
│   ├── text_encoder.py           【阶段3】BERT → [768]，cls/mean pooling（03_text_encoder(2).py）
│   ├── fusion.py                 【阶段4】Early / Late / Cross-Attention / Gated
│   ├── model.py                  编码器 + 融合 + 分类头 的组装
│   ├── train_image_only.py       【阶段2.1】Image-Only 训练 + 特征拆解
│   ├── train_text_only.py        【阶段2.2】Text-Only（BERT）
│   ├── train_multimodal.py       【阶段4】多模态融合训练 / 测试
│   ├── run_ablation.py           【阶段5】四种融合方式 + 单模态基线 一键消融
│   ├── clip_zeroshot.py          【阶段5】CLIP 零样本分类 + 图文检索 R@1/R@5
│   ├── visualize_attention.py    【阶段3/4】跨模态注意力可视化 + 图文相似度矩阵
│   ├── analyze_errors.py         【阶段5】逐类准确率 / 混淆对 / 高置信度错误 / 方法对比
│   ├── manual_check.py           数据管线自检（不需要 torch）
│   └── static_lint.py            静态体检（不需要 torch）
└── 02_singleModal_TextOnly.py    手册里那份单模态文本脚本的补全版（可直接 python 运行）
```

## 0. 环境

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# torch 请按官网命令装与 CUDA 匹配的版本；CPU 版：
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 1. 数据准备（手册 1.3：6:2:2 划分 + annotations.json）

原始结构（类别文件夹 + 与图片同名的 txt，txt 内每行一条描述）：

```
datasets/
├── 001.Black_footed_Albatross/
│   ├── Black_Footed_Albatross_0001_796111.jpg
│   └── Black_Footed_Albatross_0001_796111.txt     ← 10 行描述
└── 002.Laysan_Albatross/...
```

```bash
# 扫描 -> 分层 6:2:2 划分 -> 写 annotations.json 和 annotations_{train,val,test}.json
python -m bird_multimodal.data_utils --root datasets

# 想要手册里那种 data_root/images/... 的结构（会复制图片，占磁盘）
python -m bird_multimodal.data_utils --root datasets --copy-images

# 先小规模跑通（每类 20 张）
python -m bird_multimodal.data_utils --root datasets --limit-per-class 20
```

生成的 `annotations.json`：

```json
{
  "meta": {"source": "...", "num_classes": 100, "num_captions": 10, "split_ratio": "6:2:2"},
  "classes": ["001.Black_footed_Albatross", "..."],          // 下标 == label
  "class_prompts": {"001.Black_footed_Albatross": "a photo of a black footed albatross"},
  "splits": {"train": [{"image_path": "...", "captions": ["10 条"], "label": 0}, ...],
             "val": [...], "test": [...]}
}
```

**还没拿到真实数据也能先验证流程**（不需要 torch）：

```bash
python bird_multimodal/manual_check.py     # 造假数据跑一遍阶段 1，校验划分比例 / 路径 / 配置
python bird_multimodal/static_lint.py      # 静态体检：未定义名字、弃用 API
```

## 2. 单模态训练（手册 2）

```bash
# 2.1 Image-Only：ResNet18 / MobileNetV2，训练 + 测试 + 指标
python -m bird_multimodal.train_image_only --epochs 5

# 把模型拆开，看图像特征（[B,512,7,7] -> 49 个视觉词 -> 全局向量）
python -m bird_multimodal.train_image_only --inspect-only
# 产出 runs/image_only_resnet18/{image_feature_report.txt, layer4_feature_map.npy,
#                              image_tokens.npy, image_vector.npy}

# 解冻主干微调（GPU 上效果更好）
python -m bird_multimodal.train_image_only --set image.freeze=false train.lr=1e-4

# 2.2 Text-Only：BERT（等价于手册 02 脚本，但补齐了验证集 / 最佳权重 / 报告）
python -m bird_multimodal.train_text_only --epochs 5
python -m bird_multimodal.train_text_only --eval-only --eval-mode all   # 10 条描述投票

# 也可以用那份独立脚本（保持手册原始写法）
python 02_singleModal_TextOnly.py --epochs 5
```

## 3. 模型拆解：特征长什么样（手册 3）

```bash
# tokenizer 对照表 + BERT 逐层特征 + 每 token 向量
python -m bird_multimodal.text_encoder

# 图像：逐层特征 shape/均值/标准差，以及 [B,512,7,7]->[B,49,512] 的每一步
python -m bird_multimodal.train_image_only --inspect-only
```

要点（代码里都有注释）：

| 环节 | 形状 |
|---|---|
| 输入图片 | `[B, 3, 224, 224]` |
| ResNet 主干（去掉 avgpool/fc） | `[B, 512, 7, 7]` |
| `flatten(2)` → `transpose(1,2)` | `[B, 49, 512]`（49 个"视觉词"，Attention 融合用） |
| `AdaptiveAvgPool2d(1)` | `[B, 512]`（Early/Late/Gated 融合用） |
| BERT tokenizer | `[CLS] a small blue bird [SEP]` → id 序列（特殊标记 101/102/0/103/100） |
| BERT 输出 | `[B, L, 768]`（每个 token 一个 768 维上下文向量） |
| 池化（cls / mask-mean） | `[B, 768]` |

## 4. 多模态融合（手册 4，实验核心）

四种融合模块**接口完全一致**：`forward(pixel_values, input_ids, attention_mask) -> {"logits": ...}`，
所以换融合方式只改配置里的 `fusion.name`：

```bash
python -m bird_multimodal.train_multimodal --set fusion.name=early
python -m bird_multimodal.train_multimodal --set fusion.name=late
python -m bird_multimodal.train_multimodal --set fusion.name=cross_attention   # 推荐
python -m bird_multimodal.train_multimodal --set fusion.name=gated

# 让晚融合自己学"这次更信谁"，训练完直接读出 alpha
python -m bird_multimodal.train_multimodal --set fusion.name=late fusion.learnable_alpha=true

# 导出跨模态注意力权重（可视化"模型在看哪个词"）
python -m bird_multimodal.train_multimodal --set fusion.name=cross_attention --dump-attention
```

| 融合 | 做法 | 代码里的体现 |
|---|---|---|
| Early | `[img; txt] -> MLP` | `ConcatFusion`：`cat([512, 768]) -> 1280` |
| Late | 两分支各自分类，概率加权平均 | `LateFusion`：`logits = α·text + (1-α)·img`，α 可学习 |
| Cross-Attention | 图像 token 作 Query，文本 token 作 Key/Value | `CrossAttentionFusion`：`TransformerDecoderLayer` + padding mask，可导出 `attn_weights` |
| Gated | `a = σ(W[img;txt])`，`fusion = a·img + (1-a)·txt` | `GatedFusion`：逐样本门控（也能切成逐维门控） |

## 5. 消融、CLIP 思想实验与错误分析（手册 5）

```bash
# 四种融合 + 单模态基线 一次跑完，输出 summary.json / summary.csv 和对比表
python -m bird_multimodal.run_ablation --epochs 5 --with-baselines

# CLIP：零样本分类（类别名写提示词比相似度）+ 图文检索 R@1/R@5/R@10
python -m bird_multimodal.clip_zeroshot
python -m bird_multimodal.clip_zeroshot --prompt-mode ensemble --model-name openai/clip-vit-base-patch16

# 跨模态注意力热力图 + 图文特征相似度矩阵
python -m bird_multimodal.visualize_attention --attention runs/multimodal_cross_attention/attention.npz
python -m bird_multimodal.visualize_attention --checkpoint runs/multimodal_cross_attention/best.pt

# 错误分析：方法对比表、最差类别、最常混淆的类别对、高置信度错误、多模态"修好了哪些样本"
python -m bird_multimodal.analyze_errors
python -m bird_multimodal.analyze_errors --compare runs/image_only_resnet18/test_report.json \
                                                  runs/multimodal_cross_attention/test_report.json
```

## 6. 输出文件一览

```
runs/
├── image_only_resnet18/     best.pt, history.json, test_report.json, image_feature_report.txt, *.npy
├── text_only_bert-base-uncased/  best.pt, history.json, test_report.json
├── multimodal_<fusion>/     best.pt, history.json, test_report.json, attention.npz
├── clip_zeroshot/           clip_report.json
├── attention_viz/           attention_heatmap.png, attention_spatial.png, image_text_similarity.png
├── error_analysis/          error_analysis.json/csv, per_class_worst_*.png, confusion_*.png
└── ablation/                ablation_summary.json / .csv
```

每份 `test_report.json` 都包含：Accuracy / 宏平均 Precision / Recall / F1 / Top-5、
balanced accuracy（逐类准确率均值）、逐类准确率与样本数、混淆矩阵、逐样本预测（图片路径 + 文本 + 置信度）。

## 7. 常用命令行技巧

```bash
# 任意配置项都能用 --set 覆盖（值会自动识别 bool/int/float/list）
python -m bird_multimodal.train_multimodal --set train.epochs=3 train.batch_size=8 image.freeze=false

# 每类只用 20 张图先跑通全流程（会自动调用阶段 1 生成标注）
python -m bird_multimodal.run_ablation --limit-per-class 20 --epochs 2 --with-baselines

# 快速冒烟测试：每个 epoch 只跑 3 个 batch
python -m bird_multimodal.train_multimodal --max-batches 3 --epochs 2

# 国内网络不通 HuggingFace 时先设镜像
set HF_ENDPOINT=https://hf-mirror.com        # Windows
export HF_ENDPOINT=https://hf-mirror.com     # Linux/macOS
```

## 8. 写报告时的答题线索

| 手册问题 | 代码 / 结论位置 |
|---|---|
| Image-Only 训练、测试与指标 | `train_image_only.py` + `utils.classification_metrics` |
| 把模型拆开看图像特征 | `--inspect-only` → `image_feature_report.txt`（逐层 shape/统计） |
| `[B,512,7,7]` 怎么变 `[B,49,512]` | `flatten(2)` → `transpose(1,2)`，报告里逐步打印 |
| tokenizer 为什么返回行号 | `00_text_encoder` 的 `tokenize_show()` + 手册 3 节的词表解释 |
| [CLS] 还是 mean pooling | `text_encoder.py` 两种都实现；mask 过滤 padding 的原因写在 `masked_mean` 注释里 |
| 四种融合的本质区别 | `fusion.py` 顶部文档 + 第 4 节对照表 |
| 哪种融合最好、提升多少 | `run_ablation.py` → `ablation_summary.csv` |
| 模型在看哪个词 | `visualize_attention.py` → `attention_heatmap.png` / `attention_spatial.png` |
| 没有分类器的 CLIP 范式 | `clip_zeroshot.py` → 零样本 Accuracy + 检索 R@k |
| 多模态到底帮了什么忙 | `analyze_errors.py --compare`（修好/弄坏的样本清单） |
