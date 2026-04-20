# 深度感知视觉语言模型训练项目

这是一个基于 `Qwen/Qwen3-VL-4B-Instruct` 的中文多模态训练项目，目标是构建一个具备更强空间理解与深度感知能力的视觉语言模型。模型输入为图像与文本提示，输出为文本；训练时额外引入深度监督，让模型在生成描述、问答和场景理解结果时更好地利用几何信息。

## 项目目标

- 以 `Qwen3-VL-4B` 作为主干模型进行微调
- 保留文本生成为主任务
- 引入深度辅助监督，增强前后关系、距离、遮挡和三维结构理解
- 使用冻结的 `MoGe + LingBot-Depth` 生成深度目标特征

## 当前实现

- 主干模型：`Qwen/Qwen3-VL-4B-Instruct`
- 深度模型：
  - `Ruicheng/moge-2-vitb-normal`
  - `robbyant/lingbot-depth-pretrain-vitl-14-v0.5`
- 训练目标：
  - 当前默认只训练 `DepthHead`
  - 图像 token 的 `Depth Alignment Loss`

## 目录说明

- [tasks/vl/train_qwen3_vl_4b.py](/Users/lhy/Desktop/lingbot-vla/tasks/vl/train_qwen3_vl_4b.py)：训练入口
- [configs/vl/qwen3_vl_4b_sft.yaml](/Users/lhy/Desktop/lingbot-vla/configs/vl/qwen3_vl_4b_sft.yaml)：训练配置
- [lingbotvla/models/vl/qwen3_vl_4b](/Users/lhy/Desktop/lingbot-vla/lingbotvla/models/vl/qwen3_vl_4b)：模型封装、数据集与深度对齐实现

## 数据格式

训练数据支持 `json` 或 `jsonl`，每条样本至少包含：

```json
{"image": "images/example.jpg", "prompt": "请描述图中的空间关系。", "response": "机械臂位于桌面右侧，杯子在碗的前方。"}
```

也支持：

```json
{"images": ["images/example.jpg"], "prompt": "图里有什么？", "response": "桌面上有一个碗和一个杯子。"}
```

字段说明：

- `image`：单张图片路径
- `images`：图片路径数组，当前训练脚本默认使用第一张
- `prompt`：输入提示词
- `response`：监督文本

## 环境要求

建议环境：

- Python 3.10+
- PyTorch 2.x
- CUDA 可用
- `transformers` 版本需支持 `Qwen3VLForConditionalGeneration`

如果要启用深度监督，还需要保证本地能够正确加载：

- `Ruicheng/moge-2-vitb-normal`
- `robbyant/lingbot-depth-pretrain-vitl-14-v0.5`

## 配置说明

默认训练配置在 [configs/vl/qwen3_vl_4b_sft.yaml](/Users/lhy/Desktop/lingbot-vla/configs/vl/qwen3_vl_4b_sft.yaml)。

你至少需要修改这几个字段：

- `train_path`
- `image_root`
- `output_dir`

其余关键默认值已经配置好：

- `model_name_or_path: Qwen/Qwen3-VL-4B-Instruct`
- `moge_path: Ruicheng/moge-2-vitb-normal`
- `morgbd_path: robbyant/lingbot-depth-pretrain-vitl-14-v0.5`

## 启动训练

```bash
python3 tasks/vl/train_qwen3_vl_4b.py --config configs/vl/qwen3_vl_4b_sft.yaml
```

训练启动后会打印：

- 总参数量
- 可训练参数量
- 可训练参数占比

## 模型设计

当前训练流程是：

1. 使用冻结的 `Qwen3-VL-4B` 处理图像和文本输入
2. 从模型最后一层隐藏状态中提取图像 token
3. 使用 `DepthHead` 将图像 token 对齐到深度特征空间
4. 使用冻结的 `MoGe + LingBot-Depth` 生成深度目标特征
5. 仅优化 `DepthHead` 的深度对齐损失

说明：

- `prompt` 和 `response` 仍然会作为输入上下文进入主干模型
- 但当前默认配置不会对文本生成损失做反向传播
- 训练更新的只有 `DepthHead`

## 适用场景

- 场景描述
- 带空间关系的视觉问答
- 机器人观察结果文本化
- 强调前后、远近、遮挡关系的多模态理解任务

## 后续可扩展方向

- 支持多图输入
- 支持多轮对话监督
- 增加推理脚本与评测脚本
- 将深度监督替换为离线缓存，降低训练时延

## 参考模型

- Qwen3-VL-4B: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- MoGe: https://huggingface.co/Ruicheng/moge-2-vitb-normal
- LingBot-Depth: https://huggingface.co/robbyant/lingbot-depth-pretrain-vitl-14-v0.5
