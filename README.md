# GBRF Beep — 碧蓝幻想 Relink 精准收招检测器

检测《Granblue Fantasy: Relink》游戏中 “精准收招” UI 出现的时刻，并通过提示音提醒玩家。

> 本项目仅用于个人游戏辅助与计算机视觉学习，不包含任何修改游戏内存或注入行为。

---

## 功能

- **实时屏幕检测**：使用 Windows DXGI（dxcam）或 MSS 截取屏幕指定区域。
- **双检测方案**：
  - **推荐**：基于 SqueezeNet CNN 分类器的 ONNX 推理，对颜色/光效变化更鲁棒。
  - **旧版**：基于 OpenCV 模板匹配，部署简单但对复杂背景较敏感。
- **命中提示音**：检测到 “精准收招” 后通过 `winsound.Beep` 发出提示音，并自动进入冷却。
- **训练数据准备**：从游戏录屏中自动提取正/负样本，用于训练或扩充分类器。
- **模型训练与导出**：PyTorch 训练脚本 + ONNX 导出，CPU 推理即可跑满 60 FPS。

---

## 目录

- [快速开始](#快速开始)
- [检测区域](#检测区域)
- [脚本说明](#脚本说明)
- [训练自己的模型](#训练自己的模型)
- [模型版本](#模型版本)
- [性能指标](#性能指标)
- [依赖](#依赖)
- [文件结构](#文件结构)
- [注意事项](#注意事项)

---

## 快速开始

### 1. 安装依赖

```cmd
python -m pip install opencv-python numpy Pillow onnxruntime mss dxcam comtypes
```

若需训练模型，额外安装 PyTorch（CUDA 版）：

```cmd
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install onnx
```

### 2. 运行实时检测（推荐 ONNX 版）

```cmd
python detect_jzsz_beep_onnx.py
```

脚本会根据屏幕分辨率自动计算检测区域。若需覆盖：

```cmd
python detect_jzsz_beep_onnx.py --region 2719,1559,2993,1764
```

### 3. 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `jzsz_classifier_squeezenet.onnx` | ONNX 模型路径 |
| `--region` | 自动计算 | 检测区域 `left,top,right,bottom` |
| `--threshold` | `0.5` | 判定阈值 |
| `--interval-ms` | `16` | 采样间隔，约 60 FPS |
| `--cooldown` | `3.0` | 命中后冷却秒数 |
| `--frequency` | `1000` | 提示音频率 Hz |
| `--duration` | `500` | 提示音时长 ms |
| `--no-beep` | 否 | 只打印不发声 |

---

## 检测区域

相对坐标：

```python
SEARCH_REGION = (0.7104, 0.7222, 0.7794, 0.8148)
```

对应分辨率：

| 分辨率 | 区域（left, top, right, bottom） | 大小 |
|--------|----------------------------------|------|
| 2560×1440 | `(1818, 1039, 1995, 1173)` | 177×134 |
| 3840×2160 | `(2719, 1559, 2993, 1764)` | 274×205 |

---

## 脚本说明

| 文件 | 说明 |
|------|------|
| `detect_jzsz_beep_onnx.py` | **主程序**：ONNX 分类器 + winsound 提示音 |
| `detect_jzsz_beep.py` | 旧版：dxcam + OpenCV 模板匹配 + 提示音 |
| `monitor_onnx.py` | ONNX 分类器实时检测/截图（无提示音） |
| `train_classifier.py` | CNN 训练脚本（SqueezeNet / MobileNetV3 微调） |
| `test_classifier.py` | 测试 ONNX 分类器速度和召回率 |
| `extract_video_samples.py` | 从游戏录屏批量提取正负训练样本 |
| `extract_jzsz_template.py` | 从截图中提取 “精准收招” 模板 |
| `filter_positive_samples.py` | 对正样本进行人工复核前的筛选 |
| `debug_current_match.py` | 模板匹配调试脚本 |
| `debug_jzsz_match.py` | “精准收招” 模板匹配调试 |
| `crop_*.py` | 多种模板裁剪/预处理脚本 |
| `find_interesting_frames.py` | 从视频中找出可能出现目标 UI 的帧 |
| `locate_template.py` | 在截图中定位模板位置 |
| `tighten_template.py` | 收紧模板边界 |
| `verify_smallest.py` | 验证最小可用模板 |
| `analyze_run.py` | 分析单次运行结果 |
| `monitor.py` | 旧版实时检测脚本 |
| `screenshot_test.py` | 截图功能测试 |

---

## 训练自己的模型

### 准备数据

1. 将人工确认的 “精准收招” 帧放入 `positive_candidates/`。
2. 将不含目标 UI 的帧放入 `new_data/`（或其他负样本目录）。
3. 运行训练脚本：

```cmd
python train_classifier.py
```

训练脚本会自动：

- 对 `positive_candidates/` 和 `new_data/` 进行 MD5 去重。
- 按 8:2 划分训练集/验证集。
- 冻结 SqueezeNet 特征层，只训练最后的分类层。
- 保存最佳 PyTorch 模型为 `jzsz_classifier_squeezenet.pth`。
- 导出 ONNX 模型为 `jzsz_classifier_squeezenet.onnx`。

### 从视频提取样本

```cmd
python extract_video_samples.py "path/to/your/video.mp4"
```

默认每 5 帧采样一次，使用当前 ONNX 模型打分，取前 200 帧保存到 `positive_samples/`，后 200 帧保存到 `negative_samples/`。

> 提示：`positive_samples/` 末尾的样本建议人工复核，分类器可能把其他技能名（如 “回旋枪+”）误判为 “精准收招”。

---

## 模型版本

| 版本 | 文件 | 备份时间 | 关键指标 | 备注 |
|------|------|----------|----------|------|
| v1 | `jzsz_classifier_squeezenet_v1.onnx` | 2026-07-13 | 95.2% 召回，1.9% 误报，~60 FPS 实机 | 实测可用，推荐保留 |

当前推理模型：

- `jzsz_classifier_squeezenet.onnx` — 最新导出的 ONNX 模型。
- `jzsz_classifier_squeezenet.pth` — 对应的 PyTorch 权重。

---

## 性能指标

- **输入尺寸**：224×224
- **ONNX CPU 推理速度**：约 **1.2 ms/帧**（约 833 FPS）
- **实机运行速度**：约 **60 FPS**
- **最佳验证准确率**：96.9%
- **正样本召回率**：200/210 = 95.2%
- **负样本误报率**：16/849 ≈ 1.9%

测试环境：RTX 3080 Ti（训练）/ ONNX Runtime CPU（推理）。

---

## 依赖

- `opencv-python`
- `numpy`
- `Pillow`
- `onnxruntime`
- `mss`
- `dxcam`
- `comtypes`

训练额外依赖：

- `torch`
- `torchvision`
- `onnx`

---

## 文件结构

```
.
├── detect_jzsz_beep_onnx.py      # 推荐实时检测程序
├── detect_jzsz_beep.py           # 旧版模板匹配检测
├── monitor_onnx.py               # ONNX 检测/截图工具
├── train_classifier.py           # CNN 训练脚本
├── test_classifier.py            # 模型测试脚本
├── extract_video_samples.py      # 视频样本提取
├── candidate_jzsz.png            # “精准收招” 模板
├── jzsz_classifier_squeezenet.onnx       # 当前 ONNX 模型
├── jzsz_classifier_squeezenet.pth        # 当前 PyTorch 权重
├── jzsz_classifier_squeezenet_v1.onnx    # 实测可用版本备份
├── positive_candidates/          # 人工确认的正样本
├── new_data/                     # 负样本候选帧
├── train_data/                   # 训练集（由脚本生成）
├── val_data/                     # 验证集（由脚本生成）
├── positive_samples/             # 视频提取的正样本输出
├── negative_samples/             # 视频提取的负样本输出
├── CLAUDE.md                     # 项目开发笔记
└── README.md                     # 本文件
```

---

## 注意事项

- 本项目**仅支持 Windows**，因为提示音使用了 `winsound`，截图使用了 `dxcam`。
- 首次连接 GitHub 可能需要添加 `github.com` 到 SSH `known_hosts`。
- 若训练时磁盘空间不足，可将虚拟环境和临时目录放到空间充足的盘符。
- 导出 ONNX 前请确保模型权重已加载到 CPU，避免 `Input type and weight type should be the same` 错误。
- 如需更高召回率，可降低 `--threshold`，但可能引入更多误报。
