# CLAUDE.md — 精准收招 屏幕检测提示音

## 核心目标

- **找到游戏中 “精准收招” UI 出现的时刻。**
- 所有工作（屏幕检测、模板匹配、CNN 分类器、训练数据准备）都围绕这个时刻判定展开：当前画面里是否出现了 “精准收招” 这行字。

## 方案

- **仅 Windows**：使用 `winsound.Beep` 发声，`dxcam` 截屏。
- **必须指定区域**：`--region left,top,right,bottom`，不检测全屏。
- **最快实现**：
  - `dxcam` 走 Windows DXGI Desktop Duplication，循环外创建并启动相机。
  - 截图 + 模板都转灰度，再用 `cv2.matchTemplate(..., cv2.TM_CCOEFF_NORMED)`。
  - 默认目标帧间隔 `1/60` 秒，靠 `time.perf_counter()` 计算剩余时间补眠。
- **命中后 3 秒冷却**：beep 后立即暂停检测，3 秒后自动恢复。
- **不自动保存截图**：只有 `--debug` 才会保存单张标记图。

## 启动命令（最新版：ONNX 分类器 + 提示音）

`detect_jzsz_beep_onnx.py` 把 ONNX CNN 分类器和 `winsound.Beep` 提示音整合到了一起，比模板匹配更鲁棒。模型路径以脚本所在目录为基准，因此可以从任意路径启动：

```cmd
G:\Users\qq295\.workbuddy\binaries\python\envs\default\Scripts\python.exe G:\Users\qq295\WorkBuddy\自动录屏\detect_jzsz_beep_onnx.py
```

如需覆盖检测区域：

```cmd
G:\Users\qq295\.workbuddy\binaries\python\envs\default\Scripts\python.exe G:\Users\qq295\WorkBuddy\自动录屏\detect_jzsz_beep_onnx.py --region 2719,1559,2993,1764
```

### ONNX beep 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `jzsz_classifier_squeezenet.onnx` | ONNX 模型路径（默认在脚本目录下） |
| `--region` | 自动计算 | 检测区域，格式 `left,top,right,bottom`；省略则按 SEARCH_REGION 比例自动计算 |
| `--threshold` | `0.5` | 判定为 “精准收招” 的概率阈值 |
| `--interval-ms` | `16` | 目标帧间隔，约 60 FPS |
| `--cooldown` | `3.0` | 命中后暂停秒数 |
| `--frequency` | `1000` | 提示音频率 Hz |
| `--duration` | `500` | 提示音时长 ms |
| `--run-duration` | `0` | 运行秒数，0 为无限 |
| `--beep-crops-dir` | `beep_crops` | 每次 beep 时保存局部截图的目录，用于训练数据收集 |
| `--beep-crop-quality` | `90` | 局部截图 JPEG 质量 |
| `--no-beep` | 否 | 只打印不发声，用于测试 |

## 旧版模板匹配启动命令（保留）

```cmd
cd /d G:\Users\qq295\WorkBuddy\自动录屏
G:\Users\qq295\.workbuddy\binaries\python\envs\default\Scripts\python.exe detect_jzsz_beep.py --region 2688,1555,3264,1771
```

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--region` | 必填 | 检测区域，格式 `left,top,right,bottom` |
| `--template` | `candidate_jzsz.png` | 模板图路径 |
| `--threshold` | `0.8` | 匹配阈值 |
| `--interval` | `1/60` | 目标帧间隔 |
| `--cooldown` | `3.0` | 命中后暂停秒数 |
| `--frequency` | `1000` | 提示音频率 Hz |
| `--duration` | `500` | 提示音时长 ms |

## 性能

- 576×216 区域实测约 **3–6 ms/帧**，轻松跑满 60 FPS。
- 区域越小越快；全屏 4K 太慢，本方案不支持。

## 训练数据准备

从游戏录屏里批量提取正/负样本，用于训练或扩充 CNN 分类器。

```cmd
cd /d G:\Users\qq295\WorkBuddy\自动录屏
G:\Users\qq295\.workbuddy\binaries\python\envs\default\Scripts\python.exe extract_video_samples.py "G:\Users\qq295\Videos\Captures\Granblue Fantasy_ Relink 2026-07-13 06-26-14.mp4"
```

行为：

1. 按固定间隔（默认每 5 帧）从视频里采样，降低重复帧数量。
2. 裁剪右下角固定区域（与 `monitor.py` 的 `SEARCH_REGION` 一致，对应 2560×1440 视频中的 177×134 区域）。
3. 使用现有的 `jzsz_classifier_squeezenet.onnx` 分类器预测该区域为 “精准收招” 的概率。
4. 取概率最高的 200 帧保存到 `positive_samples/`，概率最低的 200 帧保存到 `negative_samples/`。

注意：

- 分类器本身也可能把其他技能名（如 “回旋枪+”）误判为正样本，因此 `positive_samples/` 末尾若干样本需要人工复核。
- 若需要更干净的正样本，可以提高概率阈值并减少数量；或把多段视频合并后提取。
- 文件名中包含 `prob`（分类器概率）和 `pred`（预测类别），方便快速过滤。

## CNN 分类器方案（比模板匹配更鲁棒）

当模板匹配对背景颜色/光效敏感、漏检严重时，使用小型 CNN 做二分类（“精准收招” 出现 vs 未出现）效果更好。

### 模型选择

- **SqueezeNet 1.1**：1.2M 参数，速度最快，ImageNet 预训练权重微调。
- 在 224×224 输入上，ONNX CPU 推理约 **1.5–2.3 ms/帧**，即 **400–600+ FPS**，远超 60 FPS 需求。
- 在 RTX 3080 Ti 上训练 30 个 epoch 约需 4–5 分钟。

### 训练流程

1. 准备正样本目录 `positive_candidates/`（包含 “精准收招” 的帧）。
2. 准备负样本目录 `new_data/`（不含 “精准收招” 的帧）。
3. 从 `new_data/` 删除与 `positive_candidates/` 内容重复的帧（按 MD5 去重）。
4. 运行训练脚本：

```cmd
cd /d G:\Users\qq295\WorkBuddy\自动录屏
D:\workbuddy_venv\Scripts\python.exe train_classifier.py
```

行为：

- 去重后从 `positive_candidates/` 读取正样本，从 `new_data/` 读取负样本。
- 8:2 划分训练集/验证集。
- 对训练集预生成原始 + ±1% + ±2% 平移副本（验证集保持原图）。
- 冻结 SqueezeNet 特征层，只训练最后的分类层。
- 自动保存最佳模型并导出为 `jzsz_classifier_squeezenet.onnx`。

如果还有其他负样本目录，脚本会自动识别并合并使用；目录不存在时自动忽略。

### 训练环境踩坑

- G: 盘空间不足（只剩 2.2GB），无法安装 CUDA 版 PyTorch（wheel 2.6GB）。
- 解决方案：在 D: 盘创建独立虚拟环境并安装 CUDA 版 PyTorch：

```cmd
G:\Users\qq295\.workbuddy\binaries\python\versions\3.13.12\python.exe -m venv D:\workbuddy_venv
D:\workbuddy_venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
D:\workbuddy_venv\Scripts\python.exe -m pip install onnx onnxruntime opencv-python mss pillow numpy
```

- 如果 pip 报 `No space left on device`，说明临时目录在 G: 盘。解决：把 `TMP`/`TEMP`/`TMPDIR` 和 pip cache 都改到 D: 盘。

### ONNX 导出踩坑

- 训练在 GPU 上完成后，导出 ONNX 时模型权重在 CUDA，dummy input 在 CPU，会报错：`Input type and weight type should be the same`。
- 解决：导出前重新创建模型并 `map_location='cpu'`：

```python
model = get_model(MODEL_NAME)
model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
model = model.to('cpu')
model.eval()
```

### 实时检测

```cmd
cd /d G:\Users\qq295\WorkBuddy\自动录屏
D:\workbuddy_venv\Scripts\python.exe monitor_onnx.py --once
D:\workbuddy_venv\Scripts\python.exe monitor_onnx.py --record-found --duration 300 --output-dir found_frames
```

参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--record-found` | 否 | 命中时保存截图 |
| `--record-all` | 否 | 保存每一帧 |
| `--output-dir` | `cnn_frames` | 保存目录 |
| `--duration` | 0 | 运行秒数，0 为无限 |
| `--interval-ms` | 16 | 采样间隔，约 60 FPS |
| `--threshold` | 0.5 | 判定为正样本的概率阈值 |

### 训练优化（GPU 利用率提升）

早期训练时 GPU 利用率只有约 25%。优化措施：

1. **增大 batch size**：从 8 提升到 32。
2. **启用 `pin_memory=True`**：加速 CPU→GPU 数据传输。
3. **增加 `num_workers=2`**：并行加载数据。
4. **启用混合精度训练（AMP）**：使用 `torch.amp.GradScaler('cuda')` 和 `torch.amp.autocast('cuda')`。

最终效果：RTX 3080 Ti 上 30 epoch 训练时间从数分钟降到约 **370 秒**，GPU 利用率明显提升。

### 数据增强策略

对文字检测，旋转、镜像、大幅平移都会破坏文字结构，因此不使用这些增强。

早期训练使用温和 ColorJitter，但后续改为仅做小幅度平移，以提升模型对轻微位置偏移的鲁棒性：

- 不使用 ColorJitter（颜色保持不变）。
- 训练时把每个训练样本预生成 3 份：
  - 原始图像
  - 随机平移 **±1%** 的图像
  - 随机平移 **±2%** 的图像
- 验证集不使用任何增强。

实现方式：在训练脚本内调用 `create_augmented_train_data()`，用 PIL 的 `Image.AFFINE` 生成 `train_data_aug/positive/` 和 `train_data_aug/negative/`，然后直接在这些目录上训练，不再使用在线随机增强。

> 曾尝试 ±5% 平移增强，结果在 tight crop 下文字可能移出检测区域，导致正样本召回从 98.6% 降到 95.2%。±1% / ±2% 的平移幅度小，不会破坏文字结构，同时能提升模型对轻微偏移的容忍度。

### 模型评估结果（最新：预生成 ±1% / ±2% 平移 + 正样本类别权重，数据更新后）

- 正样本：`positive_candidates/` 共 **225** 张（去重后 205 张）。
- 负样本：`new_data/` 共 **876** 张。
- 训练：RTX 3080 Ti，预生成 3 倍训练数据后 615 正 + 1845 负，30 epoch，最佳验证准确率 **98.2%**。
- 类别权重：负样本 1.00，正样本 **3.00**（正样本为少数类且漏检代价高）。
- ONNX CPU 推理：**354.2 FPS**（2.82 ms/帧），仍远超 60 FPS 需求。
- `positive_candidates/` 直接召回：**224/225 = 99.6%**。
- `new_data/` 中误报：**13/876 ≈ 1.5%**，主要包含 `beep_*.jpg` 实时截图与若干边界样本。
- 唯一未检测到的正样本：`negative_0050_frame06280_prob0.064_pred0.jpg`（概率 0.166），这是之前已确认的 hard positive，但本轮数据变化后又被漏检。

### 模型版本存档

为避免后续训练覆盖当前表现稳定的模型，已将其备份为：

- **`jzsz_classifier_squeezenet_v1.onnx`**（2.8 MB）
  - 备份时间：2026-07-13
  - 来源：当前 `jzsz_classifier_squeezenet.onnx`
  - 实测表现：实机运行 `detect_jzsz_beep_onnx.py` 能正常触发 beep，运行稳定。
  - 性能指标：见本节上方“模型评估结果（最新）”。

后续若再训练新模型，可继续备份为 `jzsz_classifier_squeezenet_v2.onnx` 等，并在下表中记录：

| 版本 | 文件 | 备份时间 | 关键指标 | 备注 |
|------|------|----------|----------|------|
| v1 | `jzsz_classifier_squeezenet_v1.onnx` | 2026-07-13 | 95.2% 召回，1.9% 误报，~60 FPS 实机 | 实测可用，推荐保留 |

### 改进建议

- 唯一未检测到的正样本 `negative_0050_frame06280_prob0.064_pred0.jpg` 概率 0.166，曾被人工确认为毋庸置疑的正样本，但本轮数据更新后模型再次漏检。建议单独放大该样本或复制多份加入训练集，也可以降低阈值到 0.15 来召回它，但会引入更多误报。
- `new_data/` 中仍包含若干 `beep_*.jpg` 高概率样本（0.5–0.95），其中可能是真正的“精准收招”触发截图，建议复核后决定是否归入 `positive_candidates/`。
- 如需进一步降低误报，可检查 `predicted_positive/` 目录中的 13 张预测正样本，做人工清洗后反向补充到负样本。
- 当前已使用类别权重（正样本 3.01）和 ±1% / ±2% 预生成平移，继续提升的空间主要在清理 `new_data/` 中的错标样本。
- 若追求更高精度，可尝试 MobileNetV3-Small（2.5M 参数），但推理速度会略低于 SqueezeNet。

## 依赖

- `dxcam`（已安装）
- `opencv-python`, `numpy`, `Pillow`
- CNN 训练额外需要：`torch` + `torchvision`（CUDA 版），`onnx`, `onnxruntime`

## 文件

- `detect_jzsz_beep_onnx.py` — 基于 ONNX CNN 分类器 + winsound 提示音的最新脚本
- `detect_jzsz_beep.py` — 基于 dxcam + 模板匹配的旧脚本
- `monitor_onnx.py` — 基于 ONNX CNN 分类器的实时检测/截图脚本（无提示音）
- `train_classifier.py` — CNN 训练脚本（SqueezeNet / MobileNetV3 微调）
- `test_classifier.py` — 测试 ONNX 分类器速度和召回
- `candidate_jzsz.png` — “精准收招” 模板
- `extract_video_samples.py` — 从视频批量提取正负训练样本
- `positive_samples/` — 正样本输出目录（CNN 预测为 “精准收招”）
- `negative_samples/` — 负样本输出目录（CNN 预测为其他 UI）
- `positive_candidates/` — 人工确认的正样本集合
- `new_data/` — 负样本候选帧
- `jzsz_classifier_squeezenet.pth` — 训练好的 PyTorch 模型
- `jzsz_classifier_squeezenet.onnx` — 已导出的 ONNX 分类器（CPU 推理用）
- `jzsz_classifier_squeezenet_v1.onnx` — 实测可用版本的备份
- `D:/workbuddy_venv/` — CUDA 版 PyTorch 训练环境
