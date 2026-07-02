# RKNN 多框问题排查与修复记录

## 现象

新训练并转换后的 `best_i8.rknn` 在香橙派上实时检测时，同一个动物会出现多个检测框。

典型表现：

- 同一个类别也会出现大框套小框。
- 有时同一个目标会出现多个相近类别框。
- PT 模型在 Windows 上检测正常，但 RKNN 上出现重复框。
- 调低 `nms_thresh` 后效果仍不稳定。

## 最终原因

根因是 **RKNN 模型输出已经做过 sigmoid，但检测代码又手动 sigmoid 了一次**。

新导出的 `best.onnx` 是 RKNN 使用的三输出 YOLO head：

```text
[1, 30, 80, 80]
[1, 30, 40, 40]
[1, 30, 20, 20]
```

检查 ONNX 输出范围后发现，输出值已经在 `0~1` 之间，说明模型输出已经经过 sigmoid。

但飞机原来的 `animal_detector_rknn.py` 仍然执行：

```python
output0 = sigmoid(outputs[0])
output1 = sigmoid(outputs[1])
output2 = sigmoid(outputs[2])
```

这会造成二次 sigmoid，把原本较低的候选框分数抬高到接近 `0.5`，导致大量候选框通过后处理，最终表现为同一个动物多个框。

## 为什么老 RKNN 没问题

老 RKNN 模型和旧代码的输出处理方式刚好匹配。

老模型可能输出的是未 sigmoid 的 raw logits，所以旧代码手动 sigmoid 是正确的。

新模型输出已经是 `0~1`，继续手动 sigmoid 就会出错。

所以问题不是训练集一定坏了，也不是 YOLO NMS 完全没调用，而是：

```text
新 RKNN 输出格式变了，旧后处理逻辑没有同步调整。
```

## 验证过程

1. 先测试 `best.pt`。

结果：PT 检测正常，一个动物基本一个框，说明训练结果本身基本没问题。

2. 检查 `best.onnx`。

确认模型是 5 类：

```text
names = {0: 'tiger', 1: 'peacock', 2: 'monkey', 3: 'elephant', 4: 'wolf'}
nc = 5
```

确认 ONNX 输出是三输出 head：

```text
output0 [1, 30, 80, 80]
286     [1, 30, 40, 40]
288     [1, 30, 20, 20]
```

确认输出范围已经是 `0~1`：

```text
min = 0.0
max ≈ 0.99
```

3. 对比是否二次 sigmoid。

不再 sigmoid 时，候选框数量明显正常。

二次 sigmoid 后，低分候选框被抬高，重复框明显增加。

## 修复方案

在 `animal_detector_rknn.py` 中加入参数：

```yaml
outputs_already_sigmoid: true
```

并根据该参数决定是否手动 sigmoid：

```python
def prepare_model_outputs(outputs, outputs_already_sigmoid):
    if outputs_already_sigmoid:
        return outputs[0], outputs[1], outputs[2]
    return sigmoid(outputs[0]), sigmoid(outputs[1]), sigmoid(outputs[2])
```

同时让 YAML 里的阈值真正生效：

```python
rknn_func.OBJ_THRESH = obj_thresh
rknn_func.NMS_THRESH = nms_thresh
```

并增加最终分数过滤：

```python
if float(detection["score"]) < score_thresh:
    continue
```

## 当前推荐配置

`mission_controller.yaml` 中：

```yaml
animal_detector:
  obj_thresh: 0.55
  score_thresh: 0.50
  nms_thresh: 0.25
  outputs_already_sigmoid: true
```

启动后应看到类似日志：

```text
animal_detector: ready model=... obj=0.55 score=0.50 nms=0.25 outputs_already_sigmoid=true
```

看到这行说明参数已经真正生效。

## 经验总结

RKNN 多框不一定是训练问题。

判断顺序应该是：

```text
PT 是否正常
ONNX 输出格式是否正确
RKNN 输出是否已经 sigmoid
后处理是否重复 sigmoid
NMS 和 score 阈值是否真正生效
```

这次问题的核心是：

```text
PT 正常，ONNX/RKNN 输出已经 sigmoid，旧 RKNN 后处理又 sigmoid 一次。
```

修复后，不需要重新训练模型，也不需要复杂重复框过滤，检测框恢复正常。
