# 技术更新报告：多目标追踪与空间选择

**日期**: 2026-03-24  
**作者**: 核心开发团队  
**主题**: 实现多目标追踪 (SORT) 与空间位置选择功能

---

## 1. 功能概述

本次更新引入了多目标追踪和空间选择功能，使用户能够通过语音指令选择特定位置的目标物体（如"左边的杯子" vs "右边的杯子"）。

### 核心特性

- **多目标追踪**: 使用 SORT 算法为检测到的物体分配稳定的追踪 ID，避免帧间目标跳变
- **空间选择**: 支持"左边/右边/中间/最近"等空间修饰符，根据物体位置智能选择
- **视角适配**: 支持前置摄像头（画面左右反转）和外置摄像头两种模式
- **抽象架构**: 使用 Protocol 接口设计，便于未来替换为 LLM 解析或其他追踪算法

---

## 2. 架构设计

### 2.1 核心接口 (`core/interfaces.py`)

```python
class ParsedIntent(NamedTuple):
    """用户意图结构化表示"""
    target_class: str          # 目标类别，如 "杯子"
    spatial_modifier: Optional[str]  # 空间修饰符，如 "left", "right"
    raw_text: str              # 原始语音文本

class IIntentParser(Protocol):
    """意图解析器接口"""
    def parse(self, text: str) -> Optional[ParsedIntent]: ...

class ISpatialResolver(Protocol):
    """空间解析器接口"""
    def resolve(self, spatial_modifier: str, tracks: List[TrackedDetection]) -> Optional[TrackedDetection]: ...
```

### 2.2 意图解析器 (`audio/intent_parser.py`)

- **RegexIntentParser**: 基于正则表达式的规则解析器
  - 支持中文: "找到左边的杯子", "右边的手机"
  - 支持英文: "find the left cup", "locate the right bottle"
  - 未来可扩展: LLMIntentParser（使用大语言模型进行语义解析）

### 2.3 空间解析器 (`core/spatial_resolver.py`)

- **StandardSpatialResolver**: 标准模式（画面左 = 用户左）
- **InvertedSpatialResolver**: 反转模式（前置摄像头，画面左 = 用户右）
- **AdaptiveSpatialResolver**: 自适应模式（根据配置切换）

### 2.4 目标追踪器 (`core/tracker.py`)

- **NorfairObjectTracker**: 使用 Norfair 库的 SORT 实现
- **SimpleCentroidTracker**: 轻量级质心追踪器（无需额外依赖）

---

## 3. 实现细节

### 3.1 追踪流程

```
摄像头帧 → OWL-ViT 检测 → 目标追踪器 → 空间解析器 → 引导控制器
                ↓                ↓              ↓
            检测框列表      稳定的追踪ID     选定的目标
```

### 3.2 语音指令解析

用户说"找到左边的杯子"：
1. ASR 转录为文本: "找到左边的杯子"
2. 意图解析器提取: `ParsedIntent(target_class="杯子", spatial_modifier="left")`
3. 系统锁定目标类别为"杯子"，空间提示为"left"
4. 追踪器为所有杯子分配稳定 ID
5. 空间解析器根据 x 坐标排序，选择最左边的杯子
6. 引导控制器追踪选定的杯子，直到目标丢失或用户更换指令

### 3.3 视角反转处理

当 `camera_facing = "user"` 时（前置摄像头）：
- 用户说"左边" → 选择画面中 x 坐标最大的目标（最右边）
- 用户说"右边" → 选择画面中 x 坐标最小的目标（最左边）

---

## 4. 测试覆盖

| 模块 | 测试文件 | 测试数量 | 状态 |
|------|----------|----------|------|
| 接口定义 | `test_interfaces.py` | 6 | ✅ 通过 |
| 意图解析 | `test_intent_parser.py` | 10 | ✅ 通过 |
| 空间解析 | `test_spatial_resolver.py` | 15 | ✅ 通过 |
| 目标追踪 | `test_tracker.py` | 27 | ✅ 通过 |
| **总计** | | **58** | **✅ 全部通过** |

---

## 5. 配置选项

在 `config.py` 中添加以下配置：

```python
# 目标追踪配置
tracker_type: str = "simple"  # "simple" 或 "norfair"

# 空间解析配置  
camera_facing: str = "outward"  # "outward" 或 "user"

# 意图解析配置
intent_parser_type: str = "regex"  # "regex" 或未来 "llm"
```

---

## 6. 未来发展方向

### 6.1 近期优化

- **追踪稳定性**: 优化 SORT 参数（距离阈值、最大丢失帧数）
- **多类别支持**: 同时追踪不同类别的物体
- **深度集成**: 结合深度信息进行三维空间选择

### 6.2 中期规划

- **LLM 意图解析**: 使用大语言模型解析复杂指令
  - "把左边那个红色的杯子给我"
  - "找到离我最近的瓶子"
- **手势交互**: 用手势代替语音进行目标选择
- **目标记忆**: 跨帧记住用户选定的目标

### 6.3 长期愿景

- **3D 空间感知**: 基于深度图的三维空间选择
- **动态视角**: 支持摄像头移动时的视角自适应
- **多模态交互**: 语音 + 手势 + 眼动追踪的融合

---

## 7. 技术债务与注意事项

- **Nofair 依赖**: 当前使用 mock 实现（norfair 未安装），实际部署需安装 `norfair`
- **追踪精度**: 简单质心追踪器在目标密集时可能误匹配
- **配置管理**: 新增配置项尚未加入 `config.yaml` 示例

---

*文档生成时间: 2026-03-24*
