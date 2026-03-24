# 未来扩展指南

本文档描述了 CV Assist 项目的扩展点和开发指南，帮助开发者快速添加新功能。

---

## 1. 扩展意图解析器

### 1.1 现有实现

当前使用 `RegexIntentParser`，基于正则表达式解析语音指令。

### 1.2 添加新的解析器

要添加新的意图解析器（如 LLM 解析器），请遵循以下步骤：

#### 步骤 1: 实现接口

创建新文件 `audio/llm_intent_parser.py`:

```python
from core.interfaces import ParsedIntent, IIntentParser
from typing import Optional

class LLMIntentParser:
    """基于大语言模型的意图解析器"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        # 初始化 LLM 客户端
        
    def parse(self, text: str) -> Optional[ParsedIntent]:
        """
        使用 LLM 解析用户意图
        
        Args:
            text: 原始语音文本
            
        Returns:
            ParsedIntent 或 None
        """
        # 构建提示词
        prompt = f"""
        从以下语音指令中提取目标物体和空间修饰符：
        指令: "{text}"
        
        返回 JSON 格式:
        {{"target": "物体名称", "spatial": "left/right/center/nearest"}}
        """
        
        # 调用 LLM
        response = self._call_llm(prompt)
        
        # 解析响应
        try:
            result = json.loads(response)
            return ParsedIntent(
                target_class=result['target'],
                spatial_modifier=result.get('spatial'),
                raw_text=text
            )
        except:
            return None
```

#### 步骤 2: 注册到工厂函数

在 `audio/intent_parser.py` 中更新 `create_intent_parser`:

```python
def create_intent_parser(parser_type: str = "regex") -> IIntentParser:
    if parser_type == "regex":
        return RegexIntentParser()
    elif parser_type == "llm":
        from audio.llm_intent_parser import LLMIntentParser
        return LLMIntentParser()
    else:
        raise ValueError(f"Unknown parser type: {parser_type}")
```

#### 步骤 3: 更新配置

在 `config.py` 中添加配置项:

```python
@dataclass
class AudioConfig:
    # ... 现有配置 ...
    intent_parser_type: str = "regex"  # "regex" 或 "llm"
    llm_model_name: str = "gpt-3.5-turbo"  # LLM 模型名称
```

---

## 2. 扩展空间解析器

### 2.1 现有实现

- `StandardSpatialResolver`: 标准模式（画面左 = 用户左）
- `InvertedSpatialResolver`: 反转模式（画面左 = 用户右）

### 2.2 添加新的解析器

创建新文件 `core/custom_spatial_resolver.py`:

```python
from core.interfaces import ISpatialResolver, TrackedDetection
from typing import List, Optional

class DepthAwareSpatialResolver:
    """基于深度的空间解析器（考虑物体距离）"""
    
    def resolve(
        self,
        spatial_modifier: str,
        tracks: List[TrackedDetection],
        reference_point: Optional[tuple] = None
    ) -> Optional[TrackedDetection]:
        """
        根据空间修饰符和深度选择目标
        """
        if not tracks:
            return None
        
        if spatial_modifier == "nearest":
            # 选择最近的物体（基于深度值）
            return min(tracks, key=lambda t: t.depth if hasattr(t, 'depth') else float('inf'))
        
        # ... 其他逻辑 ...
```

### 2.3 注册到工厂函数

在 `core/spatial_resolver.py` 中更新 `create_spatial_resolver`:

```python
def create_spatial_resolver(camera_facing: CameraFacing = "outward") -> ISpatialResolver:
    return AdaptiveSpatialResolver(camera_facing)
```

---

## 3. 扩展目标追踪器

### 3.1 现有实现

- `SimpleCentroidTracker`: 轻量级质心追踪器
- `NorfairObjectTracker`: 基于 Norfair 的 SORT 实现

### 3.2 添加新的追踪器

创建新文件 `core/deep_sort_tracker.py`:

```python
from core.interfaces import TrackedDetection
from typing import List, Dict, Any

class DeepSORTTracker:
    """DeepSORT 追踪器（带外观特征）"""
    
    def __init__(self, 
                 max_disappeared: int = 30,
                 max_distance: float = 100.0,
                 feature_dim: int = 128):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.feature_dim = feature_dim
        # 初始化外观特征提取器
        
    def update(self, detections: List[Dict[str, Any]]) -> List[TrackedDetection]:
        """
        更新追踪状态
        
        Args:
            detections: 检测结果列表
            
        Returns:
            追踪结果列表
        """
        # 提取外观特征
        features = self._extract_features(detections)
        
        # 匹配追踪目标
        # ... DeepSORT 算法 ...
        
        return tracked_detections
```

### 3.3 注册到工厂函数

在 `core/tracker.py` 中更新 `create_object_tracker`:

```python
def create_object_tracker(tracker_type: str = "simple", **kwargs) -> Any:
    if tracker_type == "simple":
        return SimpleCentroidTracker(**kwargs)
    elif tracker_type == "norfair":
        return NorfairObjectTracker(**kwargs)
    elif tracker_type == "deep_sort":
        from core.deep_sort_tracker import DeepSORTTracker
        return DeepSORTTracker(**kwargs)
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")
```

---

## 4. 添加新的空间修饰符

### 4.1 修改意图解析器

在 `audio/intent_parser.py` 中添加新的空间修饰符:

```python
class RegexIntentParser:
    def __init__(self):
        # 现有的中文空间修饰符
        self.zh_spatial = {
            '左边': 'left',
            '右边': 'right',
            # 添加新的修饰符
            '前面': 'front',
            '后面': 'back',
            '上面': 'above',
            '下面': 'below',
        }
        
        # 现有的英文空间修饰符
        self.en_spatial = {
            'left': 'left',
            'right': 'right',
            # 添加新的修饰符
            'front': 'front',
            'back': 'back',
            'above': 'above',
            'below': 'below',
        }
```

### 4.2 修改空间解析器

在 `core/spatial_resolver.py` 中处理新的空间修饰符:

```python
class StandardSpatialResolver:
    def resolve(self, spatial_modifier: str, tracks: List[TrackedDetection], 
                reference_point: Optional[tuple] = None) -> Optional[TrackedDetection]:
        if not tracks:
            return None
        
        # 现有的水平方向
        if spatial_modifier in ["left", "左边"]:
            return min(valid_tracks, key=lambda t: t.center[0])
        elif spatial_modifier in ["right", "右边"]:
            return max(valid_tracks, key=lambda t: t.center[0])
        
        # 添加垂直方向
        elif spatial_modifier in ["above", "上面"]:
            return min(valid_tracks, key=lambda t: t.center[1])
        elif spatial_modifier in ["below", "下面"]:
            return max(valid_tracks, key=lambda t: t.center[1])
        
        # 添加前后方向（基于深度）
        elif spatial_modifier in ["front", "前面"]:
            return min(valid_tracks, key=lambda t: getattr(t, 'depth', 0.5))
        elif spatial_modifier in ["back", "后面"]:
            return max(valid_tracks, key=lambda t: getattr(t, 'depth', 0.5))
```

---

## 5. 测试指南

### 5.1 为新解析器添加测试

创建测试文件 `tests/test_llm_intent_parser.py`:

```python
import unittest
from audio.llm_intent_parser import LLMIntentParser

class TestLLMIntentParser(unittest.TestCase):
    def setUp(self):
        self.parser = LLMIntentParser()
    
    def test_basic_parsing(self):
        result = self.parser.parse("找到左边的杯子")
        self.assertIsNotNone(result)
        self.assertEqual(result.target_class, "杯子")
        self.assertEqual(result.spatial_modifier, "left")
    
    def test_complex_parsing(self):
        result = self.parser.parse("把那个红色的杯子给我")
        self.assertIsNotNone(result)
        # LLM 应该能理解"红色的杯子"
```

### 5.2 运行测试

```bash
# 运行单个测试文件
python -m pytest tests/test_llm_intent_parser.py -v

# 运行所有测试
python -m pytest tests/ -v
```

---

## 6. 最佳实践

### 6.1 接口设计

- 始终使用 `Protocol` 定义接口，而不是抽象基类
- 保持接口简单，只包含必要的方法
- 使用 `NamedTuple` 或 `dataclass` 定义数据结构

### 6.2 错误处理

- 所有解析器都应返回 `Optional[...]`，处理失败情况
- 使用日志记录解析失败的原因
- 提供向后兼容的回退机制

### 6.3 配置管理

- 新功能应通过配置开关控制
- 提供合理的默认值
- 在 `config.yaml` 中添加配置示例

### 6.4 测试覆盖

- 为每个新实现编写单元测试
- 测试正常情况和边界情况
- 测试不同配置下的行为

---

## 7. 常见问题

### Q: 如何切换摄像头模式？

A: 在 `config.py` 中设置 `camera_facing = "user"`（前置摄像头）或 `"outward"`（外置摄像头）。

### Q: 如何使用 Norfair 追踪器？

A: 安装 `norfair`: `pip install norfair`，然后在配置中设置 `tracker_type = "norfair"`。

### Q: 如何添加自定义空间修饰符？

A: 修改 `audio/intent_parser.py` 中的空间修饰符映射，并在 `core/spatial_resolver.py` 中添加处理逻辑。

### Q: 如何实现 LLM 意图解析？

A: 参考第 1.2 节，实现 `LLMIntentParser` 类并注册到工厂函数。

---

*文档生成时间: 2026-03-24*
