# 更新日志

## 2026-02-26 - 日志和异常处理增强

### 新增功能

#### 1. 日志系统
- ✅ 在根目录创建 `logs` 文件夹用于存储日志文件
- ✅ 添加 `utils/logger.py` 模块，提供日志配置功能
- ✅ 日志文件按时间戳命名：`cv_assist_YYYYMMDD_HHMMSS.log`
- ✅ 支持同时输出到文件和控制台
- ✅ 配置化的日志级别和选项

#### 2. FPS 统计增强
- ✅ 新增 `FPSCounter` 类，提供完整的 FPS 统计功能
- ✅ 跟踪当前 FPS、平均 FPS、最小 FPS、最大 FPS
- ✅ 每 100 帧自动记录 FPS 统计到日志
- ✅ 系统退出时显示完整的 FPS 统计报告
- ✅ 可配置的 FPS 平滑窗口大小

#### 3. 异常处理
- ✅ 摄像头初始化异常捕获和友好提示
- ✅ 模型加载异常捕获（OWL-ViT、MiDaS、MediaPipe）
- ✅ 帧处理异常捕获，确保单帧错误不影响整体运行
- ✅ 优雅的错误退出和资源清理
- ✅ 详细的错误诊断信息和解决建议

### 文件修改

#### `config.py`
- 新增 `LoggingConfig` 数据类
- `SystemConfig` 中添加 `logging` 配置项

#### `utils/logger.py` (新建)
- `FPSCounter` 类：FPS 统计计数器
- `setup_logging()` 函数：配置日志系统

#### `core/system.py`
- 集成日志系统，使用 `setup_logging()`
- 使用 `FPSCounter` 进行 FPS 统计
- 在组件初始化时添加异常处理
- 在摄像头打开时添加异常处理和诊断
- 在主循环中添加异常处理
- 退出时显示完整的 FPS 统计

#### `detectors/owl_vit_detector.py`
- 模型加载时添加详细的异常处理
- 添加错误诊断和解决建议
- 预热函数添加异常处理

#### `detectors/depth_estimator.py`
- 模型加载时添加详细的异常处理
- 优化模型加载失败时的降级处理
- 添加错误诊断信息

#### `detectors/hand_tracker.py`
- 添加手部追踪器初始化异常处理
- 添加模型下载失败的错误处理
- 添加错误诊断信息

#### `.gitignore`
- 添加 `logs/` 和 `*.log` 排除规则

### 使用说明

#### 查看日志
日志文件保存在 `logs/` 目录下，按时间戳命名。可以直接打开查看详细的运行信息。

#### 配置日志
在创建 `SystemConfig` 时可以自定义日志配置：

```python
from config import SystemConfig, LoggingConfig

config = SystemConfig()
config.logging.log_level = "DEBUG"  # 设置日志级别
config.logging.log_to_file = True   # 输出到文件
config.logging.log_to_console = True # 输出到控制台
config.logging.enable_fps_stats = True # 启用 FPS 统计
```

#### FPS 统计
系统运行时会：
- 在屏幕上显示当前 FPS 和平均 FPS
- 每 100 帧在日志中记录一次统计信息
- 退出时显示完整的统计报告（总帧数、平均/最小/最大 FPS）

#### 错误处理
当遇到错误时：
1. 系统会记录详细的错误信息到日志
2. 显示友好的错误提示和诊断建议
3. 尽可能优雅地退出或降级运行

### 测试建议

1. **正常运行测试**：确保日志文件正确生成
2. **摄像头错误测试**：使用不存在的摄像头 ID 测试错误处理
3. **网络错误测试**：在无网络环境下测试模型加载降级
4. **FPS 统计测试**：运行一段时间后检查 FPS 统计是否正确

### 注意事项

- 日志文件会随时间累积，建议定期清理
- 首次运行可能需要下载模型，确保网络连接正常
- FPS 统计窗口大小影响平滑度，可根据需要调整
