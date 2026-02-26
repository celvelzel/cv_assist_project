# cv_assist_project

这是一个计算机视觉助手项目，整合了 **OWL-ViT 目标检测**、**MediaPipe 手部追踪**、以及 **MiDaS 深度估计**，用于演示多模型协同工作实现智能视觉分析与交互。

---

## 🛠 技术栈

- **语言**：Python 3.8+
- **主要依赖**：
  - `torch`, `torchvision`（深度学习模型）
  - `opencv-python`（图像处理）
  - `mediapipe`（手部关键点检测）
  - `midas`（深度估计模型）
  - `owl-vit`（视觉-语言检测模型）
- **项目结构**：
  ```
  cv_assist_project/
  ├─ config.py          # 配置参数
  ├─ main.py            # 程序入口
  ├─ detectors/         # 各检测模块
  ├─ core/              # 系统集成与引导逻辑
  ├─ utils/             # 共用工具函数
  ├─ test_all.py        # 单元/集成测试
  └─ README.md          # 当前文档
  ```

---

## 🚀 启动项目

1. **环境准备**  
   ```bash
   cd cv_assist_project
   # 可选：创建 Python 虚拟环境并激活
   python -m venv .venv           # 仅第一次或当需要隔离依赖时
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   # Windows cmd
   call .venv\Scripts\activate
   # Linux / macOS
   source .venv/bin/activate

   pip install -r requirements.txt
   ```
   > 🔧 如果你希望脚本自动完成以上步骤，可使用仓库根目录下的 `run.sh`（bash）或新增的 `run.bat`（Windows）。

2. **配置**  
   - 编辑 `config.py` 修改输入源、模型路径等参数（如需）。

3. **运行**  
   ```bash
   # 直接执行
   python main.py
   # 或使用运行脚本（见上文）
   ./run.sh       # Linux/macOS
   run.bat        # Windows cmd
   ```
   程序将加载检测器，打开摄像头/视频流并在画面上叠加检测结果与深度信息。

   启动后会看到类似下面的输出：
   ```
   ============================================================
    CV 视觉辅助系统
   ============================================================
    OWL-ViT: google/owlvit-base-patch32
    MiDaS: MiDaS_small
    设备: cpu
    FP16: False
   ============================================================
   …
   控制：
     q - 退出
     d - 切换深度显示
   检测目标: ['a cup', 'a bottle']
   ```
   - 按 `d` 切换深度渲染，按 `q` 退出窗口。

4. **测试**  
   验证功能是否正常：
   ```bash
   python test_all.py
   ```


---

## 🧩 功能概览

- **目标检测** 使用 OWL-ViT 模型识别图像中物体。
- **手部追踪** MediaPipe 提供掌心和手指关键点坐标。
- **深度估计** MiDaS 生成场景深度图，用于测距或增强理解。
- **模块化设计**：各 detector 互相独立，可扩展新算法或模型。
- **简单的系统逻辑** 位于 `core/`，协调各模块输入输出，进行可视化展示。

---

## 🤝 协作建议

- **分支规范**：每个新功能以 feature/xxx 命名，提交说明清晰。
- **代码风格**：遵循 PEP8，建议安装并使用 `flake8` 或 `black`。
- **文档**：新增功能或改动请更新此 README 或 `doc/summary.md`。
- **测试**：每次改动后运行 `test_all.py`，确保模块兼容无误。

---

## 📦 其他

- 可通过修改 `requirements.txt` 添加更多依赖。
- 若需在 Linux/macOS 使用，请参考 `install.sh` 或 `run.sh`，命令类似。

---