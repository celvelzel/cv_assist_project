#!/usr/bin/env python3
"""
测试日志和 FPS 统计功能
不需要摄像头，仅测试核心功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logging, FPSCounter
import logging

def test_logging():
    """测试日志系统"""
    print("\n=== 测试日志系统 ===")
    
    # 配置日志
    setup_logging(
        log_dir="logs",
        log_level="INFO",
        log_to_file=True,
        log_to_console=True
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info("这是一条信息日志")
    logger.warning("这是一条警告日志")
    logger.error("这是一条错误日志")
    
    print("✓ 日志系统测试完成")
    print("  请检查 logs/ 目录下是否生成了日志文件")


def test_fps_counter():
    """测试 FPS 计数器"""
    print("\n=== 测试 FPS 计数器 ===")
    
    fps_counter = FPSCounter(window_size=10)
    
    # 模拟一些 FPS 数据
    test_fps_values = [30.0, 28.5, 31.2, 29.8, 30.5, 27.3, 32.1, 29.9, 30.3, 28.7]
    
    for fps in test_fps_values:
        fps_counter.update(fps)
    
    stats = fps_counter.get_stats()
    
    print(f"当前 FPS: {stats['current']:.2f}")
    print(f"平均 FPS: {stats['average']:.2f}")
    print(f"最小 FPS: {stats['min']:.2f}")
    print(f"最大 FPS: {stats['max']:.2f}")
    print(f"总帧数: {stats['total_frames']}")
    
    # 验证结果
    assert stats['total_frames'] == 10, "总帧数错误"
    assert abs(stats['average'] - 29.83) < 0.1, "平均 FPS 计算错误"
    assert stats['min'] == 27.3, "最小 FPS 错误"
    assert stats['max'] == 32.1, "最大 FPS 错误"
    
    print("✓ FPS 计数器测试通过")


def test_config():
    """测试配置加载"""
    print("\n=== 测试配置加载 ===")
    
    from config import SystemConfig, LoggingConfig
    
    config = SystemConfig()
    
    print(f"日志目录: {config.logging.log_dir}")
    print(f"日志级别: {config.logging.log_level}")
    print(f"FPS 统计: {config.logging.enable_fps_stats}")
    print(f"FPS 窗口: {config.logging.fps_window_size}")
    
    assert isinstance(config.logging, LoggingConfig), "日志配置类型错误"
    assert config.logging.log_dir == "logs", "日志目录配置错误"
    
    print("✓ 配置加载测试通过")


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print(" CV 视觉辅助系统 - 日志和 FPS 功能测试")
    print("="*60)
    
    try:
        test_config()
        test_logging()
        test_fps_counter()
        
        print("\n" + "="*60)
        print(" 所有测试通过 ✓")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
