#!/usr/bin/env python3
"""
相机控制能力探测脚本。

用途:
1. 打开指定摄像头
2. 读取常见相机属性
3. 尝试设置 zoom / focus / autofocus
4. 输出每项能力是否真正生效

示例:
    python camera_control_probe.py --camera 1
    python camera_control_probe.py --camera 1 --backend dshow
    python camera_control_probe.py --camera 1 --try-autofocus 0 --try-focus 20
    python camera_control_probe.py --camera 1 --try-zoom 100
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, Optional, Tuple

import cv2


BACKEND_MAP = {
    "auto": None,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "any": cv2.CAP_ANY,
}

PROPERTY_MAP = {
    "frame_width": cv2.CAP_PROP_FRAME_WIDTH,
    "frame_height": cv2.CAP_PROP_FRAME_HEIGHT,
    "fps": cv2.CAP_PROP_FPS,
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
    "contrast": cv2.CAP_PROP_CONTRAST,
    "saturation": cv2.CAP_PROP_SATURATION,
    "sharpness": cv2.CAP_PROP_SHARPNESS,
    "gain": cv2.CAP_PROP_GAIN,
    "exposure": cv2.CAP_PROP_EXPOSURE,
    "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    "zoom": cv2.CAP_PROP_ZOOM,
    "focus": cv2.CAP_PROP_FOCUS,
    "autofocus": cv2.CAP_PROP_AUTOFOCUS,
}


def _safe_float(value: float) -> Optional[float]:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _open_camera(camera_id: int, backend_name: str) -> cv2.VideoCapture:
    backend = BACKEND_MAP.get(backend_name)
    if backend is None:
        cap = cv2.VideoCapture(camera_id)
    else:
        cap = cv2.VideoCapture(camera_id, backend)
    return cap


def _read_properties(cap: cv2.VideoCapture) -> Dict[str, Optional[float]]:
    props: Dict[str, Optional[float]] = {}
    for name, prop_id in PROPERTY_MAP.items():
        props[name] = _safe_float(cap.get(prop_id))
    return props


def _try_set_property(
    cap: cv2.VideoCapture,
    property_name: str,
    target_value: float,
) -> Dict[str, object]:
    prop_id = PROPERTY_MAP[property_name]
    before = _safe_float(cap.get(prop_id))
    set_ok = bool(cap.set(prop_id, target_value))
    after = _safe_float(cap.get(prop_id))

    changed = False
    if before is not None and after is not None:
        changed = abs(after - before) > 1e-6

    accepted = False
    if after is not None:
        accepted = abs(after - float(target_value)) <= 1e-3

    return {
        "property": property_name,
        "requested_value": target_value,
        "before": before,
        "after": after,
        "set_call_returned": set_ok,
        "changed": changed,
        "accepted_exactly": accepted,
    }


def _print_report(
    camera_id: int,
    backend_name: str,
    props_before: Dict[str, Optional[float]],
    probe_results: Dict[str, Dict[str, object]],
) -> None:
    print("=== 相机控制能力探测 ===")
    print(f"camera_id: {camera_id}")
    print(f"backend: {backend_name}")
    print()

    print("当前属性:")
    for key, value in props_before.items():
        print(f"  {key}: {value}")
    print()

    if not probe_results:
        print("未执行写入探测。可附加参数:")
        print("  --try-autofocus 0/1")
        print("  --try-focus <value>")
        print("  --try-zoom <value>")
        return

    print("写入探测结果:")
    for key, result in probe_results.items():
        print(f"  [{key}]")
        print(f"    requested_value: {result['requested_value']}")
        print(f"    before: {result['before']}")
        print(f"    after: {result['after']}")
        print(f"    set_call_returned: {result['set_call_returned']}")
        print(f"    changed: {result['changed']}")
        print(f"    accepted_exactly: {result['accepted_exactly']}")
    print()

    print("JSON 摘要:")
    print(
        json.dumps(
            {
                "camera_id": camera_id,
                "backend": backend_name,
                "properties_before": props_before,
                "probe_results": probe_results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="探测 USB 摄像头是否支持 zoom/focus/autofocus 调整")
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引，默认 0")
    parser.add_argument(
        "--backend",
        choices=sorted(BACKEND_MAP.keys()),
        default="auto",
        help="OpenCV 后端，Windows 下可尝试 dshow 或 msmf",
    )
    parser.add_argument("--try-autofocus", type=float, default=None, help="尝试设置 autofocus，例如 0 或 1")
    parser.add_argument("--try-focus", type=float, default=None, help="尝试设置 focus 值")
    parser.add_argument("--try-zoom", type=float, default=None, help="尝试设置 zoom 值")
    args = parser.parse_args()

    cap = _open_camera(args.camera, args.backend)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 {args.camera}，backend={args.backend}")
        return 1

    ret, _frame = cap.read()
    if not ret:
        print(f"[ERROR] 摄像头 {args.camera} 打开成功，但读取首帧失败")
        cap.release()
        return 2

    props_before = _read_properties(cap)
    probe_results: Dict[str, Dict[str, object]] = {}

    if args.try_autofocus is not None:
        probe_results["autofocus"] = _try_set_property(cap, "autofocus", args.try_autofocus)
    if args.try_focus is not None:
        probe_results["focus"] = _try_set_property(cap, "focus", args.try_focus)
    if args.try_zoom is not None:
        probe_results["zoom"] = _try_set_property(cap, "zoom", args.try_zoom)

    _print_report(args.camera, args.backend, props_before, probe_results)
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
