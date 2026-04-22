"""
推荐系统轻量化推理部署模块

提供基于 ONNX Runtime 的高效推理接口
"""

from .inference import RecommendationEngine

__version__ = "1.0.0"
__all__ = ["RecommendationEngine"]
