"""
ONNX Runtime 轻量化推理模块

用于快速、高效的电影推荐推理，支持单个用户或批量推理。
依赖：onnxruntime
"""

import json
from pathlib import Path
from typing import Dict, List, Union, Tuple

import numpy as np


DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parent


def resolve_path(path_value: Union[str, Path]) -> Path:
    """解析路径，兼容在项目根目录或 deploy 目录启动。"""
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj

    candidates = [
        Path.cwd() / path_obj,
        PROJECT_ROOT / path_obj,
        DEPLOY_DIR / path_obj,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # 若尚不存在（例如用户希望新建输出），默认返回以当前目录为基准的路径。
    return Path.cwd() / path_obj


class RecommendationEngine:
    """
    基于 ONNX 模型的推荐引擎
    
    特点：
    - 轻量化部署：无需 PyTorch，仅需 ONNX Runtime
    - 快速推理：CPU/GPU 可选，推理速度快
    - 灵活输入：支持单个或批量序列
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        dataset_path: Union[str, Path],
        use_gpu: bool = False,
    ):
        """
        初始化推荐引擎
        
        参数：
            model_path: ONNX 模型文件路径
            dataset_path: 数据集配置文件（dataset.json）路径
            use_gpu: 是否使用 GPU 推理（需要 ONNX Runtime GPU 版本）
        """
        try:
            import onnxruntime as rt
        except ImportError:
            raise ImportError("请安装 onnxruntime: pip install onnxruntime")
        
        self.model_path = resolve_path(model_path)
        self.dataset_path = resolve_path(dataset_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {self.dataset_path}")
        
        # 加载数据集配置（包含编码映射）
        with self.dataset_path.open("r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        
        self.num_items = self.dataset["meta"]["num_items"]
        
        # 构建映射关系：id2item（用于解码推荐结果）
        self.id2item = {int(k): v for k, v in self.dataset["id2item"].items()}
        
        # 初始化 ONNX Runtime session
        providers = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.session = rt.InferenceSession(str(self.model_path), providers=providers)
        
        # 获取模型的输入/输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # 兼容旧版 dataset.json：优先读取 meta.max_len，不存在则从 ONNX 输入形状推断。
        self.max_len = self._resolve_max_len()
        
        print(f"推荐引擎已加载")
        print(f"  - 模型：{self.model_path.name}")
        print(f"  - 物品数：{self.num_items}")
        print(f"  - 序列长度：{self.max_len}")

    def _resolve_max_len(self) -> int:
        """解析序列长度，兼容缺少 meta.max_len 的数据集。"""
        meta = self.dataset.get("meta", {})
        meta_max_len = meta.get("max_len")
        if isinstance(meta_max_len, int) and meta_max_len > 0:
            return meta_max_len

        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) >= 2 and isinstance(input_shape[1], int) and input_shape[1] > 0:
            return int(input_shape[1])

        # 兜底值与训练默认参数保持一致。
        return 50
    
    def _pad_sequence(self, seq: List[int]) -> np.ndarray:
        """
        将序列左侧补零到固定长度
        
        参数：
            seq: 电影 ID 序列
        
        返回：
            (max_len,) 的 numpy 数组
        """
        # 取最后 max_len 个元素（防止超长序列）
        seq = seq[-self.max_len:]
        # 左侧补零
        padded = [0] * (self.max_len - len(seq)) + seq
        return np.array(padded, dtype=np.int64)

    def _pad_sequence_no_zero(self, seq: List[int]) -> np.ndarray:
        """左侧使用非 0 值填充，规避部分 ONNX 图在 padding=0 时产生 NaN。"""
        seq = seq[-self.max_len :]
        if seq:
            pad_value = seq[0]
        else:
            pad_value = 1 if self.num_items >= 1 else 0
        padded = [pad_value] * (self.max_len - len(seq)) + seq
        return np.array(padded, dtype=np.int64)
    
    def _filter_and_rank(
        self,
        scores: np.ndarray,
        history: List[int],
        topk: int = 10,
        filter_history: bool = True,
    ) -> List[Tuple[int, float]]:
        """
        过滤历史物品并排序，获取 Top-K 推荐
        
        参数：
            scores: 模型输出的分数，shape (num_items+1,)
            history: 用户历史序列
            topk: 返回推荐数量
            filter_history: 是否过滤掉历史物品
        
        返回：
            [(电影ID, 分数), ...] 的列表，长度最多为 topk
        """
        # ONNX 输出存在 NaN/Inf 时，避免静默返回空推荐。
        finite_mask = np.isfinite(scores)
        if not finite_mask.any():
            raise ValueError(
                "模型输出全部为 NaN/Inf，当前 ONNX 文件可能损坏。"
                "请重新导出 outputs/model.onnx 后再推理。"
            )

        # 将 NaN/Inf 转成可排序数值，避免排序与比较阶段行为异常。
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(np.float32).max, neginf=-np.inf)

        # 过滤掉历史物品和 padding（id=0）
        if filter_history:
            history_set = set(history)
            for item_id in history_set:
                if 0 < item_id <= self.num_items:
                    scores[item_id] = -np.inf
            scores[0] = -np.inf  # 过滤 padding
        
        # 获取 Top-K 索引
        top_indices = np.argsort(-scores)[:topk]
        
        # 返回 [(id, score), ...] 列表
        results = []
        for idx in top_indices:
            if np.isfinite(scores[idx]):
                results.append((int(idx), float(scores[idx])))
        
        return results
    
    def recommend(
        self,
        history: List[int],
        topk: int = 10,
        filter_history: bool = True,
        return_scores: bool = False,
    ) -> List[Dict]:
        """
        为单个用户生成推荐
        
        参数：
            history: 用户历史序列（电影 ID 列表）
            topk: 返回推荐数量（默认 10）
            filter_history: 是否过滤掉历史物品（默认 True）
            return_scores: 是否返回推荐分数（默认 False）
        
        返回：
            [{"id": 电影ID, "title": 电影标题, "score": 分数}, ...] 的列表
        """
        # 填充序列
        padded_seq = self._pad_sequence(history).reshape(1, -1)
        
        # 模型推理
        scores = self.session.run(
            [self.output_name],
            {self.input_name: padded_seq}
        )[0][0]  # 取第一个样本的输出

        # 某些导出的 ONNX 图在 0-padding 输入下会输出全 NaN，此时自动切换到非 0 填充重试。
        if not np.isfinite(scores).any():
            padded_seq = self._pad_sequence_no_zero(history).reshape(1, -1)
            scores = self.session.run(
                [self.output_name],
                {self.input_name: padded_seq}
            )[0][0]
        
        # 过滤并排序
        top_items = self._filter_and_rank(scores, history, topk, filter_history)
        
        # 格式化输出
        results = []
        for item_id, score in top_items:
            result = {
                "id": item_id,
                "title": self.id2item.get(item_id, f"电影{item_id}"),
            }
            if return_scores:
                result["score"] = score
            results.append(result)
        
        return results
    
    def batch_recommend(
        self,
        histories: List[List[int]],
        topk: int = 10,
        filter_history: bool = True,
        return_scores: bool = False,
    ) -> List[List[Dict]]:
        """
        批量推荐（多个用户）
        
        参数：
            histories: 用户历史序列列表
            topk: 每个用户返回推荐数量
            filter_history: 是否过滤掉历史物品
            return_scores: 是否返回分数
        
        返回：
            推荐结果的嵌套列表，外层对应用户，内层为推荐列表
        """
        batch_size = len(histories)
        
        # 批量填充（标准 0-padding）
        padded_seqs = np.array([
            self._pad_sequence(h) for h in histories
        ], dtype=np.int64)
        
        # 批量推理
        scores_batch = self.session.run(
            [self.output_name],
            {self.input_name: padded_seqs}
        )[0]  # shape (batch_size, num_items+1)

        # 对全 NaN/Inf 的样本执行逐条降级重试，避免整批结果不可用。
        invalid_rows = ~np.isfinite(scores_batch).any(axis=1)
        if invalid_rows.any():
            for i in np.where(invalid_rows)[0]:
                retry_input = self._pad_sequence_no_zero(histories[i]).reshape(1, -1)
                retry_scores = self.session.run(
                    [self.output_name],
                    {self.input_name: retry_input}
                )[0][0]
                scores_batch[i] = retry_scores
        
        # 逐个用户处理
        results = []
        for i, scores in enumerate(scores_batch):
            top_items = self._filter_and_rank(scores, histories[i], topk, filter_history)
            
            user_results = []
            for item_id, score in top_items:
                result = {
                    "id": item_id,
                    "title": self.id2item.get(item_id, f"电影{item_id}"),
                }
                if return_scores:
                    result["score"] = score
                user_results.append(result)
            
            results.append(user_results)
        
        return results


if __name__ == "__main__":
    # 使用示例
    print("ONNX Runtime 推荐引擎使用示例\n")
    
    # 初始化引擎
    engine = RecommendationEngine(
        model_path="outputs/model.onnx",
        dataset_path="data/processed/dataset.json",
        use_gpu=False,
    )
    
    # 单个用户推荐
    print("\n【单用户推荐】")
    history = [1, 2, 3, 5, 7]  # 用户历史序列（电影 ID）
    recommendations = engine.recommend(history, topk=5, return_scores=True)
    print(f"用户历史：{history}")
    print("推荐结果：")
    for rec in recommendations:
        print(f"  - {rec['title']} (ID: {rec['id']}, 分数: {rec['score']:.4f})")
    
    # 批量用户推荐
    print("\n【批量推荐】")
    histories = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    batch_results = engine.batch_recommend(histories, topk=3)
    for i, recs in enumerate(batch_results):
        print(f"\n用户 {i+1}：")
        for rec in recs:
            print(f"  - {rec['title']} (ID: {rec['id']})")
