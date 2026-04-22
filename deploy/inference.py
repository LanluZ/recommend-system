"""
ONNX Runtime 轻量化推理模块

用于快速、高效的电影推荐推理，支持单个用户或批量推理。
依赖：onnxruntime
"""

import json
from pathlib import Path
from typing import Dict, List, Union, Tuple

import numpy as np


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
        
        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path)
        
        # 加载数据集配置（包含编码映射）
        with self.dataset_path.open("r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        
        self.num_items = self.dataset["meta"]["num_items"]
        self.max_len = self.dataset["meta"]["max_len"]
        
        # 构建映射关系：id2item（用于解码推荐结果）
        self.id2item = {int(k): v for k, v in self.dataset["id2item"].items()}
        
        # 初始化 ONNX Runtime session
        providers = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.session = rt.InferenceSession(str(self.model_path), providers=providers)
        
        # 获取模型的输入/输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"推荐引擎已加载")
        print(f"  - 模型：{self.model_path.name}")
        print(f"  - 物品数：{self.num_items}")
        print(f"  - 序列长度：{self.max_len}")
    
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
            if scores[idx] > -np.inf:  # 排除被过滤的物品
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
        
        # 批量填充
        padded_seqs = np.array([
            self._pad_sequence(h) for h in histories
        ], dtype=np.int64)
        
        # 批量推理
        scores_batch = self.session.run(
            [self.output_name],
            {self.input_name: padded_seqs}
        )[0]  # shape (batch_size, num_items+1)
        
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
