"""
推荐引擎 Flask API 服务

提供 HTTP 接口，支持单用户和批量推荐查询
"""

import json
from pathlib import Path
from typing import Tuple, Dict

from flask import Flask, request, jsonify

from inference import RecommendationEngine


class RecommendationAPI:
    """推荐 API 服务"""
    
    def __init__(
        self,
        model_path: str = "outputs/model.onnx",
        dataset_path: str = "data/processed/dataset.json",
        use_gpu: bool = False,
    ):
        """初始化 API 服务"""
        self.app = Flask(__name__)
        self.engine = RecommendationEngine(
            model_path=model_path,
            dataset_path=dataset_path,
            use_gpu=use_gpu,
        )
        
        # 加载数据集
        with Path(dataset_path).open("r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        
        self._setup_routes()
    
    def _setup_routes(self):
        """配置路由"""
        
        @self.app.route("/health", methods=["GET"])
        def health():
            """健康检查"""
            return jsonify({
                "status": "healthy",
                "model": "SASRec",
                "num_items": self.engine.num_items,
            })
        
        @self.app.route("/users", methods=["GET"])
        def list_users():
            """获取所有用户列表"""
            users = list(self.dataset.get("user_histories", {}).keys())
            return jsonify({
                "count": len(users),
                "users": users,
            })
        
        @self.app.route("/recommend", methods=["POST"])
        def recommend():
            """
            单用户推荐接口
            
            POST 请求体格式：
            {
                "user": "用户名" 或 "history": [电影ID列表],
                "topk": 10 (可选，默认10)
            }
            
            返回：
            {
                "user": "用户名",
                "history": [历史序列],
                "recommendations": [推荐列表]
            }
            """
            data = request.get_json()
            
            # 获取历史序列
            if "user" in data:
                user_name = data["user"]
                user_histories = self.dataset.get("user_histories", {})
                
                if user_name not in user_histories:
                    return jsonify({
                        "error": f"用户 '{user_name}' 未找到"
                    }), 404
                
                history = user_histories[user_name]
            elif "history" in data:
                history = data["history"]
                user_name = None
            else:
                return jsonify({
                    "error": "请提供 'user' 或 'history' 字段"
                }), 400
            
            topk = data.get("topk", 10)
            
            # 验证参数
            if not isinstance(history, list):
                return jsonify({"error": "history 必须是列表"}), 400
            if topk <= 0 or topk > 100:
                return jsonify({"error": "topk 必须在 1-100 之间"}), 400
            
            # 生成推荐
            try:
                recommendations = self.engine.recommend(
                    history,
                    topk=topk,
                    return_scores=True,
                )
                
                result = {
                    "user": user_name,
                    "history": history,
                    "topk": topk,
                    "recommendations": recommendations,
                }
                
                return jsonify(result)
            
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route("/recommend/batch", methods=["POST"])
        def batch_recommend():
            """
            批量推荐接口
            
            POST 请求体格式：
            {
                "histories": [[电影ID列表], ...] 或 {用户名: [列表], ...},
                "topk": 10 (可选，默认10)
            }
            
            返回：
            {
                "count": 推荐数量,
                "results": {用户索引或名称: [推荐列表], ...}
            }
            """
            data = request.get_json()
            
            if "histories" not in data:
                return jsonify({
                    "error": "请提供 'histories' 字段"
                }), 400
            
            histories_data = data["histories"]
            topk = data.get("topk", 10)
            
            # 支持字典格式 {用户名: [历史]} 或列表格式 [[历史], ...]
            if isinstance(histories_data, dict):
                user_names = list(histories_data.keys())
                histories = list(histories_data.values())
            else:
                user_names = None
                histories = histories_data
            
            # 验证参数
            if not isinstance(histories, list) or len(histories) == 0:
                return jsonify({"error": "histories 必须是非空列表"}), 400
            if topk <= 0 or topk > 100:
                return jsonify({"error": "topk 必须在 1-100 之间"}), 400
            
            # 生成推荐
            try:
                batch_results = self.engine.batch_recommend(
                    histories,
                    topk=topk,
                    return_scores=True,
                )
                
                # 组织返回结果
                results = {}
                for i, recs in enumerate(batch_results):
                    key = user_names[i] if user_names else str(i)
                    results[key] = recs
                
                return jsonify({
                    "count": len(results),
                    "topk": topk,
                    "results": results,
                })
            
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.errorhandler(404)
        def not_found(error):
            """404 处理"""
            return jsonify({"error": "请求路径未找到"}), 404
        
        @self.app.errorhandler(500)
        def server_error(error):
            """500 处理"""
            return jsonify({"error": "服务器内部错误"}), 500
    
    def run(self, host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
        """启动 API 服务"""
        print(f"推荐 API 服务启动于 http://{host}:{port}")
        print(f"可用端点：")
        print(f"  GET  /health              - 健康检查")
        print(f"  GET  /users               - 获取用户列表")
        print(f"  POST /recommend           - 单用户推荐")
        print(f"  POST /recommend/batch     - 批量推荐")
        self.app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="推荐 API 服务")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/model.onnx",
        help="ONNX 模型路径",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/dataset.json",
        help="数据集配置路径",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="服务监听地址（默认：127.0.0.1）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="服务监听端口（默认：5000）",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="使用 GPU 推理",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式",
    )
    
    args = parser.parse_args()
    
    api = RecommendationAPI(
        model_path=args.model,
        dataset_path=args.dataset,
        use_gpu=args.gpu,
    )
    api.run(host=args.host, port=args.port, debug=args.debug)
