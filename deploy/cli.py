"""
推荐引擎命令行工具

支持单用户和批量用户的推荐查询
"""

import argparse
import json
from pathlib import Path
from typing import List

from inference import RecommendationEngine


def load_user_histories(file_path: Path) -> dict:
    """
    从 JSON 文件加载用户历史（格式：{用户名: [电影ID列表]}）
    """
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def recommend_single_user(
    engine: RecommendationEngine,
    user_name: str,
    dataset: dict,
    topk: int = 10,
) -> dict:
    """
    为单个用户生成推荐
    
    参数：
        engine: 推荐引擎实例
        user_name: 用户名
        dataset: 数据集配置
        topk: 推荐数量
    
    返回：
        推荐结果字典
    """
    # 从数据集中获取用户历史
    user_histories = dataset.get("user_histories", {})
    
    if user_name not in user_histories:
        return {
            "error": f"用户 '{user_name}' 未在数据集中找到",
            "available_users": list(user_histories.keys())[:10],
        }
    
    history = user_histories[user_name]
    recommendations = engine.recommend(history, topk=topk, return_scores=True)
    
    return {
        "user": user_name,
        "history": history,
        "recommendations": recommendations,
    }


def main():
    parser = argparse.ArgumentParser(
        description="推荐引擎命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 单用户推荐
  python cli.py --user "张三" --topk 10

  # 从文件批量推荐
  python cli.py --batch users.json --topk 5

  # 查看可用用户
  python cli.py --list-users
        """,
    )
    
    parser.add_argument(
        "--model",
        type=Path,
        default="outputs/model.onnx",
        help="ONNX 模型路径（默认：outputs/model.onnx）",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default="data/processed/dataset.json",
        help="数据集配置路径（默认：data/processed/dataset.json）",
    )
    parser.add_argument(
        "--user",
        type=str,
        help="单个用户推荐（指定用户名）",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        help="批量推荐（从 JSON 文件读取用户列表或历史）",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="推荐数量（默认：10）",
    )
    parser.add_argument(
        "--list-users",
        action="store_true",
        help="列出所有可用用户",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出结果到 JSON 文件（可选）",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="使用 GPU 推理（需要 ONNX Runtime GPU 版本）",
    )
    
    args = parser.parse_args()
    
    # 初始化引擎
    print(f"加载推荐引擎...")
    engine = RecommendationEngine(
        model_path=args.model,
        dataset_path=args.dataset,
        use_gpu=args.gpu,
    )
    
    # 加载数据集
    with args.dataset.open("r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    # 执行对应操作
    if args.list_users:
        # 列出所有用户
        user_list = list(dataset.get("user_histories", {}).keys())
        print(f"\n共有 {len(user_list)} 个用户：")
        for i, user in enumerate(user_list, 1):
            print(f"  {i}. {user}")
    
    elif args.user:
        # 单用户推荐
        result = recommend_single_user(
            engine,
            args.user,
            dataset,
            topk=args.topk,
        )
        
        if "error" in result:
            print(f"\n错误：{result['error']}")
            print(f"前 10 个可用用户：{result.get('available_users', [])}")
        else:
            print(f"\n【用户：{result['user']}】")
            print(f"历史序列：{result['history']}")
            print(f"\n推荐结果（Top-{args.topk}）：")
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"  {i}. {rec['title']} (ID: {rec['id']}, 分数: {rec['score']:.4f})")
        
        # 保存输出（如果指定）
        if args.output:
            with args.output.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到 {args.output}")
    
    elif args.batch:
        # 批量推荐
        with args.batch.open("r", encoding="utf-8") as f:
            batch_data = json.load(f)
        
        # 支持两种格式：
        # 1. {"user1": [id列表], "user2": [id列表]} 
        # 2. ["user1", "user2"]
        
        if isinstance(batch_data, dict):
            # 格式 1：直接传入历史序列
            histories = list(batch_data.values())
            user_names = list(batch_data.keys())
        else:
            # 格式 2：用户名列表，从数据集中获取历史
            user_names = batch_data
            histories = [
                dataset["user_histories"][u] for u in user_names
                if u in dataset["user_histories"]
            ]
        
        print(f"\n批量推荐 {len(histories)} 个用户...")
        batch_results = engine.batch_recommend(
            histories,
            topk=args.topk,
            return_scores=True,
        )
        
        # 组织结果
        results_dict = {}
        for user_name, recs in zip(user_names, batch_results):
            results_dict[user_name] = recs
            print(f"\n{user_name}：{len(recs)} 条推荐")
            for rec in recs[:3]:  # 只显示前 3 条
                print(f"  - {rec['title']}")
        
        # 保存输出
        if args.output:
            with args.output.open("w", encoding="utf-8") as f:
                json.dump(results_dict, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到 {args.output}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
