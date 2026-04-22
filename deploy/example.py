"""
推荐引擎使用示例

演示如何使用轻量化推理模块的各种功能
"""

import json
from pathlib import Path
from inference import RecommendationEngine


def example_single_user():
    """示例 1：单用户推荐"""
    print("\n" + "="*60)
    print("【示例 1】单用户推荐")
    print("="*60)
    
    engine = RecommendationEngine(
        model_path="outputs/model.onnx",
        dataset_path="data/processed/dataset.json",
    )
    
    # 用户历史
    history = [1, 2, 3, 5, 7]
    
    # 生成推荐
    recommendations = engine.recommend(
        history=history,
        topk=5,
        return_scores=True,
    )
    
    print(f"\n用户历史序列：{history}")
    print(f"推荐结果（Top-5）：")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec['title']} (ID: {rec['id']}, 分数: {rec['score']:.4f})")


def example_batch_recommend():
    """示例 2：批量用户推荐"""
    print("\n" + "="*60)
    print("【示例 2】批量推荐（多个用户）")
    print("="*60)
    
    engine = RecommendationEngine(
        model_path="outputs/model.onnx",
        dataset_path="data/processed/dataset.json",
    )
    
    # 多个用户的历史
    histories = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    
    # 批量推荐
    batch_results = engine.batch_recommend(
        histories=histories,
        topk=3,
        return_scores=False,  # 不返回分数，仅返回电影标题
    )
    
    print(f"\n处理 {len(histories)} 个用户...")
    for i, recs in enumerate(batch_results, 1):
        print(f"\n用户 {i} 的推荐：")
        for rec in recs:
            print(f"  - {rec['title']} (ID: {rec['id']})")


def example_with_dataset():
    """示例 3：使用数据集中的真实用户"""
    print("\n" + "="*60)
    print("【示例 3】使用真实用户数据")
    print("="*60)
    
    engine = RecommendationEngine(
        model_path="outputs/model.onnx",
        dataset_path="data/processed/dataset.json",
    )
    
    # 加载数据集
    with Path("data/processed/dataset.json").open("r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    # 获取前 3 个用户的推荐
    user_histories = dataset.get("user_histories", {})
    users = list(user_histories.keys())[:3]
    
    print(f"\n数据集中的用户数：{len(user_histories)}")
    print(f"前 3 个用户：{users}")
    
    for user_name in users:
        history = user_histories[user_name]
        recommendations = engine.recommend(
            history=history,
            topk=3,
            return_scores=True,
        )
        
        print(f"\n【{user_name}】")
        print(f"  历史：{history}")
        print(f"  推荐：", end="")
        rec_titles = [rec['title'] for rec in recommendations]
        print(" → ".join(rec_titles))


def example_filter_options():
    """示例 4：推荐选项演示"""
    print("\n" + "="*60)
    print("【示例 4】推荐选项（过滤历史、返回分数）")
    print("="*60)
    
    engine = RecommendationEngine(
        model_path="outputs/model.onnx",
        dataset_path="data/processed/dataset.json",
    )
    
    history = [1, 2, 3]
    
    # 方式 1：过滤历史（默认）
    print("\n【方式 1】过滤历史 + 返回分数")
    recs1 = engine.recommend(
        history=history,
        topk=3,
        filter_history=True,
        return_scores=True,
    )
    for rec in recs1:
        print(f"  - {rec['title']} (分数: {rec['score']:.4f})")
    
    # 方式 2：不过滤历史
    print("\n【方式 2】不过滤历史")
    recs2 = engine.recommend(
        history=history,
        topk=3,
        filter_history=False,
        return_scores=False,
    )
    for rec in recs2:
        print(f"  - {rec['title']}")


def example_save_results():
    """示例 5：保存推荐结果"""
    print("\n" + "="*60)
    print("【示例 5】保存推荐结果到文件")
    print("="*60)
    
    engine = RecommendationEngine(
        model_path="outputs/model.onnx",
        dataset_path="data/processed/dataset.json",
    )
    
    # 加载数据集
    with Path("data/processed/dataset.json").open("r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    user_histories = dataset.get("user_histories", {})
    
    # 为所有用户生成推荐
    print(f"\n为 {len(user_histories)} 个用户生成推荐...")
    
    all_recommendations = {}
    for user_name, history in user_histories.items():
        recs = engine.recommend(
            history=history,
            topk=5,
            return_scores=True,
        )
        all_recommendations[user_name] = recs
    
    # 保存到文件
    output_file = Path("deploy/recommendations_output.json")
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(all_recommendations, f, ensure_ascii=False, indent=2)
    
    print(f"推荐结果已保存到 {output_file}")
    
    # 显示摘要
    print(f"\n摘要：")
    print(f"  总用户数：{len(all_recommendations)}")
    print(f"  每个用户推荐数：5")


if __name__ == "__main__":
    print("\n推荐引擎使用示例")
    print("=" * 60)
    
    try:
        # 运行所有示例
        example_single_user()
        example_batch_recommend()
        example_with_dataset()
        example_filter_options()
        example_save_results()
        
        print("\n" + "="*60)
        print("所有示例执行完毕")
        print("="*60 + "\n")
    
    except FileNotFoundError as e:
        print(f"\n错误：找不到文件")
        print(f"   请确保以下文件存在：")
        print(f"   - outputs/model.onnx")
        print(f"   - data/processed/dataset.json")
        print(f"\n   可以通过以下步骤生成：")
        print(f"   1. python -m src.preprocess --input data/raw/clean.csv --output-dir data/processed")
        print(f"   2. python -m src.train --dataset data/processed/dataset.json --output-dir outputs")
    
    except ImportError as e:
        print(f"\n错误：缺少依赖库")
        print(f"   {str(e)}")
        print(f"\n   请安装依赖：")
        print(f"   pip install -r deploy/requirements-deploy.txt")
    
    except Exception as e:
        print(f"\n未预期的错误：{str(e)}")
