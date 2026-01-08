import pandas as pd
import networkx as nx
import numpy as np

def analyze_centrality(co_occurrence_matrix_df, top_n=10):
    """
    通过共现矩阵分析 Tag 的中心性。
    
    Args:
        co_occurrence_matrix_df (pd.DataFrame): 标签共现矩阵，索引和列名均为 Tag，值为共现次数。
        top_n (int): 返回前 N 个中心 Tag。
    """
    print("正在构建标签共现网络图...")
    
    # 创建无向图
    G = nx.Graph()
    
    # 获取标签列表
    tags = co_occurrence_matrix_df.index.tolist()
    
    # 遍历矩阵添加边 (利用矩阵的对称性，只遍历上三角或直接由 nx 处理)
    # 为了效率，我们先转换为 stack 格式，然后过滤掉 0 值和自环
    stacked = co_occurrence_matrix_df.stack()
    # 过滤掉共现次数为0的，以及自己和自己的共现 (如果不需要考虑自共现)
    edges = stacked[stacked > 0].reset_index()
    edges.columns = ['source', 'target', 'weight']
    
    # 移除自环 (source == target)
    edges = edges[edges['source'] != edges['target']]
    
    # 添加带权重的边
    G.add_weighted_edges_from(edges.values)
    
    print(f"图构建完成。节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
    
    # --- 计算中心性指标 ---
    
    # 1. Degree Centrality (度中心性 - 归一化过的连接数)
    # 这里我们直接计算原始 Degree (连接了多少个不同的 Tag)
    degrees = dict(G.degree())
    
    # 2. Weighted Degree (加权度/强度 - 与其他 Tag 共现的总次数)
    weighted_degrees = dict(G.degree(weight='weight'))
    
    # 3. Eigenvector Centrality (特征向量中心性 - 连接到重要节点的节点更重要)
    # 适合用来找“核心圈子”里的 Tag
    try:
        eigen_centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
    except:
        eigen_centrality = {tag: 0 for tag in tags} # 如果不收敛则忽略

    # 汇总结果
    results = []
    for tag in G.nodes():
        results.append({
            'Tag': tag,
            'Neighbor_Count': degrees[tag],          # 连接了多少个不同的 Tag
            'Total_Cooccurrences': weighted_degrees[tag], # 共现总强度
            'Eigen_Score': eigen_centrality.get(tag, 0)
        })
    
    # 创建结果 DataFrame
    results_df = pd.DataFrame(results)
    
    # 按 "Neighbor_Count" (连接的Tag数量) 降序排列 - 这最符合你的“跟较多tag相连”的需求
    top_connected = results_df.sort_values(by='Neighbor_Count', ascending=False).head(top_n)
    
    print("\n=== Top Tags (按连接的 Tag 数量排序) ===")
    print(top_connected[['Tag', 'Neighbor_Count', 'Total_Cooccurrences']].to_string(index=False))
    
    # 按 "Total_Cooccurrences" (共现总强度) 降序排列
    top_weighted = results_df.sort_values(by='Total_Cooccurrences', ascending=False).head(top_n)
    
    print("\n=== Top Tags (按共现总强度排序) ===")
    print(top_weighted[['Tag', 'Neighbor_Count', 'Total_Cooccurrences']].to_string(index=False))

    return results_df

def calculate_document_centrality(doc_tags_map, tag_centrality_df, metric='Eigen_Score', method='mean'):
    """
    计算文档的中心性。
    
    思路:
    用户提出的 "avg" (平均值) 是一个非常合理的基准方法。它反映了文档中包含的 Tag 的“平均重要程度”。
    - 如果一个文档包含很少但很核心的 Tag，平均分会高。
    - 如果一个文档包含很多边缘 Tag，平均分会低。
    
    此外，也可以考虑 'sum' (总和)，反映文档的信息量总量（中心 Tag 越多越好）。
    
    Args:
        doc_tags_map (dict or pd.Series): 文档ID到Tag列表的映射 {doc_id: [tag1, tag2, ...]}
        tag_centrality_df (pd.DataFrame): 包含Tag中心性分数的DataFrame，必须包含 'Tag' 列和 metric 列。
        metric (str): 用于计算的中心性指标列名 (如 'Eigen_Score', 'Neighbor_Count', 'Total_Cooccurrences')
        method (str): 聚合方法 'mean' (平均值), 'sum' (总和), 'max' (最大值)
        
    Returns:
        pd.DataFrame: 包含文档ID和计算出的中心性分数的 DataFrame
    """
    print(f"\n正在计算文档中心度 (Metric: {metric}, Method: {method})...")
    
    # 创建 Tag 到 Score 的查找字典，提高效率
    if 'Tag' not in tag_centrality_df.columns:
        # 假设索引是 Tag
        tag_score_map = tag_centrality_df[metric].to_dict()
    else:
        tag_score_map = dict(zip(tag_centrality_df['Tag'], tag_centrality_df[metric]))
        
    doc_scores = []
    
    # 统一将输入转换为 items 迭代器
    iterator = doc_tags_map.items() if isinstance(doc_tags_map, dict) else doc_tags_map.items()
    
    for doc_id, tags in iterator:
        if not tags:
            doc_scores.append({'DocID': doc_id, 'Doc_Centrality': 0})
            continue
            
        # 获取该文档所有 Tag 的分数，如果 Tag 不在分析结果中（可能是低频词被过滤了），默认给 0
        scores = [tag_score_map.get(tag, 0) for tag in tags]
        
        if not scores:
            final_score = 0
        elif method == 'mean':
            final_score = np.mean(scores)
        elif method == 'sum':
            final_score = np.sum(scores)
        elif method == 'max':
            final_score = np.max(scores)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        doc_scores.append({
            'DocID': doc_id,
            'Doc_Centrality': final_score,
            'Tag_Count': len(tags),
            'Tags': str(tags) # 方便查看
        })
        
    doc_scores_df = pd.DataFrame(doc_scores)
    
    # 排序
    doc_scores_df = doc_scores_df.sort_values(by='Doc_Centrality', ascending=False)
    
    return doc_scores_df

# --- 模拟数据示例 (你可以替换为加载真实数据的代码) ---
if __name__ == "__main__":
    # 1. 模拟标签共现矩阵
    data = {
        'AI':           [5, 3, 1, 0, 4],
        'MachineLearning': [3, 6, 2, 0, 2],
        'Python':       [1, 2, 4, 1, 1],
        'Cooking':      [0, 0, 1, 2, 0],
        'BigData':      [4, 2, 1, 0, 5]
    }
    tags = ['AI', 'MachineLearning', 'Python', 'Cooking', 'BigData']
    df_mock = pd.DataFrame(data, index=tags, columns=tags)
    
    print("模拟共现矩阵:")
    print(df_mock)
    print("-" * 30)
    
    # 2. 计算 Tag 中心性
    tag_centrality_df = analyze_centrality(df_mock)
    
    # 3. 模拟文档-标签数据
    doc_tags = {
        'doc_1': ['AI', 'MachineLearning', 'BigData'], # 都是核心词
        'doc_2': ['Cooking'],                          # 边缘词
        'doc_3': ['Python', 'Cooking'],                # 混合
        'doc_4': ['AI', 'Python']                      # 核心+连接词
    }
    
    print("\n模拟文档数据:")
    for d, t in doc_tags.items():
        print(f"{d}: {t}")
        
    # 4. 计算文档中心性 (使用平均值)
    doc_centrality_df = calculate_document_centrality(
        doc_tags, 
        tag_centrality_df, 
        metric='Eigen_Score', 
        method='mean'
    )
    
    print("\n=== 文档中心度排名 (Mean Eigen_Score) ===")
    print(doc_centrality_df.to_string(index=False))