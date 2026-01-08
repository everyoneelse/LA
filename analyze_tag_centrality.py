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

# --- 模拟数据示例 (你可以替换为加载真实数据的代码) ---
if __name__ == "__main__":
    # 模拟一个简单的共现矩阵
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
    
    analyze_centrality(df_mock)
