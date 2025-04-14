import numpy as np
from scipy import sparse
import pandas as pd
from collections import defaultdict

class PageRank:
    def __init__(self, damping_factor=0.85, max_iterations=100, tolerance=1e-6, block_size=1000):
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.block_size = block_size
        
    def read_data(self, filename):
        # 读取数据并构建邻接表
        adjacency_dict = defaultdict(list)
        nodes = set()
        
        with open(filename, 'r') as f:
            for line in f:
                source, target = map(int, line.strip().split())
                adjacency_dict[source].append(target)
                nodes.add(source)
                nodes.add(target)
        
        self.nodes = sorted(list(nodes))
        self.n_nodes = len(self.nodes)
        
        # 构建节点索引映射
        self.node_to_index = {node: idx for idx, node in enumerate(self.nodes)}
        
        # 构建稀疏矩阵
        rows, cols, data = [], [], []
        for source in adjacency_dict:
            out_degree = len(adjacency_dict[source])
            for target in adjacency_dict[source]:
                rows.append(self.node_to_index[target])
                cols.append(self.node_to_index[source])
                data.append(1.0 / out_degree)
        
        self.matrix = sparse.csr_matrix((data, (rows, cols)), 
                                      shape=(self.n_nodes, self.n_nodes))
    
    def block_pagerank(self):
        # 初始化PageRank值
        scores = np.ones(self.n_nodes) / self.n_nodes
        
        for iteration in range(self.max_iterations):
            new_scores = np.zeros(self.n_nodes)
            
            # 分块计算
            for i in range(0, self.n_nodes, self.block_size):
                end = min(i + self.block_size, self.n_nodes)
                block = self.matrix[i:end, :]
                new_scores[i:end] = block.dot(scores)
            
            # 应用阻尼因子
            new_scores = self.damping_factor * new_scores + (1 - self.damping_factor) / self.n_nodes
            
            # 检查收敛
            diff = np.sum(np.abs(new_scores - scores))
            if diff < self.tolerance:
                break
                
            scores = new_scores
            
        return scores
    
    def save_results(self, scores, output_file):
        # 将结果转换回原始节点ID
        results = [(self.nodes[i], score) for i, score in enumerate(scores)]
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 保存前100个结果
        with open(output_file, 'w') as f:
            f.write("NodeID,Score\n")
            for node_id, score in results[:100]:
                f.write(f"{node_id},{score:.10f}\n")

def main():
    # 初始化PageRank计算器
    pr = PageRank(damping_factor=0.85, max_iterations=100, tolerance=1e-6, block_size=1000)
    
    # 读取数据
    input_file = "d:\\许洋计算机科学与技术\\大数据计算及应用\\pagerank大作业\\Data.txt"
    pr.read_data(input_file)
    
    # 计算PageRank
    scores = pr.block_pagerank()
    
    # 保存结果
    output_file = "d:\\许洋计算机科学与技术\\大数据计算及应用\\pagerank大作业\\Res.txt"
    pr.save_results(scores, output_file)

if __name__ == "__main__":
    main()