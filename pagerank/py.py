import numpy as np
from scipy import sparse
from collections import defaultdict
import psutil
import os
import time

class PageRank:
    def __init__(self, damping_factor=0.85, max_iterations=100, tolerance=1e-6, block_size=500):
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.block_size = block_size
        
    def read_data(self, filename):
        # 使用生成器读取文件
        def read_edges():
            with open(filename, 'r') as f:
                for line in f:
                    yield map(int, line.strip().split())
        
        adjacency_dict = defaultdict(list)
        nodes = set()
        
        # 分批处理数据
        for source, target in read_edges():
            adjacency_dict[source].append(target)
            nodes.add(source)
            nodes.add(target)
        
        self.nodes = sorted(list(nodes))
        self.n_nodes = len(self.nodes)
        self.node_to_index = {node: idx for idx, node in enumerate(self.nodes)}
        
        # 使用生成器构建稀疏矩阵数据
        def generate_matrix_data():
            for source in adjacency_dict:
                out_degree = len(adjacency_dict[source])
                for target in adjacency_dict[source]:
                    yield (self.node_to_index[target],
                          self.node_to_index[source],
                          1.0 / out_degree)
        
        # 分批构建稀疏矩阵
        rows, cols, data = zip(*generate_matrix_data())
        self.matrix = sparse.csr_matrix((data, (rows, cols)), 
                                      shape=(self.n_nodes, self.n_nodes),
                                      dtype=np.float32)
        
        # 清理不再需要的数据
        del adjacency_dict, nodes
        
    def block_pagerank(self):
        scores = np.ones(self.n_nodes, dtype=np.float32) / self.n_nodes
        
        for iteration in range(self.max_iterations):
            new_scores = np.zeros(self.n_nodes, dtype=np.float32)
            
            for i in range(0, self.n_nodes, self.block_size):
                end = min(i + self.block_size, self.n_nodes)
                block = self.matrix[i:end, :]
                new_scores[i:end] = block.dot(scores)
            
            new_scores = self.damping_factor * new_scores + (1 - self.damping_factor) / self.n_nodes
            
            diff = np.sum(np.abs(new_scores - scores))
            if diff < self.tolerance:
                print(f"收敛于第 {iteration + 1} 次迭代")
                break
                
            scores = new_scores.copy()
            
        return scores
    
    def save_results(self, scores, output_file):
        results = [(self.nodes[i], score) for i, score in enumerate(scores)]
        results.sort(key=lambda x: x[1], reverse=True)
        
        with open(output_file, 'w') as f:
            f.write("NodeID,Score\n")
            for node_id, score in results[:100]:
                f.write(f"{node_id},{score:.10f}\n")

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def main():
    start_time = time.time()
    print(f"初始内存使用: {get_memory_usage():.2f} MB")
    
    pr = PageRank(damping_factor=0.85, max_iterations=100, tolerance=1e-6, block_size=500)
    
    print("开始读取数据...")
    input_file = "D:/bigdata/pagerank/Data.txt"
    pr.read_data(input_file)
    print(f"数据读取完成，当前内存使用: {get_memory_usage():.2f} MB")
    
    print("开始计算PageRank...")
    scores = pr.block_pagerank()
    print(f"计算完成，当前内存使用: {get_memory_usage():.2f} MB")
    
    output_file = "D:/bigdata/pagerank/Res.txt"
    pr.save_results(scores, output_file)
    
    end_time = time.time()
    print(f"总运行时间: {end_time - start_time:.2f} 秒")
    print(f"最终内存使用: {get_memory_usage():.2f} MB")

if __name__ == "__main__":
    main()