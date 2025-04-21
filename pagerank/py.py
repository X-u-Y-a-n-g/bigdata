import numpy as np
from scipy import sparse
from collections import defaultdict
import psutil
import os
import time
import multiprocessing as mp
from functools import partial
import mmap
import gc

def process_block(args):
    matrix, block_range, current_scores, damping_factor, n_nodes = args
    start, end = block_range
    block = matrix[start:end, :]
    partial_scores = block.dot(current_scores)
    del block
    return start, end, partial_scores

class PageRank:
    def __init__(self, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.n_nodes = None
        self.block_size = None
    
    def read_data(self, filename):
        adjacency_dict = defaultdict(list)
        nodes = set()
        batch_size = 100000
        current_batch = []
        
        with open(filename, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            for line in iter(mm.readline, b''):
                source, target = map(int, line.decode().strip().split())
                current_batch.append((source, target))
                
                if len(current_batch) >= batch_size:
                    self._process_batch(current_batch, adjacency_dict, nodes)
                    current_batch = []
                    gc.collect()
            
            if current_batch:
                self._process_batch(current_batch, adjacency_dict, nodes)
            mm.close()
        
        self.nodes = sorted(list(nodes))
        self.n_nodes = len(self.nodes)
        self.node_to_index = {node: idx for idx, node in enumerate(self.nodes)}
        
        # 动态调整分块大小
        cpu_count = mp.cpu_count()
        self.block_size = max(1000, self.n_nodes // (cpu_count * 2))
        
        all_rows, all_cols, all_data = [], [], []
        for source in adjacency_dict:
            out_degree = len(adjacency_dict[source])
            if out_degree > 0:  # 处理dead-ends
                for target in adjacency_dict[source]:
                    all_rows.append(self.node_to_index[target])
                    all_cols.append(self.node_to_index[source])
                    all_data.append(1.0 / out_degree)
        
        self.matrix = sparse.csr_matrix((all_data, (all_rows, all_cols)), 
                                      shape=(self.n_nodes, self.n_nodes),
                                      dtype=np.float32)
        
        del adjacency_dict, nodes, self.node_to_index, all_rows, all_cols, all_data
        gc.collect()
    
    def _process_batch(self, batch, adjacency_dict, nodes):
        for source, target in batch:
            adjacency_dict[source].append(target)
            nodes.add(source)
            nodes.add(target)
    
    def parallel_block_pagerank(self):
        scores = np.ones(self.n_nodes, dtype=np.float32) / self.n_nodes
        num_processes = mp.cpu_count()
        
        # 预计算分块
        blocks = [(i, min(i + self.block_size, self.n_nodes)) 
                 for i in range(0, self.n_nodes, self.block_size)]
        
        print(f"使用 {num_processes} 个CPU核心进行并行计算")
        print(f"矩阵分块大小: {self.block_size}")
        
        with mp.Pool(num_processes) as pool:
            for iteration in range(self.max_iterations):
                iter_start = time.time()
                new_scores = np.zeros(self.n_nodes, dtype=np.float32)
                
                args = [(self.matrix, block, scores, self.damping_factor, self.n_nodes) 
                       for block in blocks]
                
                # 使用imap优化内存使用
                for start, end, partial_scores in pool.imap(process_block, args, 
                                                          chunksize=max(1, len(blocks)//num_processes)):
                    new_scores[start:end] = partial_scores
                
                # 向量化计算
                new_scores = self.damping_factor * new_scores + (1 - self.damping_factor) / self.n_nodes
                
                diff = np.sum(np.abs(new_scores - scores))
                iter_time = time.time() - iter_start
                
                print(f"迭代 {iteration + 1}: 差异={diff:.6f}, 用时={iter_time:.2f}秒, "
                      f"内存使用={get_memory_usage():.2f}MB")
                
                if diff < self.tolerance:
                    print(f"收敛于第 {iteration + 1} 次迭代")
                    break
                
                scores = new_scores.copy()
                gc.collect()
        
        return scores
    
    def save_results(self, scores, output_file):
        results = [(self.nodes[i], score) for i, score in enumerate(scores)]
        results.sort(key=lambda x: x[1], reverse=True)
        
        with open(output_file, 'w') as f:
            for node_id, score in results[:100]:
                f.write(f"{node_id} {score:.16f}\n")

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def main():
    start_time = time.time()
    initial_memory = get_memory_usage()
    print(f"初始内存使用: {initial_memory:.2f} MB")
    print(f"CPU核心数: {mp.cpu_count()}")
    
    # 设置进程启动方法
    mp.set_start_method('spawn', force=True)
    
    pr = PageRank(damping_factor=0.85, max_iterations=100, tolerance=1e-6)
    
    print("\n1. 数据读取阶段")
    print("-" * 50)
    read_start = time.time()
    input_file = "D:/bigdata/pagerank/Data.txt"
    pr.read_data(input_file)
    read_memory = get_memory_usage()
    print(f"数据读取完成:")
    print(f"- 耗时: {time.time() - read_start:.2f} 秒")
    print(f"- 内存使用: {read_memory:.2f} MB")
    print(f"- 内存增长: {read_memory - initial_memory:.2f} MB")
    
    print("\n2. PageRank计算阶段")
    print("-" * 50)
    calc_start = time.time()
    scores = pr.parallel_block_pagerank()
    calc_memory = get_memory_usage()
    print(f"计算完成:")
    print(f"- 耗时: {time.time() - calc_start:.2f} 秒")
    print(f"- 内存使用: {calc_memory:.2f} MB")
    print(f"- 内存增长: {calc_memory - read_memory:.2f} MB")
    
    print("\n3. 结果保存阶段")
    print("-" * 50)
    output_file = "D:/bigdata/pagerank/Res.txt"
    pr.save_results(scores, output_file)
    
    print(f"\n执行总结:")
    print("-" * 50)
    print(f"总运行时间: {time.time() - start_time:.2f} 秒")
    print(f"最终内存使用: {get_memory_usage():.2f} MB")

if __name__ == "__main__":
    main()