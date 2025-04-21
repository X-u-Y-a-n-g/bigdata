import networkx as nx

# 创建有向图
G = nx.DiGraph()

# 从data.txt读取边的信息
with open('d:\\bigdata\\pagerank\\data.txt', 'r') as f:
    for line in f:
        source, target = map(int, line.strip().split())
        G.add_edge(source, target)

# 计算PageRank
pr = nx.pagerank(G, alpha=0.85)

# 将结果按PageRank值排序并写入test.txt
sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
with open('d:\\bigdata\\pagerank\\test.txt', 'w') as f:
    for node, value in sorted_pr:
        f.write(f"{node} {value:.16f}\n")

# 读取Res.txt作为参考结果
ref_pr = {}
try:
    with open('d:\\bigdata\\pagerank\\Res.txt', 'r') as f:
        for line in f:
            try:
                node, value = line.strip().split()
                ref_pr[int(node)] = float(value)
            except ValueError:
                continue

    # 检查是否成功读取数据
    if not ref_pr:
        print("警告：Res.txt 为空或格式不正确")
        exit(1)

    # 比较结果
    sorted_ref = sorted(ref_pr.items(), key=lambda x: x[1], reverse=True)
    print("前100个最重要节点的比较：")
    print("序号\t节点ID\t计算值\t\t参考值\t\t差异")
    print("-" * 70)

    # 比较前100个节点
    for i, ((test_node, test_val), (ref_node, ref_val)) in enumerate(zip(sorted_pr[:100], sorted_ref[:100])):
        diff = abs(test_val - ref_val)
        print(f"{i+1}\t{test_node}\t{test_val:.8f}\t{ref_val:.8f}\t{diff:.8f}")

    # 计算总体误差
    common_nodes = set(pr.keys()) & set(ref_pr.keys())
    if common_nodes:
        max_diff = max(abs(pr[node] - ref_pr[node]) for node in common_nodes)
        avg_diff = sum(abs(pr[node] - ref_pr[node]) for node in common_nodes) / len(common_nodes)

        print("\n统计信息：")
        print(f"共同节点数量: {len(common_nodes)}")
        print(f"最大误差: {max_diff:.8f}")
        print(f"平均误差: {avg_diff:.8f}")
    else:
        print("\n错误：计算结果与参考结果没有共同的节点")

except FileNotFoundError:
    print("错误：找不到 Res.txt 文件")
except Exception as e:
    print(f"发生错误：{str(e)}")