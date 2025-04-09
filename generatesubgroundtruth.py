from py2neo import Graph

# 连接到 Neo4j 数据库
graph = Graph("neo4j://localhost:7687", auth=("neo4j", "Wsj@040308"))

# 第一个查询
query1 = """
MATCH (r1:Restaurant)-[:LOCATED_AT]->(a1:Address)-[:IN_CITY]->(c:City)<-[:IN_CITY]-(a2:Address)<-[:LOCATED_AT]-(r2:Restaurant)
WHERE (c.value CONTAINS 'new york' ) 
RETURN r1, r2
"""

# 第二个查询
query2 = """
MATCH (r1:Restaurant)-[:LOCATED_AT]->(a1:Address)-[:IN_CITY]->(c:City)<-[:IN_CITY]-(a2:Address)<-[:LOCATED_AT]-(r2:Restaurant)
WHERE (c.value = 'sacaxaxasxaxsaxla') 
RETURN r1, r2
"""
#MATCH (r1:Restaurant)-[:LOCATED_AT]->(a:Address)<-[:LOCATED_AT]-(r2:Restaurant)
# MATCH (a)-[:IN_CITY]->(c:City)
# WHERE (c.value = 'la' OR c.value CONTAINS 'los angeles' OR c.value CONTAINS 'santa monica')
# RETURN r1, r2
# 执行查询并收集所有 Restaurant ID
restaurant_ids = set()

# 执行第一个查询
data1 = graph.run(query1).data()
for record in data1:
    restaurant_ids.add(record['r1']['id'])
    restaurant_ids.add(record['r2']['id'])

# 执行第二个查询
data2 = graph.run(query2).data()
for record in data2:
    restaurant_ids.add(record['r1']['id'])
    restaurant_ids.add(record['r2']['id'])

# 输出所有不重复的 Restaurant ID
print("不重复的 Restaurant ID 列表:")
# for restaurant_id in restaurant_ids:
    # print(restaurant_id)
def generate_subgroundtruth(restaurant_ids, input_file, output_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    # 创建一个集合以存储不重复的匹配对
    subgroundtruth_pairs = set()

    current_pair = []
    for line in lines:
        # 去掉行尾的换行符并检查是否为空行
        line = line.strip()
        if line:
            current_pair.append(line)
        else:
            # 如果 current_pair 已经收集了两个 ID，处理该对
            if len(current_pair) == 2:
                id1, id2 = current_pair
                # 检查两个 ID 是否都在提供的 restaurant_ids 中
                if id1 in restaurant_ids and id2 in restaurant_ids:
                    # 使用排序的元组来避免重复的组合
                    pair = tuple(sorted((id1, id2)))
                    subgroundtruth_pairs.add(pair)
            # 重置 current_pair 以便处理下一个对
            current_pair = []

    # 确保处理文件末尾的最后一对（如果没有空行）
    if len(current_pair) == 2:
        id1, id2 = current_pair
        if id1 in restaurant_ids and id2 in restaurant_ids:
            pair = tuple(sorted((id1, id2)))
            subgroundtruth_pairs.add(pair)

    # 将不重复的匹配对写入输出文件
    with open(output_file, 'w') as outfile:
        for pair in sorted(subgroundtruth_pairs):
            outfile.write(f"{pair[0]} {pair[1]}\n\n")


# 输入和输出文件路径
input_file = 'dataset/ground_truth_fodors_zagats.txt'
output_file = 'dataset/subgroundtruth.txt'

generate_subgroundtruth(restaurant_ids, input_file, output_file)

print("subgroundtruth.txt 文件已生成。")
