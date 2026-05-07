import re
from collections import defaultdict

def parse_filtered_pattern(file_path):
    entity_similarities = defaultdict(set)

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    pattern = re.compile(
        r'Restaurant 1: \(ID: (\d+),.*?\)\nRestaurant 2: \(ID: (\d+),.*?\)\nValue Similarity: (\d\.\d+)')

    matches = pattern.findall(content)

    for match in matches:
        id1, id2, value_similarity = match
        id1, id2 = int(id1), int(id2)
        if float(value_similarity) > 0:
            entity_similarities[id1].add(id2)
            entity_similarities[id2].add(id1)

    return entity_similarities

def generate_blocks(entity_similarities):
    blocks = []
    visited = set()

    def dfs(entity, block):
        if entity not in visited:
            visited.add(entity)
            block.add(entity)
            for neighbor in entity_similarities[entity]:
                dfs(neighbor, block)

    for entity in entity_similarities:
        if entity not in visited:
            block = set()
            dfs(entity, block)
            blocks.append(block)

    return blocks

def main():
    input_file = 'output_file/filtered_pattern.txt'
    output_file = 'output_file/blocks.txt'

    entity_similarities = parse_filtered_pattern(input_file)
    blocks = generate_blocks(entity_similarities)

    with open(output_file, 'w', encoding='utf-8') as file:
        all_entities = set()
        for block in blocks:
            all_entities.update(block)

        for entity in sorted(all_entities):
            similar_entities = sorted(entity_similarities[entity])
            similar_entities_str = " ".join(map(str, similar_entities))
            file.write(f"target entity: {entity}\n")
            file.write(f"similar entities: {similar_entities_str}\n\n")

    print(f"Blocks written to {output_file}")

if __name__ == "__main__":
    main()

