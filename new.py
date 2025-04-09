from py2neo import Graph
import os
import csv


def similar(a, b):
    if a is None or b is None:
        return 0.0
    return jaccard_similarity(a, b)


def jaccard_similarity(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if len(union) > 0 else 0.0


def connect_to_neo4j(uri, user, password):
    return Graph(uri, auth=(user, password))


def execute_query(graph, base_query, batch_size=100):
    offset = 0
    batch_number = 0

    while True:
        query = f"""
        {base_query}

        SKIP {offset} LIMIT {batch_size}
        """
        cursor = graph.run(query)
        data_batch = []

        while cursor.forward():
            data_batch.append(cursor.current)

        if not data_batch:
            break

        batch_number += 1
        yield data_batch

        offset += batch_size



def process_query_results(data_batch, output_dir, iteration):
    os.makedirs(output_dir, exist_ok=True)
    filtered_output_file = os.path.join(output_dir, "filtered_pattern.txt")

    pair_count = 0
    unique_pairs = set()
    matching_pairs = []
    matched_data = []

    for record in data_batch:
        r1 = record['r1']
        r2 = record['r2']

        value_similarity = similar(r1['name'], r2['name'])
        cuisine_similarity = similar(r1['cuisine'], r2['cuisine'])
        phone_similarity = similar(r1['phone'], r2['phone'])

        pair_id = tuple(sorted((r1['id'], r2['id'])))

        if pair_id not in unique_pairs:
            unique_pairs.add(pair_id)

            output_str = {
                "restaurant_1": {"id": r1['id'], "name": r1['name'], "cuisine": r1['cuisine'], "phone": r1['phone']},
                "restaurant_2": {"id": r2['id'], "name": r2['name'], "cuisine": r2['cuisine'], "phone": r2['phone']},
                "value_similarity": value_similarity,
                "cuisine_similarity": cuisine_similarity,
                "phone_similarity": phone_similarity
            }
            matched_data.append(output_str)
            pair_count += 1

        if pair_id not in matching_pairs:
            matching_pairs.append(pair_id)

    with open(filtered_output_file, 'a', encoding='utf-8') as f:
        f.write(f"Iteration {iteration} matching pairs:\n")
        for pair_id in matching_pairs:
            r1, r2 = None, None
            for record in data_batch:
                if tuple(sorted((record['r1']['id'], record['r2']['id']))) == pair_id:
                    r1 = record['r1']
                    r2 = record['r2']
                    break

            if r1 and r2:
                value_similarity = similar(r1['name'], r2['name'])
                cuisine_similarity = similar(r1['cuisine'], r2['cuisine'])
                phone_similarity = similar(r1['phone'], r2['phone'])

                output_str = (
                    f"Restaurant 1: (ID: {r1['id']}, Name: {r1['name']}, Cuisine: {r1['cuisine']}, Phone: {r1['phone']})\n"
                    f"Restaurant 2: (ID: {r2['id']}, Name: {r2['name']}, Cuisine: {r2['cuisine']}, Phone: {r2['phone']})\n"
                    f"Value Similarity: {value_similarity:.2f}\n"
                    f"Cuisine Similarity: {cuisine_similarity:.2f}\n"
                    f"Phone Similarity: {phone_similarity:.2f}\n"
                    "--------------------------------------------\n"
                )
                f.write(output_str)
        f.write("\n")

    return pair_count, matched_data, filtered_output_file


def process_query_results2(data_batch, output_dir, filename="query_results.csv"):

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, filename)

    file_exists = os.path.isfile(output_file)

    written_pairs = set()

    headers = ["r1_id", "r1_name", "r1_phone", "r1_cuisine",
               "r2_id", "r2_name", "r2_phone", "r2_cuisine",
               "a1_value", "a2_value", "c1_value", "c2_value"]

    if file_exists:
        with open(output_file, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    written_pairs.add((row[0], row[4]))
                    written_pairs.add((row[4], row[0]))

    with open(output_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(headers)

        for record in data_batch:
            r1 = record['r1']
            r2 = record['r2']
            a1 = record['a1']
            a2 = record['a2']
            c1 = record['c1']
            c2 = record['c2']

            r1_id = r1.get('id', 'N/A')
            r2_id = r2.get('id', 'N/A')

            if (r1_id, r2_id) in written_pairs or (r2_id, r1_id) in written_pairs:
                continue
            written_pairs.add((r1_id, r2_id))
            written_pairs.add((r2_id, r1_id))

            row = [
                r1_id, r1.get('name', 'N/A'), r1.get('phone', 'N/A'), r1.get('cuisine', 'N/A'),
                r2_id, r2.get('name', 'N/A'), r2.get('phone', 'N/A'), r2.get('cuisine', 'N/A'),
                a1.get('value', 'N/A'), a2.get('value', 'N/A'),
                c1.get('value', 'N/A'), c2.get('value', 'N/A')
            ]
            writer.writerow(row)

    print(f"Results appended to {output_file}")