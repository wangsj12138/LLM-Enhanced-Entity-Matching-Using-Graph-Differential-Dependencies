from new import process_query_results2, connect_to_neo4j, execute_query, process_query_results
from pattern2blocks import parse_filtered_pattern, generate_blocks
from PPS import (
    parse_input_from_file,
    initialization_phase,
    emission_phase_for_pps,
    load_ground_truth,
    save_matched_results,
    wScheme,
)
import time
import os
def clear_previous_results(output_dir):
    filtered_output_file = os.path.join(output_dir, "filtered_pattern.txt")
    blocks_file = os.path.join(output_dir, "blocks.txt")

    if os.path.exists(filtered_output_file):
        os.remove(filtered_output_file)

    if os.path.exists(blocks_file):
        os.remove(blocks_file)


def process_and_save_results(filtered_output_file, iteration, matched_pairs, start_time):
    input_file = filtered_output_file
    output_file_blocks = 'output_file/blocks.txt'
    ground_truth_path = 'dataset/ground_truth_fodors_zagats.txt'
    results_path = 'output_file/results.txt'
    time_log_path = 'output_file/time.txt'

    print(f"Processing iteration {iteration}, reading from {input_file}...")
    ground_truth = load_ground_truth(ground_truth_path)

    entity_similarities = parse_filtered_pattern(input_file)
    blocks = generate_blocks(entity_similarities)

    with open(output_file_blocks, 'a', encoding='utf-8') as file:
        file.write(f"Iteration {iteration} blocks:\n")
        all_entities = set()
        for block in blocks:
            all_entities.update(block)

        for entity in sorted(all_entities):
            similar_entities = sorted(entity_similarities[entity])
            similar_entities_str = " ".join(map(str, similar_entities))
            file.write(f"target entity: {entity}\n")
            file.write(f"similar entities: {similar_entities_str}\n\n")

    P = parse_input_from_file(output_file_blocks)
    ComparisonList, SortedProfileList, ProfileIndex = initialization_phase(P, wScheme)

    saved_results = set()
    new_matches = 0
    total_extractions = 0
    count = 0
    last_pi = None

    while SortedProfileList or ComparisonList:
        next_best_comparison, count = emission_phase_for_pps(ComparisonList, SortedProfileList, P, ProfileIndex,
                                                             wScheme, total_extractions, count)

        if not next_best_comparison:
            break

        pi, pj, weight = next_best_comparison

        if (pi, pj) not in matched_pairs and (pj, pi) not in matched_pairs:

            pi_pj_found_via_transitivity = False
            for pk in matched_pairs:
                if (pi, pk) in matched_pairs and (pj, pk) in matched_pairs:
                    pi_pj_found_via_transitivity = True
                    break


            if (pi, pj) in ground_truth or (pj, pi) in ground_truth:
                save_matched_results(results_path, (pi, pj), saved_results)

                if pi != last_pi:
                    elapsed_time = time.time() - start_time
                    with open(time_log_path, 'a') as time_file:
                        time_file.write(f"{elapsed_time:.4f}\n")

                last_pi = pi

                new_matches += 1
            matched_pairs.add((pi, pj))

    return new_matches
def main():
    start_time = time.time()

    uri = "****"
    user = "****"
    password = "****"

    graph = connect_to_neo4j(uri, user, password)

    query1 = """
     MATCH (r1:Restaurant)-[:LOCATED_AT]->(a1:Address)-[:IN_CITY]->(c1:City),
      (r2:Restaurant)-[:LOCATED_AT]->(a2:Address)-[:IN_CITY]->(c2:City)
WHERE c1.value CONTAINS 'new york'
  AND c2.value CONTAINS 'new york'
  AND r1.id < r2.id
WITH r1, r2, apoc.text.split(r1.name, ' ') AS r1Words, apoc.text.split(r2.name, ' ') AS r2Words, a1, a2, c1, c2
WHERE SIZE(apoc.coll.intersection(r1Words, r2Words)) > 0
RETURN r1, r2, a1, a2, c1, c2




    """

    total_pairs = 0
    output_dir = "output_file"
    os.makedirs(output_dir, exist_ok=True)

    iteration = 1
    last_new_matches = None
    matched_pairs = set()

    print("Executing query 1...")
    data_stream = execute_query(graph, query1)

    query_has_new_matches = False
    for data_batch in data_stream:
        clear_previous_results(output_dir)

        pair_count, matched_data, filtered_output_file = process_query_results(data_batch, output_dir, iteration)

        process_query_results2(data_batch, output_dir)
        total_pairs += pair_count

        if pair_count > 0:
            new_matches = process_and_save_results(filtered_output_file, iteration, matched_pairs,
                                                   start_time)
            print(f"Iteration {iteration}, new matches: {new_matches}")
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"execution_time：{execution_time:.4f} s")
            iteration += 1

            if last_new_matches is not None and new_matches < last_new_matches / 6:
                print("New matches have dropped significantly, stopping execution.")


            if new_matches > 0:
                query_has_new_matches = True
                last_new_matches = new_matches
    if not query_has_new_matches:
        print("No new matches found, stopping execution.")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"total_pairs {total_pairs} ")
    print(f"time：{execution_time:.4f} ")


if __name__ == "__main__":
    main()

