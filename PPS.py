import time
import os
import re
from collections import defaultdict

def parse_input_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    profile_collection = {}
    current_entity = None

    for line in lines:
        line = line.strip()
        if line.startswith("target entity:"):
            current_entity = int(line.split(':')[1].strip())
            profile_collection[current_entity] = [current_entity]
        elif line.startswith("similar entities:"):
            similar_entities = line.split(':')[1].strip()
            if similar_entities:
                profile_collection[current_entity].extend(list(map(int, similar_entities.split())))

    return profile_collection

def buildRedundancyPositiveBlocks(profile_collection):
    return profile_collection

def buildProfileIndex(blocks):
    profile_index = {}
    for block_id, profiles in blocks.items():
        for profile in profiles:
            if profile not in profile_index:
                profile_index[profile] = []
            profile_index[profile].append(block_id)
    return profile_index

def getComparison(i, j, weight):
    return (i, j, weight)

def wScheme(profile1, profile2, block, ProfileIndex):
    block_size = len(ProfileIndex[block])
    return 1 / block_size if block_size > 0 else 0

def initialization_phase(P, wScheme):
    B = buildRedundancyPositiveBlocks(P)
    ProfileIndex = buildProfileIndex(B)

    SortedProfileList = []
    topComparisonsSet = set()

    for pi in P:
        weights = {}
        distinctNeighbors = set()

        for bk in ProfileIndex.get(pi, []):
            for pj in B[bk]:
                if pj != pi:
                    weights[pj] = weights.get(pj, 0) + wScheme(pj, pi, bk, ProfileIndex)
                    distinctNeighbors.add(pj)

        topComparison = None
        duplicationLikelihood = 0

        for j in distinctNeighbors:
            duplicationLikelihood += weights[j]
            if topComparison is None or weights[j] > topComparison[2]:
                topComparison = getComparison(pi, j, weights[j])

        if topComparison:
            topComparisonsSet.add(topComparison)

        duplicationLikelihood /= max(1, len(distinctNeighbors))
        SortedProfileList.append((pi, duplicationLikelihood))

    ComparisonList = list(topComparisonsSet)
    ComparisonList.sort(key=lambda x: x[2], reverse=True)
    SortedProfileList.sort(key=lambda x: x[1], reverse=True)

    return ComparisonList, SortedProfileList, ProfileIndex

def emission_phase_for_pps(ComparisonList, SortedProfileList, P, ProfileIndex, wScheme, Kmax=100,count=0):
    checkedEntities = set()
    B = buildRedundancyPositiveBlocks(P)

    if not ComparisonList:

        while SortedProfileList:
            count += 1

            pi, _ = SortedProfileList.pop(0)
            checkedEntities.add(pi)
            weights = {}
            distinctNeighbors = set()

            for bk in ProfileIndex.get(pi, []):
                for pj in B[bk]:

                    if pj != pi and pj not in checkedEntities:
                        weights[pj] = weights.get(pj, 0) + wScheme(pj, pi, bk, ProfileIndex)
                        distinctNeighbors.add(pj)

            for j in distinctNeighbors:
                comparison = getComparison(pi, j, weights[j])
                ComparisonList.append(comparison)

            ComparisonList.sort(key=lambda x: x[2], reverse=True)
            while ComparisonList and ComparisonList[0][2] < 0.1:
                ComparisonList.pop(0)

    return ComparisonList.pop(0) if ComparisonList else None,count

def load_ground_truth(filepath):
    ground_truth = set()
    with open(filepath, 'r', encoding='utf-8') as file:
        pairs = file.read().strip().split('\n\n')
        for pair in pairs:
            numbers = list(map(int, pair.split()))
            if len(numbers) == 2:
                ground_truth.add(tuple(numbers))
    return ground_truth

def save_matched_results(output_path, match_result, saved_results):
    if match_result in saved_results or (match_result[1], match_result[0]) in saved_results:
        return  # Already saved, skip
    saved_results.add(match_result)
    with open(output_path, 'a') as file:
        file.write(f"{match_result[0]}|{match_result[1]}\n")

