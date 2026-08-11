import argparse
import json
import os
from typing import Dict, List


def _load_jsonl_by_instance(path: str) -> Dict[str, dict]:
    data: Dict[str, dict] = {}
    if not os.path.exists(path):
        return data

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            instance_id = item.get("instance_id")
            if instance_id:
                data[instance_id] = item
    return data


def _merge_records(test_item: dict, supp_item: dict) -> dict:
    test_locs: List[str] = test_item.get("suspicious_locations", [])
    supp_locs: List[str] = supp_item.get("suppletory_locations", [])

    merged_locs = list(dict.fromkeys(test_locs + supp_locs))
    common_locs = sorted(set(test_locs) & set(supp_locs))

    return {
        "locations": merged_locs,
        "locations_from_test_base": test_locs,
        "locations_from_suppletory": supp_locs,
        "common_locations": common_locs,
        "test_base": test_item,
        "suppletory": supp_item,
    }


def merge_results(target_folder: str) -> None:
    test_based_results_path = os.path.join(
        target_folder, "blind_spot_analysis", "loc_outputs.jsonl"
    )
    suppletory_results_path = os.path.join(
        target_folder, "suppletory_retrieval", "loc_outputs.jsonl"
    )
    merge_results_path = os.path.join(target_folder, "merge", "loc_outputs.jsonl")

    test_based_data = _load_jsonl_by_instance(test_based_results_path)
    suppletory_data = _load_jsonl_by_instance(suppletory_results_path)

    all_instance_ids = sorted(set(test_based_data.keys()) | set(suppletory_data.keys()))

    os.makedirs(os.path.dirname(merge_results_path), exist_ok=True)
    with open(merge_results_path, "w", encoding="utf-8") as file:
        for instance_id in all_instance_ids:
            test_item = test_based_data.get(instance_id, {})
            supp_item = suppletory_data.get(instance_id, {})
            merged = _merge_records(test_item, supp_item)
            merged["instance_id"] = instance_id
            file.write(json.dumps(merged, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_folder", type=str, required=True)
    args = parser.parse_args()
    merge_results(args.target_folder)


if __name__ == "__main__":
    main()
