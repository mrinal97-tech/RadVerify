import json

with open(
    "data/real_eval_set.json",
    "r",
    encoding="utf-8"
) as f:
    cases = json.load(f)

fn_ids = {
    "CXR212_IM-0746-1001-0001",
    "CXR2885_IM-1287-2001",
    "CXR1459_IM-0297-1001",
    "CXR3988_IM-2041-1001",
}

for case in cases:

    if (
        case["case_id"] in fn_ids
        and case["is_corrupted"]
    ):

        print("=" * 80)
        print("CASE:", case["case_id"])
        print("ERROR:", case["error_type"])
        print("DETAIL:", case["error_detail"])

        print("\nSTRUCTURED FINDINGS:")
        for f in case["findings"]:
            print(f)

        print("\nCORRUPTED REPORT:")
        print(case["report"])

        print()