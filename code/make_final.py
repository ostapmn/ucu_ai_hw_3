import json

import common

CANDIDATES = {
    "candidate_1": ("headtail_abcdg_win3_blog", "submission_abcdg_win3_blog.csv"),
    "candidate_2": ("headtail_abcd_win", "submission_abcd_win.csv"),
}


def main():
    results = json.loads((common.RESULTS / "stack_v2.json").read_text())
    chosen = {}
    for name, (tag, csv) in CANDIDATES.items():
        row = results[tag]
        chosen[name] = {
            "tag": tag,
            "submission_file": f"code/artifacts/{csv}",
            "sources": row["sources"],
            "dev_cv_weighted_macro_f1": row["dev_cv_weighted_macro_f1"],
            "lock_weighted_macro_f1": row["lock_weighted_macro_f1"],
            "dev_cv_macro_f1": row["dev_cv_macro_f1"],
            "lock_macro_f1": row["lock_macro_f1"],
            "final_model": row["final_model"],
        }
    out = {
        "selected_by": "dev_cv_and_lock",
        "public_leaderboard_used": False,
        "rule": "candidate_1 is the best stack on dev CV with a non-negative lock difference; candidate_2 is the most stable earlier stack (dev equals lock) with a different composition",
        "candidates": chosen,
    }
    (common.RESULTS / "stack_v2_final.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: (v["dev_cv_weighted_macro_f1"], v["lock_weighted_macro_f1"]) for k, v in chosen.items()}, indent=2))


if __name__ == "__main__":
    main()
