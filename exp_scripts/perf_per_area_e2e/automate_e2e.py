import argparse
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
COMP_E2E_DIR = (SCRIPT_DIR / ".." / "comparison_e2e").resolve()


def _run(cmd: str):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=SCRIPT_DIR)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {cmd}")


def main():
    parser = argparse.ArgumentParser(description="Automate perf/area e2e workflow.")
    parser.add_argument(
        "--input",
        type=str,
        default="run_all_output_11-8",
        help="Suffix used in comparison_e2e CSV filenames.",
    )
    parser.add_argument(
        "--decode_tokens",
        type=int,
        default=128,
        help="Decode token count used in comparison_e2e outputs.",
    )
    args = parser.parse_args()

    tokens_tag = f"1024to{args.decode_tokens}"
    lat_csv = COMP_E2E_DIR / f"latencies_e2e_{tokens_tag}_{args.input}.csv"
    thr_csv = COMP_E2E_DIR / f"throughput_decode_{tokens_tag}_{args.input}.csv"
    pref_csv = COMP_E2E_DIR / f"latencies_prefill_{tokens_tag}_{args.input}.csv"

    if not lat_csv.exists():
        raise FileNotFoundError(f"Missing latencies CSV: {lat_csv}")
    if not thr_csv.exists():
        raise FileNotFoundError(f"Missing throughput CSV: {thr_csv}")
    if not pref_csv.exists():
        raise FileNotFoundError(f"Missing prefill CSV: {pref_csv}")

    perf_lat_csv = SCRIPT_DIR / f"perf_per_area_{lat_csv.stem}.csv"
    perf_thr_csv = SCRIPT_DIR / f"perf_per_area_{thr_csv.stem}.csv"
    perf_pref_csv = SCRIPT_DIR / f"perf_per_area_{pref_csv.stem}.csv"

    calc_cmd = (
        "python calculate_perf_per_area_e2e.py "
        f"--latencies \"{lat_csv}\" --latency-output \"{perf_lat_csv}\" "
        f"--throughput \"{thr_csv}\" --throughput-output \"{perf_thr_csv}\" "
        f"--prefill \"{pref_csv}\" --prefill-output \"{perf_pref_csv}\""
    )
    _run(calc_cmd)

    plot_out = SCRIPT_DIR / f"perf_per_area.pdf"
    plot_cmd = (
        "python plot_perf_per_area_e2e.py "
        f"--throughput \"{perf_thr_csv}\" "
        f"--prefill \"{perf_pref_csv}\" "
        f"--output \"{plot_out}\""
    )
    _run(plot_cmd)


if __name__ == "__main__":
    main()
