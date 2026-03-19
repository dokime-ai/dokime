"""
Launch JEPA-SCORE experiments on Vast.ai in parallel.

Creates 3 GPU instances, uploads code, runs experiments, downloads results.

Usage:
  # First: set your API key (rotate the old one!)
  vastai set api-key YOUR_NEW_KEY

  # Then launch:
  python vast_launch.py

  # Check status:
  python vast_launch.py --status

  # Download results when done:
  python vast_launch.py --download

  # Clean up (destroy instances):
  python vast_launch.py --destroy
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# Files to upload to each instance
UPLOAD_FILES = [
    "jepa_score.py",
    "run_extended.py",
    "test_jepa_score.py",
    "vast_setup.sh",
]

# Three parallel experiments
EXPERIMENTS = {
    "A": {
        "name": "baselines + full_jacobian vits14",
        "commands": [
            "cd /workspace/jepa_score && python run_extended.py --experiment baselines --n-samples 500",
            "cd /workspace/jepa_score && python run_extended.py --experiment full_jacobian --model dinov2_vits14 --n-samples 500",
        ],
        "estimated_hours": 3.5,
    },
    "B": {
        "name": "projection_sweep vits14",
        "commands": [
            "cd /workspace/jepa_score && python run_extended.py --experiment projection_sweep --model dinov2_vits14 --n-samples 200",
        ],
        "estimated_hours": 3.0,
    },
    "C": {
        "name": "full_jacobian vitb14",
        "commands": [
            "cd /workspace/jepa_score && python run_extended.py --experiment full_jacobian --model dinov2_vitb14 --ood CIFAR10_vs_SVHN --n-samples 200",
        ],
        "estimated_hours": 2.0,
    },
}


def run_cmd(cmd: str, check: bool = True) -> str:
    """Run a shell command and return stdout."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"FAILED: {cmd}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def vastai(args: str) -> str:
    """Run a vastai CLI command."""
    return run_cmd(f"vastai {args}")


def search_gpus() -> list[dict]:
    """Find cheap GPU instances (>=16GB VRAM, CUDA ready)."""
    print("Searching for GPU instances...", flush=True)
    raw = vastai('search offers "gpu_ram>=16 num_gpus=1 cuda_vers>=12.0 inet_down>=200 reliability>=0.95 rentable=true" -o "dph+" --raw')
    offers = json.loads(raw)
    # Sort by price
    offers.sort(key=lambda o: o.get("dph_total", 999))
    return offers


def create_instance(offer_id: int, label: str) -> int:
    """Create a Vast.ai instance and return its ID."""
    print(f"  Creating instance '{label}' from offer {offer_id}...", flush=True)
    raw = vastai(
        f'create instance {offer_id} '
        f'--image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime '
        f'--disk 20 '
        f'--label "jepa-{label}" '
        f'--raw'
    )
    result = json.loads(raw)
    if "new_contract" in result:
        instance_id = result["new_contract"]
    else:
        print(f"  Unexpected response: {result}", file=sys.stderr)
        sys.exit(1)
    print(f"  Instance {instance_id} created.", flush=True)
    return instance_id


def wait_for_instance(instance_id: int, timeout: int = 300):
    """Wait for instance to be ready (SSH accessible)."""
    print(f"  Waiting for instance {instance_id} to start...", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        raw = vastai(f"show instance {instance_id} --raw")
        info = json.loads(raw)
        status = info.get("actual_status", "unknown")
        if status == "running":
            ssh_host = info.get("ssh_host", "")
            ssh_port = info.get("ssh_port", "")
            if ssh_host and ssh_port:
                print(f"  Instance {instance_id} running. SSH: {ssh_host}:{ssh_port}", flush=True)
                return info
        print(f"  Status: {status}, waiting...", flush=True)
        time.sleep(15)
    print(f"  Timeout waiting for instance {instance_id}", file=sys.stderr)
    sys.exit(1)


def upload_files(instance_id: int):
    """Upload experiment files to the instance."""
    print(f"  Uploading files to instance {instance_id}...", flush=True)
    for fname in UPLOAD_FILES:
        fpath = SCRIPT_DIR / fname
        if fpath.exists():
            vastai(f"copy {fpath} {instance_id}:/workspace/jepa_score/{fname}")
    print(f"  Upload complete.", flush=True)


def run_on_instance(instance_id: int, commands: list[str], label: str):
    """Run commands on an instance via SSH. Uses nohup for long-running tasks."""
    # Combine setup + experiment commands into a single script
    all_commands = [
        "bash /workspace/jepa_score/vast_setup.sh",
        *commands,
        "echo '=== EXPERIMENT COMPLETE ==='",
    ]
    script = " && ".join(all_commands)

    # Write to a run script and execute with nohup
    run_script = f'echo \'{script}\' > /workspace/run_{label}.sh && PYTHONUNBUFFERED=1 nohup bash /workspace/run_{label}.sh > /workspace/log_{label}.txt 2>&1 &'

    print(f"  Starting experiment {label} on instance {instance_id}...", flush=True)
    vastai(f'execute {instance_id} "{run_script}"')
    print(f"  Experiment {label} launched in background.", flush=True)


def launch():
    """Launch all 3 experiments in parallel."""
    print("=" * 60)
    print("LAUNCHING JEPA-SCORE EXPERIMENTS ON VAST.AI")
    print("=" * 60)

    # Find 3 cheap offers
    offers = search_gpus()
    if len(offers) < 3:
        print(f"Only {len(offers)} offers found, need 3.", file=sys.stderr)
        sys.exit(1)

    print(f"\nFound {len(offers)} offers. Top 3 cheapest:")
    for o in offers[:3]:
        print(f"  ID {o['id']}: {o.get('gpu_name', '?')} "
              f"${o.get('dph_total', '?'):.3f}/hr "
              f"({o.get('gpu_ram', '?'):.0f}GB VRAM)")

    # Create instances
    instances = {}
    for i, (label, exp) in enumerate(EXPERIMENTS.items()):
        offer = offers[i]
        print(f"\n--- Experiment {label}: {exp['name']} ---")
        print(f"  Estimated time: {exp['estimated_hours']:.1f} hours")
        print(f"  Estimated cost: ${offer.get('dph_total', 0) * exp['estimated_hours']:.2f}")
        iid = create_instance(offer["id"], label)
        instances[label] = iid

    # Wait for all instances to be ready
    print("\nWaiting for instances to start...")
    instance_info = {}
    for label, iid in instances.items():
        instance_info[label] = wait_for_instance(iid)

    # Upload code to all instances
    print("\nUploading code...")
    for label, iid in instances.items():
        upload_files(iid)

    # Launch experiments
    print("\nLaunching experiments...")
    for label, iid in instances.items():
        exp = EXPERIMENTS[label]
        run_on_instance(iid, exp["commands"], label)

    # Save instance IDs for later
    state = {"instances": instances, "launched_at": time.time()}
    state_path = SCRIPT_DIR / "vast_state.json"
    state_path.write_text(json.dumps(state, indent=2))

    print(f"\n{'=' * 60}")
    print("ALL EXPERIMENTS LAUNCHED")
    print(f"Instance IDs saved to {state_path}")
    print(f"\nMonitor with: python vast_launch.py --status")
    print(f"Download with: python vast_launch.py --download")
    print(f"Clean up with: python vast_launch.py --destroy")
    total_cost = sum(
        offers[i].get("dph_total", 0) * exp["estimated_hours"]
        for i, exp in enumerate(EXPERIMENTS.values())
    )
    print(f"\nEstimated total cost: ${total_cost:.2f}")
    print(f"{'=' * 60}")


def check_status():
    """Check status of running experiments."""
    state_path = SCRIPT_DIR / "vast_state.json"
    if not state_path.exists():
        print("No experiments launched yet. Run without --status first.")
        return

    state = json.loads(state_path.read_text())
    print("=" * 60)
    print("EXPERIMENT STATUS")
    print("=" * 60)

    for label, iid in state["instances"].items():
        exp = EXPERIMENTS[label]
        raw = run_cmd(f"vastai show instance {iid} --raw", check=False)
        if raw:
            info = json.loads(raw)
            status = info.get("actual_status", "unknown")
        else:
            status = "error/destroyed"
        print(f"\n  [{label}] {exp['name']}")
        print(f"      Instance: {iid}")
        print(f"      Status: {status}")

        # Try to read last line of log
        if status == "running":
            log = run_cmd(f'vastai execute {iid} "tail -3 /workspace/log_{label}.txt 2>/dev/null"', check=False)
            if log:
                print(f"      Last log: {log.strip()[:120]}")


def download_results():
    """Download results from all instances."""
    state_path = SCRIPT_DIR / "vast_state.json"
    if not state_path.exists():
        print("No experiments launched yet.")
        return

    state = json.loads(state_path.read_text())
    results_dir = SCRIPT_DIR / "results_extended"
    results_dir.mkdir(exist_ok=True)

    print("Downloading results...")
    for label, iid in state["instances"].items():
        print(f"\n  [{label}] Instance {iid}...")
        # Download JSON results
        run_cmd(
            f"vastai copy {iid}:/workspace/jepa_score/results_extended/ {results_dir}/",
            check=False,
        )
        # Download logs
        run_cmd(
            f"vastai copy {iid}:/workspace/log_{label}.txt {SCRIPT_DIR}/log_{label}.txt",
            check=False,
        )
    print(f"\nResults saved to {results_dir}")


def destroy_instances():
    """Destroy all running instances."""
    state_path = SCRIPT_DIR / "vast_state.json"
    if not state_path.exists():
        print("No experiments launched yet.")
        return

    state = json.loads(state_path.read_text())
    print("Destroying instances...")
    for label, iid in state["instances"].items():
        print(f"  Destroying instance {iid} ({label})...", flush=True)
        run_cmd(f"vastai destroy instance {iid}", check=False)
    print("Done. Instances destroyed.")


def main():
    p = argparse.ArgumentParser(description="Launch JEPA-SCORE experiments on Vast.ai")
    p.add_argument("--status", action="store_true", help="Check experiment status")
    p.add_argument("--download", action="store_true", help="Download results")
    p.add_argument("--destroy", action="store_true", help="Destroy instances")
    args = p.parse_args()

    if args.status:
        check_status()
    elif args.download:
        download_results()
    elif args.destroy:
        destroy_instances()
    else:
        launch()


if __name__ == "__main__":
    main()
