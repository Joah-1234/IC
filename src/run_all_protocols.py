import os
import subprocess
import sys
from pathlib import Path


PROTOCOLS = ["OCI_to_M", "OMI_to_C", "OCM_to_I", "ICM_to_O"]
SCRIPT_DIR = Path(__file__).resolve().parent


def run_command(command, env):
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, env=env, check=True)


def main():
    base_env = os.environ.copy()
    python_executable = sys.executable

    for protocol in PROTOCOLS:
        env = base_env.copy()
        env["SRTP_PROTOCOL_NAME"] = protocol
        print("=" * 60)
        print(f"Start protocol: {protocol}")
        print("=" * 60)
        run_command([python_executable, str(SCRIPT_DIR / "train.py")], env)
        run_command([python_executable, str(SCRIPT_DIR / "test.py")], env)

    print("All protocols finished.")


if __name__ == "__main__":
    main()
