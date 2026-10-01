"""Copy a checked source snapshot and launch all workers of an EXISTING TPU pod.

This launcher never provisions resources or changes cloud/account configuration.
Worker installation and code live in a task-specific directory. Job log files
are immutable; dry-run is the default unless --execute is supplied.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shlex
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from guidon.artifacts import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]


def metadata(key: str) -> str:
    request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/" + key,
        headers={"Metadata-Flavor": "Google"},
    )
    return urllib.request.urlopen(request, timeout=5).read().decode().strip()


def discover() -> list[str]:
    if metadata("instance/attributes/accelerator-type") != "v4-32":
        raise ValueError("This launcher contract requires the existing v4-32")
    value = metadata("instance/attributes/worker-network-endpoints")
    workers = [endpoint.rsplit(":", 1)[-1] for endpoint in value.split(",")]
    if len(workers) != 4 or len(set(workers)) != 4:
        raise ValueError("v4-32 worker topology differs from the inspected contract")
    return workers


def snapshot(destination: Path) -> dict[str, Any]:
    paths = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
        )
        .decode()
        .split("\x00")
    )
    sources = [Path(p) for p in paths if p and (ROOT / p).is_file()]
    digest = hashlib.sha256()
    for path in sorted(sources):
        digest.update(str(path).encode() + b"\x00")
        digest.update((ROOT / path).read_bytes())
    identity = digest.hexdigest()
    executable = shutil.which("uv")
    if not executable:
        raise ValueError("uv executable is unavailable")
    with tarfile.open(destination, "x:gz") as archive:
        for path in sorted(sources):
            archive.add(ROOT / path, arcname=str(path), recursive=False)
        archive.add(executable, arcname=".tools/uv", recursive=False)
    return {
        "source_snapshot_sha256": identity,
        "archive_sha256": sha256_file(destination),
        "uv_binary_sha256": sha256_file(Path(executable)),
        "source_files": len(sources),
    }


def payload_snapshot(
    destination: Path, capability: Path | None, bundle: Path | None
) -> dict[str, Any]:
    """Transport only explicitly granted train/guidance and verified source assets."""
    paths: set[Path] = set()
    if capability is not None:
        from guidon.packing import training_readers

        value = json.loads(capability.read_text())
        _, access = training_readers(capability, cohort=value["cohort"])
        paths.add(capability.resolve())
        paths.update(Path(p) for p in access["reader_opened_paths"])
    if bundle is not None:
        record = json.loads((bundle / "bundle.json").read_text())
        if set(record["artifacts"]) != {
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "parity.json",
        }:
            raise ValueError("Unexpected pretrained transport bundle artifacts")
        paths.add((bundle / "bundle.json").resolve())
        for name, digest in record["artifacts"].items():
            path = (bundle / name).resolve()
            if sha256_file(path) != digest:
                raise ValueError("Pretrained transport bundle hash differs")
            paths.add(path)
    records = {}
    with tarfile.open(destination, "x:gz") as archive:
        for path in sorted(paths):
            name = path.relative_to(ROOT)
            if not path.is_file():
                raise ValueError("Transport payload must be a regular file")
            records[str(name)] = sha256_file(path)
            archive.add(path, arcname=str(name), recursive=False)
    return {
        "files": records,
        "archive_sha256": sha256_file(destination),
        "roles": ["train", "guidance"] if capability else [],
    }


def launch(
    module: str,
    arguments: list[str],
    run_dir: Path,
    *,
    execute: bool,
    workers: list[str] | None = None,
    remote_root: str = "/home/tasma/guidon-phase034",
    data_capability: Path | None = None,
    source_bundle: Path | None = None,
    timeout_seconds: int = 7200,
) -> dict[str, Any]:
    workers = workers or discover()
    if len(workers) != 4 or len(set(workers)) != 4:
        raise ValueError("Every distinct existing TPU worker must be launched")
    if module not in {"guidon.runtime", "guidon.train", "guidon.profile"}:
        raise ValueError("Unsupported checked launcher entry point")
    run_dir.mkdir(parents=True, exist_ok=False)
    source = snapshot(run_dir / "source.tar.gz")
    payload = payload_snapshot(
        run_dir / "payload.tar.gz", data_capability, source_bundle
    )
    remote = remote_root + "/" + source["source_snapshot_sha256"]
    job = hashlib.sha256(str(run_dir.resolve()).encode()).hexdigest()
    pid_root = remote + "/.jobs/" + job
    ssh_key = Path("/home/tasma/.ssh/google_compute_engine")
    ssh_options = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={run_dir.resolve() / 'known_hosts'}",
        "-i",
        str(ssh_key),
    ]
    commands = []
    for index, worker in enumerate(workers):
        command = [
            "env",
            "PYTHONUNBUFFERED=1",
            "JAX_PLATFORMS=tpu",
            "OMP_NUM_THREADS=8",
            "OPENBLAS_NUM_THREADS=1",
            f"GUIDON_COORDINATOR_ADDRESS={workers[0]}:12345",
            f"GUIDON_PROCESS_COUNT={len(workers)}",
            f"GUIDON_PROCESS_ID={index}",
            remote + "/.tools/uv",
            "run",
            "--directory",
            remote,
            "--no-sync",
            "python",
            "-m",
            module,
            *arguments,
        ]
        bootstrap = (
            "import pathlib,subprocess,sys; "
            f"root=pathlib.Path({pid_root!r}); root.mkdir(parents=True,exist_ok=True); "
            f"child=subprocess.Popen({command!r},start_new_session=True); "
            f"(root/{'worker' + str(index) + '.pid'!r}).write_text(str(child.pid)); "
            "sys.exit(child.wait())"
        )
        commands.append(
            ssh_options + [f"tasma@{worker}", shlex.join(["python3", "-c", bootstrap])]
        )
    record = source | {
        "workers": workers,
        "module": module,
        "arguments": arguments,
        "remote_root": remote,
        "commands": commands,
        "executed": execute,
        "provisioned_resources": False,
        "transported_payload": payload,
        "timeout_seconds": timeout_seconds,
    }
    atomic_json(run_dir / "launch.json", record)
    if not execute:
        return record

    def setup(worker: str) -> None:
        script = (
            "import pathlib,sys,tarfile; "
            f"root=pathlib.Path({remote!r}); root.mkdir(parents=True,exist_ok=True); "
            "archive=tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz'); "
            "[(archive.extract(m,path=root)) for m in archive "
            "if m.isfile() and not m.name.startswith('/') "
            "and '..' not in pathlib.PurePosixPath(m.name).parts]"
        )
        for name in ("source.tar.gz", "payload.tar.gz"):
            with (run_dir / name).open("rb") as archive:
                subprocess.run(
                    ssh_options
                    + [f"tasma@{worker}", shlex.join(["python3", "-c", script])],
                    stdin=archive,
                    check=True,
                    timeout=600,
                )
        command = [
            remote + "/.tools/uv",
            "sync",
            "--directory",
            remote,
            "--locked",
            "--extra",
            "dev",
            "--extra",
            "tpu",
            "--extra",
            "data",
        ]
        with (run_dir / f"setup-{worker}.log").open("x") as log:
            subprocess.run(
                ssh_options + [f"tasma@{worker}", shlex.join(command)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(setup, workers))
    logs, processes = [], []
    cancelled = False
    try:
        for index, command in enumerate(commands):
            log = (run_dir / f"worker{index}.log").open("x")
            logs.append(log)
            processes.append(
                subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            )
        deadline = time.monotonic() + timeout_seconds
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in {None, 0} for process in processes) or (
                time.monotonic() >= deadline
            ):
                cancelled = True
                break
            time.sleep(1)
        if cancelled:
            for index, worker in enumerate(workers):
                # Kill only the process group created for this exact job. Peer
                # failure must not leave another worker awaiting a collective.
                pid_path = pid_root + "/worker" + str(index) + ".pid"
                script = (
                    "import os,pathlib,signal; "
                    f"path=pathlib.Path({pid_path!r}); "
                    "pid=int(path.read_text()) if path.exists() else None; "
                    "os.killpg(pid,signal.SIGKILL) if pid else None"
                )
                subprocess.run(
                    ssh_options
                    + [f"tasma@{worker}", shlex.join(["python3", "-c", script])],
                    check=False,
                    timeout=30,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        codes = [process.wait(timeout=60) for process in processes]
        result = {
            "exit_codes": codes,
            "passed": all(code == 0 for code in codes),
            "peer_failure_or_timeout_cancelled": cancelled,
        }
        atomic_json(run_dir / "completion.json", result)
        if not result["passed"]:
            raise RuntimeError(
                f"Worker job failure: {codes}; preserve logs in {run_dir}"
            )
    finally:
        for log in logs:
            log.close()
    return record | result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", default="guidon.runtime")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--data-capability", type=Path)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("module_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.module_arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    record = launch(
        args.module,
        arguments,
        args.run_dir,
        execute=args.execute,
        data_capability=args.data_capability,
        source_bundle=args.source_bundle,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        json.dumps(
            {k: record[k] for k in ("workers", "source_snapshot_sha256", "executed")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
