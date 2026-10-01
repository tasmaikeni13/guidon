"""One-shot external key generation; run only in the registered Cloud Run job.

No private key is logged, persisted in the container, or returned to preparation.
The job can add a Secret Manager version but cannot read versions or change IAM.
Its public output is accepted only after independent permission checks.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import urllib.error
import urllib.request


def request(url: str, body: dict | None = None, *, authenticated: bool = True):
    headers = {"Content-Type": "application/json"}
    if authenticated:
        token_request = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/"
            "service-accounts/default/token",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(token_request, timeout=10) as response:
            token = json.load(response)["access_token"]
        headers["Authorization"] = "Bearer " + token
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        # Never include an API request/response body in an error: addVersion's
        # request contains the key. Metadata and HTTP status suffice for repair.
        raise RuntimeError(f"Custody API returned HTTP {error.code}") from None


def main() -> None:
    if not os.environ.get("CLOUD_RUN_EXECUTION"):
        raise RuntimeError("Generate evaluator keys only in the external Cloud Run job")
    secret = os.environ["GUIDON_CUSTODY_SECRET"]
    endpoint = "https://secretmanager.googleapis.com/v1/" + secret
    permissions = request(
        endpoint + ":testIamPermissions",
        {
            "permissions": [
                "secretmanager.versions.add",
                "secretmanager.versions.access",
                "secretmanager.secrets.setIamPolicy",
            ]
        },
    ).get("permissions", [])
    if (
        "secretmanager.versions.add" not in permissions
        or "secretmanager.versions.access" in permissions
        or "secretmanager.secrets.setIamPolicy" in permissions
    ):
        raise RuntimeError("Custody requires write-only, nonadministrative identity")
    versions = request(endpoint + "/versions")
    if versions.get("versions"):
        raise RuntimeError("A custody secret is immutable and must start empty")
    private = subprocess.run(
        [
            "openssl",
            "genpkey",
            "-algorithm",
            "RSA",
            "-pkeyopt",
            "rsa_keygen_bits:3072",
        ],
        check=True,
        capture_output=True,
    ).stdout
    public = subprocess.run(
        ["openssl", "pkey", "-pubout"],
        input=private,
        check=True,
        capture_output=True,
    ).stdout
    public_record = {
        "schema_version": 1,
        "event": "public_key_generated",
        "public_key_pem": public.decode("ascii"),
        "public_key_sha256": hashlib.sha256(public).hexdigest(),
        "bits": 3072,
        "secret": secret,
        "worker_permissions": sorted(permissions),
        "private_key_logged_or_written_to_container": False,
    }
    print(json.dumps(public_record, sort_keys=True), flush=True)
    version = request(
        endpoint + ":addVersion",
        {"payload": {"data": base64.b64encode(private).decode("ascii")}},
    )
    del private
    # Google canonicalizes a project ID to its numeric project resource name.
    # The authenticated API endpoint already selected the project; compare the
    # secret and exact version rather than a spelling of that project identity.
    if version["name"].split("/")[-4:] != secret.split("/")[-2:] + ["versions", "1"]:
        raise RuntimeError("Unexpected private-key version; retain and investigate")
    print(
        json.dumps(
            {
                "event": "private_key_deposited",
                "secret_version": version["name"],
                "public_key_sha256": public_record["public_key_sha256"],
                "state": version["state"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
