"""Small GitHub API client that leaves authentication inside the GitHub CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any
from urllib.parse import urlencode


class GitHubError(RuntimeError):
    pass


GH_TIMEOUT_SECONDS = 45


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def repository_slug(repo: Path) -> str:
    url = git(repo, "remote", "get-url", "origin").stdout.strip()
    return repository_slug_from_url(url)


def repository_slug_from_url(url: str) -> str:
    patterns = (
        r"^https://github\.com/(?P<slug>[^/]+/[^/]+?)(?:\.git)?$",
        r"^git@github\.com:(?P<slug>[^/]+/[^/]+?)(?:\.git)?$",
        r"^ssh://git@github\.com/(?P<slug>[^/]+/[^/]+?)(?:\.git)?$",
    )
    for pattern in patterns:
        match = re.match(pattern, url)
        if match:
            return match.group("slug")
    raise GitHubError(f"URL is not a supported github.com repository: {url}")


class GitHubClient:
    def __init__(self, slug: str) -> None:
        self.slug = slug
        self._viewer: str | None = None

    def viewer_login(self) -> str:
        if self._viewer is not None:
            return self._viewer
        env = os.environ.copy()
        env["GH_PROMPT_DISABLED"] = "1"
        try:
            proc = subprocess.run(
                ["gh", "api", "--hostname", "github.com", "user", "--jq", ".login"],
                check=False,
                capture_output=True,
                text=True,
                env=env,
                timeout=GH_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise GitHubError("GitHub viewer lookup timed out") from error
        if proc.returncode != 0 or not proc.stdout.strip():
            raise GitHubError(f"GitHub viewer lookup failed: {proc.stderr.strip()[:500]}")
        self._viewer = proc.stdout.strip()
        return self._viewer

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        query: dict[str, str | int] | None = None,
    ) -> Any:
        suffix = path.strip("/")
        endpoint = f"repos/{self.slug}"
        if suffix:
            endpoint = f"{endpoint}/{suffix}"
        if query:
            endpoint = f"{endpoint}?{urlencode(query)}"
        args = [
            "gh", "api", "--hostname", "github.com", "--method", method, endpoint,
            "--header", "X-GitHub-Api-Version: 2022-11-28",
        ]
        input_text = None
        if payload is not None:
            args.extend(["--input", "-"])
            input_text = json.dumps(payload)
        env = os.environ.copy()
        env["GH_PROMPT_DISABLED"] = "1"
        try:
            proc = subprocess.run(
                args,
                input=input_text,
                check=False,
                capture_output=True,
                text=True,
                env=env,
                timeout=GH_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise GitHubError(f"GitHub {method} {path} timed out") from error
        if proc.returncode != 0:
            detail = proc.stderr.strip()[:500]
            raise GitHubError(f"GitHub {method} {path} failed: {detail}")
        if not proc.stdout.strip():
            return None
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as error:
            preview = repr(proc.stdout[:160])
            raise GitHubError(
                f"GitHub {method} {path} returned invalid JSON: {preview}"
            ) from error

    def get(self, path: str, **query: str | int) -> Any:
        return self.request("GET", path, query=query or None)

    def get_all(self, path: str, *, per_page: int = 100, max_pages: int = 20) -> list[Any]:
        items: list[Any] = []
        for page in range(1, max_pages + 1):
            batch = self.get(path, per_page=per_page, page=page)
            if not isinstance(batch, list):
                raise GitHubError(f"GitHub GET {path} did not return a list")
            items.extend(batch)
            if len(batch) < per_page:
                return items
        raise GitHubError(f"GitHub GET {path} exceeded {max_pages} pages")

    def post(self, path: str, payload: dict[str, Any]) -> Any:
        return self.request("POST", path, payload=payload)

    def patch(self, path: str, payload: dict[str, Any]) -> Any:
        return self.request("PATCH", path, payload=payload)
