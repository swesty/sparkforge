#!/usr/bin/env python3
"""Push the current Worktrunk branch and create/update its Linear-linked GitHub PR."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from github_api import GitHubClient, GitHubError, git, repository_slug  # noqa: E402


DEFAULT_LINEAR_TEAM = "SHA"
LINEAR_TEAM_KEY = re.compile(r"^[A-Z][A-Z0-9]{1,9}$")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Push a verified Worktrunk branch and open/update its GitHub PR."
    )
    result.add_argument("--linear-id", help="Linear issue ID for a legacy branch without one")
    result.add_argument(
        "--linear-new",
        metavar="TEAM",
        help="Explicitly use TEAM-NEW in the PR body for genuinely unplanned work",
    )
    result.add_argument(
        "--linear-team",
        default=DEFAULT_LINEAR_TEAM,
        help=f"Linear team key (default: {DEFAULT_LINEAR_TEAM})",
    )
    result.add_argument("--title", help="PR title (default: latest commit subject)")
    result.add_argument("--body-file", type=Path, help="Additional PR body markdown")
    result.add_argument("--dry-run", action="store_true")
    return result


def linear_patterns(team: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    team = team.strip().upper()
    if not LINEAR_TEAM_KEY.fullmatch(team):
        raise ValueError(f"invalid Linear team key: {team}")
    identifier = re.compile(
        rf"(?<![A-Z0-9])({re.escape(team)}-\d+)(?![A-Z0-9])", re.I
    )
    unplanned = re.compile(
        rf"(?<![A-Z0-9]){re.escape(team)}-NEW(?![A-Z0-9])", re.I
    )
    return identifier, unplanned


def linear_link(
    branch: str,
    explicit: str | None,
    new_team: str | None,
    team: str = DEFAULT_LINEAR_TEAM,
) -> str:
    team = team.strip().upper()
    identifier, _ = linear_patterns(team)
    match = identifier.search(branch)
    if match:
        return match.group(1).upper()
    if explicit:
        if not identifier.fullmatch(explicit.strip()):
            raise ValueError(f"invalid Linear issue ID: {explicit}")
        return explicit.strip().upper()
    if new_team:
        requested_team = new_team.strip().upper()
        if requested_team != team:
            raise ValueError(
                f"this repository is linked to Linear team {team}, not {new_team}"
            )
        return f"{requested_team}-NEW"
    raise ValueError(
        "branch has no Linear issue ID; use Linear's copied branch name, --linear-id ID-123, "
        "or the explicit --linear-new TEAM escape hatch"
    )


def has_requested_link(branch: str, title: str, body: str, link: str) -> bool:
    """Match the surfaces Linear actually uses; TEAM-NEW is body-only."""
    pattern = re.compile(rf"(?<![A-Z0-9]){re.escape(link)}(?![A-Z0-9])", re.I)
    if link.endswith("-NEW"):
        return bool(pattern.search(body))
    return bool(pattern.search(" ".join((branch, title, body))))


def durable_links(
    branch: str,
    title: str,
    body: str,
    team: str = DEFAULT_LINEAR_TEAM,
) -> set[str]:
    identifier, unplanned = linear_patterns(team)
    combined = " ".join((branch, title, body))
    links = {match.group(1).upper() for match in identifier.finditer(combined)}
    links.update(match.group(0).upper() for match in unplanned.finditer(body))
    return links


def require_only_requested_link(
    branch: str,
    title: str,
    body: str,
    requested: str,
    team: str = DEFAULT_LINEAR_TEAM,
) -> None:
    links = durable_links(branch, title, body, team)
    conflicts = sorted(links - {requested})
    if conflicts:
        raise ValueError(
            f"PR metadata contains conflicting Linear link(s): {', '.join(conflicts)}; "
            f"keep only {requested}"
        )


def main() -> int:
    args = parser().parse_args()
    team = args.linear_team.strip().upper()
    repo = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").stdout.strip())
    branch = git(repo, "branch", "--show-current").stdout.strip()
    default_ref = git(
        repo, "symbolic-ref", "--short", "refs/remotes/origin/HEAD"
    ).stdout.strip()
    default_branch = default_ref.split("/", 1)[1]
    if not branch or branch == default_branch:
        raise SystemExit("wt ready must run from a non-default Worktrunk branch")
    if git(repo, "status", "--porcelain").stdout:
        raise SystemExit("worktree is dirty; commit with `wt step commit` before `wt ready`")

    try:
        linear_patterns(team)
        link = linear_link(branch, args.linear_id, args.linear_new, team)
        slug = repository_slug(repo)
    except (ValueError, GitHubError) as error:
        raise SystemExit(str(error)) from error

    subject = args.title or git(repo, "log", "-1", "--format=%s").stdout.strip()
    title = subject if link in subject.upper() or link.endswith("-NEW") else f"{link}: {subject}"
    link_line = link if link.endswith("-NEW") else f"Fixes {link}"
    extra = args.body_file.read_text() if args.body_file else ""
    body = f"{link_line}\n\n{extra.strip()}\n\nPrepared with `wt ready`.".strip()
    try:
        require_only_requested_link(branch, title, body, link, team)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if args.dry_run:
        print(f"would push {branch} and open/update PR: {title}")
        print(link_line)
        return 0

    subprocess.run(["git", "-C", str(repo), "push", "-u", "origin", branch], check=True)
    client = GitHubClient(slug)
    owner = slug.split("/", 1)[0]
    pulls = client.get("pulls", state="open", head=f"{owner}:{branch}", per_page=10)
    if pulls:
        pull = pulls[0]
        existing_title = pull.get("title") or ""
        existing_body = pull.get("body") or ""
        try:
            require_only_requested_link(
                branch, existing_title, existing_body, link, team
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        if not has_requested_link(branch, existing_title, existing_body, link):
            pull = client.patch(
                f"pulls/{pull['number']}",
                {"title": title, "body": f"{link_line}\n\n{existing_body}".strip()},
            )
            print(f"updated Linear link: {pull['html_url']}")
        else:
            print(pull["html_url"])
        return 0

    pull = client.post(
        "pulls",
        {"title": title, "head": branch, "base": default_branch, "body": body},
    )
    print(pull["html_url"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
