#!/usr/bin/env python3
"""Export the human side of a repo's AI coding dialogue to one HTML page.

Reads the local transcript stores of four coding agents and produces a
single self-contained HTML document, ordered by timestamp. The human's
prompts are the only text visible by default, each under a label naming
the human who typed it; everything machine-generated is collapsed behind a
quiet disclosure line that names the agent, model, and time. Supported
stores:

- Claude Code:  ~/.claude/projects/**/*.jsonl
- Codex:        ~/.codex/{sessions,archived_sessions}/**/*.jsonl
- Copilot Chat: VS Code User/workspaceStorage/*/chatSessions/*.{json,jsonl}
- Antigravity:  ~/.gemini/antigravity/conversations/*.pb (sealed protobuf;
                opened through the system cipher, macOS only)

Usage:
    python3 sourcery.py REPODIR OUTPUT.html [--open]

OUTPUT.html is derived from the ledgers beside it, one per human who runs
sourcery. A run rewrites the runner's own ledger, merging what the
runner's stores still hold with what only that ledger remembers, then
renders the page from every ledger, so nothing once rendered is lost when
stores are pruned or machines change. A page sourcery 5 wrote, which
carries its own snapshot instead, is refused until adopt.py has imported
it.

Nonstandard store locations can be supplied with path-separated environment
variables: AI_CHAT_CLAUDE_ROOTS, AI_CHAT_CODEX_ROOTS, AI_CHAT_VSCODE_USER_ROOTS,
AI_CHAT_ANTIGRAVITY_ROOTS.

Jargon: an "exchange" is one human prompt plus everything the agent said
back before the next human prompt; "weaving" merges every store's exchanges
into one deduplicated chronology; a "wrapper" is machine-generated text an
IDE smuggles into the user role (open-file context, system reminders); a
"canned" prompt is a fixed string a UI control fabricates in the user role
(retry/continue buttons, quick-fix templates). Neither is the human's words
and both are dropped.

More jargon: a "login" is a human's GitHub account name, as the gh CLI
saved it, lowercased; a "ledger" is one human's sourcery.LOGIN.jsonl, every
exchange of theirs the page shows; a "display name" is what the page calls
a human; a "credit" is the page's record of whose ledger holds a row.
A "project name" is what every ledger's first line calls the repo: its
repository's name, from its public home, or its checkout directory's name
when it has no public home (see project_name).
"""

from __future__ import annotations

import base64
import ctypes
import ctypes.util
import dataclasses
import datetime as dt
import functools
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

VERSION = "6.1.0"
UTC = dt.timezone.utc


class UserError(RuntimeError):
    """An actionable failure caused by input, local data, or configuration."""


class ExitMessage(RuntimeError):
    """A successful informational exit such as --help or --version."""


@dataclasses.dataclass(frozen=True)
class Ballot:
    """One answered multiple-choice question: the agent's question and
    option labels (machine prose) and which label(s) the human picked —
    empty when the human typed their own answer instead."""

    question: str
    options: tuple[str, ...]
    picked: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Exchange:
    timestamp: dt.datetime
    provider: str
    model: str
    session: str
    prompt: str
    reply: str
    source: Path
    effort: str = ""
    images: tuple[str, ...] = ()  # data: URIs of images pasted with the prompt
    elapsed: float = 0.0  # seconds the agent worked on the reply; 0 = unknown
    # Wall-clock seconds from the prompt to the last reply record; 0 = unknown.
    # Exceeds elapsed by whatever the working-time accounting cannot see: tool
    # runtime the store never recorded, and waits on the human mid-turn.
    wall: float = 0.0
    ballots: tuple[Ballot, ...] = ()  # multiple-choice questions the human answered
    # The prompt's diffstat: repo lines the agent added and deleted before the
    # next prompt, as recorded by the store (edits made through a shell tool
    # leave no record, so these are floors; Copilot records none at all).
    added: int = 0
    deleted: int = 0


# A "holding" names a prompt record the store still holds: its provider and
# timestamp. Every parser reports one for each record it attributes to the
# repo that carries typing, or that it deliberately drops as machine text
# (a canned prompt, a button click, a notification), so run() can tell a
# record the store lost (keep the ledger's copy) from one the store still
# has (the store's reading wins, purging stale ledger copies). Tool plumbing
# that carries no typing is never a holding: it was never rendered, so it
# must never purge a ledger copy that merely shares its millisecond.
Holding = tuple[str, dt.datetime]


def holding(exchange: Exchange) -> Holding:
    return (exchange.provider, exchange.timestamp)


@dataclasses.dataclass(frozen=True)
class Message:
    """One utterance inside a session, before pairing into exchanges."""

    role: str
    timestamp: dt.datetime
    text: str
    model: str = ""
    effort: str = ""
    images: tuple[str, ...] = ()
    active: float = 0.0  # seconds the agent worked to produce this message
    ballots: tuple[Ballot, ...] = ()
    # Repo lines changed by tool calls: on a reply, the lines behind it; on a
    # prompt, lines orphaned mid-turn by its arrival, which belong to the
    # exchange this prompt closes.
    added: int = 0
    deleted: int = 0


@dataclasses.dataclass(frozen=True)
class Roots:
    claude: tuple[Path, ...]
    codex: tuple[Path, ...]
    vscode: tuple[Path, ...]
    antigravity: tuple[Path, ...]

    def all(self) -> tuple[Path, ...]:
        return self.claude + self.codex + self.vscode + self.antigravity


@dataclasses.dataclass(frozen=True)
class Options:
    repo: Path
    output: Path
    open_after: bool


# ------------------------------------------------------------------- plumbing


def help_text() -> str:
    return f"""Usage:
  python3 sourcery.py REPODIR OUTPUT.html [--open]

Arguments:
  REPODIR         Directory containing the project to extract the dialog from.
  OUTPUT.html     Generate new html.
  --open          Open the generated HTML in the browser.
  -h, --help      This help text.
  --version       Show version.

Environment variables (separated by {os.pathsep!r}) for nonstandard transcript locations:
  AI_CHAT_CLAUDE_ROOTS
  AI_CHAT_CODEX_ROOTS
  AI_CHAT_VSCODE_USER_ROOTS
  AI_CHAT_ANTIGRAVITY_ROOTS
"""


def parse_args(argv: Sequence[str]) -> Options:
    positionals: list[str] = []
    flags: set[str] = set()
    for token in argv:
        match token:
            case "-h" | "--help":
                raise ExitMessage(help_text())
            case "--version":
                raise ExitMessage(VERSION)
            case "--open":
                if token in flags:
                    # TODO: Says the --open flag was given more than once.
                    raise UserError("Argumentum --open iteratum est.")
                flags.add(token)
            case _ if token.startswith("-"):
                # TODO: Says this option is not recognized.
                raise UserError(f"Argumentum ignotum: {token}\n\n{help_text()}")
            case _:
                positionals.append(token)
    if len(positionals) != 2:
        # TODO: Says exactly two arguments are required, the project
        # directory and the output file.
        raise UserError(f"Duo argumenta necessaria sunt: REPODIR et OUTPUT.html.\n\n{help_text()}")
    repo, output = positionals
    return Options(
        repo=Path(repo).expanduser(),
        output=Path(output).expanduser(),
        open_after="--open" in flags,
    )


def canonical_repo(path: Path) -> Path:
    candidate = path.resolve()
    if not candidate.is_dir():
        # TODO: Says the project directory wasn't found and to give a path
        # to an existing project directory.
        raise UserError(
            f"Directorium incepti non inventum: {candidate}\n"
            "Da viam exsistentem ad directorium incepti."
        )
    return candidate


def https_remote(url: str) -> str:
    """Normalize a git remote URL (https, ssh, or scp-style) to https, or "".

    The home is host plus path. Whatever precedes the host — a username, or
    a hosting service's token — is how this machine authenticates, not where
    the project lives, and must never reach a published page.
    """
    web = re.fullmatch(r"https?://(?:[^/@]*@)?(.+?)(?:\.git)?/?", url)
    if web:
        return f"https://{web.group(1)}"
    scp = re.fullmatch(r"(?:ssh://)?git@([^:/]+)[:/](.+?)(?:\.git)?/?", url)
    return f"https://{scp.group(1)}/{scp.group(2)}" if scp else ""


REMOTE_SECTION = re.compile(r'\[remote "(.+)"\]')


def repo_remote(repo: Path) -> str:
    """Return the repo's origin remote as an https URL, or "" when absent.

    A project with no remote lives only here, and so does one whose remote
    names no browsable page (a git:// mirror, an ssh host serving no web
    page); for both, "" is the honest answer and the page falls back to the
    local directory. A project whose public home sits under some other
    remote name is the one case that must not pass quietly: the home is
    plainly there, and only the name hides it.
    """
    config = repo / ".git" / "config"
    if not config.is_file():
        return ""
    remotes: dict[str, str] = {}
    name = ""
    for line in config.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            found = REMOTE_SECTION.fullmatch(stripped)
            name = found.group(1) if found else ""
        elif name:
            key, _, value = stripped.partition("=")
            if key.strip() == "url":
                # The first URL wins, as it does for the version-control tool
                # itself when a remote carries several.
                remotes.setdefault(name, value.strip())
    home = https_remote(remotes.pop("origin", ""))
    hidden = {other: link for other, url in remotes.items() if (link := https_remote(url))}
    if not home and hidden:
        homes = ", ".join(f"{other} ({link})" for other, link in sorted(hidden.items()))
        # TODO: Says that no remote named origin points at a public home,
        # lists the remotes that do, and asks the user to make origin point
        # at the same home — by renaming that remote, or by setting origin's
        # own URL — then rerun. Sourcery will not guess which home to show.
        # (It says nothing about whether an origin exists: one may, pointing
        # somewhere that has no web page of its own.)
        raise UserError(
            f"Nullum remotum nomine 'origin' sedem publicam monstrat: {repo}\n"
            f"Sedem publicam habent: {homes}\n"
            "Fac ut 'origin' eandem sedem monstret (aut nomen illius remoti muta, "
            "aut URL ipsius 'origin' constitue), deinde iterum curre."
        )
    return home


def project_name(repo: Path, remote: str) -> str:
    """The repo's project name: the last segment of the path of its public
    home (remote, as repo_remote found it), spelled as the URL spells it,
    or, when it has no public home, its checkout directory's name. A
    default clone names the directory after the repository, so the two
    agree until a directory is renamed; then, while there is a public home,
    the directory's name never matters. A home whose path names no
    repository (no path at all, or one ending in "/", as an origin URL
    ending in "//" or "/.git" leaves it) is refused, as configuration
    faults are."""
    if remote == "":
        return repo.name
    name = urllib.parse.urlsplit(remote).path.rsplit("/", 1)[-1]
    if name == "":
        # TODO: Says the public home sourcery read from the origin remote
        # (named, as sourcery read it) names no repository, naming this
        # checkout, and to set origin's URL to the repository's home, then
        # rerun.
        raise UserError(
            f"Sedes publica ex remoto 'origin' lecta nullum repositorium nominat: {remote!r} ({repo})\n"
            "URL ipsius 'origin' ad sedem repositorii constitue, deinde iterum curre."
        )
    return name


def env_paths(env: Mapping[str, str], key: str, defaults: Iterable[Path]) -> tuple[Path, ...]:
    raw = env.get(key)
    if raw is None:
        return tuple(defaults)
    paths = tuple(Path(piece).expanduser() for piece in raw.split(os.pathsep) if piece)
    if not paths:
        # TODO: Says this environment variable is set but empty, and to
        # either unset it or put at least one path in it.
        raise UserError(f"Variabilis {key} vacua est. Aufer eam vel saltem unam viam da.")
    return paths


def default_vscode_roots(home: Path) -> tuple[Path, ...]:
    match sys.platform:
        case "darwin":
            base = home / "Library" / "Application Support"
        case "win32":
            base = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
        case _:
            base = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config"))
    local = tuple(base / product / "User" for product in ("Code", "Code - Insiders", "VSCodium"))
    remote = (
        home / ".vscode-server" / "data" / "User",
        home / ".vscode-server-insiders" / "data" / "User",
    )
    return local + remote


def discover_roots(env: Mapping[str, str]) -> Roots:
    home = Path.home()
    claude_default = (
        Path(env["CLAUDE_CONFIG_DIR"]).expanduser() / "projects"
        if "CLAUDE_CONFIG_DIR" in env
        else home / ".claude" / "projects"
    )
    codex_default = (
        Path(env["CODEX_HOME"]).expanduser() if "CODEX_HOME" in env else home / ".codex"
    )
    return Roots(
        claude=env_paths(env, "AI_CHAT_CLAUDE_ROOTS", (claude_default,)),
        codex=env_paths(env, "AI_CHAT_CODEX_ROOTS", (codex_default,)),
        vscode=env_paths(env, "AI_CHAT_VSCODE_USER_ROOTS", default_vscode_roots(home)),
        antigravity=env_paths(env, "AI_CHAT_ANTIGRAVITY_ROOTS", (home / ".gemini" / "antigravity",)),
    )


def parse_time(value: Any) -> dt.datetime:
    match value:
        case bool():
            # TODO: Says a timestamp is in an unrecognized form.
            raise UserError(f"Forma temporis ignota: {value!r}")
        case int() | float():
            seconds = float(value)
            # Epoch milliseconds and epoch seconds are both in the wild;
            # 1e10 seconds is beyond year 2200, so larger numbers are ms.
            seconds = seconds / 1000.0 if abs(seconds) > 10_000_000_000 else seconds
            try:
                return dt.datetime.fromtimestamp(seconds, tz=UTC)
            except (OverflowError, OSError, ValueError) as exc:
                # TODO: Says a numeric timestamp is invalid.
                raise UserError(f"Tempus numericum invalidum: {value!r}") from exc
        case str():
            raw = value.strip()
            normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
            try:
                parsed = dt.datetime.fromisoformat(normalized)
            except ValueError as exc:
                # TODO: Says a timestamp is not valid ISO-8601.
                raise UserError(f"Tempus ISO-8601 invalidum: {value!r}") from exc
            parsed = parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed
            return parsed.astimezone(UTC)
        case _:
            # TODO: Says a timestamp is in an unrecognized form.
            raise UserError(f"Forma temporis ignota: {type(value).__name__}")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except PermissionError as exc:
        # TODO: Says read permission is missing for this file and to grant
        # the terminal read access, then rerun.
        raise UserError(
            f"Licentia legendi deest: {path}\n"
            "Da terminali licentiam legendi hunc fasciculum et iterum curre."
        ) from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        # TODO: Says this file contains invalid JSON.
        raise UserError(f"JSON invalidum in {path}: {exc}") from exc


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        fh = path.open("r", encoding="utf-8")
    except PermissionError as exc:
        # TODO: Says read permission is missing for this file and to grant
        # the terminal read access, then rerun.
        raise UserError(
            f"Licentia legendi deest: {path}\n"
            "Da terminali licentiam legendi hunc fasciculum et iterum curre."
        ) from exc
    except OSError as exc:
        # TODO: Says this file cannot be opened.
        raise UserError(f"Fasciculus aperiri non potest: {path}\n{exc}") from exc
    with fh:
        for line_number, line in enumerate(fh, start=1):
            if line == "\n":
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                # A line with no trailing newline is necessarily the file's
                # last, and an unparseable one is a photograph of an append
                # still in progress (live capture), not corruption: the
                # complete prefix is the transcript. Damage mid-file is real
                # corruption and stays loud.
                if not line.endswith("\n"):
                    return
                # TODO: Says this file has invalid JSONL at this line, and to
                # close Claude, Codex, and VS Code, then rerun.
                raise UserError(
                    f"JSONL invalidum: {path}:{line_number}\n{exc}\n"
                    "Claude, Codex, et VS Code claude; deinde iterum curre."
                ) from exc
            if not isinstance(record, dict):
                # TODO: Says this JSONL line is not a JSON object.
                raise UserError(f"Recordum JSONL obiectum non est: {path}:{line_number}")
            yield line_number, record


def decode_file_uri(value: str) -> str | None:
    """Return the filesystem path of a file:// URI, or None for other schemes."""
    if not value.startswith("file://"):
        return None
    parsed = urllib.parse.urlparse(value)
    path = urllib.parse.unquote(parsed.path)
    if os.name == "nt" and len(path) >= 3 and path[0] == "/" and path[2] == ":":
        return path[1:]
    return path


def under_dir(value: Any, root: Path, source: Path) -> bool:
    """Whether the path a record of the transcript at source names lies
    under root."""
    if not isinstance(value, str) or value == "":
        return False
    path = Path(value).expanduser()
    # On macOS, /home is an automount point, not a directory of this
    # machine's own: resolving a path under it (the resolve() below) asks the
    # automounter, which can hang for minutes. A path there comes from
    # another machine's transcript, such as a Claude cloud workspace's, whose
    # checkouts live under its own /home, so it is refused before anything
    # resolves it.
    if sys.platform == "darwin" and path.is_relative_to("/home"):
        # TODO: Says this transcript came from another machine (for example
        # a Claude cloud workspace), naming it and the path under /home it
        # names; its directories must first be rewritten to this machine's
        # checkout of the project, then rerun.
        raise UserError(
            f"Transcriptum ex alia machina venit (exempli gratia ex spatio operis Claude in nube): {source}\n"
            f"Via {value} sub /home iacet. Directoria transcripti prius ad exemplar incepti in hac machina "
            "rescribe, deinde iterum curre."
        )
    return path.resolve().is_relative_to(root)


# Jargon: to "tally" is to count the lines a recorded edit added and deleted;
# every tally function returns an (added, deleted) pair.
def diff_tally(lines: Iterable[str]) -> tuple[int, int]:
    """Tally unified-diff hunk lines: a +/- prefix marks an added/deleted
    line. Callers pass only hunks — never the +++/--- file headers of a full
    diff — so a line like "---" is a deletion whose content starts with
    "--", not a header."""
    added = deleted = 0
    for line in lines:
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            deleted += 1
    return (added, deleted)


# --------------------------------------------------------------- Claude Code

# Wrapper text items the Claude Code harness injects into the user role:
# <ide_opened_file>, <ide_selection>, and <system-reminder> blocks. A text
# item is a wrapper only when the tag spans the entire item.
CLAUDE_WRAPPER = re.compile(r"<(ide_[a-z_]+|system-reminder)>.*</\1>\s*", re.DOTALL)

# A slash command is stored as two redundant XML-like fields. Matching both
# fields prevents a partial or changed wrapper from becoming apparent typing.
CLAUDE_COMMAND = re.compile(
    r"\A<command-message>([^<]+)</command-message>\n<command-name>/\1</command-name>\Z"
)

# Claude Code ≥2.1.215 puts the name first, indents the wrapper, and stores
# any typed arguments in a third field. The arguments are real typing and
# must reappear after the command name; only the inter-tag whitespace, which
# the harness generates, is matched leniently.
CLAUDE_COMMAND_ARGS = re.compile(
    r"\A<command-name>/([^<]+)</command-name>\n\s*"
    r"<command-message>\1</command-message>\n\s*"
    r"<command-args>([^<]*)</command-args>\Z"
)

# A harness-local command (/model, /config, ...) records its captured output
# as a pseudo-user record. The output is machine text, and its arrival also
# reclassifies the slash-command prompt sharing its promptId: that pair was
# harness configuration, not dialogue, so both records vanish together.
CLAUDE_LOCAL_STDOUT = re.compile(
    r"\A<local-command-stdout>.*</local-command-stdout>\s*\Z", re.DOTALL
)

# Record flags that mark machine-generated pseudo-messages: subagent traffic,
# meta records, compaction summaries, and transcript-only continuation notes.
CLAUDE_SKIP_FLAGS = ("isSidechain", "isMeta", "isCompactSummary", "isVisibleInTranscriptOnly")

# Canned marker Claude Code records in the user role when the human hits
# interrupt; dropped only when it is the entire prompt.
CLAUDE_CANNED = re.compile(r"\[Request interrupted by user[^\]]*\]")

# Newer Claude Code stamps user records with an origin kind. Machine origins
# (background-task notifications, and "peer": another agent's message, e.g.
# a background agent handing its result back) are never typing; an origin kind
# that is neither human nor known-machine means the harness grew a new
# record source that must be classified deliberately. Records without an
# origin predate the field and are classified by content instead.
CLAUDE_HUMAN_ORIGINS = frozenset({"human"})
CLAUDE_MACHINE_ORIGINS = frozenset({"task-notification", "peer"})


def claude_origin_kind(record: Mapping[str, Any]) -> Any:
    """The origin kind a record states, as stated; None when it states none."""
    origin = record.get("origin")
    return origin.get("kind") if isinstance(origin, dict) else None


def claude_origin_is_machine(record: Mapping[str, Any], path: Path, line_number: int) -> bool:
    kind = claude_origin_kind(record)
    if kind is None or kind in CLAUDE_HUMAN_ORIGINS:
        return False
    if kind in CLAUDE_MACHINE_ORIGINS:
        return True
    # TODO: Says this record has an unrecognized origin kind and to add it
    # deliberately to CLAUDE_HUMAN_ORIGINS or CLAUDE_MACHINE_ORIGINS.
    raise UserError(
        f"Origo recordi ignota: {kind!r} in {path}:{line_number}\n"
        "Adde hanc originem consulto in CLAUDE_HUMAN_ORIGINS vel CLAUDE_MACHINE_ORIGINS."
    )

# Human typing that arrives wrapped in tool plumbing instead of as a chat
# message: the text typed into a tool/permission/plan denial, and the
# free-text ("Other") answers to AskUserQuestion. Recovery keys off the
# record's own toolUseResult — a tool output merely quoting these templates
# is not typing. The templates below are the complete denial family found in
# the Claude Code binary; a family member matching none of them means the
# format grew a new variant that must be classified deliberately.
CLAUDE_DENIAL_FAMILY = (
    "The tool use was rejected (eg. if it was a file edit, "
    "the new_string was NOT written to the file)."
)
# Typed-text markers, most specific first ("The user said:" is a suffix of
# another marker's neighborhood). Plan rejections ("No, keep planning" with
# feedback) arrive through these same templates via the ExitPlanMode denial.
CLAUDE_TYPED_MARKERS = (
    "The user provided the following reason for the rejection:",
    "To tell you how to proceed, the user said:",
    "The user said:",
)
# Denial tails where nothing was typed.
CLAUDE_UNTYPED_TAILS = (
    "STOP what you are doing and wait for the user to tell you how to proceed.",
    "Try a different approach or report the limitation to complete your task.",
)


def claude_denial_text(result: str, path: Path, line_number: int) -> str:
    for marker in CLAUDE_TYPED_MARKERS:
        _, found, typed = result.partition(marker)
        if found:
            typed = typed.strip()
            # The newer templates wrap the feedback in double quotes.
            unquoted = typed[1:-1] if len(typed) >= 2 and typed[0] == typed[-1] == '"' else typed
            return unquoted
    if any(tail in result for tail in CLAUDE_UNTYPED_TAILS):
        return ""
    # TODO: Says a tool denial is in an unrecognized form and to add it
    # deliberately to CLAUDE_TYPED_MARKERS or CLAUDE_UNTYPED_TAILS.
    raise UserError(
        f"Recusatio in forma ignota: {path}:{line_number}\n"
        "Adde hanc formam consulto in CLAUDE_TYPED_MARKERS vel CLAUDE_UNTYPED_TAILS."
    )


def claude_recovered(record: Mapping[str, Any], path: Path, line_number: int) -> tuple[str, tuple[Ballot, ...]]:
    """Return (typed, ballots): the human's typed words plus one Ballot per
    answered multiple-choice question. An answer matching one option label —
    or a comma-joined list of them (multi-select) — is a click; anything
    else is typing and the ballot's picked set stays empty."""
    result = record.get("toolUseResult")
    match result:
        case str() if CLAUDE_DENIAL_FAMILY in result:
            return claude_denial_text(result, path, line_number), ()
        case {"questions": list() as questions, "answers": dict() as answers}:
            ballots: list[Ballot] = []
            typed: list[str] = []
            for question in questions:
                if not isinstance(question, dict):
                    continue
                prompt_text = question.get("question")
                answer = answers.get(prompt_text)
                if not isinstance(prompt_text, str) or not isinstance(answer, str) or answer == "":
                    continue
                labels = tuple(
                    option["label"]
                    for option in question.get("options") or []
                    if isinstance(option, dict) and isinstance(option.get("label"), str)
                )
                parts = answer.split(", ")
                if answer in labels:
                    picked: tuple[str, ...] = (answer,)
                elif len(parts) > 1 and all(part in labels for part in parts):
                    picked = tuple(parts)
                else:
                    picked = ()
                    typed.append(answer)
                ballots.append(Ballot(question=prompt_text, options=labels, picked=picked))
            return "\n\n".join(typed), tuple(ballots)
        case {"questions": _}:
            # TODO: Says question answers are in an unrecognized form — the
            # answers structure seems to have changed and claude_recovered
            # needs updating.
            raise UserError(
                f"Responsa interrogationum in forma ignota: {path}:{line_number}\n"
                "Structura answers mutata videtur; claude_recovered renovandum est."
            )
        case _:
            return "", ()


def claude_denial_key(record: Mapping[str, Any], text: str) -> tuple[Any, str] | None:
    """Identify adjacent records produced by one denied parallel tool batch."""
    match record.get("toolUseResult"):
        case str() as result if CLAUDE_DENIAL_FAMILY in result:
            return (record.get("promptId"), text)
        case _:
            return None


def claude_delivery_key(attachment: Mapping[str, Any] | None, path: Path, line_number: int) -> str | int:
    """A record's "delivery key": a queued prompt's source_uuid,
    which every delivery of it shares, or else the record's own line number,
    which no other record shares (a user or assistant record, or a queued
    prompt that carries no source_uuid)."""
    match attachment:
        case {"source_uuid": str() as source_uuid} if source_uuid != "":
            return source_uuid
        case {"source_uuid": _}:
            # TODO: Says a queued prompt's source_uuid is in an unrecognized
            # form (not text, or empty) — the format seems to have changed
            # and claude_delivery_key needs updating.
            raise UserError(
                f"source_uuid in forma ignota: {path}:{line_number}\n"
                "Forma mutata videtur; claude_delivery_key renovandum est."
            )
        case _:
            return line_number


def claude_prompt(content: Any, path: Path, line_number: int) -> str:
    """Return the human-typed text of a user record, or "" if none survives."""
    match content:
        case str():
            if command := CLAUDE_COMMAND.fullmatch(content):
                return "/" + command.group(1)
            if command := CLAUDE_COMMAND_ARGS.fullmatch(content):
                name, args = command.groups()
                return f"/{name} {args}" if args else f"/{name}"
            if content.startswith(("<command-message>", "<command-name>")):
                # TODO: Says a slash-command wrapper is malformed and the
                # Claude transcript format should be inspected.
                raise UserError(
                    f"Involucrum imperii obliqui malformatum: {path}:{line_number}\n"
                    "Forma transcripti Claude inspicienda est."
                )
            return content
        case list():
            items = [item for item in content if isinstance(item, dict)]
            if any(item.get("type") == "tool_result" for item in items):
                return ""
            texts = [
                item["text"]
                for item in items
                if item.get("type") == "text" and isinstance(item.get("text"), str)
            ]
            return "".join(t for t in texts if not CLAUDE_WRAPPER.fullmatch(t))
        case _:
            return ""


def claude_reply_blocks(content: Any, path: Path, line_number: int) -> list[str]:
    """The words of an assistant record that make up its reply, in content
    order: its text blocks, and the message of each SendUserMessage tool
    call, Brief being that tool's legacy name (an agent in a Claude-desktop
    cloud workspace can speak to the human through that tool alone, writing
    no text block). Every other block stays out of the reply, even one the
    harness also shows the human, such as ExitPlanMode's plan or
    AskUserQuestion's questions. Empty strings are dropped."""
    if not isinstance(content, list):
        # TODO: Says an assistant record's content is in an unrecognized
        # form (not a list of blocks) — the format seems to have changed and
        # claude_reply_blocks needs updating.
        raise UserError(
            f"Contentum responsi in forma ignota: {path}:{line_number}\n"
            "Forma mutata videtur; claude_reply_blocks renovandum est."
        )
    words: list[str] = []
    for item in content:
        match item:
            case {"type": "text", "text": str() as text}:
                words.append(text)
            case {"type": "text"}:
                # TODO: Says a text block is in an unrecognized form (its
                # text is missing or not a string) — the format seems to have
                # changed and claude_reply_blocks needs updating.
                raise UserError(
                    f"Membrum text in forma ignota: {path}:{line_number}\n"
                    "Forma mutata videtur; claude_reply_blocks renovandum est."
                )
            case {
                "type": "tool_use",
                "name": "SendUserMessage" | "Brief",
                "input": {"message": str() as text, **rest},
            } if rest == {}:
                words.append(text)
            case {"type": "tool_use", "name": "SendUserMessage" | "Brief" as name}:
                # TODO: Says a SendUserMessage tool call (named as the record
                # names it: SendUserMessage, or its legacy name Brief) is in
                # an unrecognized form (its input is not exactly one message
                # text) — the format seems to have changed and
                # claude_reply_blocks needs updating.
                raise UserError(
                    f"Vocatio {name} in forma ignota: {path}:{line_number}\n"
                    "Forma mutata videtur; claude_reply_blocks renovandum est."
                )
            case _:
                pass  # every other block: no words for the human
    return [text for text in words if text != ""]


def claude_images(content: Any, path: Path, line_number: int) -> tuple[str, ...]:
    if not isinstance(content, list):
        return ()
    uris: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "image":
            source = item.get("source")
            source = source if isinstance(source, dict) else {}
            media = source.get("media_type")
            data = source.get("data")
            if source.get("type") != "base64" or not isinstance(media, str) or not isinstance(data, str):
                # TODO: Says a pasted image is stored in an unrecognized form.
                raise UserError(f"Imago in forma ignota: {path}:{line_number}")
            uris.append(f"data:{media};base64,{data}")
    return tuple(uris)


def claude_tally(record: Mapping[str, Any], repo: Path, path: Path, line_number: int) -> tuple[int, int]:
    """Tally the repo lines this record's tool result changed: edits and
    overwrites carry structuredPatch hunks, file creations carry the whole
    new content. Files outside the repo don't count, and neither do denied
    edits (their toolUseResult is the denial string, and nothing was
    written)."""
    result = record.get("toolUseResult")
    if not isinstance(result, dict) or not under_dir(result.get("filePath"), repo, path):
        return (0, 0)
    hunks = result.get("structuredPatch") or []
    lines = [line for hunk in hunks for line in hunk.get("lines") or []]
    if lines:
        return diff_tally(lines)
    if result.get("type") == "create":
        content = result.get("content")
        if not isinstance(content, str):
            # TODO: Says a file-creation result has no content text — the
            # format seems to have changed and claude_tally needs updating.
            raise UserError(
                f"Fructus creationis sine textu: {path}:{line_number}\n"
                "Forma mutata videtur; claude_tally renovandum est."
            )
        return (len(content.splitlines()), 0)
    return (0, 0)


def claude_tool_seconds(record: Mapping[str, Any]) -> float:
    result = record.get("toolUseResult")
    result = result if isinstance(result, dict) else {}
    # WebFetch records durationMs, WebSearch durationSeconds, Agent (subagent
    # runs) totalDurationMs; no other tool records any duration at all.
    for key, scale in (("durationMs", 1000.0), ("durationSeconds", 1.0), ("totalDurationMs", 1000.0)):
        value = result.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value / scale
    return 0.0


def claude_exchanges(
    path: Path,
    repo: Path,
    session_file: Callable[[str, Path, int, Any], None] = lambda session, path, line_number, timestamp: None,
) -> tuple[list[Exchange], set[Holding]]:
    threads: dict[str, list[Message]] = {}
    holdings: set[Holding] = set()
    # Agent working time, distinguished from waiting-for-human time by where
    # a timestamp gap ends: a gap ending at an assistant record is generation;
    # a gap ending at a tool_result hides both tool runtime and permission
    # waits, so only a duration the store itself recorded is credited there,
    # capped by the gap: a recorded run can overlap work the timeline already
    # counted (parallel tool calls, background subagents collected late).
    # `pending` accrues per session until the next emitted message.
    previous: dict[str, dt.datetime] = {}
    pending: dict[str, float] = {}
    # Repo lines added and deleted since the last emitted message, credited
    # to the next reply — or, when a new prompt arrives first, carried on
    # that prompt and credited to the exchange it closes.
    tallies: dict[str, tuple[int, int]] = {}
    last_denial: dict[str, tuple[Any, str] | None] = {}
    # A "redelivery" is a record that reaches the cwd check carrying a
    # session and delivery key (see claude_delivery_key) that an earlier
    # such record carried. Only a queued prompt's delivery key can repeat: a
    # Claude-desktop cloud workspace that restarted delivered its
    # already-delivered queued prompts again under the same source_uuids.
    # Every record that reaches the cwd check enters `first_delivery`, which
    # maps each (session, delivery key) to the line number of the first such
    # record to carry it, whether that record is attributed to this repo
    # (by cwd), and the origin kind it states (see claude_origin_kind). A
    # redelivery attributed to this repo is refused right after that check,
    # whatever its content, when its first delivery is not attributed to
    # this repo, whatever either one's origin, since which page it belongs
    # on is undecided; or when it states another origin kind than its first
    # delivery, since which origin to believe is undecided. `delivered` maps
    # each (session, delivery key) to the text and images of its first
    # delivery among the user-role records attributed to this repo that are
    # neither of machine origin nor local-command output, and only among
    # those records is a redelivery recognized: the human typed each prompt
    # once, so such a redelivery is no new prompt, and one carrying other
    # text or images than the first is refused. A redelivery that is not
    # refused states its first delivery's origin kind, so it is of machine
    # origin exactly when its first delivery is: held as machine text like
    # its first, or else judged against its first.
    delivered: dict[tuple[str, str | int], tuple[str, tuple[str, ...]]] = {}
    first_delivery: dict[tuple[str, str | int], tuple[int, bool, Any]] = {}
    # Sessions whose most recently emitted message is a recovered slash
    # command, mapped to that record's promptId, so a local-command-stdout
    # record can find and unsend its paired command.
    command_prompt: dict[str, Any] = {}
    for line_number, record in read_jsonl(path):
        role = record.get("type")
        # A message typed while the agent was mid-turn is recorded not as a
        # user record but as a queued_command attachment; other attachment
        # kinds (todo reminders, file-edit notices, ...) are machine chatter.
        attachment = record.get("attachment") if role == "attachment" else None
        if attachment is not None:
            attachment = attachment if isinstance(attachment, dict) else {}
            if attachment.get("type") != "queued_command":
                continue
            mode = attachment.get("commandMode")
            if mode == "task-notification":
                continue  # background-task wakeup queued as a command
            if mode != "prompt":
                # TODO: Says a queued command has an unrecognized mode and to
                # add it deliberately in claude_exchanges.
                raise UserError(
                    f"Modus queued_command ignotus: {mode!r} in {path}:{line_number}\n"
                    "Adde hunc modum consulto in claude_exchanges."
                )
            role = "user"
        elif role not in {"user", "assistant"}:
            continue
        if any(record.get(flag) for flag in CLAUDE_SKIP_FLAGS):
            continue
        message = record.get("message")
        if attachment is None and not isinstance(message, dict):
            # TODO: Says this record has no message object.
            raise UserError(f"Recordum message obiectum non habet: {path}:{line_number}")
        session = str(record.get("sessionId") or path.stem)
        # Every record read into a session registers its session's file,
        # whatever its cwd: everything below keeps a session's state within
        # one file, and collect() refuses a session registered from two
        # files. Read alone, a file's registrations go unheard.
        session_file(session, path, line_number, record.get("timestamp"))
        delivery = (session, claude_delivery_key(attachment, path, line_number))
        # The fields that say who sent the record: a queued prompt's
        # attachment, or the record itself.
        typed_source = attachment if attachment is not None else record
        kind = claude_origin_kind(typed_source)
        inside = under_dir(record.get("cwd"), repo, path)
        first_line, first_inside, first_kind = first_delivery.setdefault(delivery, (line_number, inside, kind))
        if not inside:
            continue
        if not first_inside:
            # TODO: Says a record was delivered again with its cwd inside
            # this project after its first delivery, in the same file and
            # session, had its cwd outside it, whatever the record's origin
            # (the human's, another agent's, a background task's, or none
            # stated) and whatever it carries (words, images, harness text,
            # or nothing), citing the first delivery and then this one, so
            # which page should show the record is undecided and
            # claude_exchanges needs updating.
            raise UserError(
                "Recordum primum extra inceptum, deinde intra traditum est, quacumque origine, quidquid fert:\n"
                f"  {path}:{first_line}\n  {path}:{line_number}\n"
                "Ad quam paginam recordum pertineat nondum decretum est; claude_exchanges renovandum est."
            )
        if kind != first_kind:
            # TODO: Says a record was delivered again inside this project,
            # in the same file and session as its first delivery, stating
            # another origin kind than its first delivery did (the human's,
            # another agent's, a background task's, or none), naming both
            # kinds and citing the first delivery and then this one;
            # sourcery does not decide which origin to believe, so the case
            # is to be examined and claude_exchanges updated.
            raise UserError(
                f"Recordum iterum traditum aliam originem fert quam primum: {first_kind!r}, deinde {kind!r}:\n"
                f"  {path}:{first_line}\n  {path}:{line_number}\n"
                "Utri originae credendum sit sourcery non decernit: casus inspiciendus, "
                "claude_exchanges renovandum est."
            )
        if "timestamp" not in record:
            # TODO: Says this record has no timestamp.
            raise UserError(f"Tempus deest in recordo: {path}:{line_number}")
        timestamp = parse_time(record["timestamp"])
        prior = previous.get(session, timestamp)
        gap = (timestamp - prior).total_seconds()
        previous[session] = max(prior, timestamp)
        if role == "assistant":
            model = message.get("model") if isinstance(message.get("model"), str) else ""
            if model == "<synthetic>":
                continue  # harness notice (auth/API errors), not model output
            last_denial.pop(session, None)
            pending[session] = pending.get(session, 0.0) + max(gap, 0.0)
        else:
            pending[session] = pending.get(session, 0.0) + min(
                max(gap, 0.0), claude_tool_seconds(record)
            )
            added, deleted = tallies.get(session, (0, 0))
            grew = claude_tally(record, repo, path, line_number)
            tallies[session] = (added + grew[0], deleted + grew[1])
        is_command = False
        if role == "user":
            content = attachment.get("prompt") if attachment is not None else message.get("content")
            if attachment is None and isinstance(content, str) and content.startswith(
                "<local-command-stdout>"
            ):
                if not CLAUDE_LOCAL_STDOUT.fullmatch(content):
                    # TODO: Says a local-command-stdout wrapper is malformed
                    # and the Claude transcript format should be inspected.
                    raise UserError(
                        f"Involucrum local-command-stdout malformatum: {path}:{line_number}\n"
                        "Forma transcripti Claude inspicienda est."
                    )
                if session not in command_prompt or command_prompt[session] != record.get(
                    "promptId"
                ):
                    # TODO: Says captured local-command output appeared without
                    # the slash command it answers and the Claude transcript
                    # format should be inspected.
                    raise UserError(
                        f"Effluxus imperii localis sine imperio compari: {path}:{line_number}\n"
                        "Forma transcripti Claude inspicienda est."
                    )
                # Unsend the command: give back the work time and edits it
                # absorbed so they ride the next real prompt instead.
                unsent = threads[session].pop()
                pending[session] = pending.get(session, 0.0) + unsent.active
                added, deleted = tallies.get(session, (0, 0))
                tallies[session] = (added + unsent.added, deleted + unsent.deleted)
                command_prompt.pop(session, None)
                holdings.add(("Claude Code", timestamp))
                continue
            if claude_origin_is_machine(typed_source, path, line_number):
                holdings.add(("Claude Code", timestamp))
                continue
            text = claude_prompt(content, path, line_number)
            # These two prefixes are harness-serialized or claude_prompt has
            # already crashed, so typed text can never be marked unsendable.
            is_command = isinstance(content, str) and content.startswith(
                ("<command-message>", "<command-name>")
            )
            ballots: tuple[Ballot, ...] = ()
            denial_key = None
            if text == "" and attachment is None:
                text, ballots = claude_recovered(record, path, line_number)
                denial_key = claude_denial_key(record, text)
            images = claude_images(content, path, line_number)
            # Compare and record every delivery before the checks below
            # decide whether it is a prompt, so a redelivery is refused or
            # recognized whatever either delivery carries. The get() yields
            # the text and images `delivered` holds for this key, or this
            # one's own if it holds none.
            if delivered.get(delivery, (text, images)) != (text, images):
                # TODO: Says a queued prompt delivered again carries other
                # words or images than its first delivery under the same
                # source_uuid — the format seems to have changed and
                # claude_exchanges needs updating.
                raise UserError(
                    f"Rogatio iterum tradita aliud fert quam prima: {path}:{line_number}\n"
                    "Forma mutata videtur; claude_exchanges renovandum est."
                )
            redelivery = delivery in delivered
            delivered[delivery] = (text, images)
            if CLAUDE_CANNED.fullmatch(text):
                holdings.add(("Claude Code", timestamp))
                continue
            if text == "" and images == () and not any(b.picked for b in ballots):
                continue  # tool plumbing with no typing: never rendered, so no holding
            holdings.add(("Claude Code", timestamp))
            # A redelivery stays held, so a ledger's stale copy of it gets
            # purged, but is no new prompt: the work pending around it rides
            # the exchange in flight.
            if redelivery:
                continue
            # One typed act can fan out across several records (a denial
            # reason stamped onto each rejected parallel tool call), so a
            # repeat of the pending unanswered prompt is not a new prompt.
            if denial_key is not None and denial_key == last_denial.get(session):
                continue
            last_denial[session] = denial_key
            added, deleted = tallies.pop(session, (0, 0))
            item = Message(role, timestamp, text, images=images, ballots=ballots,
                           active=pending.pop(session, 0.0), added=added, deleted=deleted)
        else:
            text = "\n\n".join(claude_reply_blocks(message.get("content"), path, line_number))
            if text == "":
                continue
            effort = record.get("effort") if isinstance(record.get("effort"), str) else ""
            added, deleted = tallies.pop(session, (0, 0))
            item = Message(
                role, timestamp, text, model, effort,
                active=pending.pop(session, 0.0), added=added, deleted=deleted,
            )
        threads.setdefault(session, []).append(item)
        if is_command:
            command_prompt[session] = record.get("promptId")
        else:
            command_prompt.pop(session, None)
    return [
        exchange
        for session, messages in threads.items()
        for exchange in paired(
            "Claude Code",
            session,
            messages,
            path,
            (pending.get(session, 0.0), *tallies.get(session, (0, 0))),
        )
    ], holdings


# ---------------------------------------------------------------------- Codex

# The Codex VS Code extension wraps the human's request in IDE context; the
# human's words are everything after the request heading.
CODEX_WRAP_PREFIX = "# Context from my IDE setup:"
CODEX_REQUEST_HEADING = "\n## My request for Codex:\n"

# Canned handoff message Codex fabricates in the user role when syncing agent
# history between surfaces.
CODEX_CANNED_PREFIX = "The following is the Codex agent history"

# Terminal quick-fix templates VS Code inserts into whichever chat panel has
# focus, so they appear in Copilot and Codex prompts alike — sometimes below
# hand-typed text. Everything from the template onward (the template sentence
# plus the appended terminal output) is machine text; boundary newlines are
# scaffold, not typing.
TERMINAL_FIX_TEMPLATE = re.compile(
    r"(?:^|\n)(?:Can you fix this error\?\n|I get the following error\. Please fix the error\.)"
)


def cut_canned_tail(prompt: str) -> str:
    match = TERMINAL_FIX_TEMPLATE.search(prompt)
    return prompt[: match.start()].rstrip("\n") if match else prompt


def codex_prompt(message: str, path: Path) -> str:
    if not message.startswith(CODEX_WRAP_PREFIX):
        return message
    _, found, request = message.partition(CODEX_REQUEST_HEADING)
    if not found:
        # TODO: Says an IDE context wrapper was found without its request
        # heading — the Codex transcript format seems to have changed, so
        # inspect the file.
        raise UserError(
            f"Involucrum IDE sine rogatione inventum est: {path}\n"
            "Forma transcripti Codex mutata videtur; fasciculum inspice."
        )
    return request.removesuffix("\n")


def codex_include_session(payload: Mapping[str, Any], path: Path) -> bool:
    source = payload.get("source")
    originator = payload.get("originator")
    match source:
        case dict() if "subagent" in source:
            return False  # machine-spawned subagent thread
        case "exec" if originator == "codex_vscode":
            return False  # machine side thread (UI titling and similar)
        case "vscode" | "cli" | "exec" | None:
            return True
        case _:
            # TODO: Says this Codex session has an unrecognized source marker
            # and to add it deliberately in codex_include_session.
            raise UserError(
                f"Fons sessionis Codex ignotus: {source!r} in {path}\n"
                "Adde hunc fontem consulto in codex_include_session."
            )


# Codex stores pasted images as ready-made data: URIs.
def codex_images(payload: Mapping[str, Any], path: Path, line_number: int) -> tuple[str, ...]:
    images = payload.get("images") or []
    for image in images:
        if not (isinstance(image, str) and image.startswith("data:")):
            # TODO: Says a pasted image is stored in an unrecognized form.
            raise UserError(f"Imago in forma ignota: {path}:{line_number}")
    return tuple(images)


def codex_tally(change: Any, path: Path, line_number: int) -> tuple[int, int]:
    """Tally one file's change in an applied patch: updates carry a unified
    diff starting at its first hunk (a diff bearing +++/--- file headers
    would miscount them as changes, so it falls through and fails loudly)
    or an empty diff when the patch left the file unchanged, additions and
    deletions the whole content."""
    match change:
        case {"type": "update", "unified_diff": str() as diff} if diff.startswith("@@"):
            return diff_tally(diff.splitlines())
        case {"type": "update", "unified_diff": ""}:
            # A successfully applied patch can leave a file textually
            # unchanged; nothing to count.
            return (0, 0)
        case {"type": "add", "content": str() as content}:
            return (len(content.splitlines()), 0)
        case {"type": "delete", "content": str() as content}:
            return (0, len(content.splitlines()))
        case _:
            # TODO: Says a patch change is in an unrecognized form and to add
            # it deliberately in codex_tally.
            raise UserError(
                f"Mutatio fasciculi in forma ignota: {path}:{line_number}\n"
                "Adde hanc formam consulto in codex_tally."
            )


# Payload kinds that mark the model actively producing output. A timestamp
# gap counts as working time only when it ends at one of these; gaps ending
# anywhere else hide tool runtime, system sleep, approval waits, or idle time
# between turns (tool outputs, task_started, turn boundaries, user messages).
CODEX_WORKING = frozenset({
    "message", "reasoning", "function_call", "custom_tool_call", "web_search_call",
    "agent_message", "agent_reasoning", "token_count", "task_complete",
})


def codex_exchanges(path: Path, repo: Path) -> tuple[list[Exchange], set[Holding]]:
    session = path.stem
    holdings: set[Holding] = set()
    cwd: Any = ""
    model = ""
    effort = ""
    messages: list[Message] = []
    previous: dt.datetime | None = None
    pending = 0.0
    tally = (0, 0)  # repo lines (added, deleted) since the last message
    for line_number, record in read_jsonl(path):
        payload = record.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        record_type = record.get("type")
        working = (
            record_type in {"event_msg", "response_item"}
            and payload.get("type") in CODEX_WORKING
            and payload.get("role") in (None, "assistant")
        )
        if "timestamp" in record:
            stamp = parse_time(record["timestamp"])
            if previous is not None and working:
                pending += max((stamp - previous).total_seconds(), 0.0)
            previous = stamp
        match record_type:
            case "session_meta":
                if not codex_include_session(payload, path):
                    return [], set()  # nothing in an excluded thread is held for the repo
                session = str(payload.get("id") or session)
                cwd = payload.get("cwd") or cwd
            case "turn_context":
                cwd = payload.get("cwd") or cwd
                model = payload.get("model") if isinstance(payload.get("model"), str) else model
                effort = payload.get("effort") if isinstance(payload.get("effort"), str) else effort
            case "event_msg":
                kind = payload.get("type")
                # A patch that failed to apply changed nothing, so only a
                # successful application is tallied.
                if kind == "patch_apply_end" and payload.get("success"):
                    for changed, change in (payload.get("changes") or {}).items():
                        if under_dir(changed, repo, path):
                            grew = codex_tally(change, path, line_number)
                            tally = (tally[0] + grew[0], tally[1] + grew[1])
                if kind not in {"user_message", "agent_message"}:
                    continue
                if not under_dir(cwd, repo, path):
                    continue
                text = payload.get("message")
                if not isinstance(text, str):
                    # TODO: Says this message record has no text.
                    raise UserError(f"Nuntius sine textu: {path}:{line_number}")
                if "timestamp" not in record:
                    # TODO: Says this record has no timestamp.
                    raise UserError(f"Tempus deest in recordo: {path}:{line_number}")
                timestamp = parse_time(record["timestamp"])
                if kind == "user_message":
                    holdings.add(("Codex", timestamp))
                    prompt = cut_canned_tail(codex_prompt(text, path))
                    images = codex_images(payload, path, line_number)
                    if (prompt == "" and images == ()) or prompt.startswith(CODEX_CANNED_PREFIX):
                        continue  # not an exchange boundary: the tally rides on
                    messages.append(Message("user", timestamp, prompt, images=images,
                                            active=pending, added=tally[0], deleted=tally[1]))
                    pending = 0.0
                    tally = (0, 0)
                else:
                    messages.append(
                        Message("assistant", timestamp, text, model, effort,
                                active=pending, added=tally[0], deleted=tally[1])
                    )
                    pending = 0.0
                    tally = (0, 0)
            case _:
                continue
    return paired("Codex", session, messages, path, (pending, *tally)), holdings


# ------------------------------------------------------- VS Code Copilot Chat

# Response-item kinds that carry no dialogue prose: tool plumbing, edit
# bookkeeping, progress chrome, and hidden reasoning. Visible prose arrives
# as kindless markdown chunks bearing a string "value".
VSCODE_MUTE_KINDS = frozenset({
    "thinking", "toolInvocationSerialized", "toolInvocation", "prepareToolInvocation",
    "undoStop", "textEditGroup", "workspaceEdit", "codeblockUri", "mcpServersStarting",
    "progressTaskSerialized", "progressTask", "progressMessage", "confirmation",
    "elicitationSerialized", "elicitation", "warning", "markdownVuln", "command",
    "treeData", "extensions", "hook", "systemNotification",
})


# Canned prompts VS Code chat controls fabricate in the user role, beyond
# those already marked by a "confirmation" field on the request (the
# Continue / Pause / Enable / Try Again buttons) and the terminal quick-fix
# templates handled by cut_canned_tail: the model-enable mention from older
# versions and the editor's explain-diagnostic action.
COPILOT_CANNED = re.compile(r'@\w+ Enable: "|@workspace /explain ')


# Copilot serializes pasted-image bytes as an object with numeric-string
# keys plus a mimeType. Newer sessions instead carry the whole payload
# already encoded in a "$base64" field.
def copilot_image_uri(variable: Mapping[str, Any], path: Path) -> str:
    value = variable.get("value")
    mime = variable.get("mimeType")
    if not (isinstance(value, dict) and isinstance(mime, str)):
        # TODO: Says a pasted image is stored in an unrecognized form.
        raise UserError(f"Imago in forma ignota: {path}")
    try:
        match value:
            case {"$base64": encoded}:
                data = base64.b64decode(encoded, validate=True)
            case _:
                data = bytes(value[str(i)] for i in range(len(value)))
    except (KeyError, TypeError, ValueError) as exc:
        # TODO: Says a pasted image is stored in an unrecognized form.
        raise UserError(f"Imago in forma ignota: {path}\n{exc}") from exc
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def vscode_reference_name(item: Mapping[str, Any]) -> str:
    reference = item.get("inlineReference")
    reference = reference if isinstance(reference, dict) else {}
    name = reference.get("name")
    if isinstance(name, str):
        return name
    uri = reference.get("uri")
    fs_path = uri.get("fsPath") if isinstance(uri, dict) else None
    return Path(fs_path).name if isinstance(fs_path, str) else ""


def vscode_reply(response: Any, path: Path) -> tuple[str, str]:
    parts: list[str] = []
    resolutions: list[Any] = []
    for item in response if isinstance(response, list) else []:
        if not isinstance(item, dict):
            # TODO: Says a response item is not a JSON object.
            raise UserError(f"Membrum responsi obiectum non est: {path}")
        kind = item.get("kind")
        if kind is None:
            value = item.get("value")
            if not isinstance(value, str):
                # TODO: Says a response item has no text.
                raise UserError(f"Membrum responsi sine textu: {path}")
            parts.append(value)
        elif kind == "inlineReference":
            parts.append(vscode_reference_name(item))
        elif kind == "autoModeResolution":
            resolutions.append(item.get("resolvedModel"))
        elif kind not in VSCODE_MUTE_KINDS:
            # TODO: Says a response item has an unrecognized kind and to add
            # it deliberately in VSCODE_MUTE_KINDS or vscode_reply.
            raise UserError(
                f"Genus membri responsi ignotum: {kind!r} in {path}\n"
                "Adde hoc genus consulto in VSCODE_MUTE_KINDS vel vscode_reply."
            )
    match resolutions:
        case []:
            resolved_model = ""
        case [str() as resolved_model] if resolved_model != "":
            pass
        case _:
            # TODO: Says auto-mode model resolution is malformed.
            raise UserError(f"Resolutio exemplaris auto-mode malformata: {path}")
    return "".join(parts), resolved_model


# Sessions are stored as a kind-0 snapshot followed by set (1),
# list-splice (2), and delete (3) mutations.
def replay_mutations(path: Path) -> Any:
    state: Any = None
    for line_number, entry in read_jsonl(path):
        kind = entry.get("kind")
        if kind == 0:
            state = entry.get("v")
            continue
        keys = entry.get("k")
        if state is None or not isinstance(keys, list) or keys == []:
            # TODO: Says a VS Code session mutation is malformed.
            raise UserError(f"Mutatio VS Code malformata: {path}:{line_number}")
        try:
            target = state
            for key in keys[:-1]:
                target = target[key]
            last = keys[-1]
            match kind:
                case 1:
                    if isinstance(target, list) and last == len(target):
                        target.append(entry.get("v"))
                    else:
                        target[last] = entry.get("v")
                case 2:
                    target = target[last]
                    index = entry.get("i")
                    if isinstance(index, int):
                        del target[index:]
                    target.extend(entry.get("v") or [])
                case 3:
                    del target[last]
                case _:
                    # TODO: Says a mutation has an unrecognized kind.
                    raise UserError(f"Genus mutationis ignotum: {kind!r} in {path}:{line_number}")
        except (KeyError, IndexError, TypeError) as exc:
            # TODO: Says a VS Code session mutation cannot be applied.
            raise UserError(f"Mutatio VS Code non applicari potest: {path}:{line_number}\n{exc}") from exc
    return state


def vscode_state(path: Path) -> dict[str, Any]:
    state = read_json(path) if path.suffix == ".json" else replay_mutations(path)
    if not isinstance(state, dict):
        # TODO: Says this session doesn't reconstruct to a final object.
        raise UserError(f"Sessio VS Code obiectum finale non habet: {path}")
    return state


def vscode_exchanges(
    path: Path, workspace: tuple[Path, ...], repo: Path
) -> tuple[list[Exchange], set[Holding]]:
    inside = tuple(root for root in workspace if under_dir(str(root), repo, path))
    if inside == ():
        return [], set()
    if len(inside) != len(workspace):
        # TODO: Says a multi-root workspace straddles the project boundary,
        # and to open the project as a plain folder or export this dialog
        # by hand.
        raise UserError(
            f"Workspace multiplex intra et extra inceptum est: {path}\n"
            "Aperi inceptum ut folder simplex, vel exporta hunc dialogum manu."
        )
    state = vscode_state(path)
    if state.get("version") != 3:
        # TODO: Says this session has an unrecognized schema version — the
        # storage format seems to have changed and the script needs updating.
        raise UserError(
            f"Versio sessionis VS Code ignota: {state.get('version')!r} in {path}\n"
            "Forma repositi mutata videtur; scriptum renovandum est."
        )
    requests = state.get("requests")
    if not isinstance(requests, list):
        # TODO: Says this session has no requests list.
        raise UserError(f"Sessio VS Code indicem requests non habet: {path}")
    session = str(state.get("sessionId") or path.stem)
    exchanges: list[Exchange] = []
    holdings: set[Holding] = set()
    for request in requests:
        if not isinstance(request, dict):
            # TODO: Says a requests entry is not a JSON object.
            raise UserError(f"Elementum requests obiectum non est: {path}")
        message = request.get("message")
        prompt = message.get("text") if isinstance(message, dict) else None
        if not isinstance(prompt, str):
            # TODO: Says a request has no text in message.text.
            raise UserError(f"Rogatio VS Code textum in message.text non habet: {path}")
        if "timestamp" not in request:
            # TODO: Says a request has no timestamp.
            raise UserError(f"Tempus deest in rogatione: {path}")
        timestamp = parse_time(request["timestamp"])
        holdings.add(("Copilot Chat", timestamp))
        # VS Code starts a request by itself when a terminal command the
        # agent launched exits, handing the model its completion notice plus
        # the captured output in the user role. Nobody typed it. Such a
        # request is marked isSystemInitiated (a systemInitiatedLabel and
        # terminalExecutionId ride alongside), and the whole exchange goes,
        # like the button clicks marked by "confirmation".
        match request.get("isSystemInitiated"):
            case True:
                continue
            case None | False:
                pass
            case marker:
                # TODO: Says the system-initiated marker on a request holds
                # an unrecognized value — the storage format seems to have
                # changed and the script needs updating.
                raise UserError(
                    f"Signum isSystemInitiated ignotum: {marker!r} in {path}\n"
                    "Forma repositi mutata videtur; scriptum renovandum est."
                )
        prompt = cut_canned_tail(prompt)
        if prompt == "" or request.get("confirmation") is not None or COPILOT_CANNED.match(prompt):
            continue
        model = request.get("modelId")
        result = request.get("result")
        timings = result.get("timings") if isinstance(result, dict) else None
        total_elapsed = timings.get("totalElapsed") if isinstance(timings, dict) else None
        # A confirmation or elicitation pauses the turn for human input, so
        # totalElapsed would include waiting time; show no duration instead
        # of a wrong one.
        response = request.get("response")
        reply, resolved_model = vscode_reply(response, path)
        paused = any(
            isinstance(item, dict)
            and item.get("kind") in {"confirmation", "elicitation", "elicitationSerialized"}
            for item in (response if isinstance(response, list) else [])
        )
        variable_data = request.get("variableData")
        variables = variable_data.get("variables") if isinstance(variable_data, dict) else []
        variables = variables if isinstance(variables, list) else []
        exchanges.append(
            Exchange(
                timestamp=timestamp,
                provider="Copilot Chat",
                model=resolved_model or (model if isinstance(model, str) else ""),
                session=session,
                prompt=prompt,
                reply=reply,
                source=path,
                images=tuple(
                    copilot_image_uri(variable, path)
                    for variable in variables
                    if isinstance(variable, dict) and variable.get("kind") == "image"
                ),
                elapsed=(
                    total_elapsed / 1000.0
                    if isinstance(total_elapsed, (int, float)) and not paused
                    else 0.0
                ),
            )
        )
    return exchanges, holdings


def strip_jsonc(text: str) -> str:
    """Blank out // and /* */ comments, then trailing commas, preserving strings."""
    out: list[str] = []
    i = 0
    mode = "plain"
    while i < len(text):
        char = text[i]
        pair = text[i : i + 2]
        match mode:
            case "plain" if char == '"':
                mode = "string"
                out.append(char)
                i += 1
            case "plain" if pair == "//":
                mode = "line"
                out.append("  ")
                i += 2
            case "plain" if pair == "/*":
                mode = "block"
                out.append("  ")
                i += 2
            case "string" if char == "\\" and i + 1 < len(text):
                out.append(text[i : i + 2])
                i += 2
            case "string":
                mode = "plain" if char == '"' else mode
                out.append(char)
                i += 1
            case "line":
                mode = "plain" if char == "\n" else mode
                out.append(char if char == "\n" else " ")
                i += 1
            case "block" if pair == "*/":
                mode = "plain"
                out.append("  ")
                i += 2
            case "block":
                out.append(char if char == "\n" else " ")
                i += 1
            case _:
                out.append(char)
                i += 1
    return strip_trailing_commas("".join(out))


def strip_trailing_commas(text: str) -> str:
    """Drop commas followed only by whitespace and a closing bracket — never
    touching commas inside string literals."""
    out: list[str] = []
    in_string = False
    i = 0
    while i < len(text):
        char = text[i]
        if in_string:
            out.append(char)
            if char == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 2
                continue
            in_string = char != '"'
            i += 1
        elif char == '"':
            in_string = True
            out.append(char)
            i += 1
        elif char == ",":
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j] in "]}":
                i += 1  # trailing comma: drop it, keep the whitespace
            else:
                out.append(char)
                i += 1
        else:
            out.append(char)
            i += 1
    return "".join(out)


def read_workspace_file(path: Path) -> tuple[Path, ...]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        # TODO: Says this .code-workspace file cannot be read.
        raise UserError(f"Fasciculus workspace legi non potest: {path}\n{exc}") from exc
    try:
        data = json.loads(strip_jsonc(raw))
    except json.JSONDecodeError as exc:
        # TODO: Says this .code-workspace file is invalid.
        raise UserError(f"Fasciculus workspace invalidus est: {path}\n{exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("folders"), list):
        # TODO: Says this .code-workspace file has no folders list.
        raise UserError(f"Fasciculus workspace indicem folders non habet: {path}")
    roots: list[Path] = []
    for entry in data["folders"]:
        if not isinstance(entry, dict):
            # TODO: Says a folders entry is not a JSON object.
            raise UserError(f"Elementum folders obiectum non est: {path}")
        raw_path = entry.get("path")
        if not isinstance(raw_path, str):
            uri = entry.get("uri")
            raw_path = decode_file_uri(uri) if isinstance(uri, str) else None
        if raw_path is None:
            continue  # non-file root (remote workspace): not attributable here
        candidate = Path(raw_path).expanduser()
        roots.append(candidate.resolve() if candidate.is_absolute() else (path.parent / candidate).resolve())
    return tuple(roots)


def workspace_roots(storage: Path) -> tuple[Path, ...]:
    metadata = storage / "workspace.json"
    if not metadata.exists():
        return ()
    data = read_json(metadata)
    if not isinstance(data, dict):
        # TODO: Says workspace.json is not a JSON object.
        raise UserError(f"workspace.json obiectum non est: {metadata}")
    folder = data.get("folder")
    if isinstance(folder, str):
        decoded = decode_file_uri(folder)
        return (Path(decoded).expanduser().resolve(),) if decoded is not None else ()
    pointer = data.get("workspace") or data.get("configuration")
    if isinstance(pointer, str):
        decoded = decode_file_uri(pointer)
        return read_workspace_file(Path(decoded).expanduser().resolve()) if decoded is not None else ()
    # TODO: Says workspace.json has neither a folder nor a workspace entry.
    raise UserError(f"workspace.json nec folder nec workspace habet: {metadata}")


# ------------------------------------------------------- pairing and weaving


def paired(
    provider: str,
    session: str,
    messages: Iterable[Message],
    source: Path,
    tail: tuple[float, int, int] = (0.0, 0, 0),
) -> list[Exchange]:
    """Fold a role-ordered message stream into prompt-plus-reply exchanges."""
    exchanges: list[Exchange] = []
    current: Exchange | None = None
    reply_parts: list[str] = []
    model = ""
    effort = ""

    active = 0.0
    added = 0
    deleted = 0
    ended: dt.datetime | None = None  # arrival of the exchange's last reply message

    def flush() -> None:
        if current is not None:
            exchanges.append(
                dataclasses.replace(
                    current,
                    reply="\n\n".join(reply_parts),
                    model=model,
                    effort=effort,
                    elapsed=active,
                    # No reply means the wall span is unknown, the same
                    # 0-sentinel elapsed uses.
                    wall=(ended - current.timestamp).total_seconds() if ended else 0.0,
                    added=added,
                    deleted=deleted,
                )
            )

    for message in messages:
        match message.role:
            case "user":
                # Lines the prompt orphaned mid-turn belong to the exchange
                # it closes, whose reply never arrived to claim them.
                active += message.active
                added += message.added
                deleted += message.deleted
                flush()
                current = Exchange(
                    timestamp=message.timestamp,
                    provider=provider,
                    model="",
                    session=session,
                    prompt=message.text,
                    reply="",
                    source=source,
                    images=message.images,
                    ballots=message.ballots,
                )
                reply_parts = []
                model = ""
                effort = ""
                active = 0.0
                added = 0
                deleted = 0
                ended = None
            case "assistant":
                if current is None:
                    continue  # reply to a dropped machine prompt; nothing to attach to
                reply_parts.append(message.text)
                model = message.model or model
                effort = message.effort or effort
                ended = message.timestamp
                active += message.active
                added += message.added
                deleted += message.deleted
            case _:
                raise AssertionError(message.role)
    active += tail[0]
    added += tail[1]
    deleted += tail[2]
    flush()
    return exchanges


# ---------------------------------------------------------------- Antigravity

# Antigravity (Google's agentic IDE) keeps each conversation as one protobuf
# message sealed with AES-256-GCM — twelve bytes of nonce, the ciphertext,
# sixteen bytes of tag — under a key the application ships verbatim inside
# its own language-server binary, the same for every install. It hides
# nothing from the person whose machine it is. The key is not written here:
# each run reads it out of the installed binary, recognizing it by the
# digest below, so this file never states it.
ANTIGRAVITY_KEY_DIGEST = "2516d3eccb18071460c54f73addcc3893c8d49a44c56b49b506b37a03c1e2d9f"
ANTIGRAVITY_BIN = Path("/Applications/Antigravity.app/Contents/Resources/app/extensions/antigravity/bin")
# Model enums the store stamps on each step, paired with the ids the
# application's own configuration blocks give them. An enum missing here
# is shown on the page as an unknown model with its number, never hidden.
ANTIGRAVITY_MODELS = {1007: "gemini-3-pro-low", 1008: "gemini-3-pro-high", 1018: "gemini-3-flash"}
# Step types: the human's message (payload field 19), the planner's response
# (20: visible text, thinking, tool call), the agent's notify-user message
# (94), and machine plumbing read only for its timestamps — code actions,
# file views, directory listings, grep and file-name searches, terminal
# runs, task artifacts, progress summaries, browser subtasks, ephemeral and
# history injections, image generation, and a few small bookkeeping kinds.
# An unlisted type is a format change.
ANTIGRAVITY_USER, ANTIGRAVITY_PLANNER, ANTIGRAVITY_NOTIFY = 14, 15, 82
ANTIGRAVITY_MACHINE = frozenset({4, 5, 7, 8, 9, 17, 21, 23, 25, 81, 83, 85, 90, 91, 98})
# A step's status: 3 complete; 5 emptied of content by the application while
# compacting a long conversation — its words are gone; 6 cancelled and 7
# failed, both seen only on the agent's own steps, whose words and spans
# still count. Any other status is a shape this tool has not seen.
ANTIGRAVITY_EMPTIED = 5
ANTIGRAVITY_STATUSES = frozenset({3, ANTIGRAVITY_EMPTIED, 6, 7})
# The fields each payload may carry; anything else is a shape this tool has
# not seen (an attachment, say) and must not pass quietly.
ANTIGRAVITY_USER_FIELDS = frozenset({2, 3, 4, 6, 7, 8, 12, 13})
ANTIGRAVITY_CLICK_FIELDS = frozenset({1, 5, 7})
ANTIGRAVITY_PLANNER_FIELDS = frozenset({1, 3, 4, 6, 7, 8, 11, 12})
ANTIGRAVITY_NOTIFY_FIELDS = frozenset({1, 2, 3, 4, 5, 7, 8})


def wire(data: bytes) -> list[tuple[int, int | bytes]]:
    """Protobuf wire fields as (number, value): varints as ints, everything
    else as bytes. Malformed encoding fails loudly."""
    fields: list[tuple[int, int | bytes]] = []
    at = 0

    def varint() -> int:
        nonlocal at
        result = shift = 0
        while True:
            if at >= len(data):
                raise ValueError("truncated varint")
            byte = data[at]
            at += 1
            result |= (byte & 0x7F) << shift
            shift += 7
            if byte < 0x80:
                return result

    while at < len(data):
        tag = varint()
        number, kind = tag >> 3, tag & 7
        if kind == 0:
            fields.append((number, varint()))
            continue
        length = {1: 8, 5: 4}.get(kind)
        if length is None:
            if kind != 2:
                raise ValueError(f"wire type {kind}")
            length = varint()
        if at + length > len(data):
            raise ValueError("truncated field")
        fields.append((number, data[at:at + length]))
        at += length
    return fields


def pb_get(fields: list[tuple[int, int | bytes]], number: int) -> int | bytes | None:
    """The one value of a field, None when absent; a repeated field would be
    a shape this tool has not seen."""
    values = [value for n, value in fields if n == number]
    if len(values) > 1:
        raise ValueError(f"field {number} repeated")
    return values[0] if values else None


def pb_text(fields: list[tuple[int, int | bytes]], number: int) -> str | None:
    value = pb_get(fields, number)
    if value is None:
        return None
    if not isinstance(value, bytes):
        raise ValueError(f"field {number} is not text")
    return value.decode("utf-8")


def pb_message(fields: list[tuple[int, int | bytes]], number: int) -> list[tuple[int, int | bytes]]:
    value = pb_get(fields, number)
    if value is None:
        return []
    if not isinstance(value, bytes):
        raise ValueError(f"field {number} is not a message")
    return wire(value)


@functools.cache
def antigravity_key() -> bytes:
    """The application's own store key, read out of its installed binary:
    the one 32-letter window whose digest matches."""
    for binary in sorted(ANTIGRAVITY_BIN.glob("language_server_*")):
        data = binary.read_bytes()
        for run in re.finditer(rb"[A-Za-z]{32,}", data):
            letters = run.group()
            for at in range(len(letters) - 31):
                if hashlib.sha256(letters[at:at + 32]).hexdigest() == ANTIGRAVITY_KEY_DIGEST:
                    return letters[at:at + 32]
    # TODO: Says Antigravity transcripts exist but the application that can
    # open them is not installed here (or this version of it no longer holds
    # the key this tool recognizes), naming where it looked; nothing is
    # exported until they can be read.
    raise UserError(
        f"Transcripta Antigravity adsunt, sed applicatio quae ea aperit hic non invenitur "
        f"(aut clavem notam non iam fert): {ANTIGRAVITY_BIN}\nNihil exportatur dum legi possint."
    )


def antigravity_open(path: Path, key: bytes) -> bytes:
    """The conversation's plaintext, through the system cipher."""
    blob = path.read_bytes()
    try:
        cipher = ctypes.CDLL(ctypes.util.find_library("System")).CCCryptorGCMOneshotDecrypt
    except (OSError, AttributeError, TypeError) as exc:
        # TODO: Says an Antigravity transcript exists but this platform lacks
        # the system cipher (CommonCrypto, macOS) that opens it, naming the
        # file; nothing is exported until it can be read.
        raise UserError(
            f"Transcriptum Antigravity adest, sed huic systemati cifra deest "
            f"(CommonCrypto in macOS): {path}\nNihil exportatur dum legi possit."
        ) from exc
    cipher.restype = ctypes.c_int32
    cipher.argtypes = [
        ctypes.c_uint32, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t,
        ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p,
        ctypes.c_char_p, ctypes.c_size_t,
    ]
    nonce, body, tag = blob[:12], blob[12:-16], blob[-16:]
    plain = ctypes.create_string_buffer(len(body))
    status = cipher(0, key, len(key), nonce, len(nonce), None, 0, body, len(body), plain, tag, len(tag))
    if status != 0:
        # TODO: Says an Antigravity transcript would not open (with the
        # cipher's status code): the file is damaged or sealed with another
        # key, and nothing is exported until every transcript can be read.
        raise UserError(
            f"Transcriptum Antigravity aperiri non potuit (status {status}): {path}\n"
            "Fasciculus corruptus est aut alia clave signatus; nihil exportatur "
            "dum omnia transcripta legi possint."
        )
    return plain.raw


def antigravity_time(stamp: int | bytes | None) -> dt.datetime:
    """A {seconds, nanos} stamp as a datetime; anything else fails loudly."""
    if not isinstance(stamp, bytes):
        raise ValueError("stamp missing")
    fields = wire(stamp)
    seconds, nanos = pb_get(fields, 1), pb_get(fields, 2) or 0
    if not isinstance(seconds, int) or not isinstance(nanos, int):
        raise ValueError("stamp without seconds")
    return dt.datetime.fromtimestamp(seconds, tz=UTC) + dt.timedelta(microseconds=nanos // 1000)


def antigravity_exchanges(path: Path, repo: Path, key: bytes) -> tuple[list[Exchange], set[Holding]]:
    try:
        return antigravity_parse(path, repo, key)
    except ValueError as exc:
        # TODO: Says an Antigravity transcript has a shape this tool does not
        # understand (what and where) — the store format seems to have
        # changed — and that nothing is exported until it can be read.
        raise UserError(
            f"Forma transcripti Antigravity ignota ({exc}): {path}\n"
            "Forma repositi mutata videtur; nihil exportatur dum legi possit."
        ) from exc


def antigravity_parse(path: Path, repo: Path, key: bytes) -> tuple[list[Exchange], set[Holding]]:
    top = wire(antigravity_open(path, key))
    session = pb_text(top, 6) or path.stem
    steps = [wire(step) for number, step in top if number == 2 and isinstance(step, bytes)]
    # Attribution is by conversation: the workspace its prompts name. A
    # prompt carries that context only sometimes, so every prompt inherits
    # the one workspace the conversation names; a second one is a shape
    # this tool has not seen.
    workspaces: set[str] = set()
    for step in steps:
        if pb_get(step, 1) == ANTIGRAVITY_USER:
            for number, item in pb_message(pb_message(step, 19), 4):
                uri = pb_text(wire(item), 13) if isinstance(item, bytes) else None
                if uri is not None:
                    workspaces.add(uri)
    if len(workspaces) > 1:
        raise ValueError(f"workspaces {sorted(workspaces)}")
    folder = decode_file_uri(next(iter(workspaces))) if workspaces else None
    if folder is None or not under_dir(folder, repo, path):
        return [], set()
    messages: list[Message] = []
    holdings: set[Holding] = set()
    pending = 0.0  # seconds the agent spent in steps since the last emitted message
    for step in steps:
        kind, status = pb_get(step, 1), pb_get(step, 4)
        if status not in ANTIGRAVITY_STATUSES:
            raise ValueError(f"step status {status}")
        meta = pb_message(step, 5)
        created = antigravity_time(pb_get(meta, 1))
        if kind == ANTIGRAVITY_USER:
            payload = pb_message(step, 19)
            click = pb_message(payload, 7)
            unknown = ({n for n, _ in payload} - ANTIGRAVITY_USER_FIELDS) | (
                {n for n, _ in click} - ANTIGRAVITY_CLICK_FIELDS
            )
            if unknown:
                raise ValueError(f"prompt fields {sorted(unknown)}")
            text = pb_text(payload, 2)
            if text is None and click:
                # An action on an artifact (approving a plan, say) types
                # nothing unless a comment rides along.
                text = pb_text(click, 5) or ""
            if text is None:
                if status != ANTIGRAVITY_EMPTIED:
                    raise ValueError("prompt without words")
                continue  # emptied by the application: no words left, so no holding
            holdings.add(("Antigravity", created))
            if text != "":
                messages.append(Message("user", created, text, active=pending))
                pending = 0.0
            continue
        if kind not in (ANTIGRAVITY_PLANNER, ANTIGRAVITY_NOTIFY) and kind not in ANTIGRAVITY_MACHINE:
            raise ValueError(f"step type {kind}")
        start, end = pb_get(meta, 6), pb_get(meta, 7)
        if isinstance(start, bytes) and isinstance(end, bytes):
            pending += (antigravity_time(end) - antigravity_time(start)).total_seconds()
        if kind in ANTIGRAVITY_MACHINE:
            continue
        enum = pb_get(meta, 11)
        # The model label when the store's enum is not in the table:
        model = "" if enum is None else ANTIGRAVITY_MODELS.get(enum, f"unknown model {enum}")
        if kind == ANTIGRAVITY_PLANNER:
            payload, number, allowed = pb_message(step, 20), 1, ANTIGRAVITY_PLANNER_FIELDS
        else:
            payload, number, allowed = pb_message(step, 94), 2, ANTIGRAVITY_NOTIFY_FIELDS
        unknown = {n for n, _ in payload} - allowed
        if unknown:
            raise ValueError(f"step {kind} fields {sorted(unknown)}")
        text = pb_text(payload, number)
        if not text:
            continue  # thinking or tool calls only, or a step the application emptied
        when = antigravity_time(end) if isinstance(end, bytes) else created
        messages.append(Message("assistant", when, text, model, active=pending))
        pending = 0.0
    return paired("Antigravity", session, messages, path, (pending, 0, 0)), holdings


def chronology(exchange: Exchange) -> tuple[dt.datetime, str, str, str]:
    """The page's order: by time, ties broken by provider, session, prompt."""
    return (exchange.timestamp, exchange.provider, exchange.session, exchange.prompt)


def weave(exchanges: Iterable[Exchange]) -> list[Exchange]:
    """Merge all providers into one chronology, collapsing exact duplicates
    (resumed or forked sessions replay identical records into new files)."""
    unique: dict[Exchange, Exchange] = {}
    for exchange in exchanges:
        identity = dataclasses.replace(exchange, session="", source=Path())
        unique.setdefault(identity, exchange)
    return sorted(unique.values(), key=chronology)


def require_dir(root: Path) -> bool:
    if not root.exists():
        return False
    if not root.is_dir():
        # TODO: Says a configured transcript root exists but is not a
        # directory.
        raise UserError(f"Radix transcriptuum directorium non est: {root}")
    return True


def codex_session_dirs(root: Path) -> tuple[Path, ...]:
    if root.name in {"sessions", "archived_sessions"}:
        return (root,)
    return tuple(root / name for name in ("sessions", "archived_sessions"))


def collect(repo: Path, roots: Roots) -> tuple[list[Exchange], set[Holding]]:
    """Every exchange the stores hold for the repo, unwoven, with the
    holdings: every prompt record they still hold, kept or dropped."""
    exchanges: list[Exchange] = []
    holdings: set[Holding] = set()

    def gather(parsed: tuple[list[Exchange], set[Holding]]) -> None:
        exchanges.extend(parsed[0])
        holdings.update(parsed[1])

    # One Claude Code session's records can lie in two files: a cloud
    # workspace that restarted continued its session in a second file under
    # another project directory. claude_exchanges keeps a session's state
    # within one file, so read apart the second file's redelivered prompts
    # look new and its replies to the first file's last prompt are lost.
    # Such a session is refused, for every repo alike. `session_files` maps
    # each session to the file, line, and timestamp of its first record.
    session_files: dict[str, tuple[Path, int, Any]] = {}

    def session_file(session: str, path: Path, line_number: int, timestamp: Any) -> None:
        first_path, first_line, first_timestamp = session_files.setdefault(
            session, (path, line_number, timestamp)
        )
        # Files are compared, not paths: one file reached by two paths (a
        # root listed beside a symlink to it, a transcript symlinked or hard
        # linked into another project directory) is read twice, which weave
        # collapses, and is no split session.
        if not first_path.samefile(path):
            # TODO: Says one session's records lie in two transcript files,
            # citing the session's first record in each with that record's
            # timestamp as the record holds it, so the file the session
            # begins in can be told; to merge the two files into one, the
            # file the session begins in first, or, when one file is a copy
            # of the other, to delete the copy instead of merging; then to
            # rerun.
            raise UserError(
                f"Sessio {session} in duobus fasciculis est:\n"
                f"  {first_path}:{first_line} ({first_timestamp})\n  {path}:{line_number} ({timestamp})\n"
                "Coniunge eos in unum fasciculum, prius eum in quo sessio incipit; "
                "si alter alterius exemplar est, exemplar dele, noli coniungere. Deinde iterum curre."
            )

    for root in roots.claude:
        if require_dir(root):
            for path in sorted(p for p in root.rglob("*.jsonl") if p.is_file()):
                gather(claude_exchanges(path, repo, session_file))
    for root in roots.codex:
        if require_dir(root):
            for directory in codex_session_dirs(root):
                if directory.is_dir():
                    for path in sorted(p for p in directory.rglob("*.jsonl") if p.is_file()):
                        gather(codex_exchanges(path, repo))
    for root in roots.vscode:
        storage_root = root / "workspaceStorage"
        if not require_dir(root) or not storage_root.is_dir():
            continue
        for storage in sorted(p for p in storage_root.iterdir() if p.is_dir()):
            session_dir = storage / "chatSessions"
            if not session_dir.is_dir():
                continue
            workspace = workspace_roots(storage)
            for path in sorted((*session_dir.glob("*.json"), *session_dir.glob("*.jsonl"))):
                gather(vscode_exchanges(path, workspace, repo))
    for root in roots.antigravity:
        conversations = root / "conversations"
        if not require_dir(root) or not conversations.is_dir():
            continue
        for path in sorted(p for p in conversations.glob("*.pb") if p.is_file()):
            gather(antigravity_exchanges(path, repo, antigravity_key()))
    return exchanges, holdings


# --------------------------------------------------------------------- humans

# A "login" names a human: the GitHub account sourcery credits each prompt
# to, and names each human's ledger by. It is the login the gh command-line
# tool saved when the human signed in, lowercased: GitHub ignores case in
# logins, and so do macOS file names, so a human's ledger must have one file
# name whatever capitals they registered with. A GitHub login is letters,
# digits, and single hyphens, 1 to 39 characters, no hyphen at either end.
GITHUB_LOGIN = re.compile(r"(?=.{1,39}\Z)[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")
# gh's saved login for github.com, read from its own configuration: no
# network, unlike asking the GitHub API who is signed in.
GH_USER = ("gh", "config", "get", "user", "-h", "github.com")


def gh_user() -> subprocess.CompletedProcess[str]:
    """The one call sourcery makes to gh, and the seam quals replace: they
    never call the real gh."""
    return subprocess.run(GH_USER, capture_output=True, encoding="utf-8", errors="replace", check=False)


def login_error(found: str) -> UserError:
    # TODO: Says the GitHub login the gh command-line tool saved could not
    # be read, naming the command asked and what came back instead, and to
    # run "gh auth login", then rerun.
    return UserError(
        f"Nomen GitHub a gh servatum legi non potest: {' '.join(GH_USER)}\n{found}\n"
        "Curre 'gh auth login', deinde iterum curre."
    )


def github_login() -> str:
    """The runner's login. gh missing, failing (as when no one is signed
    in), or printing anything but one GitHub login is refused."""
    try:
        done = gh_user()
    except OSError as exc:  # gh not installed, or not runnable
        raise login_error(str(exc)) from exc
    login = done.stdout.removesuffix("\n")
    if done.returncode != 0 or not GITHUB_LOGIN.fullmatch(login):
        raise login_error(f"exit {done.returncode}, stdout {done.stdout!r}, stderr {done.stderr!r}")
    return login.lower()


# A "display name" is what the page calls a human: the name this table
# gives their login, or else the login itself. The table is an expedient
# stand-in, approved by the human, for names humans would choose themselves.
DISPLAY_NAMES = {"dreeves": "dreev", "mister-person": "logan"}


def display_name(login: str) -> str:
    return DISPLAY_NAMES.get(login, login)


# ------------------------------------------------------------------ rendering


def markdown_inline(text: str) -> str:
    """Render a deliberately small, inert subset of inline Markdown."""
    escaped = html.escape(text, quote=True)
    stashed: list[str] = []

    def stash(match: re.Match[str]) -> str:
        token = f"{len(stashed)}"
        stashed.append(f"<code>{match.group(2)}</code>")
        return token

    escaped = re.sub(r"(`+)(.+?)\1", stash, escaped)
    escaped = re.sub(
        r"\[([^\]]+)\]\((https?://[^\s)]+|mailto:[^\s)]+)\)",
        lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
        escaped,
    )
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"__(.+?)__", r"<strong>\1</strong>", escaped)
    for index, fragment in enumerate(stashed):
        escaped = escaped.replace(f"{index}", fragment)
    return escaped


def markdown_html(text: str) -> str:
    """Render common assistant Markdown without accepting raw HTML."""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "":
            i += 1
            continue

        fence = re.match(r"^\s*(```+|~~~+)\s*([A-Za-z0-9_.+-]*)\s*$", line)
        if fence:
            marker, language = fence.group(1), fence.group(2)
            i += 1
            code: list[str] = []
            while i < len(lines) and not re.match(rf"^\s*{re.escape(marker)}\s*$", lines[i]):
                code.append(lines[i])
                i += 1
            i += 1 if i < len(lines) else 0
            attrs = f' data-language="{html.escape(language, quote=True)}"' if language else ""
            body = html.escape("\n".join(code), quote=True)
            out.append(f'<pre class="code"><code{attrs}>{body}</code></pre>')
            continue

        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            level = min(len(heading.group(1)) + 1, 6)
            out.append(f"<h{level}>{markdown_inline(heading.group(2))}</h{level}>")
            i += 1
            continue

        bullet = re.compile(r"^\s*[-*+]\s+(.+)$")
        numbered = re.compile(r"^\s*\d+[.)]\s+(.+)$")
        for pattern, tag in ((bullet, "ul"), (numbered, "ol")):
            if pattern.match(line):
                items: list[str] = []
                while i < len(lines) and (m := pattern.match(lines[i])):
                    items.append(f"<li>{markdown_inline(m.group(1))}</li>")
                    i += 1
                out.append(f"<{tag}>" + "".join(items) + f"</{tag}>")
                break
        else:
            if line.startswith(">"):
                quoted: list[str] = []
                while i < len(lines) and lines[i].startswith(">"):
                    quoted.append(lines[i][1:].lstrip())
                    i += 1
                inner = "<br>".join(markdown_inline(part) for part in quoted)
                out.append(f"<blockquote><p>{inner}</p></blockquote>")
                continue

            paragraph = [line]
            i += 1
            while i < len(lines) and lines[i].strip() != "":
                if re.match(r"^\s*(```+|~~~+|#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>)", lines[i]):
                    break
                paragraph.append(lines[i])
                i += 1
            out.append("<p>" + "<br>".join(markdown_inline(part) for part in paragraph) + "</p>")
    return "\n".join(out)


# The whole design brief: with nothing expanded, the page is the human's
# prompts and almost nothing else. Prompts get full ink and a reading face;
# day markers and the disclosure line (time, agent, model) are set small and
# faint; machine-written words always wear the phosphor style, and all of
# them sit inside a closed <details> except the ballot: the human's picks
# are legible only against the question and labels the agent wrote, so the
# ballot stays default-visible — in green, never mistakable for typing.
CSS = r"""
/* Beeminder hive: honey paper by day, warm black by night, goldenrod
   accents throughout. Light-mode goldenrod is darkened for text contrast;
   dark mode gets the full #FFB300. */
:root {
  color-scheme: light dark;
  --bg: #faf0d8;
  --ink: #1c1508;
  --muted: #77673f;
  --faint: #a8945e;
  --line: #eadbb4;
  --reply-bg: #f4e9cf;
  --reply-ink: #4a3f22;
  --accent: #a97b00;
  --stripe: #ffb300;
  /* Diffstat mark pair, teal for added and red for deleted, validated per
     mode against its page surface (CVD-simulated ΔE >= 10, contrast >= 3:1;
     a plain green fails deutan/protan separation against these reds). */
  --diff-add: #00785a;
  --diff-del: #cf222e;
  /* Provider identity quartet for the exchange edge stripe and meta-line
     chip, validated per mode against its page surface: within the quartet
     the worst CVD-simulated pair is 8.7 light / 7.2 dark (both inside the
     original trio; the olive's own worst pair is 11.7 / 13.6), normal-vision
     ΔE >= 16.9, contrast >= 3:1. Against the diffstat pair the olive sits
     6.3 / 6.5 under simulation — the 6–8 band that is legal only with
     secondary encoding, which these marks have: a different mark class, and
     the agent's name beside every chip. Identity never rides on color
     alone. */
  --claude: #b02777;
  --codex: #076f9e;
  --copilot: #6a2fc2;
  --antigravity: #656902;
  --measure: 44rem;
  --serif: "Iowan Old Style", Charter, Georgia, "Times New Roman", serif;
  --sans: ui-sans-serif, -apple-system, "Segoe UI", sans-serif;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16130f;
    --ink: #ece2cc;
    --muted: #a3946f;
    --faint: #5f5540;
    --line: #2d2718;
    --reply-bg: #1f1a12;
    --reply-ink: #c9bc9e;
    --accent: #ffb300;
    --stripe: #ffb300;
    --diff-add: #1fa179;
    --diff-del: #f85149;
    --claude: #d84f97;
    --codex: #1a94b4;
    --copilot: #8a75f0;
    --antigravity: #8b8806;
  }
}
* { box-sizing: border-box; }
html { background: var(--bg); }
body {
  margin: 0;
  border-top: 4px solid var(--stripe);
  background: var(--bg);
  color: var(--ink);
  font-family: var(--serif);
  font-size: 17px;
  line-height: 1.6;
  text-rendering: optimizeLegibility;
}
main {
  width: min(calc(100% - 2.5rem), var(--measure));
  margin: 0 auto;
  padding: 5rem 0 8rem;
}
.masthead { padding-bottom: 2rem; }
.generator {
  margin: 0 0 1.1rem;
  font-family: var(--sans);
  font-size: .68rem;
  letter-spacing: .06em;
  color: var(--faint);
}
.generator a { color: inherit; text-decoration: none; }
.generator a:hover { color: var(--accent); text-decoration: underline; }
h1 {
  margin: 0;
  font-size: clamp(1.9rem, 5.5vw, 2.9rem);
  font-weight: 600;
  letter-spacing: -0.02em;
  line-height: 1.1;
}
.deck, .repo-path {
  margin: .6rem 0 0;
  color: var(--faint);
  font-family: var(--sans);
  font-size: .74rem;
  letter-spacing: .03em;
  font-variant-numeric: tabular-nums;
}
.repo-path { font-family: var(--mono); font-size: .7rem; overflow-wrap: anywhere; }
.repo-path a { color: var(--muted); text-decoration: none; }
.repo-path a:hover { color: var(--accent); text-decoration: underline; }
.controls {
  margin: .9rem 0 0;
  font-family: var(--sans);
  font-size: .7rem;
  letter-spacing: .04em;
  color: var(--faint);
}
.controls a { color: var(--muted); text-decoration: none; }
.controls a:hover { color: var(--accent); text-decoration: underline; }
/* Minimap: the dialog at a glance, added lines above the midline and
   deleted below, one clickable sliver per prompt. */
.minimap { display: block; height: 48px; margin: 1.6rem 0 0; }
.minimap .hit { fill: transparent; }
.minimap .add { fill: var(--diff-add); }
.minimap .del { fill: var(--diff-del); }
.minimap .nil { fill: var(--line); }
.minimap a:hover .hit { fill: color-mix(in srgb, var(--accent) 20%, transparent); }
/* The reading-progress rail, the device Asterisk magazine runs down its
   left margin: a goldenrod column — the top stripe's vertical sibling —
   fixed to the viewport's left edge, its height the fraction of the page
   scrolled past, invisible until the reader has actually set off (the
   script adds .show past 300px of scroll). One tick per day header hangs
   on the same percent scale, so the bar's tip touches a day's tick at the
   exact moment that day reaches the top of the viewport; a tick's label
   appears on hover and clicking it jumps to the day. The bar's width
   cedes continuously as the viewport narrows toward the text column, and
   below 54rem the marks leave entirely. */
#progress .progress-bar {
  position: fixed;
  top: 0;
  left: 0;
  width: clamp(6px, calc((100vw - var(--measure)) / 2 - 12px), 32px);
  height: 0;
  opacity: 0;
  background: var(--stripe);
  transition: opacity .1s linear;
  z-index: 10;
}
#progress .daymark {
  position: fixed;
  left: 0;
  z-index: 11;
  display: flex;
  margin-top: -12px;
  color: var(--muted);
  font-family: var(--sans);
  font-size: .65rem;
  letter-spacing: .04em;
  font-variant-numeric: tabular-nums;
  text-decoration: none;
  transform: translateX(-100%);
  transition: transform .2s ease;
}
#progress .tick {
  width: 46px;
  height: 12px;
  border-bottom: 1px solid var(--ink);
}
#progress .text {
  margin-left: .5rem;
  padding: .1rem .4rem;
  border: 1px solid var(--line);
  border-radius: .3rem;
  background: var(--bg);
  opacity: 0;
  transition: opacity .2s ease-in;
}
#progress .daymark:hover .text { opacity: 1; }
#progress.show .progress-bar { opacity: 1; }
#progress.show .daymark { transform: none; }
@media (max-width: 54rem) {
  #progress .daymark { display: none; }
}
::selection { background: color-mix(in srgb, var(--accent) 25%, transparent); }
summary a.anchor { color: inherit; text-decoration: none; }
summary a.anchor:hover { text-decoration: underline; }
.exchange:target {
  background: var(--reply-bg);
  border-radius: .45rem;
  padding: 1.1rem 1.25rem;
  margin: 2.6rem -1.25rem 0;
}
/* The clock. Every time and date on the page is written in UTC, so the
   file reads the same whoever runs sourcery, and each day header names
   that zone at the far end of its rule (the script swaps in the viewer's
   own zone, see JS): the agent name's ink at regular weight, quieter than
   the date but never as faint as the meta line, since it is there to
   stop a misreading. */
.day {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 1rem;
  margin: 3.8rem 0 0;
  padding-bottom: .4rem;
  border-bottom: 1px solid color-mix(in srgb, var(--accent) 35%, var(--line));
  color: var(--accent);
  font-family: var(--sans);
  font-size: .68rem;
  font-weight: 600;
  letter-spacing: .14em;
  font-variant-numeric: tabular-nums;
}
.day .zone { color: var(--muted); font-weight: 400; }
.exchange { margin: 2.6rem 0 0; border-left: 3px solid var(--provider); padding-left: 0.85rem; }
.exchange.claude { --provider: var(--claude); }
.exchange.codex { --provider: var(--codex); }
.exchange.copilot { --provider: var(--copilot); }
.exchange.antigravity { --provider: var(--antigravity); }
.exchange.claudeai { --provider: var(--claude); }
/* The exchange's header: the speaker label — the human's display name
   set above their turn, the way a play script or an interview transcript
   names whoever speaks next — and across from it, on the label's
   baseline, the diffstat. The label is in the human's own serif, its
   lowercase as small capitals and any capitals kept full size (the
   name is data, so its case must survive), in the agent name's ink;
   the meta line no longer repeats the name. */
.exchange > header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1.1rem;
  margin: 0 0 .1rem;
}
.speaker {
  margin: 0;
  color: var(--muted);
  font-size: 1.06rem;
  font-variant-caps: small-caps;
  letter-spacing: .08em;
  line-height: 1.5;
  overflow-wrap: anywhere;
}
/* The prompt's diffstat closes the header row, never squeezed by a long
   label. Numbers
   wear the metadata ink; polarity lives in the blocks (and in the signs and
   the fixed added-first order). */
.diffstat {
  flex: none;
  display: inline-flex;
  align-items: center;
  gap: .5rem;
  color: var(--faint);
  font-family: var(--sans);
  font-size: .68rem;
  letter-spacing: .05em;
  font-variant-numeric: tabular-nums;
}
.diffstat .blocks { display: inline-flex; gap: 2px; }
.diffstat .blocks span { width: 7px; height: 7px; border-radius: 2px; }
.diffstat .add { background: var(--diff-add); }
.diffstat .del { background: var(--diff-del); }
.diffstat .nil { background: var(--line); }
pre.prompt {
  margin: 0;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  tab-size: 4;
  font: inherit;
  font-size: 1.06rem;
  line-height: 1.62;
}
.attachments {
  margin-top: .75rem;
  display: flex;
  flex-wrap: wrap;
  gap: .5rem;
}
.attachments img {
  max-width: 100%;
  max-height: 20rem;
  border: 1px solid var(--line);
  border-radius: .3rem;
}
details { margin: .55rem 0 0; }
summary {
  display: inline-flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: .6rem;
  width: fit-content;
  list-style: none;
  cursor: pointer;
  user-select: none;
  color: var(--faint);
  font-family: var(--sans);
  font-size: .68rem;
  letter-spacing: .05em;
  font-variant-numeric: tabular-nums;
}
summary::-webkit-details-marker { display: none; }
summary::before {
  content: "▸";
  color: var(--accent);
  font-size: .62rem;
  transition: transform .12s ease;
}
details[open] > summary::before { transform: rotate(90deg); }
summary:hover { color: var(--muted); }
/* The agent's marks in the meta line: a small square chip wearing the
   provider color (the same hue as the edge stripe) and the agent's name a
   step louder than the rest of the line — promoted ink, never the mark
   color, so the reading survives any color deficiency. */
.chip { display: inline-block; flex: none; width: 8px; height: 8px; border-radius: 2px; background: var(--provider); }
.agent { color: var(--muted); font-weight: 600; }
/* INVIOLABLE: machine-generated prose renders only inside a .machine
   container, in the phosphor-terminal style — monospace green on
   near-black, in both color schemes — for maximal distinction from the
   human's serif. The containers are .reply (agent replies) and .ballot
   (multiple-choice questions the agent posed and the option labels it
   wrote, including the ones the human picked). Machine styling and
   collapsing are separate rules: .reply also hides behind the closed
   disclosure, while .ballot is deliberately default-visible — collapsing
   it would orphan the human's choice — but stays in phosphor. */
.machine {
  --m-bg: #060d08;
  --m-ink: #56dd7f;
  --m-bright: #8dffab;
  --m-dim: #2f8a4f;
  --m-line: #1c4b2d;
  font-family: var(--mono);
  background: var(--m-bg);
  color: var(--m-ink);
}
.ballot {
  margin: 1rem 0 .6rem;
  padding: .65rem .85rem;
  border-radius: .3rem;
  font-size: .8rem;
  line-height: 1.6;
  overflow-wrap: anywhere;
}
.ballot-question {
  color: var(--m-dim);
  font-size: .74rem;
  margin-bottom: .35rem;
  white-space: pre-wrap;
}
.ballot .option { color: var(--m-dim); }
.ballot .option.picked { color: var(--m-bright); }
.exchange > header + .ballot { margin-top: .3rem; }
.reply {
  margin-top: .8rem;
  padding: 1rem 1.15rem;
  border-radius: .35rem;
  font-size: .82rem;
  line-height: 1.6;
  overflow-wrap: anywhere;
}
.reply > :first-child { margin-top: 0; }
.reply > :last-child { margin-bottom: 0; }
.reply p { margin: .8rem 0; }
.reply h2, .reply h3, .reply h4, .reply h5, .reply h6 {
  margin: 1.3rem 0 .55rem;
  line-height: 1.25;
}
.reply h2 { font-size: 1.05rem; }
.reply h3, .reply h4, .reply h5, .reply h6 { font-size: .95rem; }
.reply ul, .reply ol { margin: .75rem 0; padding-left: 1.4rem; }
.reply li + li { margin-top: .25rem; }
.reply blockquote {
  margin: .9rem 0;
  padding-left: .9rem;
  border-left: 2px solid var(--m-line);
  color: var(--m-dim);
}
.reply code { font-size: .95em; }
.reply :not(pre) > code {
  padding: .1em .3em;
  border: 1px solid var(--m-line);
  border-radius: .25rem;
}
.reply pre.code {
  margin: .9rem 0;
  padding: .9rem 1rem;
  overflow-x: auto;
  white-space: pre;
  border: 1px solid var(--m-line);
  border-radius: .3rem;
  background: #0a170e;
}
.reply a { color: var(--m-bright); }
.reply.empty { font-style: italic; color: var(--m-dim); }
@media print {
  :root { --bg: white; --ink: black; --faint: #777; --muted: #555; --line: #bbb; --reply-bg: white; --reply-ink: #333; }
  #progress { display: none; }
  .machine { --m-bg: white; --m-ink: #14572e; --m-bright: #14572e; --m-dim: #4d7a5d; --m-line: #9dbfa8; background: white; border: 1px solid var(--m-line); }
  .reply pre.code { background: white; }
  main { width: 100%; padding: 0; }
  .day { break-after: avoid; }
  .exchange { break-inside: avoid; }
  details > .reply { display: block !important; }
  summary::before { content: ""; }
}
"""


JS = r"""
/* The clock. sourcery writes every time and date on this page in UTC, so
   the file reads the same whoever runs it; this block re-renders each in
   the viewer's own zone: every prompt's time, the day headers (regrouping
   the prompts under the viewer's own calendar days, each header naming
   the zone as the browser names it), the rail's day marks, the deck's
   date range and the minimap's titles. With scripts off the page reads in
   UTC, as written. It runs before the rail code below, which collects the
   day marks rebuilt here. */
{
  const clock = new Intl.DateTimeFormat("en-US", {
    weekday: "long", year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", hourCycle: "h23", timeZoneName: "short",
  });
  /* An instant as sourcery writes it (ISO 8601 in UTC, to the
     microsecond), as the parts of the viewer's clock and calendar. */
  const local = (text) => {
    const utc = /^(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)(?:\.\d+)?\+00:00$/.exec(text);
    // TODO: Says the text is not a UTC instant in the form sourcery writes.
    if (!utc) throw new Error(`Instans UTC formae sourcery non est: ${text}`);
    const [, year, month, day, hour, minute, second] = utc.map(Number);
    const instant = Date.UTC(year, month - 1, day, hour, minute, second);
    return Object.fromEntries(clock.formatToParts(instant).map((part) => [part.type, part.value]));
  };
  const days = []; // [date, header text], one per day header, in page order
  for (const header of document.querySelectorAll("main > h2.day")) header.remove();
  for (const article of document.querySelectorAll("main > article.exchange")) {
    const time = article.querySelector("summary time");
    const at = local(time.dateTime);
    const date = `${at.year}-${at.month}-${at.day}`;
    time.textContent = `${at.hour}:${at.minute}`;
    const title = document.querySelector(`.minimap a[href="#${article.id}"] title`);
    title.textContent = [time.textContent, ...title.textContent.split(" ").slice(1)].join(" ");
    if (date !== days.at(-1)?.[0]) {
      days.push([date, `${date} ${at.weekday}`]);
      article.insertAdjacentHTML(
        "beforebegin",
        `<h2 class="day" id="d${date}"><time datetime="${date}">${date} ${at.weekday}</time>`
          + ` <span class="zone">${at.timeZoneName}</span></h2>`,
      );
    }
  }
  document.querySelector("#progress .daymarks").innerHTML = days.map(([date, text]) =>
    `\n<a class="daymark" href="#d${date}"><span class="tick"></span><span class="text">${text}</span></a>`,
  ).join("") + "\n";
  /* The first and last day, or the one day when they are the same (the
     Set drops the repeat), as render() writes the deck's range. */
  document.querySelector(".deck .range").innerHTML = [...new Set([days[0][0], days.at(-1)[0]])]
    .map((date) => `<time datetime="${date}">${date}</time>`).join(" – ");
}
for (const control of document.querySelectorAll("[data-omnia]"))
  control.addEventListener("click", (event) => {
    event.preventDefault();
    const open = control.dataset.omnia === "open";
    for (const details of document.querySelectorAll("details")) details.open = open;
  });
/* The reading-progress rail: the bar's height and each day mark's top
   share one scale — percent of full scroll travel — so the bar's tip
   touches a day's tick at the exact moment that day's header reaches the
   top of the viewport. A header past the last reachable scroll position
   can never get there; its mark pins to 100%, where the bar arrives at
   the very bottom of the page. */
const progress = document.getElementById("progress");
const bar = progress.querySelector(".progress-bar");
const marks = [...progress.querySelectorAll(".daymark")].map(
  (mark) => [mark, document.getElementById(mark.hash.slice(1))],
);
let ticking = false;
const sync = () => {
  if (ticking) return;
  ticking = true;
  requestAnimationFrame(() => {
    const travel = Math.max(document.documentElement.scrollHeight - innerHeight, 1);
    progress.classList.toggle("show", scrollY > 300);
    bar.style.height = 100 * scrollY / travel + "%";
    for (const [mark, header] of marks)
      mark.style.top =
        Math.min(100 * (header.getBoundingClientRect().top + scrollY) / travel, 100) + "%";
    ticking = false;
  });
};
addEventListener("scroll", sync, { passive: true });
addEventListener("resize", sync);
/* Opening or closing a disclosure reflows the whole page under the rail. */
document.addEventListener("toggle", sync, true);
sync();
"""


WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
# TODO: The zone marker on each day header, saying the day and the times
# under it are in Coordinated Universal Time. UTC is the international
# abbreviation, the same in every language, so it is kept as is rather
# than put in Latin. The script replaces it with the browser's own name
# for the viewer's zone (data, like the times).
ZONE = "UTC"


def elapsed_text(seconds: float) -> str:
    total = round(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return "".join(f"{n}{unit}" for n, unit in ((hours, "h"), (minutes, "m"), (secs, "s")) if n) or "0s"


def diffstat(added: int, deleted: int) -> str:
    """The prompt's infographic: signed line counts and five blocks split by
    the added:deleted ratio (each whole line a block when five would cover
    it, unfilled blocks padding the difference). Sign and order carry the
    same polarity as the block colors, so no reading depends on color."""
    if (added, deleted) == (0, 0):
        return ""
    total = added + deleted
    if total <= 5:
        plus, minus = added, deleted
    else:
        # Floor both shares (never overstating either side, and splitting a
        # tie symmetrically), but give any nonzero side its block.
        plus = max(5 * added // total, 1 if added else 0)
        minus = max(5 * deleted // total, 1 if deleted else 0)
    blocks = (
        '<span class="add"></span>' * plus
        + '<span class="del"></span>' * minus
        + '<span class="nil"></span>' * (5 - plus - minus)
    )
    return (
        f'\n<div class="diffstat">+{added:,} −{deleted:,}'
        f' <span class="blocks" aria-hidden="true">{blocks}</span></div>'
    )


# All rendered copy here is human-written or human-specified; everything
# else on the page (repository name/path/remote, provider, model, effort,
# timestamps, weekdays, prompts, ballots, replies) is source data.
def minimap(exchanges: Sequence[Exchange], instants: Sequence[dt.datetime]) -> str:
    """A clickable map of the whole dialog: one sliver per prompt, in order,
    linking to its permalink anchor. Added lines rise from the midline and
    deleted lines hang below it, on a square-root scale against the peak so
    small edits stay visible beside the big rewrite spikes; a prompt that
    touched no code is a tick on the midline. Each sliver's hover title is
    its time plus its counts — numerals only, so nothing here needs prose."""
    peak = max(max(e.added, e.deleted) for e in exchanges)
    unit = lambda v: f"{round(v, 1):g}"  # 30.0 -> "30"
    slots: list[str] = []
    for i, (e, at) in enumerate(zip(exchanges, instants)):
        x = 3 * i
        stamp = at.strftime("%H:%M")
        title = f"{stamp} +{e.added:,} −{e.deleted:,}" if e.added or e.deleted else stamp
        bars = f'<rect class="hit" x="{x}" y="0" width="3" height="64"/>'
        if e.added:
            rise = round(30 * (e.added / peak) ** 0.5, 1)
            bars += f'<rect class="add" x="{x}" y="{unit(32 - rise)}" width="2" height="{unit(rise)}"/>'
        if e.deleted:
            drop = 30 * (e.deleted / peak) ** 0.5
            bars += f'<rect class="del" x="{x}" y="32" width="2" height="{unit(drop)}"/>'
        if (e.added, e.deleted) == (0, 0):
            bars += f'<rect class="nil" x="{x}" y="31" width="2" height="2"/>'
        slots.append(f'<a href="#p{i + 1}"><title>{title}</title>{bars}</a>')
    width = 3 * len(exchanges)
    # TODO: The label says this is a map of the dialog, prompt by prompt.
    return (
        f'<svg class="minimap" viewBox="0 0 {width} 64" preserveAspectRatio="none"'
        f' style="width:min(100%, {2 * width}px)" aria-label="Tabula dialogi, rogatio post rogationem">'
        + "".join(slots)
        + "</svg>"
    )


# CSS class per provider, keying the exchange's edge stripe and meta-line
# chip to its agent. A provider missing here crashes render() (KeyError)
# rather than shipping an unmarked exchange.
PROVIDER_SLUGS = {
    "Claude Code": "claude", "Codex": "codex", "Copilot Chat": "copilot", "Antigravity": "antigravity",
    # claude.ai chats have no store parser: their exchanges enter a page only
    # by being seeded into a ledger from outside sourcery, and every run
    # keeps them, since no store holds them. They wear Claude Code's color.
    "claude.ai": "claudeai",
}
# page_credits reads a page's provider back from its slug, so no two
# providers may share one.
assert len(set(PROVIDER_SLUGS.values())) == len(PROVIDER_SLUGS)


def render(repo: Path, ledgers: Mapping[str, Sequence[Exchange]], remote: str = "") -> str:
    """The page for every row of the ledgers, given as login to rows, all
    in one chronology."""
    rows = sorted(
        ((login, exchange) for login, exchanges in ledgers.items() for exchange in exchanges),
        key=lambda row: chronology(row[1]),
    )
    assert rows, "render() requires at least one exchange"
    exchanges = [exchange for _, exchange in rows]
    # Every time and date on the page is written in UTC, so the file is a
    # function of the ledgers alone, whoever runs sourcery; the page's
    # script (JS) re-renders them in the viewer's own zone.
    instants = [e.timestamp.astimezone(UTC) for e in exchanges]
    first, last = instants[0].date().isoformat(), instants[-1].date().isoformat()
    count = len(exchanges)
    noun = "prompt" if count == 1 else "prompts"
    dates = (first,) if first == last else (first, last)
    total_added = sum(e.added for e in exchanges)
    total_deleted = sum(e.deleted for e in exchanges)
    totals = f" · +{total_added:,} −{total_deleted:,}" if (total_added, total_deleted) != (0, 0) else ""
    deck = f"{count} {noun} · {' – '.join(dates)}{totals}"
    # On the page, the deck's date range is time elements in a span of its
    # own, for the script to rewrite; the description meta tags carry the
    # deck as plain text.
    range_html = " – ".join(f'<time datetime="{day}">{day}</time>' for day in dates)
    deck_html = f'{count} {noun} · <span class="range">{range_html}</span>{totals}'

    chunks: list[str] = []
    days: list[tuple[str, str]] = []  # (day, header text), one per day header
    current_day = None
    for number, ((login, exchange), at) in enumerate(zip(rows, instants), start=1):
        day = at.date().isoformat()
        if day != current_day:
            weekday = WEEKDAYS[at.date().weekday()]
            chunks.append(
                f'<h2 class="day" id="d{day}"><time datetime="{day}">{day} {weekday}</time>'
                f' <span class="zone">{ZONE}</span></h2>'
            )
            days.append((day, f"{day} {weekday}"))
            current_day = day
        model = f' <span class="model">{html.escape(exchange.model)}</span>' if exchange.model else ""
        effort = f' <span class="effort">({html.escape(exchange.effort)})</span>' if exchange.effort else ""
        # The final exchange of the export is the only one that can plausibly
        # still be in flight, so an empty reply there gets the still-
        # generating note; an empty reply anywhere else means there is none.
        if exchange.reply == "" and number == count:
            reply = (
                '<div class="reply machine empty">'
                "<p>Response still generating when this transcript was captured</p></div>"
            )
        elif exchange.reply == "":
            reply = '<div class="reply machine empty"><p>No response.</p></div>'
        else:
            reply = f'<div class="reply machine">{markdown_html(exchange.reply)}</div>'
        # The wall span renders only when it reads longer than the working
        # time: an equal-at-display-precision span repeats the number beside
        # it, and a shorter one (activity credited off records later than the
        # last reply) answers a question nobody asked.
        # TODO: Says the turn's full span by the wall clock, prompt to final
        # reply, beside the active working time already shown.
        wall = (
            f" · {elapsed_text(exchange.wall)} wall-clock time"
            if exchange.wall > exchange.elapsed
            and elapsed_text(exchange.wall) != elapsed_text(exchange.elapsed)
            else ""
        )
        thought = (
            f' <span class="elapsed">thought for {elapsed_text(exchange.elapsed)}{wall}</span>'
            if exchange.elapsed >= 0.5
            else ""
        )
        summary = (
            f'<a class="anchor" href="#p{number}">'
            f'<time datetime="{exchange.timestamp.isoformat()}">{at.strftime("%H:%M")}</time></a>'
            f' <span class="chip"></span>'
            f' <span class="agent">{html.escape(exchange.provider)}</span>{model}{effort}{thought}'
        )
        attached = "".join(
            f'<img class="attachment" src="{html.escape(uri, quote=True)}" alt="">'
            for uri in exchange.images
        )
        attachments = f'\n<div class="attachments">{attached}</div>' if attached else ""
        ballots = "".join(
            '\n<div class="ballot machine">'
            f'\n<div class="ballot-question">{html.escape(ballot.question, quote=False)}</div>'
            + "".join(
                f'\n<div class="option{" picked" if label in ballot.picked else ""}">'
                f'{"✓" if label in ballot.picked else "·"} {html.escape(label, quote=False)}</div>'
                for label in ballot.options
            )
            + "\n</div>"
            for ballot in exchange.ballots
        )
        prompt = (
            f'\n<pre class="prompt">{html.escape(exchange.prompt, quote=False)}</pre>'
            if exchange.prompt != ""
            else ""
        )
        chunks.append(
            f'<article class="exchange {PROVIDER_SLUGS[exchange.provider]}" id="p{number}"'
            f' data-login="{html.escape(login, quote=True)}">'
            f'\n<header><p class="speaker">{html.escape(display_name(login))}</p>'
            f"{diffstat(exchange.added, exchange.deleted)}</header>{ballots}{prompt}{attachments}\n"
            f"<details>\n<summary>{summary}</summary>\n{reply}\n</details>\n"
            "</article>"
        )

    body = "\n".join(chunks)
    # The reading-progress rail's day marks are generated here, where the
    # days are known; the script only measures and positions them.
    marks = "".join(
        f'\n<a class="daymark" href="#d{day}">'
        f'<span class="tick"></span><span class="text">{label}</span></a>'
        for day, label in days
    )
    # TODO: The label names the day-mark rail, for assistive tech, as the
    # index of the dialog's days.
    rail = (
        '<div id="progress">\n'
        '<div class="progress-bar" aria-hidden="true"></div>\n'
        f'<nav class="daymarks" aria-label="Index dierum">{marks}\n</nav>\n'
        "</div>"
    )
    title = html.escape(repo.name)
    # The subtitle answers "where this lives": the public remote when there
    # is one, the local directory otherwise.
    if remote:
        label = html.escape(remote.removeprefix("https://").removeprefix("http://"))
        where = f'<a href="{html.escape(remote, quote=True)}">{label}</a>'
    else:
        where = html.escape(str(repo))
    # The warning comment must follow the doctype — a comment before it
    # would throw browsers into quirks mode.
    # TODO: Says this file is generated by sourcery and never hand-edited;
    # the memory of the dialog lives in the ledgers beside it
    # (sourcery.LOGIN.jsonl), from which the next generation renders it
    # anew, overwriting it, so don't edit it in place.
    return f"""<!doctype html>
<!-- Fasciculus hic a sourcery generatus est, numquam manu scriptus.
     Memoria dialogi in tabulis iuxta positis (sourcery.LOGIN.jsonl)
     servatur, ex quibus generatio proxima eum denuo reddit atque
     superscribit. Noli emendare. -->
<html lang="und">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Ctext y='14' font-size='14'%3E%F0%9F%90%9D%3C/text%3E%3C/svg%3E">
<meta name="description" content="{deck}">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{deck}">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
{rail}
<main>
<header class="masthead">
  <p class="generator"><a href="https://github.com/beeminder/sourcery">generated by sourcery</a></p>
  <h1>{title}</h1>
  <p class="deck">{deck_html}</p>
  <p class="repo-path">{where}</p>
  <p class="controls"><a href="#" data-omnia="open">expand all</a> · <a href="#" data-omnia="close">collapse all</a></p>
  {minimap(exchanges, instants)}
</header>
{body}
</main>
<script>{JS}</script>
</body>
</html>
"""


# ------------------------------------------------------------------- ledgers


# A "ledger" is one human's memory of the dialog: every exchange of theirs
# the page shows, kept beside the page in sourcery.LOGIN.jsonl. Stores get
# pruned and machines change; the ledger, committed with the page, is then
# the only copy, and nothing it once held is lost. Its first line is a
# header naming the ledger format (LEDGER_FORMAT) and the repo, by its
# project name; every further line is one exchange in freeze() form. One
# exchange per line, so a git conflict in a ledger is resolved by keeping
# both sides' lines; where both sides changed one row, that leaves two
# readings of one prompt (see check_readings). A run rewrites the runner's
# own ledger and no other.


def freeze(exchange: Exchange) -> dict[str, Any]:
    """The exchange as JSON-ready fields, minus the store path: private to
    this machine and irrelevant to the merge, so it never reaches a ledger."""
    frozen = dataclasses.asdict(exchange)
    del frozen["source"]
    frozen["timestamp"] = exchange.timestamp.isoformat()
    return frozen


# What a frozen exchange must look like, field by field: the JSON type of
# every Exchange field but the omitted source. A missing or extra field, or
# a value of another type, means a ledger row another schema wrote or a hand
# edit — never thawed into a guess.
SHAPE: dict[str, type | tuple[type, ...]] = {
    "timestamp": str, "provider": str, "model": str, "session": str, "prompt": str,
    "reply": str, "effort": str, "images": list, "elapsed": (int, float), "wall": (int, float),
    "ballots": list, "added": int, "deleted": int,
}
assert set(SHAPE) == {field.name for field in dataclasses.fields(Exchange)} - {"source"}


def shaped(value: Any, kind: type | tuple[type, ...]) -> bool:
    return isinstance(value, kind) and not isinstance(value, bool)


def thaw(frozen: Mapping[str, Any], source: Path) -> Exchange:
    """Inverse of freeze; the ledger is the source. The field set and every
    field's type and range are checked, so a row another schema wrote, or
    one edited by hand, fails loudly instead of thawing into nonsense."""
    fields = dict(frozen)
    sound = (
        set(fields) == set(SHAPE)
        and all(shaped(fields[name], kind) for name, kind in SHAPE.items())
        and fields["provider"] in PROVIDER_SLUGS
        and all(0 <= fields[name] < float("inf") for name in ("elapsed", "wall", "added", "deleted"))
        and all(isinstance(uri, str) for uri in fields["images"])
        and all(
            isinstance(ballot, dict)
            and set(ballot) == {"question", "options", "picked"}
            and isinstance(ballot["question"], str)
            and all(
                isinstance(ballot[key], list) and all(isinstance(label, str) for label in ballot[key])
                for key in ("options", "picked")
            )
            for ballot in fields["ballots"]
        )
    )
    if not sound:
        raise ValueError(f"exchange fields {sorted(fields)}")
    fields["timestamp"] = parse_time(fields["timestamp"])
    fields["images"] = tuple(fields["images"])
    fields["ballots"] = tuple(
        Ballot(ballot["question"], tuple(ballot["options"]), tuple(ballot["picked"]))
        for ballot in fields["ballots"]
    )
    return Exchange(source=source, **fields)


def ledger_path(output: Path, login: str) -> Path:
    return output.parent / f"sourcery.{login}.jsonl"


# The "ledger format": the number a ledger's header names it by, which
# changes only when the ledger format itself does (the header's keys, or
# what a row holds), never with a sourcery release. Machines running two
# releases thus write one header, so a git union merge of their ledgers
# never doubles line 1 for the release alone. read_ledger reads this format
# and refuses any other, saying that another release wrote it: a later
# format keeps the header's "ledger" key, so that a release still reading
# this one can recognize what it cannot read.
LEDGER_FORMAT = 1
# What a ledger's header must look like: each key's JSON type.
HEADER_SHAPE = {"ledger": "int", "repo": "str"}


def ledger_text(project: str, exchanges: Sequence[Exchange]) -> str:
    """A ledger's whole text: the header line, naming the ledger format
    and the project, then one exchange per line."""
    lines = [{"ledger": LEDGER_FORMAT, "repo": project}, *map(freeze, exchanges)]
    return "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)


def json_object(line: str) -> dict[str, Any]:
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError(f"{type(value).__name__} != dict")
    return value


def read_ledger(path: Path, repo: Path, remote: str) -> dict[Exchange, str]:
    """A ledger's exchanges, every line checked: a damaged line, a header
    naming a ledger format other than LEDGER_FORMAT, or a header
    naming a project other than the checkout's (repo's project name, given
    its public home, remote, as repo_remote found it), fails loudly instead
    of thawing into a guess. They are woven, as a run weaves the runner's
    own, so a row a merge left doubled shows once whoever runs. Each maps to
    its "citation", where a refusal finds it: the ledger's path and the
    number of the first line holding it."""
    project = project_name(repo, remote)
    try:
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        # TODO: Says this ledger could not be read as a UTF-8 file, gives
        # the reason, and that nothing was written.
        raise UserError(f"Tabula legi non potest: {path}\n{exc}\nNihil scriptum est.") from exc
    # Lines end at "\n" alone: JSON escapes it inside strings, but leaves
    # other characters str.splitlines would break at (U+2028, say) as typed.
    header, *rows = text.removesuffix("\n").split("\n")
    number = 1  # the line being read, cited if it fails
    try:
        head = json_object(header)
        form = ({key: type(value).__name__ for key, value in head.items()}, head.get("ledger"))
        if form != (HEADER_SHAPE, LEDGER_FORMAT):
            raise ValueError(f"{form} != {(HEADER_SHAPE, LEDGER_FORMAT)}")
        cited: dict[Exchange, str] = {}
        for number, row in enumerate(rows, start=2):
            cited.setdefault(thaw(json_object(row), path), f"{path}:{number}")
    except (ValueError, UserError) as exc:
        # TODO: Says this ledger cannot be read at this line, with the
        # reason: a ledger is a header line, then one exchange per line. If
        # the header line names a ledger format other than LEDGER_FORMAT
        # (named), another sourcery release wrote the ledger: run a release
        # that reads that format, then rerun. If it holds git conflict
        # markers, resolve the conflict by keeping both sides' lines (the
        # header line once), then rerun. With the line
        # "sourcery.*.jsonl merge=union" added to the attributes file git
        # reads for all of the human's repositories (normally
        # ~/.config/git/attributes: per gitattributes(5), the file
        # core.attributesFile names, by default
        # $XDG_CONFIG_HOME/git/attributes, or ~/.config/git/attributes when
        # XDG_CONFIG_HOME is unset or empty), git keeps both sides' lines by
        # itself; but such a merge can leave the header line twice: keep on
        # the first line the one naming this project (named), delete the
        # other, then rerun. Nothing was written.
        raise UserError(
            f"Tabula legi non potest: {path}:{number}\n{exc}\n"
            "Tabula est linea capitis, deinde una rogatio per lineam. "
            f"Si linea capitis formam tabulae aliam quam {LEDGER_FORMAT} nominat, tabulam alia editio sourcery "
            "scripsit: curre editionem quae illam formam legit, deinde iterum curre. Si signa conflictus git fert, "
            "conflictum solve utriusque partis lineas servando (lineam capitis semel), deinde iterum curre.\n"
            "Linea 'sourcery.*.jsonl merge=union' in fasciculo attributorum quem git omnibus repositoriis "
            "tuis legit (plerumque ~/.config/git/attributes) addita, git sponte utriusque partis lineas "
            f"servat; sed fusio talis lineam capitis bis relinquere potest: eam quae {project!r} nominat in "
            "prima linea serva, alteram dele, deinde iterum curre.\n"
            "Nihil scriptum est."
        ) from exc
    if head["repo"] != project:
        # TODO: Says this ledger belongs to another project, naming the
        # project its first line names, then this checkout's project and
        # directory, then the ledger; then names this checkout's public
        # home as sourcery found it from the origin remote ('' when it found
        # none); nothing was written. A project is named by the last part
        # of the path of its public home's URL (the origin remote's), or,
        # when it has no public home, by its directory's name. When a
        # repository is renamed on GitHub, sourcery refuses its ledgers
        # until each ledger's first line names the new repository. If the
        # ledger is this project's: if the repository is now named as this
        # checkout names it (named), write that name in place of the
        # ledger's (named) in each ledger's first line; if not, fix this
        # checkout: set origin's URL to the repository's home, or, when it
        # has no public home, rename the directory as the ledger names the
        # project (named). If the ledger is in fact another project's,
        # choose another output path. Then rerun.
        raise UserError(
            f"Tabula ad aliud inceptum pertinet: {head['repo']!r}, non {project!r} ({repo}): {path}\n"
            f"Sedes publica huius directorii (ex remoto 'origin'): {remote!r}\n"
            "Inceptum nominatur ultima parte viae URL sedis suae publicae (remoti 'origin'), "
            "aut, si sedem publicam non habet, nomine directorii sui.\n"
            "Repositorio in GitHub renominato, sourcery tabulas eius recusat donec prima linea "
            "cuiusque tabulae novum repositorium nominet.\n"
            f"Si tabula ad hoc inceptum pertinet: si repositorium nunc {project!r} nominatur, in prima "
            f"linea cuiusque tabulae pro {head['repo']!r} scribe {project!r}; sin minus, hic corrige: "
            "URL ipsius 'origin' ad sedem repositorii constitue, aut, si sedem publicam non habet, "
            f"directorium renomina {head['repo']!r}. Si vero re vera ad aliud inceptum pertinet, "
            "aliam viam output elige.\n"
            "Deinde iterum curre.\nNihil scriptum est."
        )
    return {exchange: cited[exchange] for exchange in weave(cited)}


# Any file named like a ledger, ignoring case. read_ledgers refuses one whose
# name is not exactly sourcery.LOGIN.jsonl for a lowercase login, so no
# capitals can split one human's ledger in two (see GITHUB_LOGIN).
LEDGER_NAME = re.compile(r"sourcery\.(.*)\.jsonl", re.IGNORECASE | re.DOTALL)


def read_ledgers(
    output: Path, repo: Path, remote: str
) -> tuple[dict[str, list[Exchange]], dict[Exchange, str]]:
    """Every ledger beside the output page, by login, in login order, and
    every row's citation (see read_ledger)."""
    ledgers: dict[str, list[Exchange]] = {}
    citations: dict[Exchange, str] = {}
    for path in sorted(output.parent.glob("*")):
        named = LEDGER_NAME.fullmatch(path.name)
        if named is None:
            continue  # no ledger
        login = named[1]
        if path.name != ledger_path(output, login.lower()).name or not GITHUB_LOGIN.fullmatch(login):
            # TODO: Says this file is named like a ledger but is none: a
            # ledger's name is sourcery.LOGIN.jsonl, all lowercase, LOGIN
            # being a GitHub login; rename the file so, or move it away, then
            # rerun. Nothing was written.
            raise UserError(
                f"Fasciculus nomen tabulae imitatur, sed tabula non est: {path}\n"
                "Nomen tabulae est sourcery.LOGIN.jsonl, totum minusculis litteris, LOGIN nomen GitHub. "
                "Fasciculum renomina vel remove, deinde iterum curre.\nNihil scriptum est."
            )
        cited = read_ledger(path, repo, remote)
        ledgers[login] = list(cited)
        citations.update(cited)
    return ledgers, citations


# A "credit" records, on the page, whose ledger holds each row it shows:
# each article carries the login in a data-login attribute, beside the row's
# provider (the article's class) and time (its summary's time element), so a
# run can check the page against the ledgers without reading display names.
# Typed text is escaped on the page, so none can forge a credit.
PROVIDER_NAMES = {slug: name for name, slug in PROVIDER_SLUGS.items()}
CREDIT = re.compile(
    rf'<article class="exchange ({"|".join(map(re.escape, PROVIDER_NAMES))})" id="p[0-9]+"'
    r' data-login="([^"]*)">.*?<time datetime="([^"]*)">',
    re.DOTALL,
)


def page_credits(output: Path) -> set[tuple[str, Holding]]:
    """Every row the existing page credits, as its login and holding;
    nothing when there is no page yet. A page that does not credit every
    row it shows cannot be checked against the ledgers and is refused: one
    sourcery 5 wrote, or no sourcery page at all."""
    if not output.exists():
        return set()
    try:
        page = output.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        # TODO: Says the existing output could not be read as a UTF-8 file
        # (it may be a directory, unreadable, or not text), gives the reason,
        # and that nothing was written.
        raise UserError(
            f"Pagina exsistens legi non potest: {output}\n{exc}\nNihil scriptum est."
        ) from exc
    credits = CREDIT.findall(page)
    if not 0 < len(credits) == page.count("<article "):
        # TODO: Says the existing output was not written by sourcery 6: it
        # does not credit every prompt it shows to a ledger, so it cannot be
        # checked against the ledgers. A page sourcery 5 wrote, carrying a
        # snapshot, is first imported by adopt.py, run as shown, LOGIN being
        # the GitHub login of the human whose sourcery 5 wrote the page,
        # perhaps not the human running sourcery now; for any other file,
        # choose another output path. Nothing was written.
        raise UserError(
            f"Pagina exsistens a sourcery 6 scripta non est: {output}\n"
            "Si pagina sourcery 5 est (memoriam snapshot fert), curre prius:\n"
            f"  python3 adopt.py REPODIR {output} {output} LOGIN\n"
            "LOGIN est nomen GitHub hominis cuius sourcery 5 paginam scripsit, fortasse non tuum.\n"
            "Aliter elige aliam viam output.\nNihil scriptum est."
        )
    return {(login, (PROVIDER_NAMES[slug], parse_time(when))) for slug, login, when in credits}


def listed(output: Path, credits: Iterable[tuple[str, Holding]]) -> str:
    """Prompts for a refusal to list, given as login and holding: under
    each login's ledger path and its count of them, each prompt by agent
    and time, one per line."""
    by_login: dict[str, list[Holding]] = {}
    for login, held in credits:
        by_login.setdefault(login, []).append(held)
    return "".join(
        f"\n  {ledger_path(output, login)}: {len(helds)}"
        + "".join(f"\n    {provider} {when.isoformat()}" for provider, when in helds)
        for login, helds in by_login.items()
    )


def check_ledgers(
    output: Path,
    login: str,
    ledgers: Mapping[str, Sequence[Exchange]],
    credits: set[tuple[str, Holding]],
    holdings: set[Holding],
) -> None:
    """Refuse, before anything is written, ledgers that disagree with the
    page, with each other, or with the runner's stores. Each prompt belongs
    to one human, so each holding belongs to one ledger."""
    held = {other: {holding(exchange) for exchange in rows} for other, rows in ledgers.items()}
    lacking = sorted(credit for credit in credits if credit[1] not in held.get(credit[0], set()))
    if lacking:
        # TODO: Says the existing page shows this many prompts that the
        # ledgers it credits them to do not hold, listing under each such
        # ledger how many and each prompt by agent and time. Such a ledger
        # may be uncommitted, deleted, or renamed: restore it, then rerun;
        # git brings back a deleted one: "git log --diff-filter=D -- PATH"
        # names the commit REV that deleted it, "git checkout REV^ -- PATH"
        # restores it. Only if the prompts were removed from their ledger on
        # purpose: delete the page, then rerun; that drops every credit the
        # page holds, every human's, so first make sure every other ledger
        # is whole. Nothing was written.
        raise UserError(
            f"Pagina exsistens {len(lacking)} rogationes ostendit quas tabulae quibus tribuuntur non tenent:"
            f"{listed(output, lacking)}\n"
            "Fortasse tabula non commissa, deleta, vel renominata est: restitue eam, deinde iterum curre. "
            "Tabulam deletam git restituit: 'git log --diff-filter=D -- VIA' commissum REV nominat quod eam "
            "delevit, 'git checkout REV^ -- VIA' eam reddit.\n"
            "Solum si rogationes consulto e tabula sua remotae sunt, paginam dele, deinde iterum curre: "
            "ita pagina omnes tributiones omnium hominum amittit, ergo prius cave ut ceterae tabulae integrae sint.\n"
            "Nihil scriptum est."
        )
    owners: dict[Holding, str] = {}
    for other, holds in held.items():
        for h in sorted(holds):
            first = owners.setdefault(h, other)
            if first != other:
                # TODO: Says one prompt (agent and time) is held by two
                # ledgers, naming both: each prompt belongs to one human, so
                # remove it from the ledger it does not belong to, then
                # rerun. Nothing was written.
                raise UserError(
                    f"Rogatio {h[0]} {h[1].isoformat()} in duabus tabulis est:\n"
                    f"  {ledger_path(output, first)}\n  {ledger_path(output, other)}\n"
                    "Rogatio unius hominis est: remove eam ex tabula ad quam non pertinet, "
                    "deinde iterum curre.\nNihil scriptum est."
                )
    claimed = sorted((owners[h], h) for h in holdings if owners.get(h, login) != login)
    if claimed:
        # TODO: Says this machine's transcript stores hold this many prompts
        # that other humans' ledgers hold, listing under each such ledger
        # how many and each prompt by agent and time: each prompt belongs to
        # one human, so none is credited to the runner (named by login) too.
        # Perhaps gh is signed in as someone else. Or that human's
        # transcripts were copied to this machine: remove the copies from
        # this machine's stores. Or this machine's transcripts reached that
        # human's machine first, and their ledger holds the prompts wrongly:
        # remove the copies from their stores and these rows from their
        # ledger by hand, and delete the page. Then rerun. Nothing was
        # written.
        raise UserError(
            f"Reposita huius machinae {len(claimed)} rogationes tenent quas tabulae aliorum iam tenent:"
            f"{listed(output, claimed)}\n"
            f"Rogatio unius hominis est, nec tibi ({login}) quoque tribuitur. Fortasse gh alium hominem nominat. "
            "Aut transcripta illius huc translata sunt: exemplaria e repositis huius machinae remove. "
            "Aut transcripta huius machinae prius ad machinam illius pervenerunt, et tabula eius ea perperam "
            "tenet: exemplaria e repositis eius et has lineas e tabula eius manu remove, paginamque dele. "
            "Deinde iterum curre.\nNihil scriptum est."
        )


# Two "readings" of one prompt are two rows alike in provider, time,
# session, and prompt text, but differing in anything else. A ledger comes
# to hold two when runs on two machines read the prompt's record
# differently (one while its reply was still arriving, say) and a git merge
# kept both sides' lines (merge=union does); a run reads two itself when
# two transcripts of the prompt's session disagree, or when one transcript
# holds the prompt's record twice and its copies read differently (the
# first with no reply, say). Exactly identical rows are no two readings:
# weave collapses them before this check.
def prompt_key(row: Exchange) -> tuple[str, dt.datetime, str, str]:
    """What every reading of one prompt shares: its agent, time, session,
    and words."""
    return (row.provider, row.timestamp, row.session, row.prompt)


def check_readings(
    ledgers: Mapping[str, Iterable[Exchange]], citations: Mapping[Exchange, str], advice: str
) -> None:
    """Refuse, before anything is written, two readings of one prompt among
    the ledgers' rows, citing each as citations gives it, then giving the
    advice the caller words to fit where it read the rows."""
    first: dict[tuple[str, dt.datetime, str, str], Exchange] = {}
    for rows in ledgers.values():
        for row in rows:
            seen = first.setdefault(prompt_key(row), row)
            if seen != row:
                # TODO: Says one prompt, named by agent, time, and session,
                # has two readings (versions), rows alike in those and in
                # its words but differing otherwise, citing each where it
                # was read: a ledger's row by the ledger's file and line, a
                # row read from a transcript, or from the snapshot of a page
                # sourcery 5 wrote, by that file; then the caller's advice
                # (see each call). Nothing was written.
                raise UserError(
                    f"Rogationis {row.provider} {row.timestamp.isoformat()} (sessio {row.session}) "
                    f"duae lectiones sunt:\n  {citations[seen]}\n  {citations[row]}\n"
                    f"{advice}\nNihil scriptum est."
                )


# ----------------------------------------------------------------------- exit


# Output (the runner's ledger, then the page) is written atomically so a
# partial document is never left behind. An existing document has already
# been read by the time this runs (the ledger's rows merged into its
# successor, the page checked against the ledgers) and is then replaced
# whole: the page is always generated, never hand-edited (the page itself
# opens with a warning comment saying so).
def write_output(path: Path, page: str) -> None:
    target = path.resolve()
    if not target.parent.is_dir():
        # TODO: Says the output directory doesn't exist and to deliberately
        # create it, then rerun.
        raise UserError(
            f"Directorium output non exsistit: {target.parent}\n"
            "Crea directorium consulto, deinde iterum curre."
        )
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent, text=True)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(page)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def no_exchanges_error(repo: Path, roots: Roots) -> UserError:
    sought = "\n".join(f"  {path}" for path in roots.all())
    # TODO: Says no prompts were attributed to this project, lists the
    # transcript roots that were searched, explains the environment
    # variables for nonstandard locations, and notes that VS Code chats
    # attribute correctly only when the project is opened as a plain folder.
    return UserError(
        f"No prompts found: {repo}\n\n"
        f"Radices inspectae:\n{sought}\n\n"
        "Si transcripta alibi sunt, variabiles AI_CHAT_CLAUDE_ROOTS, "
        "AI_CHAT_CODEX_ROOTS, AI_CHAT_VSCODE_USER_ROOTS, vel AI_CHAT_ANTIGRAVITY_ROOTS constitue.\n"
        "VS Code: inceptum ipsum ut folder aperi, non workspace multiplex."
    )


def run(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    environment = os.environ if env is None else env
    try:
        options = parse_args(args)
        repo = canonical_repo(options.repo)
        remote = repo_remote(repo)
        project = project_name(repo, remote)
        roots = discover_roots(environment)
        login = github_login()
        fresh, holdings = collect(repo, roots)
        credits = page_credits(options.output)
        ledgers, citations = read_ledgers(options.output, repo, remote)
        check_ledgers(options.output, login, ledgers, credits, holdings)
        # The store is the truth for every record it still holds and the
        # runner's ledger for the rest: a ledger copy of a record the store
        # still has is dropped (the store's reading wins, so parser fixes
        # purge stale copies), a ledger copy of a record the store lost is
        # kept (nothing rendered is ever lost). Fresh first, so weave keeps
        # the store's copy. The runner's ledger is absent until their first
        # run: then it holds nothing.
        old = ledgers.get(login, [])
        kept = [exchange for exchange in old if holding(exchange) not in holdings]
        mine = weave(fresh + kept)
        ledgers = dict(sorted({**ledgers, login: mine}.items()))
        # Two readings of one prompt are sought only now, among the rows
        # this run would write and render: two the runner's ledger held of
        # a prompt the runner's stores hold were both just replaced by the
        # stores' reading, so a rerun on the machine whose transcripts hold
        # the prompt's session heals them. A row read from the stores has no
        # ledger line yet, so it is cited by its transcript. Two readings
        # both read from the stores are sought first, among the stores' rows
        # alone: no ledger line holds either, and no run elsewhere settles
        # them, so their advice is their own.
        sources = {exchange: str(exchange.source) for exchange in fresh}
        # TODO: Says both readings were read from this machine's own
        # transcripts, which disagree about the prompt, and that if one file
        # is a stale copy of the other, to move the copy out of the
        # transcript store, then rerun; if not, as when one file is cited
        # twice (it holds both readings), sourcery does not decide which
        # reading to believe: the case is to be examined and the script
        # updated.
        check_readings(
            {login: weave(fresh)},
            sources,
            "Ambae lectiones e transcriptis huius machinae lectae sunt, quae de rogatione dissentiunt. "
            "Si alter fasciculus alterius exemplar obsoletum est, exemplar e reposito transcriptorum alio move, "
            "deinde iterum curre. Sin minus, ut cum unus fasciculus bis citatur, utri lectioni credendum sit "
            "sourcery non decernit: casus inspiciendus, scriptum renovandum est.",
        )
        # TODO: Says to delete the stale line, then rerun; or to rerun on
        # the machine whose transcripts hold that session.
        check_readings(
            ledgers,
            {**citations, **sources},
            "Lineam obsoletam dele, deinde iterum curre; aut iterum curre in machina cuius "
            "transcripta illam sessionem tenent.",
        )
        if not any(ledgers.values()):
            raise no_exchanges_error(repo, roots)
        # The page is rendered before anything is written, so a refusal found
        # while rendering it writes nothing.
        page = render(repo, ledgers, remote)
        ledger = ledger_path(options.output, login)
        write_output(ledger, ledger_text(project, mine))
        write_output(options.output, page)
        output = options.output.resolve()
        # TODO: Reports success with each path written (the runner's
        # ledger, then the page), the number of exported prompts, and how
        # many of them each human's ledger holds, by login.
        print(
            f"Written: {ledger.resolve()}\nWritten: {output}\nPrompts: {sum(map(len, ledgers.values()))}"
            + "".join(f"\n  {who}: {len(rows)}" for who, rows in ledgers.items())
        )
        # A ledger prompt whose record the store still holds but no longer
        # reads as a prompt was dropped by the merge above. It is named here,
        # every run, so a parser change that dropped typed words by mistake
        # cannot pass unseen: the diff of the ledger shows the loss, this
        # line says why.
        # Prompts are counted by prompt_key, so two readings the ledger
        # held of one prompt count as the one prompt they read.
        yielded = {holding(exchange) for exchange in fresh}
        dropped = sorted(
            {prompt_key(exchange) for exchange in old if holding(exchange) in holdings and holding(exchange) not in yielded}
        )
        # Reports how many prompts the runner's ledger held that the store
        # now reads as machine text, each named by agent and time.
        print(f"Prompts deleted: {len(dropped)}" + "".join(f"\n  {provider} {when.isoformat()}" for provider, when, _, _ in dropped))
        if options.open_after and not webbrowser.open(output.as_uri()):
            # Says the browser refused to open the file.
            raise UserError(f"Browser failed to open: {output}")
        return 0
    except ExitMessage as exc:
        print(exc)
        return 0
    except UserError as exc:
        print(f"Error:\n{exc}", file=sys.stderr)
        return 2


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
