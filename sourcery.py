#!/usr/bin/env python3
"""Export the human side of a repo's AI coding dialogue to one HTML page.

Reads the local transcript stores of four coding agents and produces a
single self-contained HTML document, ordered by timestamp. The human's
prompts are the only text visible by default; everything machine-generated
is collapsed behind a quiet disclosure line that names the agent, model,
and time. Supported stores:

- Claude Code:  ~/.claude/projects/**/*.jsonl
- Codex:        ~/.codex/{sessions,archived_sessions}/**/*.jsonl
- Copilot Chat: VS Code User/workspaceStorage/*/chatSessions/*.{json,jsonl}
- Antigravity:  ~/.gemini/antigravity/conversations/*.pb (sealed protobuf;
                opened through the system cipher, macOS only)

Usage:
    python3 sourcery.py REPODIR OUTPUT.html [--open]

OUTPUT.html is read before it is written: the page carries a snapshot of
every exchange it shows, and a run merges what the stores still hold with
what only the page remembers, so nothing once rendered is lost when stores
are pruned or machines change. A page from before snapshots is refused
until unrender.py has imported it.

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
import sys
import tempfile
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

VERSION = "5.5.0"
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
# record the store lost (keep the page's copy) from one the store still has
# (the store's reading wins, purging stale page copies). Tool plumbing that
# carries no typing is never a holding: it was never rendered, so it must
# never purge a page copy that merely shares its millisecond.
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


def under_dir(value: Any, root: Path) -> bool:
    if not isinstance(value, str) or value == "":
        return False
    return Path(value).expanduser().resolve().is_relative_to(root)


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
# (background-task notifications, so far) are never typing; an origin kind
# that is neither human nor known-machine means the harness grew a new
# record source that must be classified deliberately. Records without an
# origin predate the field and are classified by content instead.
CLAUDE_HUMAN_ORIGINS = frozenset({"human"})
CLAUDE_MACHINE_ORIGINS = frozenset({"task-notification"})


def claude_origin_is_machine(record: Mapping[str, Any], path: Path, line_number: int) -> bool:
    origin = record.get("origin")
    kind = origin.get("kind") if isinstance(origin, dict) else None
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


def claude_reply_blocks(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []
    return [
        item["text"]
        for item in content
        if isinstance(item, dict)
        and item.get("type") == "text"
        and isinstance(item.get("text"), str)
        and item["text"] != ""
    ]


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
    if not isinstance(result, dict) or not under_dir(result.get("filePath"), repo):
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


def claude_exchanges(path: Path, repo: Path) -> tuple[list[Exchange], set[Holding]]:
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
        if not under_dir(record.get("cwd"), repo):
            continue
        if "timestamp" not in record:
            # TODO: Says this record has no timestamp.
            raise UserError(f"Tempus deest in recordo: {path}:{line_number}")
        timestamp = parse_time(record["timestamp"])
        session = str(record.get("sessionId") or path.stem)
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
            typed_source = attachment if attachment is not None else record
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
            if CLAUDE_CANNED.fullmatch(text):
                holdings.add(("Claude Code", timestamp))
                continue
            images = claude_images(content, path, line_number)
            if text == "" and images == () and not any(b.picked for b in ballots):
                continue  # tool plumbing with no typing: never rendered, so no holding
            holdings.add(("Claude Code", timestamp))
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
            text = "\n\n".join(claude_reply_blocks(message.get("content")))
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
                        if under_dir(changed, repo):
                            grew = codex_tally(change, path, line_number)
                            tally = (tally[0] + grew[0], tally[1] + grew[1])
                if kind not in {"user_message", "agent_message"}:
                    continue
                if not under_dir(cwd, repo):
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
    inside = tuple(root for root in workspace if under_dir(str(root), repo))
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
    if folder is None or not under_dir(folder, repo):
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


def weave(exchanges: Iterable[Exchange]) -> list[Exchange]:
    """Merge all providers into one chronology, collapsing exact duplicates
    (resumed or forked sessions replay identical records into new files)."""
    unique: dict[Exchange, Exchange] = {}
    for exchange in exchanges:
        identity = dataclasses.replace(exchange, session="", source=Path())
        unique.setdefault(identity, exchange)
    return sorted(unique.values(), key=lambda e: (e.timestamp, e.provider, e.session, e.prompt))


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

    for root in roots.claude:
        if require_dir(root):
            for path in sorted(p for p in root.rglob("*.jsonl") if p.is_file()):
                gather(claude_exchanges(path, repo))
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
.day {
  clear: both;
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
.exchange { margin: 2.6rem 0 0; clear: both; border-left: 3px solid var(--provider); padding-left: 0.85rem; }
.exchange.claude { --provider: var(--claude); }
.exchange.codex { --provider: var(--codex); }
.exchange.copilot { --provider: var(--copilot); }
.exchange.antigravity { --provider: var(--antigravity); }
/* The prompt's diffstat floats right of the prompt's first lines. Numbers
   wear the metadata ink; polarity lives in the blocks (and in the signs and
   the fixed added-first order). */
.diffstat {
  float: right;
  display: inline-flex;
  align-items: center;
  gap: .5rem;
  margin: .25rem 0 .5rem 1.1rem;
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
.chip { display: inline-block; width: 8px; height: 8px; border-radius: 2px; background: var(--provider); }
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
  clear: right; /* its filled box must not run under a floated diffstat */
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
.exchange > .ballot:first-child { margin-top: 0; }
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
def minimap(exchanges: Sequence[Exchange], locals_: Sequence[dt.datetime]) -> str:
    """A clickable map of the whole dialog: one sliver per prompt, in order,
    linking to its permalink anchor. Added lines rise from the midline and
    deleted lines hang below it, on a square-root scale against the peak so
    small edits stay visible beside the big rewrite spikes; a prompt that
    touched no code is a tick on the midline. Each sliver's hover title is
    its time plus its counts — numerals only, so nothing here needs prose."""
    peak = max(max(e.added, e.deleted) for e in exchanges)
    unit = lambda v: f"{round(v, 1):g}"  # 30.0 -> "30"
    slots: list[str] = []
    for i, (e, local) in enumerate(zip(exchanges, locals_)):
        x = 3 * i
        stamp = local.strftime("%H:%M")
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
}


def render(repo: Path, exchanges: Sequence[Exchange], remote: str = "") -> str:
    assert exchanges, "render() requires at least one exchange"
    locals_ = [e.timestamp.astimezone() for e in exchanges]
    first, last = locals_[0].date().isoformat(), locals_[-1].date().isoformat()
    count = len(exchanges)
    noun = "prompt" if count == 1 else "prompts"
    range_text = first if first == last else f"{first} – {last}"
    total_added = sum(e.added for e in exchanges)
    total_deleted = sum(e.deleted for e in exchanges)
    deck = f"{count} {noun} · {range_text}"
    if (total_added, total_deleted) != (0, 0):
        deck += f" · +{total_added:,} −{total_deleted:,}"

    chunks: list[str] = []
    days: list[tuple[str, str]] = []  # (day, header text), one per day header
    current_day = None
    for number, (exchange, local) in enumerate(zip(exchanges, locals_), start=1):
        day = local.date().isoformat()
        if day != current_day:
            weekday = WEEKDAYS[local.date().weekday()]
            chunks.append(
                f'<h2 class="day" id="d{day}"><time datetime="{day}">{day} {weekday}</time></h2>'
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
            f'<time datetime="{exchange.timestamp.isoformat()}">{local.strftime("%H:%M")}</time></a>'
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
            f'<article class="exchange {PROVIDER_SLUGS[exchange.provider]}" id="p{number}">'
            f"{diffstat(exchange.added, exchange.deleted)}{ballots}{prompt}{attachments}\n"
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
    # it carries its own memory of the dialog (the snapshot at its end),
    # which the next generation reads back and then overwrites, so don't
    # edit it in place.
    return f"""<!doctype html>
<!-- Fasciculus hic a sourcery generatus est, numquam manu scriptus.
     Memoriam dialogi ipse fert, quam generatio proxima legit atque
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
  <p class="deck">{deck}</p>
  <p class="repo-path">{where}</p>
  <p class="controls"><a href="#" data-omnia="open">expand all</a> · <a href="#" data-omnia="close">collapse all</a></p>
  {minimap(exchanges, locals_)}
</header>
{body}
</main>
<script>{JS}</script>
{SNAPSHOT_OPEN}
{snapshot(repo, exchanges)}
</script>
</body>
</html>
"""


# ------------------------------------------------------------------ snapshot


# The page's own memory: every exchange it renders, embedded as JSON so the
# next generation can read it back. Stores get pruned and machines change;
# the page is then the only copy, and nothing it once rendered is lost.
SNAPSHOT_OPEN = '<script type="application/json" id="snapshot">'
SNAPSHOT = re.compile(re.escape(SNAPSHOT_OPEN) + r"\n(.*?)\n</script>", re.DOTALL)


def freeze(exchange: Exchange) -> dict[str, Any]:
    """The exchange as JSON-ready fields, minus the store path: private to
    this machine and irrelevant to the merge, so it never reaches the page."""
    frozen = dataclasses.asdict(exchange)
    del frozen["source"]
    frozen["timestamp"] = exchange.timestamp.isoformat()
    return frozen


# What a frozen exchange must look like, field by field: the JSON type of
# every Exchange field but the omitted source. A missing or extra field, or
# a value of another type, means a snapshot another schema wrote or a hand
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
    """Inverse of freeze; the page is the source. The field set and every
    field's type and range are checked, so a snapshot another schema wrote,
    or one edited by hand, fails loudly instead of thawing into nonsense."""
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


def snapshot(repo: Path, exchanges: Sequence[Exchange]) -> str:
    """One exchange per line, so a refresh diffs as appended lines. Every "<"
    is escaped: the block can then neither end its script element early nor
    open a comment inside it, whatever a prompt or reply contains."""
    rows = ",\n".join(json.dumps(freeze(exchange), ensure_ascii=False) for exchange in exchanges)
    # The version is provenance for a human reading the page; thaw checks the
    # schema itself, so no version is ever refused or trusted on its own.
    text = f'{{"sourcery": {json.dumps(VERSION)}, "repo": {json.dumps(repo.name)}, "exchanges": [\n{rows}\n]}}'
    return text.replace("<", "\\u003c")


def inherit(output: Path, repo: Path) -> list[Exchange]:
    """Everything a previous generation of this page rendered, read back from
    its snapshot; nothing when there is no page yet."""
    if not output.exists():
        return []
    try:
        # Bytes, not text: universal newlines would rewrite a carriage return
        # someone typed, and the page must give back exactly what it holds.
        page = output.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        # TODO: Says the existing output could not be read as a UTF-8 file
        # (it may be a directory, unreadable, or not text), gives the reason,
        # and that nothing was written.
        raise UserError(
            f"Pagina exsistens legi non potest: {output}\n{exc}\nNihil scriptum est."
        ) from exc
    found = SNAPSHOT.findall(page)
    if len(found) != 1:
        # TODO: Says the existing output carries no snapshot (or several), so
        # nothing can be merged into it: either it predates snapshots — run
        # unrender.py on it first — or it is not a sourcery page — choose
        # another output path. Nothing was written.
        raise UserError(
            f"Pagina exsistens memoriam (snapshot) non fert: {output}\n"
            "Aut pagina vetus est — curre prius unrender.py — aut pagina sourcery non est: "
            "elige aliam viam output.\nNihil scriptum est."
        )
    try:
        data = json.loads(found[0])
        name = data["repo"]
        exchanges = [thaw(frozen, output) for frozen in data["exchanges"]]
    except (ValueError, KeyError, TypeError) as exc:
        # TODO: Says the page's snapshot is malformed and nothing was written.
        raise UserError(f"Memoria paginae corrupta est: {output}\nNihil scriptum est.") from exc
    if name != repo.name:
        # TODO: Says the page's snapshot belongs to another project, naming
        # both, and nothing was written.
        raise UserError(
            f"Memoria paginae ad aliud inceptum pertinet: {name!r}, non {repo.name!r}: {output}\n"
            "Nihil scriptum est."
        )
    return exchanges


# ----------------------------------------------------------------------- exit


# Output is written atomically so a partial document is never left behind.
# An existing document has already been read back by inherit() by the time
# this runs — its snapshot is merged into the new page — and is then
# replaced whole: it is always generated, never hand-edited (the page itself
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
        roots = discover_roots(environment)
        fresh, holdings = collect(repo, roots)
        inherited = inherit(options.output, repo)
        # The store is the truth for every record it still holds and the page
        # for the rest: a page copy of a record the store still has is
        # dropped (the store's reading wins, so parser fixes purge stale
        # copies), a page copy of a record the store lost is kept (nothing
        # rendered is ever lost). Fresh first, so weave keeps the store's copy.
        kept = [exchange for exchange in inherited if holding(exchange) not in holdings]
        exchanges = weave(fresh + kept)
        if exchanges == []:
            raise no_exchanges_error(repo, roots)
        write_output(options.output, render(repo, exchanges, repo_remote(repo)))
        output = options.output.resolve()
        # TODO: Reports success with the output path and the number of
        # exported prompts.
        print(f"Written: {output}\nPrompts: {len(exchanges)}")
        if options.open_after and not webbrowser.open(output.as_uri()):
            # TODO: Says the browser refused to open the file.
            raise UserError(f"Navigatrum fasciculum aperire recusavit: {output}")
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
