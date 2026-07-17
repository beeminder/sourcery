#!/usr/bin/env python3
"""Export the human side of a repo's AI coding dialogue to one HTML page.

Reads the local transcript stores of three coding agents and produces a
single self-contained HTML document, ordered by timestamp. The human's
prompts are the only text visible by default; everything machine-generated
is collapsed behind a quiet disclosure line that names the agent, model,
and time. Supported stores:

- Claude Code:  ~/.claude/projects/**/*.jsonl
- Codex:        ~/.codex/{sessions,archived_sessions}/**/*.jsonl
- Copilot Chat: VS Code User/workspaceStorage/*/chatSessions/*.{json,jsonl}

Usage:
    python3 sourcery.py --repo /path/to/project [--output out.html] [--open]

Nonstandard store locations can be supplied with path-separated environment
variables: AI_CHAT_CLAUDE_ROOTS, AI_CHAT_CODEX_ROOTS, AI_CHAT_VSCODE_USER_ROOTS.

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
import dataclasses
import datetime as dt
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

VERSION = "4.4.0"
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
    ballots: tuple[Ballot, ...] = ()  # multiple-choice questions the human answered


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


@dataclasses.dataclass(frozen=True)
class Roots:
    claude: tuple[Path, ...]
    codex: tuple[Path, ...]
    vscode: tuple[Path, ...]

    def all(self) -> tuple[Path, ...]:
        return self.claude + self.codex + self.vscode


@dataclasses.dataclass(frozen=True)
class Options:
    repo: Path
    output: Path
    open_after: bool


# ------------------------------------------------------------------- plumbing


def help_text() -> str:
    return f"""Usage:
  python3 sourcery.py --repo PATH [--output FILE.html] [--open]

Arguments:
  --repo PATH     Directory containing the project to extract the dialog from.
  --output FILE   Generate new html.
  --open          Open the generated HTML in the browser.
  -h, --help      This help text.
  --version       Show version.

Environment variables (separated by {os.pathsep!r}) for nonstandard transcript locations:
  AI_CHAT_CLAUDE_ROOTS
  AI_CHAT_CODEX_ROOTS
  AI_CHAT_VSCODE_USER_ROOTS
"""


def parse_args(argv: Sequence[str]) -> Options:
    values: dict[str, str] = {}
    flags: set[str] = set()
    i = 0
    while i < len(argv):
        token = argv[i]
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
                i += 1
            case "--repo" | "--output":
                if token in values:
                    # TODO: Says this option was given more than once.
                    raise UserError(f"Argumentum {token} iteratum est.")
                if i + 1 >= len(argv):
                    # TODO: Says this option requires a path argument.
                    raise UserError(f"Argumentum {token} viam postulat.\n\n{help_text()}")
                values[token] = argv[i + 1]
                i += 2
            case _:
                # TODO: Says this option is not recognized.
                raise UserError(f"Argumentum ignotum: {token}\n\n{help_text()}")
    if "--repo" not in values:
        # TODO: Says the --repo option is required.
        raise UserError(f"Argumentum --repo necessarium est.\n\n{help_text()}")
    repo = Path(values["--repo"]).expanduser()
    output = Path(values.get("--output", f"{repo.name}-ai-dialogus.html")).expanduser()
    return Options(repo=repo, output=output, open_after="--open" in flags)


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
    """Normalize a git remote URL (https, ssh, or scp-style) to https, or ""."""
    for prefix in ("https://", "http://"):
        if url.startswith(prefix):
            return url.removesuffix(".git")
    scp = re.fullmatch(r"(?:ssh://)?git@([^:/]+)[:/](.+?)(?:\.git)?/?", url)
    return f"https://{scp.group(1)}/{scp.group(2)}" if scp else ""


def repo_remote(repo: Path) -> str:
    """Return the repo's origin remote as an https URL, or "" when absent."""
    config = repo / ".git" / "config"
    if not config.is_file():
        return ""
    url = ""
    section = ""
    for line in config.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            section = stripped
        elif section == '[remote "origin"]':
            key, _, value = stripped.partition("=")
            if key.strip() == "url":
                url = value.strip()
    return https_remote(url)


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


# --------------------------------------------------------------- Claude Code

# Wrapper text items the Claude Code harness injects into the user role:
# <ide_opened_file>, <ide_selection>, and <system-reminder> blocks. A text
# item is a wrapper only when the tag spans the entire item.
CLAUDE_WRAPPER = re.compile(r"<(ide_[a-z_]+|system-reminder)>.*</\1>\s*", re.DOTALL)

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
            raise UserError(
                f"Responsa interrogationum in forma ignota: {path}:{line_number}\n"
                "Structura answers mutata videtur; claude_recovered renovandum est."
            )
        case _:
            return "", ()


def claude_prompt(content: Any) -> str:
    """Return the human-typed text of a user record, or "" if none survives."""
    match content:
        case str():
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


# TODO: Identify a pasted image stored in an unrecognized form rather than
# silently dropping it.
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
                raise UserError(f"Imago in forma ignota: {path}:{line_number}")
            uris.append(f"data:{media};base64,{data}")
    return tuple(uris)


def claude_tool_seconds(record: Mapping[str, Any]) -> float:
    result = record.get("toolUseResult")
    result = result if isinstance(result, dict) else {}
    for key, scale in (("durationMs", 1000.0), ("durationSeconds", 1.0)):
        value = result.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value / scale
    return 0.0


def claude_exchanges(path: Path, repo: Path) -> list[Exchange]:
    threads: dict[str, list[Message]] = {}
    # Agent working time, distinguished from waiting-for-human time by where
    # a timestamp gap ends: a gap ending at an assistant record is generation;
    # a gap ending at a tool_result hides both tool runtime and permission
    # waits, so only a duration the store itself recorded is credited there.
    # `pending` accrues per session until the next emitted message.
    previous: dict[str, dt.datetime] = {}
    pending: dict[str, float] = {}
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
            raise UserError(f"Recordum message obiectum non habet: {path}:{line_number}")
        if not under_dir(record.get("cwd"), repo):
            continue
        if "timestamp" not in record:
            raise UserError(f"Tempus deest in recordo: {path}:{line_number}")
        timestamp = parse_time(record["timestamp"])
        session = str(record.get("sessionId") or path.stem)
        gap = (timestamp - previous[session]).total_seconds() if session in previous else 0.0
        previous[session] = timestamp
        if role == "assistant":
            pending[session] = pending.get(session, 0.0) + max(gap, 0.0)
        else:
            pending[session] = pending.get(session, 0.0) + claude_tool_seconds(record)
        if role == "user":
            typed_source = attachment if attachment is not None else record
            content = attachment.get("prompt") if attachment is not None else message.get("content")
            if claude_origin_is_machine(typed_source, path, line_number):
                continue
            text = claude_prompt(content)
            ballots: tuple[Ballot, ...] = ()
            if text == "" and attachment is None:
                text, ballots = claude_recovered(record, path, line_number)
            if CLAUDE_CANNED.fullmatch(text):
                continue
            images = claude_images(content, path, line_number)
            if text == "" and images == () and not any(b.picked for b in ballots):
                continue
            item = Message(role, timestamp, text, images=images, ballots=ballots)
            pending[session] = 0.0
        else:
            model = message.get("model") if isinstance(message.get("model"), str) else ""
            if model == "<synthetic>":
                continue  # harness notice (auth/API errors), not model output
            text = "\n\n".join(claude_reply_blocks(message.get("content")))
            if text == "":
                continue
            effort = record.get("effort") if isinstance(record.get("effort"), str) else ""
            item = Message(role, timestamp, text, model, effort, active=pending.pop(session, 0.0))
        threads.setdefault(session, []).append(item)
    return [
        exchange
        for session, messages in threads.items()
        for exchange in paired("Claude Code", session, messages, path)
    ]


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
        raise UserError(
            f"Involucrum IDE sine rogatione inventum est: {path}\n"
            "Forma transcripti Codex mutata videtur; fasciculum inspice."
        )
    return request.removesuffix("\n")


# TODO: Identify a session whose source marker is unrecognized so new Codex
# session kinds are classified deliberately instead of guessed at.
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
            raise UserError(
                f"Fons sessionis Codex ignotus: {source!r} in {path}\n"
                "Adde hunc fontem consulto in codex_include_session."
            )


# TODO: Identify a pasted image stored in an unrecognized form rather than
# silently dropping it; Codex stores them as ready-made data: URIs.
def codex_images(payload: Mapping[str, Any], path: Path, line_number: int) -> tuple[str, ...]:
    images = payload.get("images") or []
    for image in images:
        if not (isinstance(image, str) and image.startswith("data:")):
            raise UserError(f"Imago in forma ignota: {path}:{line_number}")
    return tuple(images)


# Payload kinds that mark the model actively producing output. A timestamp
# gap counts as working time only when it ends at one of these; gaps ending
# anywhere else hide tool runtime, system sleep, approval waits, or idle time
# between turns (tool outputs, task_started, turn boundaries, user messages).
CODEX_WORKING = frozenset({
    "message", "reasoning", "function_call", "custom_tool_call", "web_search_call",
    "agent_message", "agent_reasoning", "token_count", "task_complete",
})


def codex_exchanges(path: Path, repo: Path) -> list[Exchange]:
    session = path.stem
    cwd: Any = ""
    model = ""
    effort = ""
    messages: list[Message] = []
    previous: dt.datetime | None = None
    pending = 0.0
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
                    return []
                session = str(payload.get("id") or session)
                cwd = payload.get("cwd") or cwd
            case "turn_context":
                cwd = payload.get("cwd") or cwd
                model = payload.get("model") if isinstance(payload.get("model"), str) else model
                effort = payload.get("effort") if isinstance(payload.get("effort"), str) else effort
            case "event_msg":
                kind = payload.get("type")
                if kind not in {"user_message", "agent_message"}:
                    continue
                if not under_dir(cwd, repo):
                    continue
                text = payload.get("message")
                if not isinstance(text, str):
                    raise UserError(f"Nuntius sine textu: {path}:{line_number}")
                if "timestamp" not in record:
                    raise UserError(f"Tempus deest in recordo: {path}:{line_number}")
                timestamp = parse_time(record["timestamp"])
                if kind == "user_message":
                    prompt = cut_canned_tail(codex_prompt(text, path))
                    images = codex_images(payload, path, line_number)
                    pending = 0.0
                    if (prompt == "" and images == ()) or prompt.startswith(CODEX_CANNED_PREFIX):
                        continue
                    messages.append(Message("user", timestamp, prompt, images=images))
                else:
                    messages.append(Message("assistant", timestamp, text, model, effort, active=pending))
                    pending = 0.0
            case _:
                continue
    return paired("Codex", session, messages, path)


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


# TODO: Identify a pasted image stored in an unrecognized form rather than
# silently dropping it; Copilot serializes the raw bytes as an object with
# numeric-string keys plus a mimeType.
def copilot_image_uri(variable: Mapping[str, Any], path: Path) -> str:
    value = variable.get("value")
    mime = variable.get("mimeType")
    if not (isinstance(value, dict) and isinstance(mime, str)):
        raise UserError(f"Imago in forma ignota: {path}")
    try:
        data = bytes(value[str(i)] for i in range(len(value)))
    except (KeyError, TypeError, ValueError) as exc:
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


# TODO: Identify an unrecognized response-item kind so new VS Code chat
# machinery is classified deliberately instead of silently dropped.
def vscode_reply(response: Any, path: Path) -> str:
    parts: list[str] = []
    for item in response if isinstance(response, list) else []:
        if not isinstance(item, dict):
            raise UserError(f"Membrum responsi obiectum non est: {path}")
        kind = item.get("kind")
        if kind is None:
            value = item.get("value")
            if not isinstance(value, str):
                raise UserError(f"Membrum responsi sine textu: {path}")
            parts.append(value)
        elif kind == "inlineReference":
            parts.append(vscode_reference_name(item))
        elif kind not in VSCODE_MUTE_KINDS:
            raise UserError(
                f"Genus membri responsi ignotum: {kind!r} in {path}\n"
                "Adde hoc genus consulto in VSCODE_MUTE_KINDS vel vscode_reply."
            )
    return "".join(parts)


# TODO: Identify a malformed VS Code mutation entry instead of repairing or
# skipping it; sessions are stored as a kind-0 snapshot followed by
# set (1), list-splice (2), and delete (3) mutations.
def replay_mutations(path: Path) -> Any:
    state: Any = None
    for line_number, entry in read_jsonl(path):
        kind = entry.get("kind")
        if kind == 0:
            state = entry.get("v")
            continue
        keys = entry.get("k")
        if state is None or not isinstance(keys, list) or keys == []:
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
                    raise UserError(f"Genus mutationis ignotum: {kind!r} in {path}:{line_number}")
        except (KeyError, IndexError, TypeError) as exc:
            raise UserError(f"Mutatio VS Code non applicari potest: {path}:{line_number}\n{exc}") from exc
    return state


def vscode_state(path: Path) -> dict[str, Any]:
    state = read_json(path) if path.suffix == ".json" else replay_mutations(path)
    if not isinstance(state, dict):
        raise UserError(f"Sessio VS Code obiectum finale non habet: {path}")
    return state


# TODO: Identify an unsupported session schema version rather than guessing
# at a changed storage format.
def vscode_exchanges(path: Path, workspace: tuple[Path, ...], repo: Path) -> list[Exchange]:
    inside = tuple(root for root in workspace if under_dir(str(root), repo))
    if inside == ():
        return []
    if len(inside) != len(workspace):
        raise UserError(
            f"Workspace multiplex intra et extra inceptum est: {path}\n"
            "Aperi inceptum ut folder simplex, vel exporta hunc dialogum manu."
        )
    state = vscode_state(path)
    if state.get("version") != 3:
        raise UserError(
            f"Versio sessionis VS Code ignota: {state.get('version')!r} in {path}\n"
            "Forma repositi mutata videtur; scriptum renovandum est."
        )
    requests = state.get("requests")
    if not isinstance(requests, list):
        raise UserError(f"Sessio VS Code indicem requests non habet: {path}")
    session = str(state.get("sessionId") or path.stem)
    exchanges: list[Exchange] = []
    for request in requests:
        if not isinstance(request, dict):
            raise UserError(f"Elementum requests obiectum non est: {path}")
        message = request.get("message")
        prompt = message.get("text") if isinstance(message, dict) else None
        if not isinstance(prompt, str):
            raise UserError(f"Rogatio VS Code textum in message.text non habet: {path}")
        prompt = cut_canned_tail(prompt)
        if prompt == "" or request.get("confirmation") is not None or COPILOT_CANNED.match(prompt):
            continue
        if "timestamp" not in request:
            raise UserError(f"Tempus deest in rogatione: {path}")
        model = request.get("modelId")
        result = request.get("result")
        timings = result.get("timings") if isinstance(result, dict) else None
        total_elapsed = timings.get("totalElapsed") if isinstance(timings, dict) else None
        # A confirmation or elicitation pauses the turn for human input, so
        # totalElapsed would include waiting time; show no duration instead
        # of a wrong one.
        response = request.get("response")
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
                timestamp=parse_time(request["timestamp"]),
                provider="Copilot Chat",
                model=model if isinstance(model, str) else "",
                session=session,
                prompt=prompt,
                reply=vscode_reply(request.get("response"), path),
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
    return exchanges


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


# TODO: Identify an unreadable or structurally invalid .code-workspace file
# and the folder entry that prevents repository attribution.
def read_workspace_file(path: Path) -> tuple[Path, ...]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise UserError(f"Fasciculus workspace legi non potest: {path}\n{exc}") from exc
    try:
        data = json.loads(strip_jsonc(raw))
    except json.JSONDecodeError as exc:
        raise UserError(f"Fasciculus workspace invalidus est: {path}\n{exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("folders"), list):
        raise UserError(f"Fasciculus workspace indicem folders non habet: {path}")
    roots: list[Path] = []
    for entry in data["folders"]:
        if not isinstance(entry, dict):
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


# TODO: Identify malformed VS Code workspace metadata rather than guessing
# which repository owns a chat session.
def workspace_roots(storage: Path) -> tuple[Path, ...]:
    metadata = storage / "workspace.json"
    if not metadata.exists():
        return ()
    data = read_json(metadata)
    if not isinstance(data, dict):
        raise UserError(f"workspace.json obiectum non est: {metadata}")
    folder = data.get("folder")
    if isinstance(folder, str):
        decoded = decode_file_uri(folder)
        return (Path(decoded).expanduser().resolve(),) if decoded is not None else ()
    pointer = data.get("workspace") or data.get("configuration")
    if isinstance(pointer, str):
        decoded = decode_file_uri(pointer)
        return read_workspace_file(Path(decoded).expanduser().resolve()) if decoded is not None else ()
    raise UserError(f"workspace.json nec folder nec workspace habet: {metadata}")


# ------------------------------------------------------- pairing and weaving


def paired(
    provider: str,
    session: str,
    messages: Iterable[Message],
    source: Path,
) -> list[Exchange]:
    """Fold a role-ordered message stream into prompt-plus-reply exchanges."""
    exchanges: list[Exchange] = []
    current: Exchange | None = None
    reply_parts: list[str] = []
    model = ""
    effort = ""

    active = 0.0

    def flush() -> None:
        if current is not None:
            exchanges.append(
                dataclasses.replace(
                    current,
                    reply="\n\n".join(reply_parts),
                    model=model,
                    effort=effort,
                    elapsed=active,
                )
            )

    for message in messages:
        match message.role:
            case "user":
                # One typed act can fan out across several records (a denial
                # reason stamped onto each rejected parallel tool call), so a
                # repeat of the pending unanswered prompt is not a new prompt.
                if (
                    current is not None
                    and reply_parts == []
                    and (current.prompt, current.images, current.ballots)
                    == (message.text, message.images, message.ballots)
                ):
                    continue
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
            case "assistant":
                if current is None:
                    continue  # reply to a dropped machine prompt; nothing to attach to
                reply_parts.append(message.text)
                model = message.model or model
                effort = message.effort or effort
                active += message.active
            case _:
                raise AssertionError(message.role)
    flush()
    return exchanges


def weave(exchanges: Iterable[Exchange]) -> list[Exchange]:
    """Merge all providers into one chronology, collapsing exact duplicates
    (resumed or forked sessions replay identical records into new files)."""
    unique: dict[tuple[str, dt.datetime, str, tuple[Ballot, ...], str], Exchange] = {}
    for exchange in exchanges:
        unique.setdefault(
            (exchange.provider, exchange.timestamp, exchange.prompt, exchange.ballots, exchange.reply),
            exchange,
        )
    return sorted(unique.values(), key=lambda e: (e.timestamp, e.provider, e.session, e.prompt))


# TODO: Identify a configured transcript root that exists but is not a
# directory.
def require_dir(root: Path) -> bool:
    if not root.exists():
        return False
    if not root.is_dir():
        raise UserError(f"Radix transcriptuum directorium non est: {root}")
    return True


def codex_session_dirs(root: Path) -> tuple[Path, ...]:
    if root.name in {"sessions", "archived_sessions"}:
        return (root,)
    return tuple(root / name for name in ("sessions", "archived_sessions"))


def collect(repo: Path, roots: Roots) -> list[Exchange]:
    exchanges: list[Exchange] = []
    for root in roots.claude:
        if require_dir(root):
            for path in sorted(p for p in root.rglob("*.jsonl") if p.is_file()):
                exchanges.extend(claude_exchanges(path, repo))
    for root in roots.codex:
        if require_dir(root):
            for directory in codex_session_dirs(root):
                if directory.is_dir():
                    for path in sorted(p for p in directory.rglob("*.jsonl") if p.is_file()):
                        exchanges.extend(codex_exchanges(path, repo))
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
                exchanges.extend(vscode_exchanges(path, workspace, repo))
    return weave(exchanges)


# ------------------------------------------------------------------ rendering


def markdown_inline(text: str) -> str:
    """Render a deliberately small, inert subset of inline Markdown."""
    escaped = html.escape(text, quote=False)
    stashed: list[str] = []

    def stash(match: re.Match[str]) -> str:
        token = f"{len(stashed)}"
        stashed.append(f"<code>{match.group(2)}</code>")
        return token

    escaped = re.sub(r"(`+)(.+?)\1", stash, escaped)
    escaped = re.sub(
        r"\[([^\]]+)\]\((https?://[^\s)]+|mailto:[^\s)]+)\)",
        lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{m.group(1)}</a>',
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
# faint; every machine-written word sits inside a closed <details>.
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
.exchange { margin: 2.6rem 0 0; }
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
/* INVIOLABLE: machine-generated prose renders only inside a .machine
   container, in the phosphor-terminal style — monospace green on
   near-black, in both color schemes — for maximal distinction from the
   human's serif. The containers are .reply (agent replies), .asked
   (questions the agent posed), and .chosen (option labels the agent wrote
   and the human clicked). */
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
  .machine { --m-bg: white; --m-ink: #14572e; --m-bright: #14572e; --m-dim: #4d7a5d; --m-line: #9dbfa8; background: white; border: 1px solid var(--m-line); }
  .reply pre.code { background: white; }
  main { width: 100%; padding: 0; }
  .day { break-after: avoid; }
  .exchange { break-inside: avoid; }
  details > .reply { display: block !important; }
  summary::before { content: ""; }
}
"""


WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def elapsed_text(seconds: float) -> str:
    total = round(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return "".join(f"{n}{unit}" for n, unit in ((hours, "h"), (minutes, "m"), (secs, "s")) if n) or "0s"


# TODO: The deck line under the title states the number of human prompts and
# the date range. The collapsed line under a prompt with no recorded reply
# says the response was not found. The "thought for 5m27s" phrase (exact copy
# specified by the human) conveys how long the agent worked on the reply.
# The controls line lets the reader open or close every reply at once:
# "aperi omnia" = open all, "claude omnia" = close all. All other rendered
# text (repository name/path/remote, provider, model, timestamps, weekdays,
# prompts, replies) is source data.
def render(repo: Path, exchanges: Sequence[Exchange], remote: str = "") -> str:
    assert exchanges, "render() requires at least one exchange"
    locals_ = [e.timestamp.astimezone() for e in exchanges]
    first, last = locals_[0].date().isoformat(), locals_[-1].date().isoformat()
    count = len(exchanges)
    noun = "prompt" if count == 1 else "prompts"
    range_text = first if first == last else f"{first} – {last}"

    chunks: list[str] = []
    current_day = None
    for number, (exchange, local) in enumerate(zip(exchanges, locals_), start=1):
        day = local.date().isoformat()
        if day != current_day:
            weekday = WEEKDAYS[local.date().weekday()]
            chunks.append(f'<h2 class="day"><time datetime="{day}">{day} {weekday}</time></h2>')
            current_day = day
        model = f' <span class="model">{html.escape(exchange.model)}</span>' if exchange.model else ""
        effort = f' <span class="effort">({html.escape(exchange.effort)})</span>' if exchange.effort else ""
        # TODO: The final exchange of the export is the only one that can
        # plausibly still be in flight, so an empty reply there says the
        # response was still generating at capture time (exact copy specified
        # by the human); an empty reply anywhere else says there is none.
        if exchange.reply == "" and number == count:
            reply = (
                '<div class="reply machine empty">'
                "<p>Response still generating when this transcript was captured</p></div>"
            )
        elif exchange.reply == "":
            reply = '<div class="reply machine empty" lang="la"><p>No response.</p></div>'
        else:
            reply = f'<div class="reply machine">{markdown_html(exchange.reply)}</div>'
        thought = (
            f' <span class="elapsed">thought for {elapsed_text(exchange.elapsed)}</span>'
            if exchange.elapsed >= 0.5
            else ""
        )
        summary = (
            f'<a class="anchor" href="#p{number}">'
            f'<time datetime="{exchange.timestamp.isoformat()}">{local.strftime("%H:%M")}</time></a>'
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
            f'<article class="exchange" id="p{number}">'
            f"{ballots}{prompt}{attachments}\n"
            f"<details>\n<summary>{summary}</summary>\n{reply}\n</details>\n"
            "</article>"
        )

    body = "\n".join(chunks)
    title = html.escape(repo.name)
    # The subtitle answers "where this lives": the public remote when there
    # is one, the local directory otherwise.
    if remote:
        label = html.escape(remote.removeprefix("https://").removeprefix("http://"))
        where = f'<a href="{html.escape(remote, quote=True)}">{label}</a>'
    else:
        where = html.escape(str(repo))
    return f"""<!doctype html>
<html lang="und">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Ctext y='14' font-size='14'%3E%F0%9F%90%9D%3C/text%3E%3C/svg%3E">
<meta name="description" content="{count} {noun} · {range_text}">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{count} {noun} · {range_text}">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<main>
<header class="masthead">
  <h1>{title}</h1>
  <p class="deck" lang="la">{count} {noun} · {range_text}</p>
  <p class="repo-path">{where}</p>
  <p class="controls" lang="la"><a href="#" data-omnia="open">expand all</a> · <a href="#" data-omnia="close">collapse all</a></p>
</header>
{body}
</main>
<script>
for (const control of document.querySelectorAll("[data-omnia]"))
  control.addEventListener("click", (event) => {{
    event.preventDefault();
    const open = control.dataset.omnia === "open";
    for (const details of document.querySelectorAll("details")) details.open = open;
  }});
</script>
</body>
</html>
"""


# ----------------------------------------------------------------------- exit


# TODO: Explain that output is never overwritten and its parent directory
# must already exist; write atomically so a partial document is never left
# behind.
def write_output(path: Path, page: str) -> None:
    target = path.resolve()
    if target.exists():
        raise UserError(
            f"Fasciculus iam exsistit: {target}\n"
            "Elige aliam viam cum --output, vel fasciculum veterem consulto remove."
        )
    if not target.parent.is_dir():
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
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise UserError(
                f"Fasciculus inter scribendum creatus est: {target}\n"
                "Fasciculus novus non superscriptus est; aliam viam cum --output elige."
            ) from exc
        except OSError as exc:
            raise UserError(
                f"Fasciculus scribi non potest: {target}\n{exc}\n"
                "Systema fasciculorum vincula dura non fert; aliam viam cum --output elige."
            ) from exc
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


# TODO: Explain exactly where transcript roots were sought and how to
# configure nonstandard roots when no prompt belongs to the repository.
def no_exchanges_error(repo: Path, roots: Roots) -> UserError:
    sought = "\n".join(f"  {path}" for path in roots.all())
    return UserError(
        f"Nullae rogationes huic incepto attributae sunt: {repo}\n\n"
        f"Radices inspectae:\n{sought}\n\n"
        "Si transcripta alibi sunt, variabiles AI_CHAT_CLAUDE_ROOTS, "
        "AI_CHAT_CODEX_ROOTS, vel AI_CHAT_VSCODE_USER_ROOTS constitue.\n"
        "VS Code: inceptum ipsum ut folder aperi, non workspace multiplex."
    )


# TODO: Report success with the exact output path and the number of exported
# prompts; report actionable failures without leaving a partial document.
def run(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    environment = os.environ if env is None else env
    try:
        options = parse_args(args)
        repo = canonical_repo(options.repo)
        roots = discover_roots(environment)
        exchanges = collect(repo, roots)
        if exchanges == []:
            raise no_exchanges_error(repo, roots)
        write_output(options.output, render(repo, exchanges, repo_remote(repo)))
        output = options.output.resolve()
        print(f"Scriptum: {output}\nRogationes: {len(exchanges)}")
        if options.open_after and not webbrowser.open(output.as_uri()):
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
