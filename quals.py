#!/usr/bin/env python3
"""Quals for sourcery.py. Run: python3 quals.py

Each qual replicates a transcript shape observed in the real stores (Claude
Code, Codex, VS Code / Copilot Chat), states the expected extraction, and
lets unittest report what happened instead.
"""

import ast
import binascii
import contextlib
import dataclasses
import datetime as dt
import hashlib
import io
import itertools
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

import adopt
import sourcery as ace

T0 = "2026-03-01T10:00:00.000Z"
T1 = "2026-03-01T10:05:00.000Z"
T2 = "2026-03-01T10:10:00.000Z"
T3 = "2026-03-01T10:15:00.000Z"


def utc(iso: str) -> dt.datetime:
    return dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))


def write_jsonl(path: Path, records: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def exchange(**kw) -> "ace.Exchange":
    base = dict(
        timestamp=utc(T0),
        provider="Claude Code",
        model="claude-opus-4-8",
        session="s",
        prompt="p",
        reply="r",
        source=Path("/x"),
    )
    base.update(kw)
    return ace.Exchange(**base)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()).resolve()
        self.repo = self.tmp / "repo"
        self.repo.mkdir()


# The login every qual credits unless it says otherwise: what CliQuals' gh
# seam reports, and the ledger RenderQuals render from.
LOGIN = "dreeves"


def gh_says(stdout, returncode=0, stderr=""):
    """A stand-in for the gh seam: gh ran, exiting with returncode and
    printing stdout and stderr. Quals never call the real gh."""
    return lambda: subprocess.CompletedProcess(ace.GH_USER, returncode, stdout, stderr)


def gh_missing():
    """A stand-in for the gh seam on a machine without gh."""
    raise FileNotFoundError(2, "No such file or directory", "gh")


def ledger_header(project):
    """A ledger's first line as sourcery writes it: the ledger format, 1,
    and the project name."""
    return {"ledger": 1, "repo": project}


def ledger_text(repo_name, exchanges, header=None):
    """A ledger written by hand, line by line: the header (ledger_header's
    for repo_name unless one is given), then each exchange in freeze()
    form."""
    head = ledger_header(repo_name) if header is None else header
    return "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in [head, *map(ace.freeze, exchanges)])


def ledger_rows(path):
    """A ledger's lines as JSON values, the header first."""
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n"), text[-80:]
    return [json.loads(line) for line in text.removesuffix("\n").split("\n")]


def jsonable(exchange):
    """The exchange's freeze() form as a ledger line gives it back."""
    return json.loads(json.dumps(ace.freeze(exchange)))


def five_page(repo, exchanges, version="5.5.2", name=None):
    """A page as sourcery 5 wrote it: its articles crediting no one and
    naming no human, and before </body> its snapshot, the block 5.5.2's
    snapshot() wrote (every exchange in freeze() form, one per line, every
    "<" escaped), naming the repo `name`, or repo's own name."""
    bare = re.sub(r' data-login="[^"]*"| <span class="human">[^<]*</span>', "", ace.render(repo, {LOGIN: exchanges}))
    rows = ",\n".join(json.dumps(ace.freeze(e), ensure_ascii=False) for e in exchanges)
    named = json.dumps(repo.name if name is None else name)
    snapshot = f'{{"sourcery": {json.dumps(version)}, "repo": {named}, "exchanges": [\n{rows}\n]}}'
    escaped = snapshot.replace("<", "\\u003c")
    head, tail = bare.rsplit("</body>", 1)
    return f'{head}<script type="application/json" id="snapshot">\n{escaped}\n</script>\n</body>{tail}'


def with_snapshot(page, text):
    """A page sourcery 5 wrote, its snapshot's JSON replaced by text."""
    return re.sub(
        r'(<script type="application/json" id="snapshot">\n).*?(\n</script>)',
        lambda found: found[1] + text.replace("<", "\\u003c") + found[2],
        page,
        flags=re.DOTALL,
    )


@contextlib.contextmanager
def on_platform(name):
    """Run as on the platform sys.platform calls name, with every path under
    /home resolving to itself, as a plain directory would: quals never ask
    this machine's automounter to resolve one. Yields the list of every path
    under /home resolved meanwhile."""
    resolved = []
    original = sys.platform, Path.resolve

    def resolve(path, strict=False):
        if path.is_relative_to("/home"):
            resolved.append(path)
            return path
        return original[1](path, strict)

    sys.platform, Path.resolve = name, resolve
    try:
        yield resolved
    finally:
        sys.platform, Path.resolve = original


# ---------------------------------------------------------------- Claude Code

def cu(text_or_content, ts=T0, cwd=None, session="cs1", **extra):
    record = {
        "type": "user",
        "message": {"role": "user", "content": text_or_content},
        "cwd": cwd,
        "sessionId": session,
        "uuid": "u1",
    }
    if ts is not None:
        record["timestamp"] = ts
    record.update(extra)
    return record


def ca(blocks, ts=T1, cwd=None, session="cs1", mid="m1", model="claude-opus-4-8", effort=None):
    record = {
        "type": "assistant",
        "message": {"role": "assistant", "id": mid, "model": model, "content": blocks},
        "timestamp": ts,
        "cwd": cwd,
        "sessionId": session,
        "uuid": "a1",
    }
    if effort is not None:
        record["effort"] = effort
    return record


def send_user_message(message, tid="tu1"):
    """A SendUserMessage tool call: how an agent in a Claude-desktop cloud
    workspace can speak to the human without writing a text block."""
    return {"type": "tool_use", "id": tid, "name": "SendUserMessage", "input": {"message": message}}


def cq(text, ts, cwd, session="cs1", images=(), **attachment):
    """A queued_command attachment in prompt mode: words the human typed
    while the agent was mid-turn, then `images`, its image items. `attachment`
    adds fields to the attachment itself, such as source_uuid and
    delivery_id."""
    assert "prompt" not in attachment
    return {
        "type": "attachment",
        "attachment": {
            "type": "queued_command",
            "commandMode": "prompt",
            "prompt": [{"type": "text", "text": text}, *images],
            "origin": {"kind": "human"},
            "humanTurn": True,
            "timestamp": ts,
            **attachment,
        },
        "timestamp": ts,
        "cwd": cwd,
        "sessionId": session,
        "uuid": "q1",
    }


class ClaudeQuals(Fixture):
    def path(self, records):
        return write_jsonl(self.tmp / "claude" / "p1" / "sess.jsonl", records)

    def test_string_prompt_kept_character_exact(self):
        text = "  two  spaces\n\ttab, trailing blank line\n\n"
        got = ace.claude_exchanges(self.path([cu(text, cwd=str(self.repo))]), self.repo)[0]
        self.assertEqual([e.prompt for e in got], [text])
        self.assertEqual(got[0].provider, "Claude Code")
        self.assertEqual(got[0].timestamp, utc(T0))

    def test_injected_wrapper_items_dropped_human_item_exact(self):
        content = [
            {"type": "text", "text": "<ide_opened_file>The user opened /x.</ide_opened_file>"},
            {"type": "text", "text": "<ide_selection>lines 1-2</ide_selection>"},
            {"type": "text", "text": "<system-reminder>recall</system-reminder>"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA"}},
            {"type": "text", "text": "real prompt, exactly  this"},
        ]
        got = ace.claude_exchanges(self.path([cu(content, cwd=str(self.repo))]), self.repo)[0]
        self.assertEqual([e.prompt for e in got], ["real prompt, exactly  this"])

    def test_wrapper_only_message_yields_nothing(self):
        content = [{"type": "text", "text": "<ide_opened_file>x</ide_opened_file>"}]
        got = ace.claude_exchanges(self.path([cu(content, cwd=str(self.repo))]), self.repo)[0]
        self.assertEqual(got, [])

    def test_tool_results_and_flagged_records_skipped(self):
        cwd = str(self.repo)
        records = [
            cu([{"type": "tool_result", "tool_use_id": "t", "content": "out"}], cwd=cwd),
            cu("sidechain", cwd=cwd, isSidechain=True),
            cu("meta", cwd=cwd, isMeta=True),
            cu("This session is being continued...", cwd=cwd, isCompactSummary=True),
            cu("transcript-only", cwd=cwd, isVisibleInTranscriptOnly=True),
        ]
        self.assertEqual(ace.claude_exchanges(self.path(records), self.repo)[0], [])

    def test_reply_joins_text_blocks_across_streamed_records(self):
        cwd = str(self.repo)
        records = [
            cu("go", cwd=cwd),
            ca([{"type": "thinking", "thinking": "hmm"}], cwd=cwd, mid="mA"),
            ca([{"type": "text", "text": "Part one."}], cwd=cwd, mid="mA"),
            ca([{"type": "tool_use", "id": "t", "name": "Bash", "input": {}}], cwd=cwd, mid="mA"),
            ca([{"type": "text", "text": "Part two."}], ts=T2, cwd=cwd, mid="mB", effort="xhigh"),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].prompt, "go")
        self.assertEqual(got[0].reply, "Part one.\n\nPart two.")
        self.assertEqual(got[0].model, "claude-opus-4-8")
        self.assertEqual(got[0].effort, "xhigh")
        self.assertEqual(got[0].elapsed, 600.0)  # prompt at T0, last reply block at T2

    def test_send_user_message_alone_is_the_reply(self):
        # Replicata: an agent in a Claude-desktop cloud workspace answers
        # only through a SendUserMessage tool call, writing no text block.
        # Expectata: the call's message is the reply, under the record's
        # model. Resultata (v5.5.0): "No response." and no model, because
        # only text blocks were read.
        cwd = str(self.repo)
        records = [
            cu("go", cwd=cwd),
            ca([send_user_message("Here is what I found.")], cwd=cwd, model="claude-opus-5-5"),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.reply, e.model) for e in got],
            [("go", "Here is what I found.", "claude-opus-5-5")],
        )

    def test_text_and_send_user_message_join_in_content_order(self):
        # Replicata: one assistant record interleaves text blocks with a
        # SendUserMessage call and a SendMessage call (agent to agent, with
        # a message field of its own); the next two records carry one
        # SendUserMessage call and one text block. Expectata: every text
        # block and SendUserMessage message, joined in content order across
        # the records; SendMessage, like every other tool call, says nothing
        # to the human. Resultata (v5.5.0): both SendUserMessage messages
        # were missing.
        cwd = str(self.repo)
        records = [
            cu("go", cwd=cwd),
            ca(
                [
                    {"type": "text", "text": "One."},
                    send_user_message("Two."),
                    {"type": "tool_use", "id": "t2", "name": "SendMessage",
                     "input": {"to": "helper", "message": "between agents"}},
                    {"type": "text", "text": "Three."},
                ],
                cwd=cwd,
                mid="mA",
            ),
            ca([send_user_message("Four.", tid="t3")], ts=T2, cwd=cwd, mid="mB"),
            ca([{"type": "text", "text": "Five."}], ts=T3, cwd=cwd, mid="mC"),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.reply) for e in got],
            [("go", "One.\n\nTwo.\n\nThree.\n\nFour.\n\nFive.")],
        )

    def test_workspace_reply_after_handoff_names_the_exchange_model(self):
        # Replicata: a conversation begun in claude.ai chat moves into a
        # cloud workspace. The transcript opens with the chat part copied in
        # (handedOff, every record stamped with one shared timestamp), its
        # one text reply stamped claude-sonnet-4-6; a system marker sits
        # between prompt and reply; the workspace model then answers only
        # through SendUserMessage. Expectata: one exchange whose reply is the
        # chat text and then the workspace message, under the model of the
        # last reply-bearing record, claude-opus-5-5. Resultata (v5.5.0):
        # the chat text alone, labeled claude-sonnet-4-6.
        cwd = str(self.repo)
        records = [
            cu("pick up where we left off", ts=T0, cwd=cwd, handedOff=True),
            {"type": "system", "subtype": "upgrade_relay_marker", "content": "moved",
             "timestamp": T0, "cwd": cwd, "sessionId": "cs1"},
            {
                **ca(
                    [
                        {"type": "text", "text": "From the chat."},
                        {"type": "tool_use", "id": "t1", "name": "WebSearch", "input": {"query": "q"}},
                    ],
                    ts=T0,
                    cwd=cwd,
                    mid="m-chat",
                    model="claude-sonnet-4-6",
                ),
                "handedOff": True,
            },
            ca([send_user_message("From the workspace.")], ts=T1, cwd=cwd, mid="m-work",
               model="claude-opus-5-5"),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.reply, e.model) for e in got],
            [("pick up where we left off", "From the chat.\n\nFrom the workspace.", "claude-opus-5-5")],
        )

    def test_malformed_send_user_message_fails_loudly(self):
        # Replicata: a SendUserMessage call whose input is not exactly
        # {"message": <text>}: no message, an extra field, a message that is
        # not text, an input that is not an object, no input at all.
        # Expectata: a loud error citing the record, never a guessed reply.
        # Resultata (v5.5.0): silently ignored like every tool call.
        cwd = str(self.repo)
        call = {"type": "tool_use", "id": "t1", "name": "SendUserMessage"}
        for block in (
            {**call, "input": {}},
            {**call, "input": {"message": "hi", "attachments": []}},
            {**call, "input": {"message": 7}},
            {**call, "input": {"message": None}},
            {**call, "input": "hi"},
            {**call, "input": None},
            call,
        ):
            path = self.path([cu("go", cwd=cwd), ca([block], cwd=cwd)])
            with self.subTest(block=block):
                with self.assertRaises(ace.UserError) as ctx:
                    ace.claude_exchanges(path, self.repo)
                self.assertIn(f"{path}:2", str(ctx.exception))

    def test_empty_send_user_message_skipped_like_empty_text(self):
        # Replicata: SendUserMessage calls with an empty message, one beside
        # an empty text block and a nonempty message, one alone in the next
        # exchange. Expectata: an empty message adds no paragraph, and an
        # exchange whose only words are empty has no reply and so no model,
        # exactly as with an empty text block. Resultata (v5.5.0): the
        # nonempty message was lost too.
        cwd = str(self.repo)
        records = [
            cu("go", ts=T0, cwd=cwd),
            ca(
                [{"type": "text", "text": ""}, send_user_message(""), send_user_message("Only this.", tid="t2")],
                ts=T1,
                cwd=cwd,
                model="claude-opus-5-5",
            ),
            cu("again", ts=T2, cwd=cwd),
            ca([send_user_message("", tid="t3")], ts=T3, cwd=cwd, mid="m2", model="claude-opus-5-5"),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.reply, e.model) for e in got],
            [("go", "Only this.", "claude-opus-5-5"), ("again", "", "")],
        )

    def test_lone_brief_call_is_the_reply(self):
        # Replicata: an agent answers only through a tool call named Brief,
        # the legacy name of SendUserMessage in Claude Code's tool-name map,
        # writing no text block. Expectata: the call's message is the reply,
        # under the record's model, exactly as for SendUserMessage.
        # Resultata (v5.5.0): no reply and no model, because Brief was read
        # like any other tool call.
        cwd = str(self.repo)
        brief = {"type": "tool_use", "id": "tu1", "name": "Brief", "input": {"message": "Here is what I found."}}
        records = [cu("go", cwd=cwd), ca([brief], cwd=cwd, model="claude-opus-5-5")]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.reply, e.model) for e in got],
            [("go", "Here is what I found.", "claude-opus-5-5")],
        )

    def test_malformed_brief_call_fails_loudly(self):
        # Replicata: a Brief call (SendUserMessage under its legacy name)
        # whose input is not exactly {"message": <text>}: no message, an
        # extra field, a message that is not text, an input that is not an
        # object, no input at all. Expectata: the loud error SendUserMessage
        # gets, citing the record and naming the call as the record names
        # it. Resultata (v5.5.0): silently ignored like every tool call.
        cwd = str(self.repo)
        call = {"type": "tool_use", "id": "t1", "name": "Brief"}
        for block in (
            {**call, "input": {}},
            {**call, "input": {"message": "hi", "attachments": []}},
            {**call, "input": {"message": 7}},
            {**call, "input": {"message": None}},
            {**call, "input": "hi"},
            {**call, "input": None},
            call,
        ):
            path = self.path([cu("go", cwd=cwd), ca([block], cwd=cwd)])
            with self.subTest(block=block):
                with self.assertRaises(ace.UserError) as ctx:
                    ace.claude_exchanges(path, self.repo)
                self.assertIn(f"{path}:2", str(ctx.exception))
                self.assertIn("Brief", str(ctx.exception))

    def test_brief_mode_status_fails_loudly_under_either_name(self):
        # Replicata: in brief mode, Claude Code's SendUserMessage schema
        # requires a status ("normal" or "proactive") beside the message,
        # and the call can arrive under either name. Expectata: a loud error
        # citing the record, until a real transcript of that shape shows
        # what the status means for the page. Resultata (v5.5.0): silently
        # ignored like every tool call.
        cwd = str(self.repo)
        for name in ("SendUserMessage", "Brief"):
            block = {"type": "tool_use", "id": "t1", "name": name, "input": {"message": "hi", "status": "normal"}}
            path = self.path([cu("go", cwd=cwd), ca([block], cwd=cwd)])
            with self.subTest(name=name):
                with self.assertRaises(ace.UserError) as ctx:
                    ace.claude_exchanges(path, self.repo)
                self.assertIn(f"{path}:2", str(ctx.exception))

    def test_reply_content_not_a_list_fails_loudly(self):
        # Replicata: an assistant record whose message content is not a
        # list of blocks: a string, an object, a number, null, or no content
        # at all. Expectata: a loud error citing the record, never a reply
        # read as empty. Resultata (v5.5.1): no error; the record was read
        # as saying nothing, and its words, if any, were lost.
        cwd = str(self.repo)
        absent = ca([], cwd=cwd)
        del absent["message"]["content"]
        for record in (
            ca("Done.", cwd=cwd),
            ca({"type": "text", "text": "Done."}, cwd=cwd),
            ca(7, cwd=cwd),
            ca(None, cwd=cwd),
            absent,
        ):
            path = self.path([cu("go", cwd=cwd), record])
            with self.subTest(message=record["message"]):
                with self.assertRaises(ace.UserError) as ctx:
                    ace.claude_exchanges(path, self.repo)
                self.assertIn(f"{path}:2", str(ctx.exception))

    def test_text_block_whose_text_is_not_a_string_fails_loudly(self):
        # Replicata: an assistant record holding a text block whose text is
        # a number, null, a list, or an object, or is missing, either alone
        # or after a well-formed text block. Expectata: a loud error citing
        # the record. Resultata (v5.5.1): no error; the block was skipped
        # like a tool call, and its words, if any, were lost.
        cwd = str(self.repo)
        sound = {"type": "text", "text": "Fine."}
        for block in (
            {"type": "text", "text": 7},
            {"type": "text", "text": None},
            {"type": "text", "text": ["Done."]},
            {"type": "text", "text": {"value": "Done."}},
            {"type": "text"},
        ):
            for blocks in ([block], [sound, block]):
                path = self.path([cu("go", cwd=cwd), ca(blocks, cwd=cwd)])
                with self.subTest(blocks=blocks):
                    with self.assertRaises(ace.UserError) as ctx:
                        ace.claude_exchanges(path, self.repo)
                    self.assertIn(f"{path}:2", str(ctx.exception))

    def test_pasted_images_recovered_as_data_uris(self):
        cwd = str(self.repo)
        png = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}}
        records = [
            cu([png, {"type": "text", "text": "look at this"}], cwd=cwd),
            cu([dict(png)], ts=T1, cwd=cwd),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.images) for e in got],
            [("look at this", ("data:image/png;base64,AAA",)), ("", ("data:image/png;base64,AAA",))],
        )

    def test_unrecognized_image_form_fails_loudly(self):
        content = [{"type": "image", "source": {"type": "url", "url": "https://x"}}]
        with self.assertRaises(ace.UserError):
            ace.claude_exchanges(self.path([cu(content, cwd=str(self.repo))]), self.repo)[0]

    def test_synthetic_harness_notices_dropped(self):
        cwd = str(self.repo)
        records = [
            cu("go", cwd=cwd),
            ca([{"type": "text", "text": "API Error: 401"}], cwd=cwd, model="<synthetic>"),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.reply, e.model, e.elapsed) for e in got],
            [("go", "", "", 0.0)],
        )

    def test_identical_consecutive_prompts_are_two_human_acts(self):
        cwd = str(self.repo)
        records = [cu("retry", ts=T0, cwd=cwd), cu("retry", ts=T1, cwd=cwd)]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual([(e.prompt, e.reply) for e in got], [("retry", ""), ("retry", "")])

    def test_cwd_outside_repo_excluded_subdir_included(self):
        records = [
            cu("outside", cwd="/somewhere/else"),
            cu("inside", ts=T1, cwd=str(self.repo / "sub" / "dir")),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual([e.prompt for e in got], ["inside"])

    def test_elapsed_excludes_permission_wait_credits_recorded_tool_time(self):
        cwd = str(self.repo)
        records = [
            cu("go", ts=T0, cwd=cwd),
            ca([{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}], ts=T1, cwd=cwd),
            cu(  # tool result arrives after a wait; only recorded runtime counts
                [{"type": "tool_result", "tool_use_id": "t1", "content": "ran"}],
                ts=T2,
                cwd=cwd,
                toolUseResult={"stdout": "ran", "durationMs": 60000},
            ),
            ca([{"type": "text", "text": "done"}], ts="2026-03-01T10:15:00.000Z", cwd=cwd, mid="m9"),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        # 300s (prompt -> tool_use) + 60s recorded runtime + 300s (result -> reply);
        # the T1 -> T2 gap itself (wait + run) is never counted.
        self.assertEqual([(e.prompt, e.elapsed) for e in got], [("go", 660.0)])

    def test_subagent_total_duration_credited(self):
        cwd = str(self.repo)
        records = [
            cu("go", ts=T0, cwd=cwd),
            ca([{"type": "tool_use", "id": "t1", "name": "Agent", "input": {}}], ts=T1, cwd=cwd),
            cu(  # Agent results record totalDurationMs, not durationMs
                [{"type": "tool_result", "tool_use_id": "t1", "content": "done"}],
                ts=T2,
                cwd=cwd,
                toolUseResult={"totalDurationMs": 240000},
            ),
            ca([{"type": "text", "text": "report"}], ts="2026-03-01T10:15:00.000Z", cwd=cwd, mid="m9"),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        # 300s (prompt -> tool_use) + 240s recorded subagent runtime + 300s
        # (result -> reply).
        self.assertEqual([(e.prompt, e.elapsed) for e in got], [("go", 840.0)])

    def test_recorded_duration_capped_by_open_timeline_gap(self):
        cwd = str(self.repo)
        records = [
            cu("go", ts=T0, cwd=cwd),
            ca([{"type": "tool_use", "id": "t1", "name": "Agent", "input": {}}], ts=T1, cwd=cwd),
            cu(  # a 240s run collected 60s after launch overlapped work the
                # timeline already counted, so only the open 60s is credited
                [{"type": "tool_result", "tool_use_id": "t1", "content": "done"}],
                ts="2026-03-01T10:06:00.000Z",
                cwd=cwd,
                toolUseResult={"totalDurationMs": 240000},
            ),
            ca([{"type": "text", "text": "report"}], ts="2026-03-01T10:07:00.000Z", cwd=cwd, mid="m9"),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        # 300s (prompt -> tool_use) + min(60s gap, 240s recorded) + 60s
        # (result -> reply).
        self.assertEqual([(e.prompt, e.elapsed) for e in got], [("go", 420.0)])

    def test_wall_spans_prompt_to_last_reply(self):
        cwd = str(self.repo)
        records = [
            cu("go", ts=T0, cwd=cwd),
            ca([{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}], ts=T1, cwd=cwd),
            cu([{"type": "tool_result", "tool_use_id": "t1", "content": "ran"}],
               ts=T2, cwd=cwd, toolUseResult={"stdout": "ran"}),
            ca([{"type": "text", "text": "done"}], ts="2026-03-01T10:15:00.000Z", cwd=cwd, mid="m9"),
            cu("next", ts="2026-03-01T18:00:00.000Z", cwd=cwd),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        # Wall clock runs from the prompt to its final reply record (900s),
        # exceeding elapsed (600s) by the unrecorded tool-result gap; the
        # second exchange never got a reply, so its wall is unknown.
        self.assertEqual(
            [(e.prompt, e.elapsed, e.wall) for e in got],
            [("go", 600.0, 900.0), ("next", 0.0, 0.0)],
        )

    def test_edits_and_tool_time_at_eof_credited_to_inflight_exchange(self):
        cwd = str(self.repo)
        patch = {
            "filePath": str(self.repo / "a.py"),
            "structuredPatch": [{"lines": ["-old", "+new"]}],
            "durationMs": 60000,
        }
        records = [
            cu("change it", ts=T0, cwd=cwd),
            ca([{"type": "tool_use", "id": "t1", "name": "Edit", "input": {}}], ts=T1, cwd=cwd),
            cu([{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
               ts=T2, cwd=cwd, toolUseResult=patch),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.reply, e.elapsed, e.added, e.deleted) for e in got],
            [("change it", "", 360.0, 1, 1)],
        )

    def test_edit_lines_tallied_to_prompting_exchange(self):
        cwd = str(self.repo)
        patch = {
            "filePath": str(self.repo / "a.py"),
            "oldString": "old",
            "newString": "new one\nnew two",
            "originalFile": "ctx\nold\n",
            "structuredPatch": [
                {"oldStart": 1, "oldLines": 2, "newStart": 1, "newLines": 3,
                 "lines": [" ctx", "-old", "+new one", "+new two"]}
            ],
            "userModified": False,
            "replaceAll": False,
        }
        create = {
            "type": "create",
            "filePath": str(self.repo / "b.py"),
            "content": "one\ntwo\nthree\n",
            "originalFile": None,
            "structuredPatch": [],
            "userModified": False,
        }
        records = [
            cu("build it", ts=T0, cwd=cwd),
            cu([{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
               ts=T1, cwd=cwd, toolUseResult=patch),
            cu([{"type": "tool_result", "tool_use_id": "t2", "content": "ok"}],
               ts=T1, cwd=cwd, toolUseResult=create),
            ca([{"type": "text", "text": "built"}], ts=T2, cwd=cwd),
            cu("now a question, no edits", ts="2026-03-01T10:15:00.000Z", cwd=cwd),
            ca([{"type": "text", "text": "answered"}], ts="2026-03-01T10:16:00.000Z", mid="m2", cwd=cwd),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.added, e.deleted) for e in got],
            [("build it", 5, 1), ("now a question, no edits", 0, 0)],
        )

    def test_edits_outside_repo_and_denied_edits_not_tallied(self):
        cwd = str(self.repo)
        elsewhere = {
            "filePath": "/somewhere/else/a.py",
            "structuredPatch": [
                {"oldStart": 1, "oldLines": 1, "newStart": 1, "newLines": 1, "lines": ["-x", "+y"]}
            ],
        }
        denial = (
            "The user doesn't want to proceed with this tool use. "
            "The tool use was rejected (eg. if it was a file edit, the new_string "
            "was NOT written to the file). STOP what you are doing and wait for "
            "the user to tell you how to proceed."
        )
        records = [
            cu("go", ts=T0, cwd=cwd),
            cu([{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
               ts=T1, cwd=cwd, toolUseResult=elsewhere),
            cu([{"type": "tool_result", "tool_use_id": "t2", "content": denial}],
               ts=T1, cwd=cwd, toolUseResult=denial),
            ca([{"type": "text", "text": "hm"}], ts=T2, cwd=cwd),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual([(e.prompt, e.added, e.deleted) for e in got], [("go", 0, 0)])

    def test_hunk_lines_that_look_like_diff_headers_still_counted(self):
        # Deleting a line whose content is "--" or adding one starting with
        # "++" stores hunk lines "---"/"+++...". Hunks never contain the
        # file headers of a full diff, so every +/- prefix is a change.
        cwd = str(self.repo)
        patch = {
            "filePath": str(self.repo / "notes.md"),
            "structuredPatch": [
                {"oldStart": 1, "oldLines": 2, "newStart": 1, "newLines": 2,
                 "lines": ["---", "+++x;", " ctx"]}
            ],
        }
        records = [
            cu("go", ts=T0, cwd=cwd),
            cu([{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
               ts=T1, cwd=cwd, toolUseResult=patch),
            ca([{"type": "text", "text": "done"}], ts=T2, cwd=cwd),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual([(e.added, e.deleted) for e in got], [(1, 1)])

    def test_orphaned_edits_credited_to_interrupted_exchange(self):
        cwd = str(self.repo)
        patch = {
            "filePath": str(self.repo / "a.py"),
            "structuredPatch": [
                {"oldStart": 1, "oldLines": 2, "newStart": 1, "newLines": 3,
                 "lines": [" ctx", "-old", "+new one", "+new two"]}
            ],
        }
        create = {
            "type": "create",
            "filePath": str(self.repo / "b.py"),
            "content": "one\ntwo\nthree\n",
            "structuredPatch": [],
        }
        queued = {
            "type": "attachment",
            "attachment": {
                "type": "queued_command",
                "commandMode": "prompt",
                "prompt": [{"type": "text", "text": "wait, also do X"}],
                "origin": {"kind": "human"},
            },
            "timestamp": T2,
            "cwd": cwd,
            "sessionId": "cs1",
            "uuid": "q1",
        }
        records = [
            cu("start the work", ts=T0, cwd=cwd),
            cu([{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
               ts=T1, cwd=cwd, toolUseResult=patch),
            queued,  # arrives before the agent has said anything
            cu([{"type": "tool_result", "tool_use_id": "t2", "content": "ok"}],
               ts=T2, cwd=cwd, toolUseResult=create),
            ca([{"type": "text", "text": "Did X."}], ts="2026-03-01T10:15:00.000Z", cwd=cwd),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.reply, e.added, e.deleted) for e in got],
            [("start the work", "", 2, 1), ("wait, also do X", "Did X.", 3, 0)],
        )

    def test_orphaned_work_time_credited_to_interrupted_exchange(self):
        cwd = str(self.repo)
        queued = {
            "type": "attachment",
            "attachment": {
                "type": "queued_command",
                "commandMode": "prompt",
                "prompt": [{"type": "text", "text": "second"}],
                "origin": {"kind": "human"},
            },
            "timestamp": T2,
            "cwd": cwd,
            "sessionId": "cs1",
        }
        records = [
            cu("first", ts=T0, cwd=cwd),
            ca([{"type": "tool_use", "id": "t1", "name": "Edit", "input": {}}], ts=T1, cwd=cwd),
            queued,
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual([(e.prompt, e.elapsed) for e in got], [("first", 300.0), ("second", 0.0)])

    def test_file_creation_without_content_fails_loudly(self):
        cwd = str(self.repo)
        broken = {"type": "create", "filePath": str(self.repo / "b.py"), "structuredPatch": []}
        records = [
            cu("go", ts=T0, cwd=cwd),
            cu([{"type": "tool_result", "tool_use_id": "t", "content": "ok"}],
               ts=T1, cwd=cwd, toolUseResult=broken),
        ]
        with self.assertRaises(ace.UserError):
            ace.claude_exchanges(self.path(records), self.repo)[0]

    def test_queued_midturn_message_recovered_as_prompt(self):
        cwd = str(self.repo)
        def queued(mode, prompt, ts):
            return {
                "type": "attachment",
                "attachment": {
                    "type": "queued_command",
                    "commandMode": mode,
                    "prompt": prompt,
                    "origin": {"kind": "human" if mode == "prompt" else "task-notification"},
                },
                "timestamp": ts,
                "cwd": cwd,
                "sessionId": "cs1",
                "uuid": "q1",
            }
        records = [
            cu("start the work", ts=T0, cwd=cwd),
            ca([{"type": "text", "text": "Working."}], ts=T1, cwd=cwd),
            queued("prompt", [{"type": "text", "text": "wait, also do X"}], T2),
            queued("task-notification", "<task-notification>done</task-notification>", T2),
            {"type": "attachment", "attachment": {"type": "todo_reminder"}, "timestamp": T2, "cwd": cwd},
            ca([{"type": "text", "text": "Doing X."}], ts="2026-03-01T10:15:00.000Z", cwd=cwd, mid="m8"),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.reply) for e in got],
            [("start the work", "Working."), ("wait, also do X", "Doing X.")],
        )
        bad = [queued("hologram-mode", [{"type": "text", "text": "x"}], T0)]
        with self.assertRaises(ace.UserError):
            ace.claude_exchanges(self.path(bad), self.repo)[0]

    def test_queued_prompt_timestamp_does_not_rewind_activity_clock(self):
        cwd = str(self.repo)
        queued = {
            "type": "attachment",
            "attachment": {
                "type": "queued_command",
                "commandMode": "prompt",
                "prompt": [{"type": "text", "text": "second"}],
                "origin": {"kind": "human"},
            },
            "timestamp": "2026-03-01T10:06:00.000Z",
            "cwd": cwd,
            "sessionId": "cs1",
        }
        records = [
            cu("first", ts=T0, cwd=cwd),
            ca(
                [
                    {"type": "text", "text": "working"},
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}},
                ],
                ts=T1,
                cwd=cwd,
            ),
            cu([{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}], ts=T2, cwd=cwd),
            queued,
            ca([{"type": "text", "text": "done"}], ts="2026-03-01T10:11:00.000Z", cwd=cwd),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.elapsed) for e in got],
            [("first", 300.0), ("second", 60.0)],
        )

    def test_machine_origin_records_dropped_human_kept_unknown_loud(self):
        cwd = str(self.repo)
        notification = (
            "<task-notification>\n<task-id>bw86bdoa3</task-id>\n"
            "<status>completed</status>\n</task-notification>"
        )
        records = [
            cu(notification, cwd=cwd, origin={"kind": "task-notification"}),
            cu("typed with origin", ts=T1, cwd=cwd, origin={"kind": "human"}),
            cu("typed without origin", ts=T2, cwd=cwd),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual([e.prompt for e in got], ["typed with origin", "typed without origin"])
        with self.assertRaises(ace.UserError):
            ace.claude_exchanges(
                self.path([cu("x", cwd=cwd, origin={"kind": "hologram"})]), self.repo
            )[0]

    def test_peer_agent_handback_queued_midturn_is_machine_not_prompt(self):
        # Replicata: a background agent hands its result back mid-turn; Claude
        # Code records it as a queued_command attachment in prompt mode whose
        # origin kind is "peer" (isMeta sits inside the attachment, not on the
        # record). Expectata: no prompt, no error. Resultata (v5.5.0): UserError
        # "Origo recordi ignota: 'peer'".
        cwd = str(self.repo)
        body = "Findings: all green."
        handback = {
            "type": "attachment",
            "attachment": {
                "type": "queued_command",
                "prompt": f'<agent-message from="a7k2m9q4x1c8v5b3n">\n{body}\n</agent-message>',
                "source_uuid": "u9",
                "commandMode": "prompt",
                "origin": {
                    "kind": "peer",
                    "from": "a7k2m9q4x1c8v5b3n",
                    "senderTaskId": "a7k2m9q4x1c8v5b3n",
                    "body": body,
                    "handback": True,
                },
                "timestamp": T2,
                "isMeta": True,
            },
            "timestamp": T2,
            "cwd": cwd,
            "sessionId": "cs1",
            "uuid": "q1",
        }
        records = [
            cu("start the work", ts=T0, cwd=cwd),
            ca([{"type": "text", "text": "Working."}], ts=T1, cwd=cwd),
            handback,
            ca([{"type": "text", "text": "Done."}], ts="2026-03-01T10:15:00.000Z", cwd=cwd, mid="m8"),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual([e.prompt for e in got], ["start the work"])

    def test_queued_prompts_redelivered_after_restart_are_not_new_prompts(self):
        # Replicata: two prompts are queued mid-turn, each delivered with a
        # source_uuid and a delivery_id; after the last reply, an edit (with
        # recorded tool time) is still unclaimed when the cloud workspace
        # restarts and delivers both prompts again: same source_uuid, new
        # delivery_id, new timestamp. Then another edit and a reply.
        # Expectata: each prompt once, at its first delivery; the work
        # around the redeliveries, before and after, credited to the
        # exchange in flight (the second prompt's); both redeliveries still
        # held, so a ledger's stale copies of them get purged. Resultata
        # (v5.5.0): each prompt shown twice, the work split between the
        # second prompt and the second redelivery.
        cwd = str(self.repo)
        before = {
            "filePath": str(self.repo / "a.py"),
            "structuredPatch": [{"lines": ["-old", "+new"]}],
            "durationMs": 20000,
        }
        after = {
            "filePath": str(self.repo / "a.py"),
            "structuredPatch": [{"lines": ["+newer"]}],
            "durationMs": 30000,
        }
        first, second = "2026-03-01T10:06:00.000Z", "2026-03-01T10:08:00.000Z"
        again, again2 = "2026-03-01T10:30:00.000Z", "2026-03-01T10:30:00.002Z"
        records = [
            cu("start the work", ts=T0, cwd=cwd),
            ca([{"type": "text", "text": "Working."}], ts=T1, cwd=cwd),
            cq("first aside", first, cwd, source_uuid="src-1", delivery_id="dlv-1"),
            ca([{"type": "text", "text": "Noted the first."}], ts="2026-03-01T10:07:00.000Z", cwd=cwd, mid="m2"),
            cq("second aside", second, cwd, source_uuid="src-2", delivery_id="dlv-2"),
            ca([{"type": "text", "text": "Noted the second."}], ts="2026-03-01T10:09:00.000Z", cwd=cwd, mid="m3"),
            cu([{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
               ts=T2, cwd=cwd, toolUseResult=before),
            cq("first aside", again, cwd, source_uuid="src-1", delivery_id="dlv-3"),
            cq("second aside", again2, cwd, source_uuid="src-2", delivery_id="dlv-4"),
            cu([{"type": "tool_result", "tool_use_id": "t2", "content": "ok"}],
               ts="2026-03-01T10:31:00.000Z", cwd=cwd, toolUseResult=after),
            ca([{"type": "text", "text": "Did both."}], ts="2026-03-01T10:32:00.000Z", cwd=cwd, mid="m4"),
        ]
        exchanges, holdings = ace.claude_exchanges(self.path(records), self.repo)
        self.assertEqual(
            [(e.prompt, e.reply, e.elapsed, e.wall, e.added, e.deleted) for e in exchanges],
            [
                ("start the work", "Working.", 300.0, 300.0, 0, 0),
                ("first aside", "Noted the first.", 60.0, 60.0, 0, 0),
                # 60s to its first reply, then 20s and 30s of recorded tool
                # time on either side of the restart and 60s to the last
                # reply; both edits' lines.
                ("second aside", "Noted the second.\n\nDid both.", 170.0, 1440.0, 2, 1),
            ],
        )
        self.assertEqual(
            holdings,
            {("Claude Code", utc(ts)) for ts in (T0, first, second, again, again2)},
        )

    def test_queued_prompts_with_distinct_or_no_source_uuid_are_each_kept(self):
        # Replicata: the human queues the same words twice mid-turn, either
        # under two different source_uuids or with no source_uuid at all.
        # Expectata: two prompts each time; only a repeated source_uuid marks
        # a redelivery. Resultata (v5.5.0): as expected; this guards the
        # redelivery fix against keying on the words, or treating a missing
        # source_uuid as one.
        cwd = str(self.repo)
        for fields in (({"source_uuid": "src-1"}, {"source_uuid": "src-2"}), ({}, {})):
            records = [
                cu("start the work", ts=T0, cwd=cwd),
                ca([{"type": "text", "text": "Working."}], ts=T1, cwd=cwd),
                cq("check again", "2026-03-01T10:06:00.000Z", cwd, **fields[0]),
                ca([{"type": "text", "text": "Checked once."}], ts="2026-03-01T10:07:00.000Z", cwd=cwd, mid="m2"),
                cq("check again", "2026-03-01T10:08:00.000Z", cwd, **fields[1]),
                ca([{"type": "text", "text": "Checked twice."}], ts="2026-03-01T10:09:00.000Z", cwd=cwd, mid="m3"),
            ]
            with self.subTest(fields=fields):
                got = ace.claude_exchanges(self.path(records), self.repo)[0]
                self.assertEqual(
                    [(e.prompt, e.reply) for e in got],
                    [
                        ("start the work", "Working."),
                        ("check again", "Checked once."),
                        ("check again", "Checked twice."),
                    ],
                )

    def test_queued_prompt_with_non_text_source_uuid_fails_loudly(self):
        # Replicata: a queued prompt whose source_uuid is present but not
        # text. Expectata: a loud error citing the record, never a guess at
        # whether it repeats an earlier delivery. Resultata (v5.5.0): the
        # field was never read.
        cwd = str(self.repo)
        for source in (7, None, ["src-1"], {"id": "src-1"}):
            path = self.path([cu("start the work", cwd=cwd), cq("aside", T1, cwd, source_uuid=source)])
            with self.subTest(source=source):
                with self.assertRaises(ace.UserError) as ctx:
                    ace.claude_exchanges(path, self.repo)
                self.assertIn(f"{path}:2", str(ctx.exception))

    def test_source_uuid_repeated_in_another_session_is_that_sessions_own(self):
        # Replicata: one file holds two sessions, each delivering a queued
        # prompt under the same source_uuid. Expectata: both prompts kept;
        # deliveries are tracked per session, like all of claude_exchanges'
        # bookkeeping. Resultata (v5.5.0): as expected; this guards the
        # redelivery fix against tracking deliveries across sessions.
        cwd = str(self.repo)
        records = [
            cu("start one", ts=T0, cwd=cwd, session="cs1"),
            cu("start two", ts=T0, cwd=cwd, session="cs2"),
            cq("aside", T1, cwd, session="cs1", source_uuid="src-1"),
            cq("aside", T2, cwd, session="cs2", source_uuid="src-1"),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.session, e.prompt) for e in got],
            [("cs1", "start one"), ("cs1", "aside"), ("cs2", "start two"), ("cs2", "aside")],
        )

    def test_redelivery_carrying_other_words_or_images_fails_loudly(self):
        # Replicata: a queued prompt of words and an image is delivered, then
        # delivered again under the same source_uuid carrying other words,
        # another image, no image, or an extra image. Expectata: a loud error
        # citing the second delivery, never a silent drop of what it
        # carries. Resultata (v5.5.0): no error; the second delivery became a
        # prompt of its own.
        cwd = str(self.repo)

        def image(data):
            return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}

        for text, images in (
            ("other words", [image("AA")]),
            ("aside", [image("BB")]),
            ("aside", []),
            ("aside", [image("AA"), image("BB")]),
        ):
            path = self.path([
                cu("start the work", ts=T0, cwd=cwd),
                cq("aside", T1, cwd, source_uuid="src-1", images=[image("AA")]),
                cq(text, T2, cwd, source_uuid="src-1", images=images),
            ])
            with self.subTest(text=text, images=images):
                with self.assertRaises(ace.UserError) as ctx:
                    ace.claude_exchanges(path, self.repo)
                self.assertIn(f"{path}:3", str(ctx.exception))

    def test_identical_redelivery_with_image_held_not_refused(self):
        # Replicata: a queued prompt of words and an image is delivered twice
        # under one source_uuid, the deliveries differing only in timestamp
        # and delivery_id. Expectata: no error; one prompt, with its image, at
        # the first delivery; both deliveries held. Resultata (v5.5.0): the
        # prompt shown twice. This guards the refusal of differing
        # redeliveries against comparing more than words and images.
        cwd = str(self.repo)
        png = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA"}}
        records = [
            cu("start the work", ts=T0, cwd=cwd),
            cq("aside", T1, cwd, source_uuid="src-1", delivery_id="dlv-1", images=[png]),
            cq("aside", T2, cwd, source_uuid="src-1", delivery_id="dlv-2", images=[png]),
        ]
        exchanges, holdings = ace.claude_exchanges(self.path(records), self.repo)
        self.assertEqual(
            [(e.timestamp, e.prompt, e.images) for e in exchanges],
            [(utc(T0), "start the work", ()), (utc(T1), "aside", ("data:image/png;base64,AA",))],
        )
        self.assertEqual(holdings, {("Claude Code", utc(ts)) for ts in (T0, T1, T2)})

    def test_queued_prompt_with_empty_source_uuid_fails_loudly(self):
        # Replicata: a queued prompt whose source_uuid is the empty string.
        # Expectata: exactly the loud error a source_uuid that is not text
        # gets, citing the record. Resultata (v5.5.0): the field was never
        # read.
        cwd = str(self.repo)
        errors = []
        for source in ("", 7):
            path = self.path([cu("start the work", cwd=cwd), cq("aside", T1, cwd, source_uuid=source)])
            with self.assertRaises(ace.UserError, msg=source) as ctx:
                ace.claude_exchanges(path, self.repo)
            errors.append(str(ctx.exception))
        self.assertIn(f"{path}:2", errors[0])
        self.assertEqual(errors[0], errors[1])

    def test_redelivery_between_local_command_and_its_stdout_leaves_it_unsendable(self):
        # Replicata: after a queued prompt's first delivery, the human runs a
        # local slash command, and the prompt's redelivery lands between the
        # command and the record of the command's captured output.
        # Expectata: no error; the output still finds and unsends its
        # command, so neither the command nor the redelivery is a prompt.
        # Resultata (v5.5.0): a loud error, the redelivery having become a
        # prompt that left the output no command to unsend.
        cwd = str(self.repo)
        wrapped = (
            "<command-name>/remit</command-name>\n            "
            "<command-message>remit</command-message>\n            "
            "<command-args>everything owed</command-args>"
        )
        records = [
            cu("start the work", ts=T0, cwd=cwd),
            ca([{"type": "text", "text": "Working."}], ts=T1, cwd=cwd),
            cq("aside", "2026-03-01T10:06:00.000Z", cwd, source_uuid="src-1"),
            ca([{"type": "text", "text": "Noted."}], ts="2026-03-01T10:07:00.000Z", cwd=cwd, mid="m2"),
            cu(wrapped, ts="2026-03-01T10:08:00.000Z", cwd=cwd, promptId="pq1"),
            cq("aside", "2026-03-01T10:08:00.001Z", cwd, source_uuid="src-1"),
            cu("<local-command-stdout>Remitted.</local-command-stdout>",
               ts="2026-03-01T10:08:00.002Z", cwd=cwd, promptId="pq1"),
            cu("carry on", ts=T2, cwd=cwd),
            ca([{"type": "text", "text": "Done."}], ts=T3, cwd=cwd, mid="m3"),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.reply) for e in got],
            [("start the work", "Working."), ("aside", "Noted."), ("carry on", "Done.")],
        )

    def test_redelivery_inside_denial_fanout_still_renders_the_denial_once(self):
        # Replicata: after a queued prompt's first delivery, the human denies
        # two parallel tool calls with one typed reason, which Claude Code
        # stamps onto both tool results, and the prompt's redelivery lands
        # between the two. Expectata: the reason is one prompt and the
        # redelivery none. Resultata (v5.5.0): the redelivery became a
        # prompt that split the fan-out, so the reason showed twice.
        cwd = str(self.repo)
        denial = (
            "Error: The user doesn't want to proceed with this tool use. "
            "The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file). "
            "The user provided the following reason for the rejection:  hold tight"
        )

        def denial_record(ts, tid):
            return cu([{"type": "tool_result", "tool_use_id": tid, "content": denial[7:]}],
                      ts=ts, cwd=cwd, toolUseResult=denial, promptId="act-1")

        records = [
            cu("start the work", ts=T0, cwd=cwd),
            ca([{"type": "text", "text": "Working."}], ts=T1, cwd=cwd),
            cq("aside", "2026-03-01T10:06:00.000Z", cwd, source_uuid="src-1"),
            ca(
                [
                    {"type": "text", "text": "Noted."},
                    {"type": "tool_use", "id": "t1", "name": "Edit", "input": {}},
                    {"type": "tool_use", "id": "t2", "name": "Edit", "input": {}},
                ],
                ts="2026-03-01T10:07:00.000Z",
                cwd=cwd,
                mid="m2",
            ),
            denial_record("2026-03-01T10:08:00.000Z", "t1"),
            cq("aside", "2026-03-01T10:08:00.001Z", cwd, source_uuid="src-1"),
            denial_record("2026-03-01T10:08:00.002Z", "t2"),
            ca([{"type": "text", "text": "Standing by."}], ts=T2, cwd=cwd, mid="m3"),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.reply) for e in got],
            [("start the work", "Working."), ("aside", "Noted."), ("hold tight", "Standing by.")],
        )

    # Queued prompt contents that make no prompt: a harness wrapper item
    # alone, nothing at all, and the interrupt marker alone.
    NO_PROMPT_CONTENTS = (
        "<system-reminder>recall</system-reminder>",
        "",
        "[Request interrupted by user]",
    )

    def test_malformed_source_uuid_fails_loudly_even_on_a_delivery_that_is_no_prompt(self):
        # Replicata: a queued prompt whose source_uuid is empty or not text
        # carries only a harness wrapper item, nothing at all, or only the
        # interrupt marker. Expectata: the loud error the same source_uuid
        # gets on a queued prompt of words, citing the record. Resultata
        # (v5.5.1): no error; such a record was dropped before its
        # source_uuid was read.
        cwd = str(self.repo)
        for source in ("", 7):
            path = self.path([cu("start the work", cwd=cwd), cq("aside", T1, cwd, source_uuid=source)])
            with self.assertRaises(ace.UserError) as worded:
                ace.claude_exchanges(path, self.repo)
            self.assertIn(f"{path}:2", str(worded.exception))
            for text in self.NO_PROMPT_CONTENTS:
                path = self.path([
                    cu("start the work", cwd=cwd),
                    cq(text, T1, cwd, source_uuid=source),
                ])
                with self.subTest(source=source, text=text):
                    with self.assertRaises(ace.UserError) as ctx:
                        ace.claude_exchanges(path, self.repo)
                    self.assertEqual(str(ctx.exception), str(worded.exception))

    def test_differing_redelivery_fails_loudly_even_if_either_delivery_is_no_prompt(self):
        # Replicata: a queued prompt is delivered twice under one
        # source_uuid; one delivery carries words, the other only a harness
        # wrapper item, nothing at all, or only the interrupt marker, in
        # either order. Expectata: the loud error a redelivery carrying
        # other words gets, citing the second delivery. Resultata (v5.5.1):
        # no error; a redelivery that is no prompt vanished, and words
        # delivered again after a first delivery that is no prompt became a
        # prompt of their own.
        cwd = str(self.repo)
        aside = "aside"

        def deliveries(first, second):
            return self.path([
                cu("start the work", ts=T0, cwd=cwd),
                cq(first, T1, cwd, source_uuid="src-1"),
                cq(second, T2, cwd, source_uuid="src-1"),
            ])

        path = deliveries(aside, "other words")
        with self.assertRaises(ace.UserError) as worded:
            ace.claude_exchanges(path, self.repo)
        self.assertIn(f"{path}:3", str(worded.exception))
        for other in self.NO_PROMPT_CONTENTS:
            for first, second in ((aside, other), (other, aside)):
                path = deliveries(first, second)
                with self.subTest(first=first, second=second):
                    with self.assertRaises(ace.UserError) as ctx:
                        ace.claude_exchanges(path, self.repo)
                    self.assertEqual(str(ctx.exception), str(worded.exception))

    def test_identical_redelivery_of_a_delivery_that_is_no_prompt_is_held_as_its_first(self):
        # Replicata: a queued prompt carrying only a harness wrapper item,
        # nothing at all, or only the interrupt marker is delivered twice
        # under one source_uuid. Expectata: no error and no prompt; the
        # redelivery is held exactly when its first delivery is: the
        # interrupt marker both times, a delivery without typing neither
        # time. Resultata (v5.5.1): as expected; this guards the comparison
        # of such deliveries against holding a redelivery differently from
        # its first, or refusing an identical one.
        cwd = str(self.repo)
        wrapper, nothing, marker = self.NO_PROMPT_CONTENTS
        for text, held in ((wrapper, (T0,)), (nothing, (T0,)), (marker, (T0, T1, T2))):
            records = [
                cu("start the work", ts=T0, cwd=cwd),
                cq(text, T1, cwd, source_uuid="src-1"),
                cq(text, T2, cwd, source_uuid="src-1"),
            ]
            with self.subTest(text=text):
                exchanges, holdings = ace.claude_exchanges(self.path(records), self.repo)
                self.assertEqual([e.prompt for e in exchanges], ["start the work"])
                self.assertEqual(holdings, {("Claude Code", utc(ts)) for ts in held})

    def test_redelivery_first_delivered_outside_the_repo_fails_loudly(self):
        # Replicata: mid-session the agent's cwd leaves the repo and a
        # queued prompt is delivered there; the cwd comes back, and after a
        # restart the prompt is redelivered, identical, with its cwd inside
        # the repo. Expectata: a loud error citing the file at the first
        # delivery's line and at the redelivery's, never a guess at which
        # page the prompt belongs on. Resultata (v5.5.1): no error; the
        # redelivery became a prompt on this repo's page at its own time,
        # while the other directory's page would show the first delivery.
        cwd, elsewhere = str(self.repo), str(self.tmp / "elsewhere")
        path = self.path([
            cu("start the work", ts=T0, cwd=cwd),
            ca([{"type": "text", "text": "Working."}], ts=T1, cwd=elsewhere),
            cq("aside", "2026-03-01T10:06:00.000Z", elsewhere, source_uuid="src-1", delivery_id="dlv-1"),
            ca([{"type": "text", "text": "Noted."}], ts="2026-03-01T10:07:00.000Z", cwd=cwd, mid="m2"),
            cq("aside", "2026-03-01T10:30:00.000Z", cwd, source_uuid="src-1", delivery_id="dlv-2"),
            ca([{"type": "text", "text": "Done."}], ts="2026-03-01T10:31:00.000Z", cwd=cwd, mid="m3"),
        ])
        with self.assertRaises(ace.UserError) as ctx:
            ace.claude_exchanges(path, self.repo)
        self.assertIn(f"{path}:3", str(ctx.exception))
        self.assertIn(f"{path}:5", str(ctx.exception))

    def test_redelivery_after_several_deliveries_outside_the_repo_cites_the_first(self):
        # Replicata: a queued prompt is delivered with its cwd outside the
        # repo, delivered there again, then redelivered, identical, with its
        # cwd inside the repo. Expectata: the loud error citing the file at
        # the first delivery's line and at the inside redelivery's, and not
        # at the second delivery outside: the first delivery is the one that
        # decides. Resultata (v5.5.1): no error; the inside delivery became a
        # prompt.
        cwd, elsewhere = str(self.repo), str(self.tmp / "elsewhere")
        path = self.path([
            cu("start the work", ts=T0, cwd=cwd),
            cq("aside", T1, elsewhere, source_uuid="src-1"),
            cq("aside", T2, elsewhere, source_uuid="src-1"),
            cq("aside", T3, cwd, source_uuid="src-1"),
        ])
        with self.assertRaises(ace.UserError) as ctx:
            ace.claude_exchanges(path, self.repo)
        self.assertIn(f"{path}:2", str(ctx.exception))
        self.assertIn(f"{path}:4", str(ctx.exception))
        self.assertNotIn(f"{path}:3", str(ctx.exception))

    def test_redelivery_first_delivered_outside_the_repo_fails_loudly_whatever_it_carries(self):
        # Replicata: a queued prompt carrying only a harness wrapper item,
        # nothing at all, or only the interrupt marker is delivered with its
        # cwd outside the repo, then redelivered, identical, with its cwd
        # inside it. Expectata: the loud error the same two deliveries get
        # when they carry words, citing both: the refusal comes right after
        # the cwd check, before what the redelivery carries is read.
        # Resultata (v5.5.1): no error; the redelivery was taken for a first
        # delivery (the interrupt marker held, the others dropped).
        cwd, elsewhere = str(self.repo), str(self.tmp / "elsewhere")

        def deliveries(text):
            return self.path([
                cu("start the work", ts=T0, cwd=cwd),
                cq(text, T1, elsewhere, source_uuid="src-1"),
                cq(text, T2, cwd, source_uuid="src-1"),
            ])

        path = deliveries("aside")
        with self.assertRaises(ace.UserError) as worded:
            ace.claude_exchanges(path, self.repo)
        self.assertIn(f"{path}:2", str(worded.exception))
        self.assertIn(f"{path}:3", str(worded.exception))
        for text in self.NO_PROMPT_CONTENTS:
            path = deliveries(text)
            with self.subTest(text=text):
                with self.assertRaises(ace.UserError) as ctx:
                    ace.claude_exchanges(path, self.repo)
                self.assertEqual(str(ctx.exception), str(worded.exception))

    def test_redelivery_first_delivered_outside_the_repo_fails_loudly_whatever_its_origin(self):
        # Replicata: another agent's handback (a queued prompt of origin
        # "peer"), a background task's notification queued as a prompt
        # (origin "task-notification"), or a queued prompt with no origin
        # field (one that predates the field) is delivered with its cwd
        # outside the repo, then redelivered, identical, with its cwd inside
        # it. Expectata: the loud error the same two deliveries get when the
        # human typed them (origin "human"), citing both: the refusal comes
        # right after the cwd check, before the record's origin is read.
        # Resultata (v5.5.1): no error; the redelivery was held as machine
        # text, or, with no origin field, became a prompt.
        cwd, elsewhere = str(self.repo), str(self.tmp / "elsewhere")

        def deliveries(fields):
            # Both deliveries, each attachment's origin field (cq() gives
            # the kind "human") replaced by `fields`: an origin, or none.
            records = [
                cu("start the work", ts=T0, cwd=cwd),
                cq("Findings: all green.", T1, elsewhere, source_uuid="src-1"),
                cq("Findings: all green.", T2, cwd, source_uuid="src-1"),
            ]
            for queued in records[1:]:
                del queued["attachment"]["origin"]
                queued["attachment"].update(fields)
            return self.path(records)

        path = deliveries({"origin": {"kind": "human"}})
        with self.assertRaises(ace.UserError) as typed:
            ace.claude_exchanges(path, self.repo)
        self.assertIn(f"{path}:2", str(typed.exception))
        self.assertIn(f"{path}:3", str(typed.exception))
        for fields in ({"origin": {"kind": "peer"}}, {"origin": {"kind": "task-notification"}}, {}):
            path = deliveries(fields)
            with self.subTest(fields=fields):
                with self.assertRaises(ace.UserError) as ctx:
                    ace.claude_exchanges(path, self.repo)
                self.assertEqual(str(ctx.exception), str(typed.exception))

    def test_redelivery_first_delivered_outside_the_repo_is_refused_before_its_timestamp_is_read(self):
        # Replicata: a queued prompt is delivered with its cwd outside the
        # repo, then redelivered, identical, with its cwd inside it and no
        # timestamp on its record. Expectata: the loud error the same two
        # deliveries get when the redelivery's record has its timestamp,
        # citing both: the refusal comes right after the cwd check, before
        # the record's timestamp is read. Resultata (v5.5.1): the loud error
        # for a record with no timestamp, citing only the redelivery.
        cwd, elsewhere = str(self.repo), str(self.tmp / "elsewhere")
        records = [
            cu("start the work", ts=T0, cwd=cwd),
            cq("aside", T1, elsewhere, source_uuid="src-1"),
            cq("aside", T2, cwd, source_uuid="src-1"),
        ]
        path = self.path(records)
        with self.assertRaises(ace.UserError) as timed:
            ace.claude_exchanges(path, self.repo)
        del records[2]["timestamp"]
        path = self.path(records)
        with self.assertRaises(ace.UserError) as untimed:
            ace.claude_exchanges(path, self.repo)
        self.assertIn(f"{path}:2", str(untimed.exception))
        self.assertIn(f"{path}:3", str(untimed.exception))
        self.assertEqual(str(untimed.exception), str(timed.exception))

    def test_redelivery_outside_the_repo_after_a_first_delivery_inside_changes_nothing(self):
        # Replicata: a queued prompt is delivered with its cwd inside the
        # repo; the agent's cwd leaves the repo, and after a restart the
        # prompt is redelivered with its cwd outside it. Expectata: no error;
        # one prompt, at its first delivery, and nothing of the redelivery on
        # this repo's page, that record being another directory's. Resultata
        # (v5.5.1): as expected; this guards the refusal of a redelivery
        # first delivered elsewhere against refusing the reverse.
        cwd, elsewhere = str(self.repo), str(self.tmp / "elsewhere")
        first = "2026-03-01T10:06:00.000Z"
        records = [
            cu("start the work", ts=T0, cwd=cwd),
            ca([{"type": "text", "text": "Working."}], ts=T1, cwd=cwd),
            cq("aside", first, cwd, source_uuid="src-1", delivery_id="dlv-1"),
            ca([{"type": "text", "text": "Noted."}], ts="2026-03-01T10:07:00.000Z", cwd=cwd, mid="m2"),
            cq("aside", "2026-03-01T10:30:00.000Z", elsewhere, source_uuid="src-1", delivery_id="dlv-2"),
            ca([{"type": "text", "text": "Done."}], ts="2026-03-01T10:31:00.000Z", cwd=elsewhere, mid="m3"),
        ]
        exchanges, holdings = ace.claude_exchanges(self.path(records), self.repo)
        self.assertEqual(
            [(e.prompt, e.reply) for e in exchanges],
            [("start the work", "Working."), ("aside", "Noted.")],
        )
        self.assertEqual(holdings, {("Claude Code", utc(ts)) for ts in (T0, first)})

    def test_redelivery_inside_after_one_outside_is_ordinary_since_the_first_decides(self):
        # Replicata: a queued prompt is delivered with its cwd inside the
        # repo, redelivered with its cwd outside it, then redelivered again
        # inside. Expectata: no error; the first delivery decides, so the
        # last is an ordinary redelivery: one prompt, at the first delivery,
        # the reply after the last delivery credited to it, and both
        # deliveries inside held. Resultata (v5.5.1): as expected; this
        # guards the refusal of a redelivery first delivered elsewhere
        # against reading "first" as "previous".
        cwd, elsewhere = str(self.repo), str(self.tmp / "elsewhere")
        first, again = "2026-03-01T10:06:00.000Z", "2026-03-01T10:31:00.000Z"
        records = [
            cu("start the work", ts=T0, cwd=cwd),
            ca([{"type": "text", "text": "Working."}], ts=T1, cwd=cwd),
            cq("aside", first, cwd, source_uuid="src-1", delivery_id="dlv-1"),
            ca([{"type": "text", "text": "Noted."}], ts="2026-03-01T10:07:00.000Z", cwd=cwd, mid="m2"),
            cq("aside", "2026-03-01T10:30:00.000Z", elsewhere, source_uuid="src-1", delivery_id="dlv-2"),
            cq("aside", again, cwd, source_uuid="src-1", delivery_id="dlv-3"),
            ca([{"type": "text", "text": "Done."}], ts="2026-03-01T10:32:00.000Z", cwd=cwd, mid="m3"),
        ]
        exchanges, holdings = ace.claude_exchanges(self.path(records), self.repo)
        self.assertEqual(
            [(e.prompt, e.reply) for e in exchanges],
            [("start the work", "Working."), ("aside", "Noted.\n\nDone.")],
        )
        self.assertEqual(holdings, {("Claude Code", utc(ts)) for ts in (T0, first, again)})

    def test_source_uuid_first_delivered_outside_in_another_session_is_no_redelivery(self):
        # Replicata: one file holds two sessions: one delivers a queued
        # prompt with its cwd outside the repo, then the other delivers a
        # queued prompt under the same source_uuid with its cwd inside it.
        # Expectata: no error; the inside delivery is its own session's first
        # and so a prompt, deliveries being tracked per session. Resultata
        # (v5.5.1): as expected; this guards the refusal of a redelivery
        # first delivered elsewhere against tracking deliveries across
        # sessions.
        cwd, elsewhere = str(self.repo), str(self.tmp / "elsewhere")
        records = [
            cu("start one", ts=T0, cwd=cwd, session="cs1"),
            cq("aside", T1, elsewhere, session="cs2", source_uuid="src-1"),
            cq("aside", T2, cwd, session="cs1", source_uuid="src-1"),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual([(e.session, e.prompt) for e in got], [("cs1", "start one"), ("cs1", "aside")])

    def test_queued_prompt_outside_the_repo_never_redelivered_inside_changes_nothing(self):
        # Replicata: a session's transcript, and the same transcript with a
        # queued prompt carrying a source_uuid delivered mid-session with its
        # cwd outside the repo, and never delivered again. Expectata: the
        # same exchanges and holdings from both, that record being another
        # directory's. Resultata (v5.5.1): as expected; this guards the
        # bookkeeping of deliveries outside the repo against reaching this
        # repo's page.
        cwd, elsewhere = str(self.repo), str(self.tmp / "elsewhere")
        records = [
            cu("start the work", ts=T0, cwd=cwd),
            ca([{"type": "text", "text": "Working."}], ts=T1, cwd=cwd),
            ca([{"type": "text", "text": "Still working."}], ts=T3, cwd=cwd, mid="m2"),
        ]
        without = ace.claude_exchanges(self.path(records), self.repo)
        outside = cq("aside", T2, elsewhere, source_uuid="src-1")
        self.assertEqual(ace.claude_exchanges(self.path([*records[:2], outside, records[2]]), self.repo), without)

    def test_malformed_source_uuid_fails_loudly_even_outside_the_repo_or_from_a_peer(self):
        # Replicata: a queued prompt whose source_uuid is empty or not text
        # is delivered with its cwd outside the repo, or inside it as
        # another agent's handback (origin "peer"). Expectata: the loud
        # error the same source_uuid gets on a queued prompt typed inside
        # the repo, citing the record: a queued prompt's source_uuid is read
        # before its cwd or origin is looked at, a redelivery inside the repo
        # being judged by its first delivery wherever that was. Resultata
        # (v5.5.1): no error; such a record was dropped before its
        # source_uuid was read.
        cwd, elsewhere = str(self.repo), str(self.tmp / "elsewhere")

        def handback(source):
            record = cq("Findings: all green.", T1, cwd, source_uuid=source)
            record["attachment"]["origin"] = {"kind": "peer"}
            return record

        for source in ("", 7):
            path = self.path([cu("start the work", cwd=cwd), cq("aside", T1, cwd, source_uuid=source)])
            with self.assertRaises(ace.UserError) as typed:
                ace.claude_exchanges(path, self.repo)
            self.assertIn(f"{path}:2", str(typed.exception))
            for queued in (cq("aside", T1, elsewhere, source_uuid=source), handback(source)):
                path = self.path([cu("start the work", cwd=cwd), queued])
                with self.subTest(source=source, cwd=queued["cwd"], origin=queued["attachment"]["origin"]):
                    with self.assertRaises(ace.UserError) as ctx:
                        ace.claude_exchanges(path, self.repo)
                    self.assertEqual(str(ctx.exception), str(typed.exception))

    # The origin fields a queued prompt's attachment can carry: the human's
    # typing, another agent's message, a background task's notification,
    # and none at all (a record predating the field).
    ORIGINS = (
        {"origin": {"kind": "human"}},
        {"origin": {"kind": "peer"}},
        {"origin": {"kind": "task-notification"}},
        {},
    )

    def stated(self, queued, fields):
        """The queued prompt with its attachment's origin field (cq() gives
        the kind "human") replaced by `fields`: an origin, or none."""
        del queued["attachment"]["origin"]
        queued["attachment"].update(fields)
        return queued

    def test_redelivery_whose_origin_differs_from_its_first_deliverys_fails_loudly(self):
        # Replicata: a queued prompt is delivered inside the repo, then
        # redelivered there, identical, under the same source_uuid, the two
        # deliveries stating different origin kinds: every ordered pair of
        # the human's ("human"), another agent's ("peer"), a background
        # task's ("task-notification"), and none (no origin field).
        # Expectata: a loud error citing the file at the first delivery's
        # line and at the redelivery's, naming both kinds; sourcery picks no
        # rule for which delivery to believe. Resultata (v5.5.2): no error;
        # the redelivery was held as machine text, became a prompt of its
        # own, or was taken for an ordinary redelivery, as its own origin
        # decided.
        cwd = str(self.repo)
        for first, second in itertools.permutations(self.ORIGINS, 2):
            path = self.path([
                cu("start the work", ts=T0, cwd=cwd),
                self.stated(cq("Findings: all green.", T1, cwd, source_uuid="src-1"), first),
                self.stated(cq("Findings: all green.", T2, cwd, source_uuid="src-1"), second),
            ])
            kinds = [repr(fields.get("origin", {}).get("kind")) for fields in (first, second)]
            with self.subTest(first=first, second=second):
                with self.assertRaises(ace.UserError) as ctx:
                    ace.claude_exchanges(path, self.repo)
                for cited in (f"{path}:2", f"{path}:3", *kinds):
                    self.assertIn(cited, str(ctx.exception))

    def test_redelivery_stating_its_first_deliverys_origin_is_judged_as_before(self):
        # Replicata: a queued prompt is delivered inside the repo, then
        # redelivered there, identical, under the same source_uuid, both
        # deliveries stating the same origin kind: the human's, another
        # agent's, a background task's, or none. Expectata: no error; both
        # deliveries held; of machine origin, neither is a prompt; typed, or
        # of no stated origin, the first delivery is the prompt and the
        # redelivery none. Resultata (v5.5.2): as expected; this guards the
        # refusal of a redelivery of another origin against refusing one of
        # the same.
        cwd = str(self.repo)
        aside = "Findings: all green."
        for fields, prompts in zip(self.ORIGINS, ([aside], [], [], [aside])):
            path = self.path([
                cu("start the work", ts=T0, cwd=cwd),
                self.stated(cq(aside, T1, cwd, source_uuid="src-1"), fields),
                self.stated(cq(aside, T2, cwd, source_uuid="src-1"), fields),
            ])
            with self.subTest(fields=fields):
                exchanges, holdings = ace.claude_exchanges(path, self.repo)
                self.assertEqual([e.prompt for e in exchanges], ["start the work", *prompts])
                self.assertEqual(holdings, {("Claude Code", utc(ts)) for ts in (T0, T1, T2)})

    def test_redelivery_outside_the_repo_stating_another_origin_changes_nothing(self):
        # Replicata: a queued prompt is delivered inside the repo, then
        # redelivered, identical, with its cwd outside the repo, stating
        # another origin kind (another agent's). Expectata: no error, and
        # the same exchanges and holdings as when the redelivery states the
        # first delivery's origin: one prompt, at its first delivery, and
        # nothing of the redelivery, that record being another directory's.
        # Resultata (v5.5.2): as expected; this guards the refusal of a
        # redelivery of another origin against records this repo's page
        # never shows.
        cwd, elsewhere = str(self.repo), str(self.tmp / "elsewhere")

        def read(fields):
            return ace.claude_exchanges(self.path([
                cu("start the work", ts=T0, cwd=cwd),
                cq("aside", T1, cwd, source_uuid="src-1"),
                self.stated(cq("aside", T2, elsewhere, source_uuid="src-1"), fields),
            ]), self.repo)

        same = read(self.ORIGINS[0])
        self.assertEqual([e.prompt for e in same[0]], ["start the work", "aside"])
        self.assertEqual(read(self.ORIGINS[1]), same)

    def test_interrupt_markers_dropped_but_typed_text_around_them_kept(self):
        cwd = str(self.repo)
        records = [
            cu("[Request interrupted by user]", cwd=cwd),
            cu("[Request interrupted by user for tool use]", ts=T1, cwd=cwd),
            cu("[Request interrupted by user] but i typed this", ts=T2, cwd=cwd),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual([e.prompt for e in got], ["[Request interrupted by user] but i typed this"])

    def test_slash_command_wrapper_recovered_as_typed_command(self):
        wrapped = "<command-message>insights</command-message>\n<command-name>/insights</command-name>"
        got = ace.claude_exchanges(self.path([cu(wrapped, cwd=str(self.repo))]), self.repo)[0]
        self.assertEqual([e.prompt for e in got], ["/insights"])

    def test_malformed_slash_command_wrapper_fails_loudly(self):
        wrapped = "<command-message>insights</command-message>\n<command-name>/different</command-name>"
        with self.assertRaises(ace.UserError):
            ace.claude_exchanges(self.path([cu(wrapped, cwd=str(self.repo))]), self.repo)[0]

    def test_partial_slash_command_wrapper_fails_loudly(self):
        wrapped = "<command-name>/insights</command-name>"
        with self.assertRaises(ace.UserError):
            ace.claude_exchanges(self.path([cu(wrapped, cwd=str(self.repo))]), self.repo)[0]

    def test_slash_command_wrapper_with_args_recovered_with_typed_args(self):
        wrapped = (
            "<command-name>/remit</command-name>\n            "
            "<command-message>remit</command-message>\n            "
            "<command-args>everything owed</command-args>"
        )
        got = ace.claude_exchanges(self.path([cu(wrapped, cwd=str(self.repo))]), self.repo)[0]
        self.assertEqual([e.prompt for e in got], ["/remit everything owed"])

    def test_slash_command_wrapper_with_empty_args_recovered_as_bare_command(self):
        wrapped = (
            "<command-name>/remit</command-name>\n            "
            "<command-message>remit</command-message>\n            "
            "<command-args></command-args>"
        )
        got = ace.claude_exchanges(self.path([cu(wrapped, cwd=str(self.repo))]), self.repo)[0]
        self.assertEqual([e.prompt for e in got], ["/remit"])

    def test_slash_command_args_wrapper_with_mismatched_name_fails_loudly(self):
        wrapped = (
            "<command-name>/remit</command-name>\n            "
            "<command-message>different</command-message>\n            "
            "<command-args>everything owed</command-args>"
        )
        with self.assertRaises(ace.UserError):
            ace.claude_exchanges(self.path([cu(wrapped, cwd=str(self.repo))]), self.repo)[0]

    def test_local_command_and_its_stdout_both_vanish(self):
        cwd = str(self.repo)
        wrapped = (
            "<command-name>/remit</command-name>\n            "
            "<command-message>remit</command-message>\n            "
            "<command-args>everything owed</command-args>"
        )
        records = [
            cu("real question", ts=T0, cwd=cwd),
            ca([{"type": "text", "text": "Real answer."}], ts=T1, cwd=cwd),
            cu(wrapped, ts=T2, cwd=cwd, promptId="pq1"),
            cu("<local-command-stdout>Remitted.</local-command-stdout>",
               ts=T2, cwd=cwd, promptId="pq1"),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual([(e.prompt, e.reply) for e in got], [("real question", "Real answer.")])

    def test_unsent_local_command_gives_back_orphaned_edits(self):
        cwd = str(self.repo)
        patch = {
            "filePath": str(self.repo / "a.py"),
            "structuredPatch": [
                {"oldStart": 1, "oldLines": 1, "newStart": 1, "newLines": 1,
                 "lines": ["-old", "+new"]}
            ],
        }
        wrapped = (
            "<command-message>remit</command-message>\n"
            "<command-name>/remit</command-name>"
        )
        records = [
            cu("start the work", ts=T0, cwd=cwd),
            cu([{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
               ts=T1, cwd=cwd, toolUseResult=patch),
            cu(wrapped, ts=T2, cwd=cwd, promptId="pq1"),
            cu("<local-command-stdout>ok</local-command-stdout>", ts=T2, cwd=cwd, promptId="pq1"),
            cu("carry on", ts=T2, cwd=cwd),
            ca([{"type": "text", "text": "Done."}], ts="2026-03-01T10:15:00.000Z", cwd=cwd),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.reply, e.added, e.deleted) for e in got],
            [("start the work", "", 1, 1), ("carry on", "Done.", 0, 0)],
        )

    def test_local_stdout_without_its_command_fails_loudly(self):
        record = cu("<local-command-stdout>ok</local-command-stdout>", cwd=str(self.repo))
        with self.assertRaises(ace.UserError):
            ace.claude_exchanges(self.path([record]), self.repo)[0]

    def test_malformed_local_stdout_fails_loudly(self):
        record = cu("<local-command-stdout>truncated", cwd=str(self.repo))
        with self.assertRaises(ace.UserError):
            ace.claude_exchanges(self.path([record]), self.repo)[0]

    def test_typed_text_resembling_a_command_is_never_unsent(self):
        # Only harness-serialized command wrappers may be unsent by a stdout
        # record; pasted text that merely looks command-ish must not be, so
        # the stdout finds no paired command and the export crashes loudly.
        cwd = str(self.repo)
        records = [
            cu("<command-args>i pasted this myself</command-args>", ts=T0, cwd=cwd,
               promptId="pq1"),
            cu("<local-command-stdout>ok</local-command-stdout>", ts=T1, cwd=cwd,
               promptId="pq1"),
        ]
        with self.assertRaises(ace.UserError):
            ace.claude_exchanges(self.path(records), self.repo)[0]

    def test_rejection_reason_recovered_as_prompt(self):
        cwd = str(self.repo)
        denial = (
            "Error: The user doesn't want to proceed with this tool use. "
            "The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file). "
            "The user provided the following reason for the rejection:  hold tight, i'm catching you up"
        )
        records = [
            cu([{"type": "tool_result", "tool_use_id": "t1", "content": denial[7:]}], cwd=cwd, toolUseResult=denial),
            ca([{"type": "text", "text": "Understood."}], cwd=cwd),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.reply) for e in got],
            [("hold tight, i'm catching you up", "Understood.")],
        )

    def test_denial_reason_fanned_across_parallel_tools_collapses_to_one(self):
        cwd = str(self.repo)
        denial = (
            "Error: The user doesn't want to proceed with this tool use. "
            "The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file). "
            "The user provided the following reason for the rejection:  hold tight"
        )
        def denial_record(ts, tid):
            return cu(
                [{"type": "tool_result", "tool_use_id": tid, "content": denial[7:]}],
                ts=ts,
                cwd=cwd,
                toolUseResult=denial,
            )
        records = [
            denial_record(T0, "t1"),
            denial_record(T1, "t2"),
            ca([{"type": "text", "text": "Standing by."}], ts=T2, cwd=cwd),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual([(e.prompt, e.reply) for e in got], [("hold tight", "Standing by.")])
        self.assertEqual(got[0].timestamp, utc(T0))

    def test_identical_adjacent_denial_feedback_with_distinct_prompt_ids_stays_distinct(self):
        cwd = str(self.repo)
        denial = (
            "Error: The user doesn't want to proceed with this tool use. "
            "The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file). "
            "The user provided the following reason for the rejection:  hold tight"
        )
        records = [
            cu([{"type": "tool_result", "tool_use_id": "t1", "content": denial[7:]}],
               ts=T0, cwd=cwd, toolUseResult=denial, promptId="act-1"),
            cu([{"type": "tool_result", "tool_use_id": "t2", "content": denial[7:]}],
               ts=T1, cwd=cwd, toolUseResult=denial, promptId="act-2"),
            ca([{"type": "text", "text": "Standing by."}], ts=T2, cwd=cwd),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.reply) for e in got],
            [("hold tight", ""), ("hold tight", "Standing by.")],
        )

    def test_denial_fanout_with_differing_machine_wrappers_collapses(self):
        cwd = str(self.repo)
        family = (
            "The tool use was rejected (eg. if it was a file edit, "
            "the new_string was NOT written to the file). "
        )
        denials = [
            "Error: The user doesn't want to proceed with this tool use. "
            f"{family}The user provided the following reason for the rejection: hold tight",
            "Error: Permission for this tool use was denied. "
            f'{family}To tell you how to proceed, the user said: "hold tight"',
        ]
        records = [
            cu([{"type": "tool_result", "tool_use_id": f"t{i}", "content": denial[7:]}],
               ts=ts, cwd=cwd, toolUseResult=denial, promptId="one-human-act")
            for i, (ts, denial) in enumerate(zip((T0, T1), denials), start=1)
        ]
        records.append(ca([{"type": "text", "text": "Standing by."}], ts=T2, cwd=cwd))
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual([(e.prompt, e.reply) for e in got], [("hold tight", "Standing by.")])

    def test_identical_denial_feedback_after_more_agent_work_stays_distinct(self):
        cwd = str(self.repo)
        denial = (
            "Error: The user doesn't want to proceed with this tool use. "
            "The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file). "
            "The user provided the following reason for the rejection:  hold tight"
        )
        records = [
            cu([{"type": "tool_result", "tool_use_id": "t1", "content": denial[7:]}],
               ts=T0, cwd=cwd, toolUseResult=denial, promptId="reused-prompt"),
            ca([{"type": "tool_use", "id": "t2", "name": "Edit", "input": {}}],
               ts="2026-03-01T10:01:00.000Z", cwd=cwd),
            cu([{"type": "tool_result", "tool_use_id": "t2", "content": denial[7:]}],
               ts=T1, cwd=cwd, toolUseResult=denial, promptId="reused-prompt"),
            ca([{"type": "text", "text": "Standing by."}], ts=T2, cwd=cwd),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.reply) for e in got],
            [("hold tight", ""), ("hold tight", "Standing by.")],
        )

    def test_reasonless_denials_and_lookalike_tool_output_not_recovered(self):
        cwd = str(self.repo)
        family = (
            "The tool use was rejected (eg. if it was a file edit, "
            "the new_string was NOT written to the file). "
        )
        lookalike = "grep output: The user provided the following reason for the rejection: fake"
        records = [
            cu(
                [{"type": "tool_result", "tool_use_id": "t1", "content": "denied"}],
                cwd=cwd,
                toolUseResult=(
                    f"Error: The user doesn't want to proceed with this tool use. {family}"
                    "STOP what you are doing and wait for the user to tell you how to proceed.\n\n"
                    "Note: The user's next message may contain a correction."
                ),
            ),
            cu(
                [{"type": "tool_result", "tool_use_id": "t2", "content": "denied"}],
                ts=T1,
                cwd=cwd,
                toolUseResult=(
                    f"Error: Permission for this tool use was denied. {family}"
                    "Try a different approach or report the limitation to complete your task."
                ),
            ),
            cu(
                [{"type": "tool_result", "tool_use_id": "t3", "content": lookalike}],
                ts=T2,
                cwd=cwd,
                toolUseResult={"stdout": lookalike, "stderr": ""},
            ),
        ]
        self.assertEqual(ace.claude_exchanges(self.path(records), self.repo)[0], [])

    def test_plan_and_permission_feedback_variants_recovered(self):
        cwd = str(self.repo)
        family = (
            "The tool use was rejected (eg. if it was a file edit, "
            "the new_string was NOT written to the file). "
        )
        records = [
            cu(
                [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}],
                cwd=cwd,
                toolUseResult=(
                    f"Error: The user doesn't want to proceed with this tool use. {family}"
                    'The user said: "make the plan shorter"'
                ),
            ),
            cu(
                [{"type": "tool_result", "tool_use_id": "t2", "content": "x"}],
                ts=T1,
                cwd=cwd,
                toolUseResult=(
                    f"Error: Permission for this tool use was denied. {family}"
                    "To tell you how to proceed, the user said: use uv instead"
                ),
            ),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual([e.prompt for e in got], ["make the plan shorter", "use uv instead"])

    def test_mutated_answers_structure_fails_loudly(self):
        records = [
            cu(
                [{"type": "tool_result", "tool_use_id": "t1", "content": "answered"}],
                cwd=str(self.repo),
                toolUseResult={"questions": [{"question": "Q"}], "answers": "mutated-into-a-string"},
            ),
        ]
        with self.assertRaises(ace.UserError):
            ace.claude_exchanges(self.path(records), self.repo)[0]

    def test_unrecognized_denial_variant_fails_loudly(self):
        denial = (
            "Error: The user doesn't want to proceed with this tool use. "
            "The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file). "
            "Some brand new tail the binary grew overnight."
        )
        records = [
            cu(
                [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}],
                cwd=str(self.repo),
                toolUseResult=denial,
            ),
        ]
        with self.assertRaises(ace.UserError):
            ace.claude_exchanges(self.path(records), self.repo)[0]

    def test_answers_split_into_typed_asked_and_chosen(self):
        cwd = str(self.repo)
        result = {
            "questions": [
                {"question": "Q1", "header": "h", "options": [{"label": "Yes (Recommended)", "description": "d"}]},
                {"question": "Q2", "header": "h", "options": [{"label": "A", "description": "d"}]},
            ],
            "answers": {"Q1": "Yes (Recommended)", "Q2": "my own typed answer"},
        }
        records = [
            cu(
                [{"type": "tool_result", "tool_use_id": "t1", "content": "Your questions have been answered..."}],
                cwd=cwd,
                toolUseResult=result,
            ),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.ballots) for e in got],
            [(
                "my own typed answer",
                (
                    ace.Ballot("Q1", ("Yes (Recommended)",), ("Yes (Recommended)",)),
                    ace.Ballot("Q2", ("A",), ()),
                ),
            )],
        )

    def test_pure_click_answer_becomes_an_exchange(self):
        cwd = str(self.repo)
        result = {
            "questions": [{"question": "Q1", "header": "h", "options": [{"label": "Delete it", "description": "d"}]}],
            "answers": {"Q1": "Delete it"},
        }
        records = [
            cu(
                [{"type": "tool_result", "tool_use_id": "t1", "content": "Your questions have been answered..."}],
                cwd=cwd,
                toolUseResult=result,
            ),
            ca([{"type": "text", "text": "Deleting."}], cwd=cwd),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.ballots, e.reply) for e in got],
            [("", (ace.Ballot("Q1", ("Delete it",), ("Delete it",)),), "Deleting.")],
        )

    def test_multiselect_comma_joined_labels_detected_as_clicks(self):
        cwd = str(self.repo)
        result = {
            "questions": [{
                "question": "Q1",
                "header": "h",
                "options": [{"label": "Tutorial"}, {"label": "Sandbox"}, {"label": "Version tag"}],
            }],
            "answers": {"Q1": "Tutorial, Sandbox"},
        }
        records = [
            cu(
                [{"type": "tool_result", "tool_use_id": "t1", "content": "answered"}],
                cwd=cwd,
                toolUseResult=result,
            ),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(got[0].ballots[0].picked, ("Tutorial", "Sandbox"))
        self.assertEqual(got[0].prompt, "")

    def test_missing_timestamp_fails_loudly(self):
        path = self.path([cu("go", ts=None, cwd=str(self.repo))])
        with self.assertRaises(ace.UserError):
            ace.claude_exchanges(path, self.repo)[0]

    def test_live_capture_tolerates_truncated_final_line_only(self):
        path = self.tmp / "claude" / "p1" / "live.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        good = json.dumps(cu("hi", cwd=str(self.repo)))
        path.write_text(good + '\n{"type": "user", "mess', encoding="utf-8")
        got = ace.claude_exchanges(path, self.repo)[0]
        self.assertEqual([e.prompt for e in got], ["hi"])
        path.write_text('not json\n' + good + "\n", encoding="utf-8")
        with self.assertRaises(ace.UserError):
            ace.claude_exchanges(path, self.repo)[0]

    def test_malformed_jsonl_cites_file_and_line(self):
        path = self.tmp / "claude" / "p1" / "bad.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"type": "system"}\nnot json\n', encoding="utf-8")
        with self.assertRaises(ace.UserError) as ctx:
            ace.claude_exchanges(path, self.repo)[0]
        self.assertIn(str(path), str(ctx.exception))
        self.assertIn(":2", str(ctx.exception))


# --------------------------------------------------------------------- Codex

def cxmeta(cwd, source="vscode", originator="codex_vscode", sid="cx1"):
    return {
        "type": "session_meta",
        "timestamp": T0,
        "payload": {"id": sid, "cwd": cwd, "source": source, "originator": originator},
    }


def cxturn(model, cwd, effort=None, ts=T0):
    payload = {"model": model, "cwd": cwd}
    if effort is not None:
        payload["effort"] = effort
    return {"type": "turn_context", "timestamp": ts, "payload": payload}


def cxuser(message, ts=T0, images=None):
    payload = {"type": "user_message", "message": message}
    if images is not None:
        payload["images"] = images
    return {"type": "event_msg", "timestamp": ts, "payload": payload}


def cxagent(message, ts=T1):
    return {"type": "event_msg", "timestamp": ts, "payload": {"type": "agent_message", "message": message}}


def cxpatch(changes, success=True, ts=T1):
    payload = {
        "type": "patch_apply_end",
        "call_id": "c1",
        "stdout": "",
        "stderr": "",
        "success": success,
        "changes": changes,
    }
    return {"type": "event_msg", "timestamp": ts, "payload": payload}


class CodexQuals(Fixture):
    def path(self, records, name="rollout-1.jsonl"):
        return write_jsonl(self.tmp / "codex" / "sessions" / "2026" / name, records)

    def test_ide_wrapper_unwrapped_to_bare_request(self):
        wrapped = (
            "# Context from my IDE setup:\n\n## Active file: quals/README\n\n"
            "## Open tabs:\n- README: quals/README\n\n"
            "## My request for Codex:\nsure, let's see how it looks\n"
        )
        records = [
            cxmeta(str(self.repo)),
            cxturn("gpt-5.3-codex", str(self.repo), effort="xhigh"),
            cxuser(wrapped),
            cxagent("Looks fine."),
        ]
        got = ace.codex_exchanges(self.path(records), self.repo)[0]
        self.assertEqual([e.prompt for e in got], ["sure, let's see how it looks"])
        self.assertEqual(got[0].reply, "Looks fine.")
        self.assertEqual(got[0].model, "gpt-5.3-codex")
        self.assertEqual(got[0].effort, "xhigh")
        self.assertEqual(got[0].provider, "Codex")
        self.assertEqual(got[0].elapsed, 300.0)  # user at T0, agent_message at T1

    def test_plain_message_kept_verbatim(self):
        records = [cxmeta(str(self.repo)), cxuser("fix the  bug\nplease")]
        got = ace.codex_exchanges(self.path(records), self.repo)[0]
        self.assertEqual([e.prompt for e in got], ["fix the  bug\nplease"])

    def test_identical_consecutive_prompts_are_two_human_acts(self):
        records = [
            cxmeta(str(self.repo)),
            cxuser("retry", ts=T0),
            cxuser("retry", ts=T1),
            cxagent("done", ts=T2),
        ]
        got = ace.codex_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.reply) for e in got],
            [("retry", ""), ("retry", "done")],
        )

    def test_reasoning_and_token_counts_ignored(self):
        records = [
            cxmeta(str(self.repo)),
            cxuser("go"),
            {"type": "event_msg", "timestamp": T0, "payload": {"type": "agent_reasoning", "text": "mull"}},
            {"type": "event_msg", "timestamp": T0, "payload": {"type": "token_count", "info": {}}},
            cxagent("done"),
        ]
        got = ace.codex_exchanges(self.path(records), self.repo)[0]
        self.assertEqual([(e.prompt, e.reply) for e in got], [("go", "done")])

    def test_machine_sessions_skipped(self):
        subagent = [cxmeta(str(self.repo), source={"subagent": {"other": "guardian"}}), cxuser("audit")]
        titler = [cxmeta(str(self.repo), source="exec"), cxuser("Generate a concise UI title")]
        self.assertEqual(ace.codex_exchanges(self.path(subagent, "a.jsonl"), self.repo)[0], [])
        self.assertEqual(ace.codex_exchanges(self.path(titler, "b.jsonl"), self.repo)[0], [])

    def test_unknown_source_fails_loudly(self):
        path = self.path([cxmeta(str(self.repo), source="quantum"), cxuser("hi")])
        with self.assertRaises(ace.UserError) as ctx:
            ace.codex_exchanges(path, self.repo)[0]
        self.assertIn("quantum", str(ctx.exception))

    def test_cwd_outside_repo_excluded(self):
        records = [cxmeta("/elsewhere"), cxuser("hi")]
        self.assertEqual(ace.codex_exchanges(self.path(records), self.repo)[0], [])

    def test_agent_history_handoff_dropped(self):
        records = [
            cxmeta(str(self.repo)),
            cxuser("The following is the Codex agent history added since your last message."),
            cxuser("real question", ts=T1),
        ]
        got = ace.codex_exchanges(self.path(records), self.repo)[0]
        self.assertEqual([e.prompt for e in got], ["real question"])

    def test_pasted_image_data_uris_pass_through(self):
        records = [
            cxmeta(str(self.repo)),
            cxuser("see image", images=["data:image/png;base64,QUJD"]),
        ]
        got = ace.codex_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(got[0].images, ("data:image/png;base64,QUJD",))

    def test_elapsed_excludes_overnight_idle_before_next_turn(self):
        records = [
            cxmeta(str(self.repo)),
            cxuser("q1", ts=T0),
            cxagent("a1", ts=T1),
            cxturn("gpt-5.3-codex", str(self.repo), ts="2026-03-02T09:00:00.000Z"),
            cxuser("q2", ts="2026-03-02T09:00:01.000Z"),
            cxagent("a2", ts="2026-03-02T09:01:01.000Z"),
        ]
        got = ace.codex_exchanges(self.path(records), self.repo)[0]
        self.assertEqual([(e.prompt, e.elapsed) for e in got], [("q1", 300.0), ("q2", 60.0)])

    def test_elapsed_excludes_tool_output_gaps(self):
        records = [
            cxmeta(str(self.repo)),
            cxuser("go", ts=T0),
            {"type": "response_item", "timestamp": T1, "payload": {"type": "function_call", "name": "shell"}},
            {"type": "response_item", "timestamp": T2, "payload": {"type": "function_call_output", "output": "x"}},
            cxagent("done", ts="2026-03-01T10:15:00.000Z"),
        ]
        got = ace.codex_exchanges(self.path(records), self.repo)[0]
        # 300s (prompt -> call) + 300s (output -> reply); the call -> output
        # gap (tool runtime, possibly spanning a sleep) is never counted.
        self.assertEqual([(e.prompt, e.elapsed) for e in got], [("go", 600.0)])

    def test_wall_spans_prompt_to_last_reply(self):
        records = [
            cxmeta(str(self.repo)),
            cxuser("go", ts=T0),
            {"type": "response_item", "timestamp": T1, "payload": {"type": "function_call", "name": "shell"}},
            {"type": "response_item", "timestamp": T2, "payload": {"type": "function_call_output", "output": "x"}},
            cxagent("done", ts="2026-03-01T10:15:00.000Z"),
        ]
        got = ace.codex_exchanges(self.path(records), self.repo)[0]
        # Wall clock runs from the prompt to the final agent_message (900s),
        # exceeding elapsed (600s) by the uncounted tool-output gap.
        self.assertEqual([(e.prompt, e.elapsed, e.wall) for e in got], [("go", 600.0, 900.0)])

    def test_patch_lines_tallied_to_prompting_exchange(self):
        changes = {
            str(self.repo / "a.js"): {
                "type": "update",
                "move_path": None,
                "unified_diff": "@@ -1,2 +1,3 @@\n ctx\n-old\n+new one\n+new two",
            },
            str(self.repo / "b.js"): {"type": "add", "content": "one\ntwo\nthree"},
            str(self.repo / "c.js"): {"type": "delete", "content": "bye\nbye"},
        }
        records = [
            cxmeta(str(self.repo)),
            cxuser("build it", ts=T0),
            cxpatch(changes, ts=T1),
            cxagent("built", ts=T2),
            cxuser("just a question", ts="2026-03-01T10:15:00.000Z"),
            cxagent("answered", ts="2026-03-01T10:16:00.000Z"),
        ]
        got = ace.codex_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.added, e.deleted) for e in got],
            [("build it", 5, 3), ("just a question", 0, 0)],
        )

    def test_patch_at_eof_credited_to_inflight_exchange(self):
        changes = {
            str(self.repo / "a.js"): {
                "type": "update",
                "move_path": None,
                "unified_diff": "@@ -1 +1 @@\n-old\n+new",
            },
        }
        records = [
            cxmeta(str(self.repo)),
            cxuser("change it", ts=T0),
            {"type": "response_item", "timestamp": T1,
             "payload": {"type": "function_call", "name": "apply_patch"}},
            cxpatch(changes, ts=T2),
        ]
        got = ace.codex_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.reply, e.elapsed, e.added, e.deleted) for e in got],
            [("change it", "", 300.0, 1, 1)],
        )

    def test_orphaned_work_time_credited_to_interrupted_exchange(self):
        records = [
            cxmeta(str(self.repo)),
            cxuser("first", ts=T0),
            {"type": "response_item", "timestamp": T1,
             "payload": {"type": "function_call", "name": "shell"}},
            cxuser("second", ts=T2),
        ]
        got = ace.codex_exchanges(self.path(records), self.repo)[0]
        self.assertEqual([(e.prompt, e.elapsed) for e in got], [("first", 300.0), ("second", 0.0)])

    def test_failed_and_foreign_patches_not_tallied(self):
        outside = {"/somewhere/else/a.js": {"type": "add", "content": "x\ny"}}
        failed = {str(self.repo / "a.js"): {"type": "add", "content": "x\ny"}}
        records = [
            cxmeta(str(self.repo)),
            cxuser("go", ts=T0),
            cxpatch(outside, ts=T1),
            cxpatch(failed, success=False, ts=T1),
            cxagent("hm", ts=T2),
        ]
        got = ace.codex_exchanges(self.path(records), self.repo)[0]
        self.assertEqual([(e.added, e.deleted) for e in got], [(0, 0)])

    def test_orphaned_patch_credited_to_interrupted_exchange(self):
        changes = {
            str(self.repo / "a.js"): {
                "type": "update",
                "move_path": None,
                "unified_diff": "@@ -1,2 +1,3 @@\n ctx\n-old\n+new one\n+new two",
            },
        }
        records = [
            cxmeta(str(self.repo)),
            cxuser("q1", ts=T0),
            cxpatch(changes, ts=T1),
            cxuser("q2, before any reply", ts=T2),
            cxagent("done", ts="2026-03-01T10:15:00.000Z"),
        ]
        got = ace.codex_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.reply, e.added, e.deleted) for e in got],
            [("q1", "", 2, 1), ("q2, before any reply", "done", 0, 0)],
        )

    def test_canned_handoff_does_not_erase_pending_tally(self):
        changes = {str(self.repo / "a.js"): {"type": "add", "content": "x\ny\nz"}}
        records = [
            cxmeta(str(self.repo)),
            cxuser("q1", ts=T0),
            cxpatch(changes, ts=T1),
            cxuser("The following is the Codex agent history added since your last message.", ts=T2),
            cxagent("done", ts="2026-03-01T10:15:00.000Z"),
        ]
        got = ace.codex_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.reply, e.added, e.deleted) for e in got],
            [("q1", "done", 3, 0)],
        )

    def test_update_diff_with_file_headers_fails_loudly(self):
        # Every observed update diff starts at its first hunk; a diff bearing
        # +++/--- file headers means the format changed, and counting it
        # naively would miscount the headers as changed lines.
        changes = {
            str(self.repo / "a.js"): {
                "type": "update",
                "move_path": None,
                "unified_diff": "--- a/a.js\n+++ b/a.js\n@@ -1 +1 @@\n-x\n+y",
            },
        }
        records = [cxmeta(str(self.repo)), cxuser("go"), cxpatch(changes)]
        with self.assertRaises(ace.UserError) as ctx:
            ace.codex_exchanges(self.path(records), self.repo)[0]
        self.assertIn("codex_tally", str(ctx.exception))

    def test_update_with_empty_diff_counts_nothing(self):
        # A successfully applied patch can leave a file textually unchanged,
        # recorded as an update with an empty diff.
        changes = {
            str(self.repo / "a.js"): {
                "type": "update",
                "move_path": None,
                "unified_diff": "",
            },
        }
        records = [
            cxmeta(str(self.repo)),
            cxuser("q1", ts=T0),
            cxpatch(changes, ts=T1),
            cxagent("done", ts=T2),
        ]
        got = ace.codex_exchanges(self.path(records), self.repo)[0]
        self.assertEqual(
            [(e.prompt, e.reply, e.added, e.deleted) for e in got],
            [("q1", "done", 0, 0)],
        )

    def test_unknown_patch_change_form_fails_loudly(self):
        changes = {str(self.repo / "a.js"): {"type": "transmogrify"}}
        records = [cxmeta(str(self.repo)), cxuser("go"), cxpatch(changes)]
        with self.assertRaises(ace.UserError) as ctx:
            ace.codex_exchanges(self.path(records), self.repo)[0]
        self.assertIn("codex_tally", str(ctx.exception))

    def test_terminal_fix_template_dropped_even_in_codex(self):
        records = [
            cxmeta(str(self.repo)),
            cxuser("I get the following error. Please fix the error. Completed code only, no commentary.\n\n$ x\nboom"),
        ]
        self.assertEqual(ace.codex_exchanges(self.path(records), self.repo)[0], [])


# ---------------------------------------------------- paths under /home

class HomeQuals(Fixture):
    # A cwd as a Claude cloud workspace records it: another machine's
    # checkout, under that machine's /home.
    CLOUD = "/home/user/svenn"

    def transcripts(self):
        """A Claude Code transcript and a Codex session, each one prompt
        typed with its cwd at CLOUD."""
        claude = write_jsonl(
            self.tmp / "claude" / "p" / "sess.jsonl",
            [cu("start the work", ts=T0, cwd=self.CLOUD), ca([{"type": "text", "text": "Working."}], cwd=self.CLOUD)],
        )
        codex = write_jsonl(
            self.tmp / "codex" / "rollout-1.jsonl", [cxmeta(self.CLOUD), cxuser("start the work", ts=T0)]
        )
        return ((ace.claude_exchanges, claude), (ace.codex_exchanges, codex))

    def test_cwd_under_home_on_macos_refused_at_once(self):
        # Replicata: on macOS, a Claude Code transcript copied from another
        # machine (a Claude cloud workspace), its records' cwd under /home,
        # or a Codex session whose cwd is. Expectata: a loud error naming
        # the transcript and the path, saying it came from another machine
        # and its directories must first be rewritten to this machine's
        # checkout, raised before any path under /home is resolved: on
        # macOS /home is an automount point, where resolving a path can
        # hang for minutes. Resultata (v5.5.2): the cwd resolved, which on
        # this machine has hung for minutes, and the transcript then read
        # as outside the repo.
        with on_platform("darwin") as resolved:
            for parse, path in self.transcripts():
                with self.subTest(path=path):
                    with self.assertRaises(ace.UserError) as ctx:
                        parse(path, self.repo)
                    self.assertIn(str(path), str(ctx.exception))
                    self.assertIn(self.CLOUD, str(ctx.exception))
        self.assertEqual(resolved, [])

    def test_cwd_under_home_elsewhere_is_this_machines_own(self):
        # Replicata: on Linux, where /home holds this machine's own
        # checkouts, the same transcripts, read for the repo at their cwd.
        # Expectata: no error; each yields its prompt. Resultata (v5.5.2): as
        # expected; this guards the refusal of paths under /home on macOS
        # against reaching other platforms.
        with on_platform("linux"):
            for parse, path in self.transcripts():
                with self.subTest(path=path):
                    self.assertEqual([e.prompt for e in parse(path, Path(self.CLOUD))[0]], ["start the work"])


# ------------------------------------------------------ VS Code / Copilot Chat

def md(value):
    return {"value": value, "supportThemeIcons": False, "supportHtml": False}


def vsreq(text, response, model="copilot/gemini-3.1-pro-preview", ts=1772576233307, rid="r1", confirmation=None, variables=None, elapsed_ms=None):
    request = {
        "requestId": rid,
        "message": {"text": text, "parts": []},
        "modelId": model,
        "timestamp": ts,
        "response": response,
    }
    if confirmation is not None:
        request["confirmation"] = confirmation
    if variables is not None:
        request["variableData"] = {"variables": variables}
    if elapsed_ms is not None:
        request["result"] = {"timings": {"firstProgress": 1, "totalElapsed": elapsed_ms}}
    return request


def vssession(requests, version=3, sid="v1"):
    return {
        "version": version,
        "sessionId": sid,
        "creationDate": 1772576000000,
        "requesterUsername": "dreeves",
        "responderUsername": "GitHub Copilot",
        "requests": requests,
    }


def vsterminal_notice(response, rid="r2", ts=1772576233307, elapsed_ms=None):
    # VS Code starts a request by itself when a terminal command the agent
    # launched exits: the user-role text is its completion notice plus the
    # captured output, and the request carries three marker fields no typed
    # request has (shape observed 2026-09-10).
    request = vsreq(
        "[Terminal 00000000-0000-4000-8000-000000000000 notification: command "
        "completed with exit code 143. The terminal has been cleaned up.]\n"
        "Terminal output:\n$ python3 -m http.server 8000\n"
        "Serving HTTP on :: port 8000 (http://[::]:8000/) ...\nTerminated: 15\n",
        response,
        ts=ts,
        rid=rid,
        elapsed_ms=elapsed_ms,
    )
    request.update(
        isSystemInitiated=True,
        systemInitiatedLabel="` python3 -m http.server 8000` completed",
        terminalExecutionId="00000000-0000-4000-8000-000000000000",
    )
    return request


class VscodeQuals(Fixture):
    def write_session(self, state, name="s.json"):
        path = self.tmp / "chatSessions" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state), encoding="utf-8")
        return path

    def test_prompt_exact_reply_from_markdown_chunks(self):
        response = [
            {"kind": "mcpServersStarting"},
            {"kind": "thinking", "value": "hidden"},
            md("Chunk one "),
            {"kind": "toolInvocationSerialized", "toolId": "x"},
            md("chunk two."),
            {"kind": "prepareToolInvocation"},
            {"kind": "undoStop", "id": "u"},
            {"kind": "textEditGroup", "edits": []},
            {"kind": "codeblockUri"},
            {"kind": "progressTaskSerialized"},
            {"kind": "progressMessage"},
            {"kind": "confirmation"},
            {"kind": "elicitationSerialized"},
            {"kind": "elicitation"},
            {"kind": "warning", "content": md("w")},
            {"kind": "markdownVuln"},
        ]
        path = self.write_session(vssession([vsreq("do the  thing", response)]))
        got = ace.vscode_exchanges(path, (self.repo,), self.repo)[0]
        self.assertEqual([e.prompt for e in got], ["do the  thing"])
        self.assertEqual(got[0].reply, "Chunk one chunk two.")
        self.assertEqual(got[0].provider, "Copilot Chat")
        self.assertEqual(got[0].model, "copilot/gemini-3.1-pro-preview")
        self.assertEqual(got[0].timestamp, dt.datetime.fromtimestamp(1772576233307 / 1000, tz=dt.timezone.utc))

    def test_auto_mode_resolution_supplies_actual_model(self):
        response = [
            {
                "kind": "autoModeResolution",
                "resolvedModel": "gpt-5.4",
                "resolvedModelName": "GPT-5.4",
                "predictedLabel": "no_reasoning",
                "confidence": 0.95,
            },
            md("done"),
        ]
        path = self.write_session(vssession([vsreq("q", response, model="copilot/auto")]))
        got = ace.vscode_exchanges(path, (self.repo,), self.repo)[0]
        self.assertEqual([(e.reply, e.model) for e in got], [("done", "gpt-5.4")])

    def test_malformed_or_duplicate_auto_mode_resolution_fails_loudly(self):
        malformed = {"kind": "autoModeResolution"}
        valid = {"kind": "autoModeResolution", "resolvedModel": "gpt-5.4"}
        for response in ([malformed], [valid, valid]):
            with self.subTest(response=response):
                path = self.write_session(vssession([vsreq("q", response, model="copilot/auto")]))
                with self.assertRaises(ace.UserError) as ctx:
                    ace.vscode_exchanges(path, (self.repo,), self.repo)[0]
                self.assertIn("auto-mode", str(ctx.exception))

    def test_inline_reference_name_spliced_into_reply(self):
        response = [
            md("see "),
            {"kind": "inlineReference", "inlineReference": {"name": "foo.py", "location": {}}},
            md(" for details"),
        ]
        path = self.write_session(vssession([vsreq("q", response)]))
        got = ace.vscode_exchanges(path, (self.repo,), self.repo)[0]
        self.assertEqual(got[0].reply, "see foo.py for details")

    def test_unknown_response_kind_fails_loudly(self):
        path = self.write_session(vssession([vsreq("q", [{"kind": "hologram"}])]))
        with self.assertRaises(ace.UserError) as ctx:
            ace.vscode_exchanges(path, (self.repo,), self.repo)[0]
        self.assertIn("hologram", str(ctx.exception))
        self.assertIn(str(path), str(ctx.exception))

    def test_unknown_version_fails_loudly(self):
        path = self.write_session(vssession([vsreq("q", [])], version=2))
        with self.assertRaises(ace.UserError):
            ace.vscode_exchanges(path, (self.repo,), self.repo)[0]

    def test_mutation_log_replayed(self):
        base = vssession([])
        req0 = vsreq("first", [], ts=1772576233307, rid="r0")
        req1 = vsreq("second", [], ts=1772576240000, rid="r1")
        lines = [
            {"kind": 0, "v": base},
            {"kind": 1, "k": ["requests", 0], "v": req0},
            {"kind": 1, "k": ["requests", 1], "v": req1},
            {"kind": 2, "k": ["requests", 1, "response"], "i": 0, "v": [md("late reply")]},
            {"kind": 1, "k": ["customTitle"], "v": "t"},
            {"kind": 3, "k": ["customTitle"]},
        ]
        path = write_jsonl(self.tmp / "chatSessions" / "m.jsonl", lines)
        got = ace.vscode_exchanges(path, (self.repo,), self.repo)[0]
        self.assertEqual([e.prompt for e in got], ["first", "second"])
        self.assertEqual(got[1].reply, "late reply")

    def test_system_initiated_terminal_notice_dropped_typed_kept(self):
        # Replicata: a typed prompt, then the request VS Code fabricates when
        # the agent's terminal command exits (marked isSystemInitiated), then
        # another typed prompt. Expectata: exchanges for the two typed prompts
        # only. Resultata before the fix: the notice rendered as a third
        # human prompt, terminal log and all.
        requests = [
            vsreq("start the server", [md("started")], rid="r1", ts=1772576233307),
            vsterminal_notice([md("The server stopped.")], rid="r2", ts=1772576300000, elapsed_ms=26000),
            vsreq("thanks", [md("ok")], rid="r3", ts=1772576400000),
        ]
        path = self.write_session(vssession(requests))
        got = ace.vscode_exchanges(path, (self.repo,), self.repo)[0]
        self.assertEqual([e.prompt for e in got], ["start the server", "thanks"])

    def test_explicitly_not_system_initiated_request_kept(self):
        # An explicit false marker is a typed request: VS Code does store
        # explicit false for other boolean request fields.
        request = vsreq("typed", [md("r")])
        request["isSystemInitiated"] = False
        path = self.write_session(vssession([request]))
        got = ace.vscode_exchanges(path, (self.repo,), self.repo)[0]
        self.assertEqual([e.prompt for e in got], ["typed"])

    def test_unrecognized_system_initiated_marker_fails_loudly(self):
        # A marker that is neither true nor false could hide typing behind a
        # truthy value, so it must crash rather than be coerced either way.
        for marker in ("true", 1):
            request = vsreq("typed", [md("r")])
            request["isSystemInitiated"] = marker
            path = self.write_session(vssession([request]))
            with self.assertRaises(ace.UserError):
                ace.vscode_exchanges(path, (self.repo,), self.repo)[0]

    def test_button_and_template_requests_dropped_typed_kept(self):
        requests = [
            vsreq('@agent Continue: "Continue to iterate?"', [md("went on")], confirmation="Continue", rid="r1"),
            vsreq("@agent Try Again", [], confirmation="Try Again", rid="r2"),
            vsreq('@GitHubCopilot Enable: "Enable Gemini 2.5 Pro (Preview) for all clients"', [], rid="r3"),
            vsreq('@workspace /explain Expected ":"', [], rid="r4"),
            vsreq("Can you fix this error?\n\n$ python3 x.py\nTraceback", [], rid="r5"),
            vsreq(
                "I get the following error. Please fix the error. Completed code only, no commentary.\n\n$ x",
                [],
                rid="r6",
            ),
            vsreq("can you fix this:\n\n$ ./x.py\nTraceback", [md("done")], rid="r7"),
            vsreq(
                "(general reminder about AGENTS.md)\n\nCan you fix this error?\n\n$ python3 x.py\nTraceback",
                [],
                rid="r8",
            ),
        ]
        path = self.write_session(vssession(requests))
        got = ace.vscode_exchanges(path, (self.repo,), self.repo)[0]
        self.assertEqual(
            [e.prompt for e in got],
            ["can you fix this:\n\n$ ./x.py\nTraceback", "(general reminder about AGENTS.md)"],
        )

    def test_elapsed_taken_from_result_timings(self):
        path = self.write_session(vssession([vsreq("q", [md("r")], elapsed_ms=327000)]))
        got = ace.vscode_exchanges(path, (self.repo,), self.repo)[0]
        self.assertEqual(got[0].elapsed, 327.0)

    def test_elapsed_suppressed_when_turn_paused_for_confirmation(self):
        response = [md("r"), {"kind": "confirmation"}]
        path = self.write_session(vssession([vsreq("q", response, elapsed_ms=41000000)]))
        got = ace.vscode_exchanges(path, (self.repo,), self.repo)[0]
        self.assertEqual(got[0].elapsed, 0.0)

    def test_pasted_image_bytes_rebuilt_file_attachments_ignored(self):
        variables = [
            {"kind": "file", "id": "f", "name": "x.py", "value": {"path": "/x.py"}},
            {"kind": "image", "id": "i", "name": "Pasted Image", "mimeType": "image/png",
             "isPasted": True, "value": {"0": 65, "1": 66, "2": 67}},
        ]
        path = self.write_session(vssession([vsreq("q", [], variables=variables)]))
        got = ace.vscode_exchanges(path, (self.repo,), self.repo)[0]
        self.assertEqual(got[0].images, ("data:image/png;base64,QUJD",))

    def test_pasted_image_base64_field_decoded(self):
        variables = [
            {"kind": "image", "id": "i", "name": "Pasted Image", "mimeType": "image/png",
             "isPasted": True, "value": {"$base64": "QUJD"}},
        ]
        path = self.write_session(vssession([vsreq("q", [], variables=variables)]))
        got = ace.vscode_exchanges(path, (self.repo,), self.repo)[0]
        self.assertEqual(got[0].images, ("data:image/png;base64,QUJD",))

    def test_unrecognized_image_form_fails_loudly(self):
        # Each shape names the cause it must fail on, so a shape that fails for
        # some unrelated reason cannot pass by accident.
        shapes = [
            ("QUJD", type(None)),
            ({"$base64": "!!"}, binascii.Error),
            ({"$base64": 3}, TypeError),
            ({"nope": 1}, KeyError),
            ({"0": "A"}, TypeError),
        ]
        for value, cause in shapes:
            with self.subTest(value=value):
                variables = [{"kind": "image", "id": "i", "name": "Pasted Image",
                              "mimeType": "image/png", "value": value}]
                path = self.write_session(vssession([vsreq("q", [], variables=variables)]))
                with self.assertRaises(ace.UserError) as ctx:
                    ace.vscode_exchanges(path, (self.repo,), self.repo)[0]
                self.assertIn(str(path), str(ctx.exception))
                self.assertIsInstance(ctx.exception.__cause__, cause)

    def test_workspace_outside_repo_excluded_and_empty_session_ok(self):
        path = self.write_session(vssession([vsreq("q", [])]))
        other = self.tmp / "other"
        other.mkdir()
        self.assertEqual(ace.vscode_exchanges(path, (other,), self.repo)[0], [])
        empty = self.write_session(vssession([]), "e.json")
        self.assertEqual(ace.vscode_exchanges(empty, (self.repo,), self.repo)[0], [])

    def test_straddling_multiroot_workspace_fails_loudly(self):
        other = self.tmp / "other"
        other.mkdir()
        path = self.write_session(vssession([vsreq("q", [])]))
        with self.assertRaises(ace.UserError):
            ace.vscode_exchanges(path, (self.repo, other), self.repo)[0]

    def test_strip_jsonc_preserves_commas_inside_strings(self):
        raw = '{\n  // note\n  "folders": [\n    {"path": "we,]ird, }name"}, /* x */\n  ],\n}\n'
        parsed = json.loads(ace.strip_jsonc(raw))
        self.assertEqual(parsed["folders"][0]["path"], "we,]ird, }name")

    def test_workspace_roots_folder_and_jsonc_workspace_file(self):
        storage_a = self.tmp / "storageA"
        storage_a.mkdir()
        (storage_a / "workspace.json").write_text(
            json.dumps({"folder": self.repo.as_uri()}), encoding="utf-8"
        )
        self.assertEqual(ace.workspace_roots(storage_a), (self.repo,))

        code_workspace = self.tmp / "multi.code-workspace"
        code_workspace.write_text(
            '{\n  // comment\n  "folders": [\n    {"path": "repo"}, /* inline */\n  ],\n}\n',
            encoding="utf-8",
        )
        storage_b = self.tmp / "storageB"
        storage_b.mkdir()
        (storage_b / "workspace.json").write_text(
            json.dumps({"workspace": code_workspace.as_uri()}), encoding="utf-8"
        )
        self.assertEqual(ace.workspace_roots(storage_b), (self.repo,))


# ------------------------------------------------------------ weave and render

class WeaveQuals(Fixture):
    def test_orders_across_providers_by_timestamp(self):
        a = exchange(timestamp=utc(T1), provider="Codex", prompt="b")
        b = exchange(timestamp=utc(T0), provider="Copilot Chat", prompt="a")
        c = exchange(timestamp=utc(T2), provider="Claude Code", prompt="c")
        self.assertEqual([e.prompt for e in ace.weave([a, b, c])], ["a", "b", "c"])

    def test_duplicate_content_collapsed_across_sessions_and_files(self):
        a = exchange(session="s1", source=Path("/f1"))
        b = exchange(session="s2", source=Path("/f2"))
        self.assertEqual(len(ace.weave([a, b])), 1)

    def test_differing_exchange_metadata_is_not_deduplicated(self):
        original = exchange(session="s1", source=Path("/f1"))
        variants = [
            exchange(session="s2", source=Path("/f2"), added=1),
            exchange(session="s2", source=Path("/f2"), deleted=1),
            exchange(session="s2", source=Path("/f2"), elapsed=1.0),
            exchange(session="s2", source=Path("/f2"), model="another-model"),
            exchange(session="s2", source=Path("/f2"), effort="high"),
            exchange(session="s2", source=Path("/f2"), images=("data:image/png;base64,AA",)),
        ]
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertEqual(len(ace.weave([original, variant])), 2)


class RenderQuals(Fixture):
    def test_prompt_visible_exact_reply_collapsed(self):
        page = ace.render(
            self.repo,
            {LOGIN: [exchange(prompt="a<b>&c\n  indented", reply="<script>alert(1)</script>")]},
        )
        self.assertIn("a&lt;b&gt;&amp;c\n  indented", page)
        self.assertNotIn("<script>alert(1)", page)
        self.assertIn("<details>", page)
        self.assertNotIn("<details open", page)
        prompt_at = page.index("a&lt;b&gt;")
        details_at = page.index("<details>")
        self.assertLess(prompt_at, details_at)

    def test_meta_line_carries_time_provider_model(self):
        page = ace.render(self.repo, {LOGIN: [exchange()]})
        local = utc(T0).astimezone().strftime("%H:%M")
        summary = page[page.index("<summary") : page.index("</summary>")]
        self.assertIn(local, summary)
        self.assertIn("Claude Code", summary)
        self.assertIn("claude-opus-4-8", summary)
        self.assertNotIn("(", summary)

    def test_meta_line_names_each_prompts_human_by_display_name(self):
        # Replicata: a page rendered from three humans' ledgers, given in no
        # particular order: dreeves and mister-person, whom the display
        # table names, and a login it does not name. Expectata: the rows
        # interleave by time; each article records its login invisibly, in
        # a data-login attribute; each meta line shows its human's display
        # name (the table's, else the login itself) right after the time.
        # Resultata (v5.5.2): render took no humans, and no line named one.
        page = ace.render(
            self.repo,
            {
                "mister-person": [exchange(timestamp=utc(T1), prompt="b")],
                "dreeves": [exchange(timestamp=utc(T3), prompt="d"), exchange(prompt="a")],
                "someone-else": [exchange(timestamp=utc(T2), prompt="c")],
            },
        )
        articles = re.findall(
            r'<article class="exchange claude" id="p(\d)" data-login="([a-z-]+)">'
            r'.*?<pre class="prompt">(\w)</pre>.*?<summary>(.*?)</summary>',
            page,
            re.DOTALL,
        )
        self.assertEqual(
            [(number, login, prompt) for number, login, prompt, _ in articles],
            [("1", "dreeves", "a"), ("2", "mister-person", "b"), ("3", "someone-else", "c"), ("4", "dreeves", "d")],
        )
        for (_, _, _, summary), name in zip(articles, ("dreev", "logan", "someone-else", "dreev")):
            self.assertIn(f'</time></a> <span class="human">{name}</span> <span class="chip"></span>', summary)

    def test_single_human_page_names_its_human_too(self):
        # Replicata: a page rendered from one ledger. Expectata: its meta
        # line still shows the human's display name. Resultata (v5.5.2): no
        # human named.
        page = ace.render(self.repo, {"dreeves": [exchange()]})
        self.assertIn('data-login="dreeves">', page)
        self.assertIn('<span class="human">dreev</span>', page)

    def test_display_names_are_the_approved_table(self):
        # Replicata: the display table, an expedient stand-in for names
        # humans choose. Expectata: exactly the two names the human
        # approved; every other login displays as itself. Resultata
        # (v5.5.2): no table.
        self.assertEqual(ace.DISPLAY_NAMES, {"dreeves": "dreev", "mister-person": "logan"})
        self.assertEqual(
            [ace.display_name(login) for login in ("dreeves", "mister-person", "dreev", "logan")],
            ["dreev", "logan", "dreev", "logan"],
        )

    def test_effort_shown_in_parens_after_model(self):
        page = ace.render(self.repo, {LOGIN: [exchange(effort="xhigh")]})
        summary = page[page.index("<summary") : page.index("</summary>")]
        self.assertIn("claude-opus-4-8", summary)
        self.assertIn("(xhigh)", summary)
        self.assertLess(summary.index("claude-opus-4-8"), summary.index("(xhigh)"))

    def test_one_day_header_per_local_day_with_weekday(self):
        page = ace.render(
            self.repo,
            {LOGIN: [exchange(prompt="x"), exchange(timestamp=utc(T1), prompt="y")]},
        )
        self.assertEqual(page.count('class="day"'), 1)
        local = utc(T0).astimezone().date()
        self.assertIn(f"{local.isoformat()} {ace.WEEKDAYS[local.weekday()]}<", page)

    def test_thought_duration_shown_when_known(self):
        page = ace.render(self.repo, {LOGIN: [exchange(elapsed=327.0)]})
        summary = page[page.index("<summary") : page.index("</summary>")]
        self.assertIn("thought for 5m27s", summary)
        page = ace.render(self.repo, {LOGIN: [exchange()]})
        self.assertNotIn("thought for", page)

    def test_provider_identity_stripe_class_and_chip(self):
        page = ace.render(
            self.repo,
            {LOGIN: [
                exchange(),
                exchange(timestamp=utc(T1), provider="Codex", prompt="b"),
                exchange(timestamp=utc(T2), provider="Copilot Chat", prompt="c"),
            ]},
        )
        # Each exchange wears its provider's class so the CSS edge stripe and
        # meta-line chip can color by agent.
        for slug in ("claude", "codex", "copilot"):
            self.assertIn(f'<article class="exchange {slug}"', page)
        # The chip is a mark beside the agent name — the name itself stays in
        # ink (text never wears mark colors), one chip per exchange.
        self.assertEqual(page.count('<span class="chip"></span>'), 3)
        self.assertLess(page.index('class="chip"'), page.index('class="agent"'))

    def test_meta_line_wraps_between_items_and_the_chip_keeps_its_size(self):
        # Replicata: a page read at phone width (390px), where an exchange's
        # meta line (time, chip, agent, model, effort, working time) is
        # wider than the column. Expectata: the chip stays an 8px square,
        # and the line wraps between its items, never inside one. Resultata
        # (v5.5.1): the items were squeezed into one row, the chip shrunk to
        # a sliver and the model name broken at its hyphens.
        self.assertIn(
            ".chip { display: inline-block; flex: none; width: 8px; height: 8px;"
            " border-radius: 2px; background: var(--provider); }",
            ace.CSS,
        )
        rule = ace.CSS[ace.CSS.index("\nsummary {") :]
        rule = rule[: rule.index("}")]
        self.assertIn("\n  flex-wrap: wrap;\n", rule)

    def test_display_name_wears_the_agent_names_ink(self):
        # Replicata: a meta line, where the human's display name (span.human)
        # sits beside the agent's name (span.agent). Expectata: the
        # stylesheet gives .human the ink .agent wears, var(--muted), and
        # not .agent's weight. Resultata (v6.0.0): no rule for .human, so the
        # name wore the summary's faint ink.
        self.assertIn("\n.human { color: var(--muted); }\n", ace.CSS)
        self.assertIn("\n.agent { color: var(--muted); font-weight: 600; }\n", ace.CSS)

    def test_unknown_provider_fails_loudly(self):
        with self.assertRaises(KeyError):
            ace.render(self.repo, {LOGIN: [exchange(provider="Quantum")]})

    def test_claude_ai_exchange_wears_claude_codes_color(self):
        # Replicata: an exchange from a claude.ai chat, which reaches a page
        # only through a ledger (no store parser reads claude.ai).
        # Expectata: its article carries the class claudeai, whose stripe
        # and chip take Claude Code's color, defined for both color schemes,
        # and its meta line names claude.ai. Resultata (v5.5.0): KeyError,
        # claude.ai being no known provider.
        page = ace.render(self.repo, {LOGIN: [exchange(provider="claude.ai", model="claude-opus-5-5")]})
        self.assertIn('<article class="exchange claudeai"', page)
        self.assertIn('<span class="agent">claude.ai</span>', page)
        self.assertIn(".exchange.claudeai { --provider: var(--claude); }", ace.CSS)
        self.assertEqual(ace.CSS.count("--claude:"), 2)

    def test_provider_slugs_are_unique(self):
        # Replicata: invert PROVIDER_SLUGS, as page_credits does to read a
        # page's provider back from its article class. Expectata: no slug
        # lost, so every slug names one provider. Resultata (v5.5.0): as
        # expected; this guards each new provider's slug.
        self.assertEqual(len(set(ace.PROVIDER_SLUGS.values())), len(ace.PROVIDER_SLUGS))

    def test_autogenerated_warning_comment_after_doctype(self):
        # The warning must follow the doctype: a comment before it would
        # throw browsers into quirks mode.
        page = ace.render(self.repo, {LOGIN: [exchange()]})
        self.assertTrue(page.startswith("<!doctype html>\n<!-- "), page[:60])
        self.assertLess(page.index("-->"), page.index("<html"))

    def test_wall_clock_shown_only_when_beyond_thought(self):
        page = ace.render(self.repo, {LOGIN: [exchange(elapsed=327.0, wall=540.0)]})
        summary = page[page.index("<summary") : page.index("</summary>")]
        self.assertIn("thought for 5m27s · 9m wall-clock time", summary)
        # A wall span matching the working time at display precision adds
        # nothing and stays off,
        page = ace.render(self.repo, {LOGIN: [exchange(elapsed=327.0, wall=327.4)]})
        self.assertNotIn("wall-clock time", page)
        # as does one the working time exceeds (activity credited off records
        # later than the last reply),
        page = ace.render(self.repo, {LOGIN: [exchange(elapsed=327.0, wall=300.0)]})
        self.assertNotIn("wall-clock time", page)
        # and one on an exchange with no working time shown at all.
        page = ace.render(self.repo, {LOGIN: [exchange(wall=540.0)]})
        self.assertNotIn("thought for", page)
        self.assertNotIn("wall-clock time", page)

    def test_diffstat_shown_with_ratio_blocks(self):
        page = ace.render(self.repo, {LOGIN: [exchange(added=1234, deleted=45, prompt="hi")]})
        self.assertIn("+1,234 −45", page)
        self.assertEqual(page.count('<span class="add"></span>'), 4)  # round(5·1234/1279)
        self.assertEqual(page.count('<span class="del"></span>'), 1)
        self.assertEqual(page.count('<span class="nil"></span>'), 0)
        # The stat precedes the prompt so it floats beside the prompt's top.
        self.assertLess(page.index('class="diffstat"'), page.index('class="prompt"'))
        # A prompt that touched no code gets no diffstat at all.
        page = ace.render(self.repo, {LOGIN: [exchange()]})
        self.assertNotIn('<div class="diffstat">', page)

    def test_diffstat_tiny_and_onesided_ratios(self):
        page = ace.render(self.repo, {LOGIN: [exchange(added=1, deleted=1)]})
        self.assertEqual(page.count('<span class="add"></span>'), 1)
        self.assertEqual(page.count('<span class="del"></span>'), 1)
        self.assertEqual(page.count('<span class="nil"></span>'), 3)
        page = ace.render(self.repo, {LOGIN: [exchange(added=0, deleted=7)]})
        self.assertIn("+0 −7", page)
        self.assertEqual(page.count('<span class="add"></span>'), 0)
        self.assertEqual(page.count('<span class="del"></span>'), 5)
        # A nonzero side always gets at least one block.
        page = ace.render(self.repo, {LOGIN: [exchange(added=1, deleted=999)]})
        self.assertEqual(page.count('<span class="add"></span>'), 1)
        self.assertEqual(page.count('<span class="del"></span>'), 4)
        # An even split renders symmetrically, remainder unfilled.
        page = ace.render(self.repo, {LOGIN: [exchange(added=10, deleted=10)]})
        self.assertEqual(page.count('<span class="add"></span>'), 2)
        self.assertEqual(page.count('<span class="del"></span>'), 2)
        self.assertEqual(page.count('<span class="nil"></span>'), 1)

    def test_deck_totals_shown_when_any_lines_counted(self):
        page = ace.render(
            self.repo,
            {LOGIN: [exchange(added=2, deleted=1), exchange(timestamp=utc(T1), prompt="b", added=3)]},
        )
        local = utc(T0).astimezone().date().isoformat()
        deck = f"2 prompts · {local} · +5 −1"
        self.assertIn(f'<p class="deck">{deck}</p>', page)
        self.assertIn(f'<meta name="description" content="{deck}">', page)
        page = ace.render(self.repo, {LOGIN: [exchange()]})
        self.assertNotIn("+0 −0", page)

    def test_minimap_sliver_per_prompt_linked_and_scaled(self):
        page = ace.render(
            self.repo,
            {LOGIN: [
                exchange(added=100, deleted=0),
                exchange(timestamp=utc(T1), prompt="b", added=25, deleted=4),
                exchange(timestamp=utc(T2), prompt="c"),
            ]},
        )
        svg = page[page.index('<svg class="minimap"') : page.index("</svg>")]
        self.assertEqual(svg.count("<a "), 3)
        for anchor in ('href="#p1"', 'href="#p2"', 'href="#p3"'):
            self.assertIn(anchor, svg)
        # Square-root scale against the peak side (100): 100 -> 30 units,
        # 25 -> 15, and the second prompt's 4 deleted -> 6.
        self.assertIn('height="30"', svg)
        self.assertIn('height="15"', svg)
        self.assertIn('height="6"', svg)
        # A prompt that touched no code still gets a tick and its link,
        # and every sliver has a full-height hit target.
        self.assertIn('class="nil"', svg)
        self.assertEqual(svg.count('class="hit"'), 3)
        tick_time = utc(T2).astimezone().strftime("%H:%M")
        self.assertIn(f"<title>{tick_time}</title>", svg)

    def test_elapsed_text_formats(self):
        for seconds, expected in ((327, "5m27s"), (45, "45s"), (300, "5m"), (3661, "1h1m1s"), (0.4, "0s")):
            self.assertEqual(ace.elapsed_text(seconds), expected)

    def test_pasted_images_rendered_with_prompt_not_collapsed(self):
        page = ace.render(self.repo, {LOGIN: [exchange(images=("data:image/png;base64,QUJD",))]})
        img_at = page.index('src="data:image/png;base64,QUJD"')
        self.assertLess(page.index('class="prompt"'), img_at)
        self.assertLess(img_at, page.index("<details>"))

    def test_generator_attribution_links_home(self):
        page = ace.render(self.repo, {LOGIN: [exchange()]})
        self.assertIn(
            '<a href="https://github.com/beeminder/sourcery">generated by sourcery</a>',
            page,
        )

    def test_remote_link_replaces_directory_when_known(self):
        page = ace.render(self.repo, {LOGIN: [exchange()]}, remote="https://github.com/dreeves/crashla")
        self.assertIn('<a href="https://github.com/dreeves/crashla">github.com/dreeves/crashla</a>', page)
        self.assertNotIn(str(self.repo), page)
        page = ace.render(self.repo, {LOGIN: [exchange()]})
        self.assertIn(str(self.repo), page)
        self.assertNotIn("github.com/dreeves/crashla", page)

    def test_expand_controls_and_permalink_anchors(self):
        page = ace.render(
            self.repo,
            {LOGIN: [exchange(prompt="a"), exchange(timestamp=utc(T1), prompt="b")]},
        )
        self.assertIn('data-omnia="open"', page)
        self.assertIn('data-omnia="close"', page)
        self.assertIn("<script>", page)
        self.assertIn('id="p1"', page)
        self.assertIn('href="#p2"', page)

    def test_empty_reply_marked(self):
        page = ace.render(self.repo, {LOGIN: [exchange(reply="")]})
        self.assertIn('class="reply machine empty"', page)

    def test_only_final_empty_reply_marked_still_generating(self):
        page = ace.render(
            self.repo,
            {LOGIN: [exchange(prompt="a", reply=""), exchange(timestamp=utc(T1), prompt="b", reply="")]},
        )
        self.assertEqual(page.count("Response still generating when this transcript was captured"), 1)
        self.assertIn("No response.", page)
        self.assertLess(page.index("No response."), page.index("Response still generating"))
        page = ace.render(self.repo, {LOGIN: [exchange(prompt="a", reply=""), exchange(timestamp=utc(T1), prompt="b", reply="done")]})
        self.assertNotIn("Response still generating", page)

    def test_machine_prose_only_inside_machine_containers(self):
        ballot = ace.Ballot("Q?", ("A label", "B label"), ("A label",))
        page = ace.render(
            self.repo,
            {LOGIN: [exchange(prompt="typed words", reply="agent words", ballots=(ballot,))]},
        )
        self.assertIn('<div class="reply machine">', page)
        self.assertIn('<div class="ballot machine">', page)
        self.assertIn('<div class="ballot-question">Q?</div>', page)
        self.assertIn('<div class="option picked">✓ A label</div>', page)
        self.assertIn('<div class="option">· B label</div>', page)
        # The human's prompt must not sit inside any machine container.
        pre = page[page.index('<pre class="prompt">') : page.index("</pre>")]
        self.assertIn("typed words", pre)
        self.assertNotIn("machine", pre)

    def test_ballot_stays_default_visible_outside_the_disclosure(self):
        # Deliberate exception to machine-words-collapsed: the human's picks
        # are legible only against the agent's question and labels, so the
        # ballot renders before the closed <details>, still in phosphor.
        ballot = ace.Ballot("Q?", ("A label", "B label"), ("A label",))
        page = ace.render(self.repo, {LOGIN: [exchange(prompt="typed words", ballots=(ballot,))]})
        article = page[page.index("<article") : page.index("</article>")]
        self.assertLess(article.index('<div class="ballot machine">'), article.index("<details>"))

    def test_click_only_answer_renders_without_prompt_block(self):
        ballot = ace.Ballot("Q?", ("Delete it", "Keep it"), ("Delete it",))
        page = ace.render(self.repo, {LOGIN: [exchange(prompt="", ballots=(ballot,))]})
        self.assertNotIn('<pre class="prompt">', page)
        self.assertIn("✓ Delete it", page)
        self.assertIn("· Keep it", page)

    def test_progress_rail_bar_and_one_mark_per_day(self):
        # Replicata: a dialog spanning two days. Expectata: the page carries
        # the Asterisk-style reading-progress rail — a fixed bar plus one
        # clickable day mark per day header, each mark's label the header's
        # exact text and its target an anchor the header itself carries.
        page = ace.render(
            self.repo,
            {LOGIN: [exchange(prompt="a"), exchange(timestamp=utc("2026-03-05T10:00:00.000Z"), prompt="b")]},
        )
        self.assertIn('<div id="progress">', page)
        self.assertIn('class="progress-bar"', page)
        days = re.findall(
            r'<h2 class="day" id="([^"]+)"><time datetime="[0-9-]+">([^<]+)</time></h2>', page
        )
        self.assertEqual(len(days), 2)
        self.assertEqual(days[0][0], f"d{utc(T0).astimezone().date().isoformat()}")
        marks = re.findall(
            r'<a class="daymark" href="#([^"]+)">'
            r'<span class="tick"></span><span class="text">([^<]+)</span></a>',
            page,
        )
        self.assertEqual(marks, days)
        # Two prompts on one day are one day header, so one mark.
        page = ace.render(
            self.repo, {LOGIN: [exchange(prompt="a"), exchange(timestamp=utc(T1), prompt="b")]}
        )
        self.assertEqual(page.count('class="daymark"'), 1)

    def test_progress_rail_revealed_and_driven_by_script(self):
        # The rail appears only once the reader has actually set off: the
        # stylesheet ships the bar transparent and the marks offscreen, and
        # the script shows them past 300px of scroll, resizing the bar on
        # scroll and resize and after any disclosure opens or closes (which
        # reflows the whole page under the rail).
        page = ace.render(self.repo, {LOGIN: [exchange()]})
        self.assertIn("scrollY > 300", page)
        self.assertIn('addEventListener("scroll", sync, { passive: true })', page)
        self.assertIn('addEventListener("resize", sync)', page)
        self.assertIn('document.addEventListener("toggle", sync, true)', page)
        self.assertIn("opacity: 0;", ace.CSS)
        self.assertIn("#progress.show .progress-bar { opacity: 1; }", ace.CSS)
        self.assertIn("#progress.show .daymark { transform: none; }", ace.CSS)

    def test_progress_rail_absent_from_print(self):
        # Scroll progress means nothing on paper; the print stylesheet drops
        # the whole rail.
        self.assertLess(ace.CSS.index("@media print"), ace.CSS.index("#progress { display: none; }"))

    def test_reply_markdown_subset(self):
        html = ace.markdown_html("intro `x` **b** [l](https://e.com)\n\n```py\nx = 1 < 2\n```\n- item")
        self.assertIn("<code>x</code>", html)
        self.assertIn("<strong>b</strong>", html)
        self.assertIn('<a href="https://e.com">l</a>', html)
        self.assertIn("x = 1 &lt; 2", html)
        self.assertIn("<li>item</li>", html)
        self.assertNotIn("[l]", html)

    def test_reply_markdown_link_query_is_escaped_once(self):
        got = ace.markdown_inline("[query](https://example.test/search?a=1&b=2)")
        self.assertEqual(got, '<a href="https://example.test/search?a=1&amp;b=2">query</a>')

    def test_reply_markdown_link_cannot_inject_an_attribute_through_code(self):
        class AnchorParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.attrs = []

            def handle_starttag(self, tag, attrs):
                if tag == "a":
                    self.attrs.append(attrs)

        parser = AnchorParser()
        parser.feed(ace.markdown_inline('[x](https://e.test/`" onmouseover="alert(1)`)'))
        self.assertEqual([[name for name, _ in attrs] for attrs in parser.attrs], [["href"]])


# ------------------------------------------------------------------------ CLI

class CliQuals(Fixture):
    def setUp(self):
        super().setUp()
        self.claude_root = self.tmp / "claude"
        self.codex_root = self.tmp / "codex"
        self.vscode_root = self.tmp / "vscode"
        self.antigravity_root = self.tmp / "antigravity"
        for root in (self.claude_root, self.codex_root, self.vscode_root, self.antigravity_root):
            root.mkdir()
        self.env = {
            "AI_CHAT_CLAUDE_ROOTS": str(self.claude_root),
            "AI_CHAT_CODEX_ROOTS": str(self.codex_root),
            "AI_CHAT_VSCODE_USER_ROOTS": str(self.vscode_root),
            "AI_CHAT_ANTIGRAVITY_ROOTS": str(self.antigravity_root),
        }
        # Fixtures are sealed with the test key; the real finder would look
        # for the application's key in the installed binary.
        self.finder = ace.antigravity_key
        ace.antigravity_key = lambda: AG_TEST_KEY
        # The runner is LOGIN; the real seam would ask the installed gh.
        self.gh_user = ace.gh_user
        ace.gh_user = gh_says(f"{LOGIN}\n")

    def tearDown(self):
        ace.antigravity_key = self.finder
        ace.gh_user = self.gh_user

    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = ace.run(argv, self.env)
        return code, out.getvalue(), err.getvalue()

    def populate(self):
        write_jsonl(
            self.claude_root / "p" / "s.jsonl",
            [
                cu("claude prompt", ts=T0, cwd=str(self.repo)),
                ca([{"type": "text", "text": "claude reply"}], cwd=str(self.repo)),
            ],
        )
        write_jsonl(
            self.codex_root / "sessions" / "2026" / "rollout-1.jsonl",
            [cxmeta(str(self.repo)), cxuser("codex prompt", ts=T1), cxagent("codex reply", ts=T2)],
        )
        storage = self.vscode_root / "workspaceStorage" / "h1"
        storage.mkdir(parents=True)
        (storage / "workspace.json").write_text(
            json.dumps({"folder": self.repo.as_uri()}), encoding="utf-8"
        )
        (storage / "chatSessions").mkdir()
        (storage / "chatSessions" / "a.json").write_text(
            json.dumps(vssession([vsreq("copilot prompt", [md("copilot reply")], ts=int(utc(T2).timestamp() * 1000))])),
            encoding="utf-8",
        )

    def test_end_to_end_all_three_providers_in_order(self):
        self.populate()
        out_path = self.tmp / "out.html"
        code, out, err = self.run_cli([str(self.repo), str(out_path)])
        self.assertEqual(code, 0, err)
        page = out_path.read_text(encoding="utf-8")
        order = [page.index("claude prompt"), page.index("codex prompt"), page.index("copilot prompt")]
        self.assertEqual(order, sorted(order))
        for label in ("Claude Code", "Codex", "Copilot Chat"):
            self.assertIn(label, page)
        self.assertIn(str(out_path), out)

    def test_wrong_arity_and_unknown_arguments_rejected(self):
        out = str(self.tmp / "out.html")
        self.assertEqual(self.run_cli([])[0], 2)
        self.assertEqual(self.run_cli([str(self.repo)])[0], 2)
        self.assertEqual(self.run_cli([str(self.repo), out, "extra"])[0], 2)
        self.assertEqual(self.run_cli(["--bogus", str(self.repo), out])[0], 2)

    def test_retired_flag_spellings_rejected(self):
        code, _, err = self.run_cli(["--repo", str(self.repo), "--output", str(self.tmp / "o.html")])
        self.assertEqual(code, 2)
        self.assertIn("--repo", err)

    def test_missing_repo_directory_rejected(self):
        code, _, err = self.run_cli([str(self.tmp / "absent"), str(self.tmp / "o.html")])
        self.assertEqual(code, 2)
        self.assertIn("absent", err)

    def test_existing_output_crediting_no_ledger_rejected(self):
        # Replicata: the output path already holds a file crediting no
        # ledger — a page from before snapshots, or not a sourcery page at
        # all. Expectata: exit 2 naming the path, file left byte-identical.
        # Resultata before snapshots: silently overwritten.
        self.populate()
        out_path = self.tmp / "out.html"
        out_path.write_text("stale generation", encoding="utf-8")
        code, _, err = self.run_cli([str(self.repo), str(out_path)])
        self.assertEqual(code, 2)
        self.assertIn(str(out_path), err)
        self.assertEqual(out_path.read_text(encoding="utf-8"), "stale generation")

    def test_failed_write_leaves_no_debris(self):
        target = self.tmp / "out.html"
        target.write_text("precious", encoding="utf-8")
        original = ace.os.replace
        ace.os.replace = lambda src, dst: (_ for _ in ()).throw(OSError("denied"))
        try:
            with self.assertRaises(OSError):
                ace.write_output(target, "page")
        finally:
            ace.os.replace = original
        self.assertEqual(target.read_text(encoding="utf-8"), "precious")
        self.assertEqual(list(self.tmp.glob(".out.html.*")), [])

    def test_no_exchanges_error_lists_roots(self):
        code, _, err = self.run_cli([str(self.repo), str(self.tmp / "o.html")])
        self.assertEqual(code, 2)
        self.assertIn(str(self.claude_root), err)

    def generate(self, out_path):
        code, out, err = self.run_cli([str(self.repo), str(out_path)])
        self.assertEqual(code, 0, err)
        return out, out_path.read_text(encoding="utf-8")

    def test_pruned_session_kept_from_ledger(self):
        # Replicata: run, then the Claude session file vanishes from the store
        # (pruned, or this is another machine). Expectata: the next run keeps
        # that exchange from the ledger. Resultata before snapshots: gone.
        self.populate()
        out_path = self.tmp / "out.html"
        self.generate(out_path)
        (self.claude_root / "p" / "s.jsonl").unlink()
        out, page = self.generate(out_path)
        for text in ("claude prompt", "claude reply", "codex prompt", "copilot prompt"):
            self.assertIn(text, page)
        self.assertIn("Prompts: 3", out)
        # The kept exchange keeps its place in time among the fresh ones.
        order = [page.index("claude prompt"), page.index("codex prompt"), page.index("copilot prompt")]
        self.assertEqual(order, sorted(order))

    def test_empty_stores_regenerate_from_ledger_alone(self):
        # A new machine: no transcript roots exist at all, only the page and
        # its ledger.
        self.populate()
        out_path = self.tmp / "out.html"
        _, first = self.generate(out_path)
        for root in (self.claude_root, self.codex_root, self.vscode_root):
            shutil.rmtree(root)
        _, again = self.generate(out_path)
        self.assertEqual(again, first)

    def test_store_record_supersedes_ledger_copy(self):
        # The store is the truth for a record it still holds: the record's
        # text changes under the same timestamp, so the ledger copy is
        # replaced.
        self.populate()
        out_path = self.tmp / "out.html"
        self.generate(out_path)
        write_jsonl(
            self.claude_root / "p" / "s.jsonl",
            [
                cu("revised claude prompt", ts=T0, cwd=str(self.repo)),
                ca([{"type": "text", "text": "revised reply"}], cwd=str(self.repo)),
            ],
        )
        out, page = self.generate(out_path)
        self.assertIn("revised claude prompt", page)
        self.assertIn("revised reply", page)
        self.assertNotIn('<pre class="prompt">claude prompt</pre>', page)
        self.assertNotIn(">claude reply<", page)
        self.assertIn("Prompts: 3", out)

    def test_reclassified_record_purged_even_when_session_yields_nothing(self):
        # Replicata: after a run, a parser fix classes the Copilot session's
        # only request as machine text (system-initiated). Expectata: the
        # store still holds the record, so the ledger's copy is purged even
        # though the session now yields no exchange at all. Resultata under a
        # session-level rule: the session looked pruned and the stale copy
        # stayed forever.
        self.populate()
        out_path = self.tmp / "out.html"
        self.generate(out_path)
        request = vsreq("copilot prompt", [md("copilot reply")], ts=int(utc(T2).timestamp() * 1000))
        request["isSystemInitiated"] = True
        (self.vscode_root / "workspaceStorage" / "h1" / "chatSessions" / "a.json").write_text(
            json.dumps(vssession([request])), encoding="utf-8"
        )
        out, page = self.generate(out_path)
        self.assertNotIn("copilot prompt", page)
        self.assertIn("Prompts: 2", out)

    def test_run_names_prompts_the_store_now_reads_as_nothing(self):
        # Replicata: a run; then the store's Copilot request is reclassified
        # as machine text (system-initiated) while a Claude prompt's text is
        # merely revised in place. Expectata: the rerun drops the Copilot
        # ledger copy and NAMES it on stdout by agent and time; the revised
        # Claude prompt, replaced rather than dropped, is not named; a run
        # with nothing dropped says so with a count of zero. Resultata
        # before: the drop happened silently, so a parser change that
        # dropped typed prompts by mistake would have passed unseen.
        self.populate()
        out_path = self.tmp / "out.html"
        out, _ = self.generate(out_path)
        self.assertIn("Prompts deleted: 0", out)
        request = vsreq("copilot prompt", [md("copilot reply")], ts=int(utc(T2).timestamp() * 1000))
        request["isSystemInitiated"] = True
        (self.vscode_root / "workspaceStorage" / "h1" / "chatSessions" / "a.json").write_text(
            json.dumps(vssession([request])), encoding="utf-8"
        )
        write_jsonl(
            self.claude_root / "p" / "s.jsonl",
            [cu("revised claude prompt", ts=T0, cwd=str(self.repo)), ca([{"type": "text", "text": "r"}], cwd=str(self.repo))],
        )
        out, page = self.generate(out_path)
        self.assertIn("Prompts deleted: 1", out)
        self.assertIn(f"Copilot Chat {utc(T2).isoformat()}", out)
        self.assertNotIn("Claude Code", out.split("Prompts deleted")[1])
        self.assertNotIn("copilot prompt", page)

    def test_two_readings_of_a_prompt_the_store_now_reads_as_nothing_reported_deleted_once(self):
        # Replicata: dreeves's ledger holds two readings of the Copilot
        # prompt his store holds, their replies cut short at two points, as
        # a merge keeping both sides' lines leaves them, and two Codex
        # prompts of one time and session whose words differ, the shape of
        # a pair crashla's sourcery 5 page holds; the store's records of both
        # times now read as machine text (the Copilot request
        # system-initiated, the Codex message a canned handoff). Expectata:
        # exit 0, none of the four rows shown, and stdout naming the deleted
        # prompts, the Copilot prompt once, each Codex prompt once:
        # "Prompts deleted: 3", then each by agent and time on its own line.
        # Resultata (v6.0.0): "Prompts deleted: 4", the Copilot prompt named
        # twice. Resultata counting prompts by agent and time alone:
        # "Prompts deleted: 2", the two Codex prompts named as one.
        self.populate()
        request = vsreq("copilot prompt", [md("copilot reply")], ts=int(utc(T2).timestamp() * 1000))
        request["isSystemInitiated"] = True
        (self.vscode_root / "workspaceStorage" / "h1" / "chatSessions" / "a.json").write_text(
            json.dumps(vssession([request])), encoding="utf-8"
        )
        pair_time = "2026-02-21T04:18:45.833Z"
        write_jsonl(
            self.codex_root / "sessions" / "2026" / "rollout-2.jsonl",
            [cxmeta(str(self.repo), sid="cr1"),
             cxuser("The following is the Codex agent history added since your last message.", ts=pair_time)],
        )
        readings = [
            exchange(timestamp=utc(T2), provider="Copilot Chat", model="copilot/gemini-3.1-pro-preview", session="v1",
                     prompt="copilot prompt", reply=reply)
            for reply in ("copilot", "copilot rep")
        ]
        pair = [
            exchange(timestamp=utc(pair_time), provider="Codex", model="gpt-5.4", session="cr1", prompt=words)
            for words in ("first words", "second words")
        ]
        (self.tmp / f"sourcery.{LOGIN}.jsonl").write_text(ledger_text(self.repo.name, readings + pair), encoding="utf-8")
        out, page = self.generate(self.tmp / "out.html")
        self.assertTrue(
            out.endswith(
                f"Prompts: 2\n  {LOGIN}: 2\nPrompts deleted: 3\n  Codex {utc(pair_time).isoformat()}"
                f"\n  Codex {utc(pair_time).isoformat()}\n  Copilot Chat {utc(T2).isoformat()}\n"
            ),
            out,
        )
        for text in ("copilot prompt", "first words", "second words"):
            self.assertNotIn(text, page)

    def test_rerun_purges_redelivered_copy_and_relabels_handed_off_exchange(self):
        # Replicata: a ledger holding what v5.5.0 read from a cloud-workspace
        # session whose chat part was handed off and whose queued prompt was
        # redelivered after a restart: the first exchange labeled with the
        # chat model, the queued prompt shown twice. Expectata: a rerun
        # shows the queued prompt once, names the redelivery's ledger copy
        # on stdout as deleted, and labels the first exchange with the
        # workspace model. Resultata (v5.5.0): the rerun changed nothing.
        cwd = str(self.repo)
        redelivered = "2026-03-01T10:30:00.000Z"
        write_jsonl(
            self.claude_root / "p" / "s.jsonl",
            [
                cu("pick up where we left off", ts=T0, cwd=cwd, handedOff=True),
                {
                    **ca([{"type": "text", "text": "From the chat."}], ts=T0, cwd=cwd,
                         model="claude-sonnet-4-6"),
                    "handedOff": True,
                },
                ca([send_user_message("From the workspace.")], ts=T1, cwd=cwd, mid="m2",
                   model="claude-opus-5-5"),
                cq("also this", T2, cwd, source_uuid="src-1", delivery_id="dlv-1"),
                cq("also this", redelivered, cwd, source_uuid="src-1", delivery_id="dlv-2"),
                ca([send_user_message("Done.", tid="t2")], ts="2026-03-01T10:31:00.000Z", cwd=cwd,
                   mid="m3", model="claude-opus-5-5"),
            ],
        )
        stale = [
            exchange(timestamp=utc(T0), model="claude-sonnet-4-6", session="cs1",
                     prompt="pick up where we left off", reply="From the chat."),
            exchange(timestamp=utc(T2), model="", session="cs1", prompt="also this", reply=""),
            exchange(timestamp=utc(redelivered), model="", session="cs1", prompt="also this", reply=""),
        ]
        out_path = self.tmp / "out.html"
        (self.tmp / f"sourcery.{LOGIN}.jsonl").write_text(ledger_text(self.repo.name, stale), encoding="utf-8")
        out, page = self.generate(out_path)
        self.assertIn("Prompts: 2", out)
        self.assertIn(f"Prompts deleted: 1\n  Claude Code {utc(redelivered).isoformat()}", out)
        self.assertEqual(page.count('<pre class="prompt">also this</pre>'), 1)
        self.assertNotIn("claude-sonnet-4-6", page)
        self.assertIn("From the workspace.", page)

    def test_redelivery_first_delivered_outside_the_repo_refused_and_page_untouched(self):
        # Replicata: a page is written from a session whose queued prompt
        # was delivered while the agent's cwd was outside the repo; then the
        # workspace restarts and redelivers the prompt with its cwd inside
        # the repo, and the page is regenerated. Expectata: exit 2, the error
        # citing the transcript at both deliveries' lines, and the page left
        # byte-identical. Resultata (v5.5.1): the regenerated page showed the
        # prompt at the redelivery's time.
        cwd, elsewhere = str(self.repo), str(self.tmp / "elsewhere")
        records = [
            cu("start the work", ts=T0, cwd=cwd),
            ca([{"type": "text", "text": "Working."}], ts=T1, cwd=elsewhere),
            cq("aside", "2026-03-01T10:06:00.000Z", elsewhere, source_uuid="src-1", delivery_id="dlv-1"),
            ca([{"type": "text", "text": "Noted."}], ts="2026-03-01T10:07:00.000Z", cwd=cwd, mid="m2"),
        ]
        path = write_jsonl(self.claude_root / "p" / "s.jsonl", records)
        out_path = self.tmp / "out.html"
        self.generate(out_path)
        page = out_path.read_bytes()
        redelivery = cq("aside", "2026-03-01T10:30:00.000Z", cwd, source_uuid="src-1", delivery_id="dlv-2")
        write_jsonl(path, [*records, redelivery])
        code, _, err = self.run_cli([str(self.repo), str(out_path)])
        self.assertEqual(code, 2)
        self.assertIn(f"{path}:3", err)
        self.assertIn(f"{path}:5", err)
        self.assertEqual(out_path.read_bytes(), page)

    def test_seeded_claude_ai_exchange_survives_rerun(self):
        # Replicata: a ledger seeded with a claude.ai chat exchange (no
        # store parser reads claude.ai) is rerun beside a Claude Code
        # session. Expectata: the claude.ai exchange is kept from the
        # ledger, since no store holds it, and rendered in time order
        # beside the store's exchange; nothing is reported deleted.
        # Resultata (v5.5.0): the page was refused as corrupt, claude.ai
        # being no known provider.
        cwd = str(self.repo)
        write_jsonl(
            self.claude_root / "p" / "s.jsonl",
            [
                cu("claude prompt", ts=T1, cwd=cwd),
                ca([{"type": "text", "text": "claude reply"}], ts=T2, cwd=cwd),
            ],
        )
        seed = exchange(provider="claude.ai", model="claude-opus-5-5", effort="max", session="chat-1",
                        prompt="chat prompt", reply="chat reply")
        out_path = self.tmp / "out.html"
        ledger = self.tmp / f"sourcery.{LOGIN}.jsonl"
        ledger.write_text(ledger_text(self.repo.name, [seed]), encoding="utf-8")
        out, page = self.generate(out_path)
        self.assertIn(f"Prompts: 2\n  {LOGIN}: 2\nPrompts deleted: 0\n", out)
        self.assertLess(page.index('<article class="exchange claudeai"'), page.index('<article class="exchange claude"'))
        self.assertEqual(
            [(row["provider"], row["prompt"]) for row in ledger_rows(ledger)[1:]],
            [("claude.ai", "chat prompt"), ("Claude Code", "claude prompt")],
        )

    def test_shrunk_live_session_keeps_typed_prompt(self):
        # Replicata: a store restored from an older backup still holds the
        # session file but not its later records. Expectata: the page's copies
        # of those records survive. Resultata under a session-level rule: the
        # live session purged them, and nothing replaced them.
        records = [
            cu("first", ts=T0, cwd=str(self.repo)),
            ca([{"type": "text", "text": "reply one"}], ts=T1, cwd=str(self.repo)),
            cu("second", ts=T2, cwd=str(self.repo)),
            ca([{"type": "text", "text": "reply two"}], ts=T3, cwd=str(self.repo)),
        ]
        path = write_jsonl(self.claude_root / "p" / "s.jsonl", records)
        out_path = self.tmp / "out.html"
        out, _ = self.generate(out_path)
        self.assertIn("Prompts: 2", out)
        write_jsonl(path, records[:2])
        out, page = self.generate(out_path)
        self.assertIn("second", page)
        self.assertIn("reply two", page)
        self.assertIn("Prompts: 2", out)

    def test_existing_output_from_other_repo_rejected(self):
        self.populate()
        out_path = self.tmp / "out.html"
        self.generate(out_path)
        other = self.tmp / "other"
        other.mkdir()
        before = out_path.read_bytes()
        code, _, err = self.run_cli([str(other), str(out_path)])
        self.assertEqual(code, 2)
        self.assertIn("other", err)
        self.assertIn(self.repo.name, err)
        self.assertEqual(out_path.read_bytes(), before)

    def test_rerun_with_unchanged_stores_byte_identical(self):
        self.populate()
        out_path = self.tmp / "out.html"
        _, first = self.generate(out_path)
        _, again = self.generate(out_path)
        self.assertEqual(again, first)

    def test_forked_session_copy_not_duplicated_after_original_pruned(self):
        # Replicata: a fork replays identical records into a new session
        # file; the ledger holds one copy; then the original file is pruned.
        # Expectata: the fork's live record supersedes the ledger copy and the
        # prompt renders once. Resultata with session in the holding: the page
        # copy, tagged with the pruned session, survived beside the fork's.
        self.populate()
        write_jsonl(
            self.claude_root / "p" / "fork.jsonl",
            [
                cu("claude prompt", ts=T0, cwd=str(self.repo), session="cs2"),
                ca([{"type": "text", "text": "claude reply"}], cwd=str(self.repo), session="cs2"),
            ],
        )
        out_path = self.tmp / "out.html"
        out, _ = self.generate(out_path)
        self.assertIn("Prompts: 3", out)
        (self.claude_root / "p" / "s.jsonl").unlink()
        out, page = self.generate(out_path)
        self.assertIn("Prompts: 3", out)
        self.assertEqual(page.count('<pre class="prompt">claude prompt</pre>'), 1)

    def test_in_flight_reply_completed_on_rerun(self):
        # Replicata: the page captured the last exchange mid-generation; the
        # store then gains the reply. Expectata: the rerun shows the reply,
        # once. Resultata under a pure union: the prompt rendered twice, once
        # still generating.
        path = write_jsonl(self.claude_root / "p" / "s.jsonl", [cu("claude prompt", ts=T0, cwd=str(self.repo))])
        out_path = self.tmp / "out.html"
        _, page = self.generate(out_path)
        self.assertIn("Response still generating", page)
        write_jsonl(
            path,
            [
                cu("claude prompt", ts=T0, cwd=str(self.repo)),
                ca([{"type": "text", "text": "late reply"}], cwd=str(self.repo)),
            ],
        )
        _, page = self.generate(out_path)
        self.assertIn("late reply", page)
        self.assertNotIn("still generating", page)
        self.assertEqual(page.count('<pre class="prompt">claude prompt</pre>'), 1)

    def test_ledger_and_page_omit_store_paths(self):
        self.populate()
        out_path = self.tmp / "out.html"
        _, page = self.generate(out_path)
        ledger = (self.tmp / f"sourcery.{LOGIN}.jsonl").read_text(encoding="utf-8")
        for root in (self.claude_root, self.codex_root, self.vscode_root):
            self.assertNotIn(str(root), ledger)
            self.assertNotIn(str(root), page)
        self.assertNotIn('"source"', ledger)

    def test_malformed_ledger_rejected_and_nothing_written(self):
        self.populate()
        out_path = self.tmp / "out.html"
        _, page = self.generate(out_path)
        ledger = self.tmp / f"sourcery.{LOGIN}.jsonl"
        header, first, *rest = ledger.read_text(encoding="utf-8").removesuffix("\n").split("\n")
        foreign = json.dumps({**json.loads(first), "author": "someone"})
        for bad in ("{not json", foreign):
            broken = "\n".join([header, bad, *rest]) + "\n"
            ledger.write_text(broken, encoding="utf-8")
            code, _, err = self.run_cli([str(self.repo), str(out_path)])
            self.assertEqual(code, 2)
            self.assertIn(f"{ledger}:2", err)
            self.assertEqual(ledger.read_text(encoding="utf-8"), broken)
            self.assertEqual(out_path.read_text(encoding="utf-8"), page)

    def test_prompt_quoting_the_credit_markup_does_not_confuse_the_reader(self):
        # Replicata: a typed prompt is an article's opening tag, crediting
        # another login, and a summary's time, as the page writes them.
        # Expectata: the rerun succeeds, the page crediting one row, and the
        # ledger gives the prompt back byte-exact. Resultata with an
        # unescaped prompt: a second credit, naming a ledger that holds no
        # such row, and the rerun refused.
        text = (
            '<article class="exchange claude" id="p1" data-login="mister-person">'
            '<time datetime="2026-03-01T09:00:00+00:00">'
        )
        write_jsonl(self.claude_root / "p" / "s.jsonl", [cu(text, ts=T0, cwd=str(self.repo))])
        out_path = self.tmp / "out.html"
        self.generate(out_path)
        self.generate(out_path)
        self.assertEqual(ace.page_credits(out_path), {(LOGIN, ("Claude Code", utc(T0)))})
        self.assertEqual(
            [row["prompt"] for row in ledger_rows(self.tmp / f"sourcery.{LOGIN}.jsonl")[1:]], [text]
        )

    def test_collect_reports_holdings_and_unwoven_exchanges(self):
        self.populate()
        write_jsonl(
            self.claude_root / "p" / "fork.jsonl",
            [
                cu("claude prompt", ts=T0, cwd=str(self.repo), session="cs2"),
                ca([{"type": "text", "text": "claude reply"}], cwd=str(self.repo), session="cs2"),
            ],
        )
        exchanges, holdings = ace.collect(self.repo, ace.discover_roots(self.env))
        self.assertEqual(len(exchanges), 4)
        self.assertEqual(len(ace.weave(exchanges)), 3)
        self.assertEqual(
            holdings, {("Claude Code", utc(T0)), ("Codex", utc(T1)), ("Copilot Chat", utc(T2))}
        )

    def split_session(self):
        """A session Claude Code continued in a second file after a cloud
        workspace restarted: a file of the same name, carrying the same
        session id, under another project directory, opening with the reply
        to the first file's last prompt, then redelivering the prompt queued
        before the restart. Returns both paths."""
        cwd = str(self.repo)
        first = write_jsonl(
            self.claude_root / "p" / "cs1.jsonl",
            [
                cu("start the work", ts=T0, cwd=cwd),
                ca([{"type": "text", "text": "Working."}], ts=T1, cwd=cwd),
                cq("aside", "2026-03-01T10:06:00.000Z", cwd, source_uuid="src-1", delivery_id="dlv-1"),
                ca([{"type": "text", "text": "Noted."}], ts="2026-03-01T10:07:00.000Z", cwd=cwd, mid="m2"),
            ],
        )
        second = write_jsonl(
            self.claude_root / "q" / "cs1.jsonl",
            [
                ca([{"type": "text", "text": "Still on it."}], ts="2026-03-01T10:29:00.000Z", cwd=cwd, mid="m3"),
                cq("aside", "2026-03-01T10:30:00.000Z", cwd, source_uuid="src-1", delivery_id="dlv-2"),
                ca([{"type": "text", "text": "Done."}], ts="2026-03-01T10:31:00.000Z", cwd=cwd, mid="m4"),
            ],
        )
        return first, second

    def test_session_split_across_two_files_refused_naming_both(self):
        # Replicata: a cloud workspace restarts mid-session, and Claude Code
        # continues the session in a second file (see split_session).
        # Expectata: a loud error naming both files, each at the session's
        # first record there, never a reading of either file alone.
        # Resultata (v5.5.1): no error; read apart, the files showed the
        # queued prompt twice, the second time at its redelivery, and lost
        # the reply the second file opens with, which had no prompt there.
        first, second = self.split_session()
        with self.assertRaises(ace.UserError) as ctx:
            ace.collect(self.repo, ace.discover_roots(self.env))
        self.assertIn(f"{first}:1", str(ctx.exception))
        self.assertIn(f"{second}:1", str(ctx.exception))

    def test_session_split_refusal_times_each_file_and_says_to_delete_a_copy(self):
        # Replicata: a session continued in a second file (see
        # split_session), that file opening with a record whose cwd is
        # inside the repo or outside it. Expectata: the loud error gives,
        # beside each file's citation, the timestamp of the session's first
        # record there, as the record holds it, so the human can tell which
        # file the session begins in; and it says to delete a file that is a
        # copy of the other instead of merging the two. Resultata (v5.5.1):
        # no error. Resultata with the refusal's first message: each file
        # cited by line alone, with no timestamp and no word about a copy.
        first, second = self.split_session()
        opening, *rest = [json.loads(line) for line in second.read_text(encoding="utf-8").splitlines()]
        for where in (str(self.repo), str(self.tmp / "elsewhere")):
            write_jsonl(second, [{**opening, "cwd": where}, *rest])
            with self.subTest(where=where):
                with self.assertRaises(ace.UserError) as ctx:
                    ace.collect(self.repo, ace.discover_roots(self.env))
                message = str(ctx.exception)
                for path, ts in ((first, T0), (second, "2026-03-01T10:29:00.000Z")):
                    [line] = [line for line in message.splitlines() if f"{path}:1" in line]
                    self.assertIn(ts, line)
                self.assertIn("exemplar dele", message)

    def test_session_split_refused_by_run_with_nothing_written(self):
        # Replicata: a page is written from a session's transcript; then the
        # session continues in a second file (see split_session), and the
        # page is regenerated, and a page is generated at a new path too.
        # Expectata: exit 2 both times, the error naming both files, the
        # page left byte-identical, and no page at the new path. Resultata
        # (v5.5.1): exit 0, and both pages showed the queued prompt twice.
        first, second = self.split_session()
        continued = second.read_bytes()
        second.unlink()
        out_path = self.tmp / "out.html"
        self.generate(out_path)
        page = out_path.read_bytes()
        second.write_bytes(continued)
        fresh = self.tmp / "fresh.html"
        for target in (out_path, fresh):
            code, out, err = self.run_cli([str(self.repo), str(target)])
            with self.subTest(target=target.name):
                self.assertEqual((code, out), (2, ""))
                self.assertIn(str(first), err)
                self.assertIn(str(second), err)
        self.assertEqual(out_path.read_bytes(), page)
        self.assertFalse(fresh.exists())

    def test_every_record_read_into_a_session_registers_its_file_wherever_its_cwd(self):
        # Replicata: a session's transcript, and a second file under another
        # project directory holding one record of the same session that
        # claude_exchanges reads into it: a typed prompt, a tool result, a
        # reply, a harness notice, a queued prompt, or another agent's
        # handback, its cwd inside the repo or outside it. Expectata: the
        # loud error naming both files, whichever record it is and wherever
        # its cwd: a session split across files is read wrongly for every
        # repo, so the refusal is the store's, not one page's. Resultata
        # (v5.5.1): no error.
        cwd, elsewhere = str(self.repo), str(self.tmp / "elsewhere")
        first = write_jsonl(
            self.claude_root / "p" / "cs1.jsonl",
            [cu("start the work", ts=T0, cwd=cwd), ca([{"type": "text", "text": "Working."}], ts=T1, cwd=cwd)],
        )

        def handback(where):
            record = cq("Findings: all green.", T2, where, source_uuid="src-9")
            record["attachment"]["origin"] = {"kind": "peer"}
            return record

        for where in (cwd, elsewhere):
            for record in (
                cu("more", ts=T2, cwd=where),
                cu([{"type": "tool_result", "tool_use_id": "t1", "content": "out"}], ts=T2, cwd=where),
                ca([{"type": "text", "text": "More."}], ts=T2, cwd=where),
                ca([{"type": "text", "text": "API Error: 401"}], ts=T2, cwd=where, model="<synthetic>"),
                cq("aside", T2, where, source_uuid="src-1"),
                handback(where),
            ):
                second = write_jsonl(self.claude_root / "q" / "cs1.jsonl", [record])
                with self.subTest(where=where, record=record):
                    with self.assertRaises(ace.UserError) as ctx:
                        ace.collect(self.repo, ace.discover_roots(self.env))
                    self.assertIn(f"{first}:1", str(ctx.exception))
                    self.assertIn(f"{second}:1", str(ctx.exception))

    def test_records_never_read_into_a_session_register_no_file(self):
        # Replicata: a session's transcript, and a second file under another
        # project directory holding one record of the same session that
        # claude_exchanges never reads into it: subagent traffic, a meta
        # record, a compaction summary, a transcript-only note, a system
        # record, an attachment that is no queued prompt, or a queued
        # background-task wakeup. Expectata: no error, and the transcript
        # read exactly as alone. Resultata (v5.5.1): as expected; this guards
        # the refusal of a split session against records that leave no mark
        # on a session's reading.
        cwd = str(self.repo)
        first = write_jsonl(
            self.claude_root / "p" / "cs1.jsonl",
            [cu("start the work", ts=T0, cwd=cwd), ca([{"type": "text", "text": "Working."}], ts=T1, cwd=cwd)],
        )
        alone = ace.claude_exchanges(first, self.repo)
        stamp = {"timestamp": T2, "cwd": cwd, "sessionId": "cs1"}
        wakeup = {
            "type": "queued_command",
            "commandMode": "task-notification",
            "prompt": "<task-notification>done</task-notification>",
        }
        for record in (
            cu("subagent prompt", ts=T2, cwd=cwd, isSidechain=True),
            {**ca([{"type": "text", "text": "Subagent reply."}], ts=T2, cwd=cwd), "isSidechain": True},
            cu("meta", ts=T2, cwd=cwd, isMeta=True),
            cu("This session is being continued...", ts=T2, cwd=cwd, isCompactSummary=True),
            cu("transcript-only", ts=T2, cwd=cwd, isVisibleInTranscriptOnly=True),
            {"type": "system", "subtype": "upgrade_relay_marker", "content": "moved", **stamp},
            {"type": "attachment", "attachment": {"type": "todo_reminder", "content": []}, **stamp},
            {"type": "attachment", "attachment": wakeup, **stamp},
        ):
            write_jsonl(self.claude_root / "q" / "cs1.jsonl", [record])
            with self.subTest(record=record):
                self.assertEqual(ace.collect(self.repo, ace.discover_roots(self.env)), alone)

    def test_subagent_transcript_carrying_its_parents_session_not_refused(self):
        # Replicata: a session whose subagent's transcript, nested under the
        # session's directory, carries the parent's session id on every
        # record, each record marked isSidechain. Expectata: no error, and
        # the session read exactly as its own file alone. Resultata
        # (v5.5.1): as expected; this guards the refusal of a split session
        # against subagent transcripts.
        cwd = str(self.repo)
        parent = write_jsonl(
            self.claude_root / "p" / "cs1.jsonl",
            [cu("start the work", ts=T0, cwd=cwd), ca([{"type": "text", "text": "Working."}], ts=T1, cwd=cwd)],
        )
        write_jsonl(
            self.claude_root / "p" / "cs1" / "subagents" / "agent-a1.jsonl",
            [
                cu("Survey the repo.", ts=T1, cwd=cwd, isSidechain=True, agentId="a1"),
                {**ca([{"type": "text", "text": "Surveyed."}], ts=T2, cwd=cwd), "isSidechain": True, "agentId": "a1"},
                {
                    "type": "attachment",
                    "attachment": {"type": "date", "content": "today"},
                    "isSidechain": True,
                    "timestamp": T2,
                    "cwd": cwd,
                    "sessionId": "cs1",
                },
            ],
        )
        self.assertEqual(
            ace.collect(self.repo, ace.discover_roots(self.env)), ace.claude_exchanges(parent, self.repo)
        )

    def test_workflow_journals_sharing_a_name_not_refused(self):
        # Replicata: two workflow runs under one session, each keeping a
        # journal.jsonl of agent launches and results: no user or assistant
        # record and no session id, so a session named for the file would be
        # "journal" in both. Expectata: no error, and the session read
        # exactly as its own file alone. Resultata (v5.5.1): as expected;
        # this guards the refusal of a split session against files that
        # share a name but hold no session.
        cwd = str(self.repo)
        parent = write_jsonl(
            self.claude_root / "p" / "cs1.jsonl",
            [cu("start the work", ts=T0, cwd=cwd), ca([{"type": "text", "text": "Working."}], ts=T1, cwd=cwd)],
        )
        for run in ("wf_1", "wf_2"):
            write_jsonl(
                self.claude_root / "p" / "cs1" / "subagents" / "workflows" / run / "journal.jsonl",
                [
                    {"type": "started", "agentId": "a1", "key": "k1"},
                    {"type": "result", "agentId": "a1", "key": "k1", "result": "done"},
                ],
            )
        self.assertEqual(
            ace.collect(self.repo, ace.discover_roots(self.env)), ace.claude_exchanges(parent, self.repo)
        )

    def test_one_file_holding_two_sessions_reads_the_same_through_collect(self):
        # Replicata: one transcript holding two sessions' records,
        # interleaved, several records each. Expectata: no error, and
        # collect reads the file exactly as claude_exchanges reads it alone.
        # Resultata (v5.5.1): as expected; this guards the refusal of a
        # split session against a session's records within one file.
        cwd = str(self.repo)
        path = write_jsonl(
            self.claude_root / "p" / "cs1.jsonl",
            [
                cu("start one", ts=T0, cwd=cwd, session="cs1"),
                cu("start two", ts=T0, cwd=cwd, session="cs2"),
                ca([{"type": "text", "text": "One."}], ts=T1, cwd=cwd, session="cs1"),
                ca([{"type": "text", "text": "Two."}], ts=T1, cwd=cwd, session="cs2", mid="m2"),
                cu("again one", ts=T2, cwd=cwd, session="cs1"),
                ca([{"type": "text", "text": "Again."}], ts=T3, cwd=cwd, session="cs1", mid="m3"),
            ],
        )
        got = ace.collect(self.repo, ace.discover_roots(self.env))
        self.assertEqual(got, ace.claude_exchanges(path, self.repo))
        self.assertEqual(
            [(e.session, e.prompt) for e in got[0]],
            [("cs1", "start one"), ("cs1", "again one"), ("cs2", "start two")],
        )

    def test_other_agents_repeating_session_ids_not_refused(self):
        # Replicata: Codex, Copilot Chat, and Antigravity each hold two files
        # under one session id, an id a Claude Code session carries too.
        # Expectata: no error, and every file read. Resultata (v5.5.1): as
        # expected; this guards the refusal of a split session, a shape of
        # Claude Code's store, against other agents' stores.
        cwd = str(self.repo)
        write_jsonl(self.claude_root / "p" / "cs1.jsonl", [cu("claude prompt", ts=T0, cwd=cwd)])
        write_jsonl(self.codex_root / "sessions" / "rollout-1.jsonl", [cxmeta(cwd, sid="cs1"), cxuser("codex one", ts=T0)])
        write_jsonl(
            self.codex_root / "archived_sessions" / "rollout-2.jsonl", [cxmeta(cwd, sid="cs1"), cxuser("codex two", ts=T1)]
        )
        for storage, text, ts in (("h1", "copilot one", T0), ("h2", "copilot two", T1)):
            folder = self.vscode_root / "workspaceStorage" / storage
            (folder / "chatSessions").mkdir(parents=True)
            (folder / "workspace.json").write_text(json.dumps({"folder": self.repo.as_uri()}), encoding="utf-8")
            (folder / "chatSessions" / "a.json").write_text(
                json.dumps(vssession([vsreq(text, [md("r")], ts=int(utc(ts).timestamp() * 1000))], sid="cs1")),
                encoding="utf-8",
            )
        for name, text, ts in (("conv-1", "antigravity one", T0), ("conv-2", "antigravity two", T1)):
            write_ag(self.antigravity_root, name, agconv([aguser(text, ts, workspace=self.repo.as_uri())], cid="cs1"))
        exchanges, _ = ace.collect(self.repo, ace.discover_roots(self.env))
        self.assertEqual(
            sorted((e.provider, e.session, e.prompt) for e in exchanges),
            [
                ("Antigravity", "cs1", "antigravity one"),
                ("Antigravity", "cs1", "antigravity two"),
                ("Claude Code", "cs1", "claude prompt"),
                ("Codex", "cs1", "codex one"),
                ("Codex", "cs1", "codex two"),
                ("Copilot Chat", "cs1", "copilot one"),
                ("Copilot Chat", "cs1", "copilot two"),
            ],
        )

    def test_one_transcript_reached_by_two_paths_not_refused(self):
        # Replicata: a session's transcript, reached by a second path too:
        # AI_CHAT_CLAUDE_ROOTS lists the store beside a symlink to it, or
        # beside itself spelled through "..", or the store holds a symlink
        # or a hard link to the transcript under another project directory.
        # Expectata: no error, and the session read exactly as its own file
        # alone, the one file being read twice. Resultata (v5.5.1): as
        # expected. Resultata with files told apart by their paths: the loud
        # error for a session split across two files, citing the one file
        # under both paths and asking that it be merged with itself.
        cwd = str(self.repo)
        first = write_jsonl(
            self.claude_root / "p" / "cs1.jsonl",
            [cu("start the work", ts=T0, cwd=cwd), ca([{"type": "text", "text": "Working."}], ts=T1, cwd=cwd)],
        )
        exchanges, holdings = ace.claude_exchanges(first, self.repo)
        alone = (ace.weave(exchanges), holdings)

        def read(*roots):
            self.env["AI_CHAT_CLAUDE_ROOTS"] = os.pathsep.join(str(root) for root in roots)
            exchanges, holdings = ace.collect(self.repo, ace.discover_roots(self.env))
            return ace.weave(exchanges), holdings

        alias = self.tmp / "alias"
        alias.symlink_to(self.claude_root)
        for second in (alias, self.claude_root / "p" / ".."):
            with self.subTest(second=second):
                self.assertEqual(read(self.claude_root, second), alone)
        link = self.claude_root / "q" / "cs1.jsonl"
        link.parent.mkdir()
        for make in (link.symlink_to, link.hardlink_to):
            make(first)
            with self.subTest(link=make.__name__):
                self.assertEqual(read(self.claude_root), alone)
            link.unlink()

    def test_every_json_escape_survives_ledger_round_trip(self):
        # Replicata: a prompt and reply holding every character JSON escapes,
        # every character a reader splitting lines on more than "\n" would
        # break at, and markup the page writes; then the store is pruned, so
        # the ledger alone remembers them. Expectata: the ledger gives both
        # back byte-exact, one line for the exchange, and the page holds one
        # script. Resultata with lines split as str.splitlines splits them:
        # the row broken across lines, and the ledger refused.
        text = (
            'quote " backslash \\ slash / bs \b ff \f nl \n cr \r tab \t nul \x00 del \x7f '
            "ls   ps   lit \\u003c amp & lt < gt > tag </script> cmt <!-- --> cdata ]]> "
            "astral \U0001F41D combining é crlf \r\n vt \x0b fs \x1c gs \x1d rs \x1e nel \x85 "
            'credit <article class="exchange claude" id="p1" data-login="x">'
        )
        write_jsonl(
            self.claude_root / "p" / "s.jsonl",
            [cu(text, ts=T0, cwd=str(self.repo)), ca([{"type": "text", "text": text}], cwd=str(self.repo))],
        )
        out_path = self.tmp / "out.html"
        self.generate(out_path)
        (self.claude_root / "p" / "s.jsonl").unlink()
        _, page = self.generate(out_path)
        ledger = self.tmp / f"sourcery.{LOGIN}.jsonl"
        self.assertEqual([(row["prompt"], row["reply"]) for row in ledger_rows(ledger)[1:]], [(text, text)])
        self.assertEqual(ledger.read_bytes().count(b"\n"), 2)
        self.assertEqual(count_scripts(page), 1)

    def test_tool_result_at_same_millisecond_elsewhere_never_purges_ledger_copy(self):
        # Replicata: the ledger holds a Claude prompt at T0 from session A; A
        # is pruned; session B, still in the store, has a bare tool_result
        # record at T0 (tool plumbing, no typing). Expectata: the ledger copy
        # is kept — a record that could never have been rendered is no
        # holding. Resultata before the fix: purged, and nothing replaced it.
        write_jsonl(
            self.claude_root / "p" / "a.jsonl",
            [
                cu("precious prompt", ts=T0, cwd=str(self.repo), session="A"),
                ca([{"type": "text", "text": "reply"}], cwd=str(self.repo), session="A"),
            ],
        )
        out_path = self.tmp / "out.html"
        self.generate(out_path)
        (self.claude_root / "p" / "a.jsonl").unlink()
        write_jsonl(
            self.claude_root / "p" / "b.jsonl",
            [
                cu("other prompt", ts="2026-03-01T09:59:00.000Z", cwd=str(self.repo), session="B"),
                ca(
                    [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}],
                    ts="2026-03-01T09:59:30.000Z", cwd=str(self.repo), session="B",
                ),
                cu(
                    [{"type": "tool_result", "tool_use_id": "t1", "content": "file contents"}],
                    ts=T0, cwd=str(self.repo), session="B", toolUseResult={"type": "text", "file": {}},
                ),
                ca([{"type": "text", "text": "other reply"}], ts=T1, cwd=str(self.repo), session="B"),
            ],
        )
        out, page = self.generate(out_path)
        self.assertIn("precious prompt", page)
        self.assertIn("Prompts: 2", out)

    def test_store_copy_wins_even_when_less_complete(self):
        # Replicata: the ledger holds a prompt with its reply; the store's copy
        # of the session then loses the reply record but keeps the prompt
        # (a restore from an older backup). Expectata, deliberately: the
        # store's reading wins and the page now shows the prompt awaiting a
        # reply — a parser fix that empties a reply (harness notices were once
        # rendered as replies) must purge the stale text, and the merge cannot
        # tell that case from this one. The typed prompt itself is kept.
        records = [
            cu("first", ts=T0, cwd=str(self.repo)),
            ca([{"type": "text", "text": "reply one"}], ts=T1, cwd=str(self.repo)),
        ]
        path = write_jsonl(self.claude_root / "p" / "s.jsonl", records)
        out_path = self.tmp / "out.html"
        self.generate(out_path)
        write_jsonl(path, records[:1])
        out, page = self.generate(out_path)
        self.assertIn("first", page)
        self.assertNotIn("reply one", page)
        self.assertIn("Prompts: 1", out)

    def test_ledger_missing_field_rejected_and_nothing_written(self):
        # A ledger another schema wrote: a row lacking a field, even one the
        # dataclass would default, is refused rather than thawed into a guess.
        self.populate()
        out_path = self.tmp / "out.html"
        _, page = self.generate(out_path)
        ledger = self.tmp / f"sourcery.{LOGIN}.jsonl"
        header, *rows = ledger_rows(ledger)
        for row in rows:
            del row["effort"]
        broken = "".join(json.dumps(line) + "\n" for line in [header, *rows])
        ledger.write_text(broken, encoding="utf-8")
        code, _, err = self.run_cli([str(self.repo), str(out_path)])
        self.assertEqual(code, 2)
        self.assertIn(f"{ledger}:2", err)
        self.assertEqual(ledger.read_text(encoding="utf-8"), broken)
        self.assertEqual(out_path.read_text(encoding="utf-8"), page)

    def test_ledger_first_line_names_the_format_not_the_release(self):
        # Replicata: a run, then a rerun on unchanged stores by another
        # sourcery release (VERSION replaced), as on two machines a release
        # apart. Expectata: the ledger's first line is {"ledger": 1, "repo":
        # PROJECT}, the ledger format and the project name, naming no
        # release, so both runs write the ledger byte-identical, and a union
        # merge between the two machines' ledgers never doubles line 1.
        # Resultata (v6.0.0): the first line named the release that wrote
        # it, {"sourcery": VERSION, "repo": PROJECT}, so each release
        # rewrote line 1.
        self.populate()
        out_path = self.tmp / "out.html"
        self.generate(out_path)
        ledger = self.ledger()
        self.assertEqual(ledger_rows(ledger)[0], ledger_header(self.repo.name))
        written = ledger.read_bytes()
        release = ace.VERSION
        ace.VERSION = "9.9.9"
        try:
            out, _ = self.generate(out_path)
        finally:
            ace.VERSION = release
        self.assertIn("Prompts: 3", out)
        self.assertEqual(ledger.read_bytes(), written)

    def test_ledger_of_another_format_refused(self):
        # Replicata: beside the page, the runner's ledger or mister-person's,
        # its first line naming a ledger format other than 1: 2, as a later
        # format would; "1", 1.0, or true, no format number at all; or the
        # first line sourcery 6.0.0 wrote before ledgers named their format,
        # {"sourcery": "6.0.0", "repo": PROJECT}. Expectata: exit 2 citing
        # the ledger at line 1; nothing written. Resultata (v6.0.0): no
        # format named; a first line naming any release was read alike.
        self.populate()
        out_path = self.tmp / "out.html"
        for login in (LOGIN, "mister-person"):
            for header in (
                {"ledger": 2, "repo": self.repo.name},
                {"ledger": "1", "repo": self.repo.name},
                {"ledger": 1.0, "repo": self.repo.name},
                {"ledger": True, "repo": self.repo.name},
                {"sourcery": "6.0.0", "repo": self.repo.name},
            ):
                self.ledger(login).write_text(ledger_text(self.repo.name, [self.theirs()], header), encoding="utf-8")
                with self.subTest(login=login, header=header):
                    err = self.refused([str(self.repo), str(out_path)])
                    self.assertIn(f"Tabula legi non potest: {self.ledger(login)}:1\n", err)
            self.ledger(login).unlink()

    def test_ledger_of_a_later_format_refused_saying_another_release_wrote_it(self):
        # Replicata: beside the page, mister-person's ledger or the
        # runner's, as a later sourcery release writing ledger format 2
        # would leave it on a machine that upgraded: its first line
        # {"ledger": 2, "repo": PROJECT}. Expectata: exit 2 citing the
        # ledger at line 1, saying that when the header line names a ledger
        # format other than 1, another sourcery release wrote the ledger,
        # and to run a release that reads that format, then rerun; nothing
        # written. Resultata (v6.0.0): only the advice for a git conflict or
        # a union merge, though no merge made the ledger; and advice added
        # once format 2 exists never reaches a machine still running this
        # release, which is the one that meets such a ledger.
        self.populate()
        out_path = self.tmp / "out.html"
        for login in ("mister-person", LOGIN):
            self.ledger(login).write_text(
                ledger_text(self.repo.name, [self.theirs()], {"ledger": 2, "repo": self.repo.name}), encoding="utf-8"
            )
            with self.subTest(login=login):
                err = self.refused([str(self.repo), str(out_path)])
                self.assertIn(f"Tabula legi non potest: {self.ledger(login)}:1\n", err)
                self.assertIn(
                    "Si linea capitis formam tabulae aliam quam 1 nominat, tabulam alia editio sourcery scripsit: "
                    "curre editionem quae illam formam legit, deinde iterum curre.",
                    err,
                )
            self.ledger(login).unlink()

    def test_header_line_a_union_merge_left_twice_refused_saying_which_to_keep(self):
        # Replicata: github.com/dreeves/blog is renamed newblog, and this
        # checkout's origin names newblog. dreeves's ledger is as git's
        # union merge (merge=union) left it, line 1 twice: on one machine he
        # wrote newblog in place of blog on its first line, as sourcery's
        # refusal says to, while on another, whose origin still named blog,
        # a run added a prompt beside it; the merge kept both sides' lines,
        # so line 1 names blog and line 3 newblog. Expectata: exit 2 citing
        # the ledger at line 3, saying that such a union merge can leave the
        # header line twice, and to keep on the first line the one naming
        # 'newblog' and delete the other, then rerun; nothing written. With
        # that done, the run succeeds, showing both prompts. Resultata
        # (v6.0.0): no word of a header line left twice.
        self.origin("https://github.com/dreeves/newblog.git")
        older = exchange(timestamp=utc("2026-03-01T08:00:00.000Z"), session="cs0", prompt="older prompt")
        original = exchange(timestamp=utc("2026-03-01T09:00:00.000Z"), session="cs0", prompt="original prompt")
        row = lambda e: json.dumps(ace.freeze(e), ensure_ascii=False)
        lines = [json.dumps(ledger_header("blog")), row(older), json.dumps(ledger_header("newblog")), row(original)]
        self.ledger().write_text("\n".join(lines) + "\n", encoding="utf-8")
        out_path = self.tmp / "out.html"
        err = self.refused([str(self.repo), str(out_path)])
        self.assertIn(f"Tabula legi non potest: {self.ledger()}:3\n", err)
        self.assertIn(
            "git sponte utriusque partis lineas servat; sed fusio talis lineam capitis bis relinquere potest: "
            "eam quae 'newblog' nominat in prima linea serva, alteram dele, deinde iterum curre.\n",
            err,
        )
        self.ledger().write_text("\n".join([lines[2], lines[1], lines[3]]) + "\n", encoding="utf-8")
        out, page = self.generate(out_path)
        self.assertIn(f"Prompts: 2\n  {LOGIN}: 2\n", out)
        for text in ("older prompt", "original prompt"):
            self.assertIn(f'<pre class="prompt">{text}</pre>', page)

    def test_copilot_button_click_without_timestamp_fails_loudly(self):
        # Every request is a holding, so its timestamp is checked before the
        # button-click skip: a click without one is loud where it was skipped.
        storage = self.vscode_root / "workspaceStorage" / "h1"
        (storage / "chatSessions").mkdir(parents=True)
        (storage / "workspace.json").write_text(json.dumps({"folder": self.repo.as_uri()}), encoding="utf-8")
        click = vsreq("@agent Try Again", [], confirmation="Try Again")
        del click["timestamp"]
        typed = vsreq("typed", [md("reply")], rid="r9")
        (storage / "chatSessions" / "a.json").write_text(json.dumps(vssession([click, typed])), encoding="utf-8")
        code, _, err = self.run_cli([str(self.repo), str(self.tmp / "o.html")])
        self.assertEqual(code, 2)
        self.assertIn("Tempus deest", err)
        self.assertFalse((self.tmp / "o.html").exists())

    def test_repo_whose_public_home_is_not_named_origin_rejected(self):
        # The refusal reaches the CLI: nothing is written, and the message
        # says which remote holds the home and what to rename.
        self.populate()
        git = self.repo / ".git"
        git.mkdir()
        (git / "config").write_text(
            '[remote "efme"]\n\turl = git@github.com:beeminder/efme.git\n', encoding="utf-8"
        )
        out_path = self.tmp / "out.html"
        code, _, err = self.run_cli([str(self.repo), str(out_path)])
        self.assertEqual(code, 2)
        self.assertIn("efme", err)
        self.assertIn("origin", err)
        self.assertFalse(out_path.exists())

    def test_open_flag_opens_the_written_page(self):
        self.populate()
        out_path = self.tmp / "out.html"
        opened = []
        original = ace.webbrowser.open
        ace.webbrowser.open = lambda uri: opened.append(uri) or True
        try:
            code, _, err = self.run_cli([str(self.repo), str(out_path), "--open"])
            self.assertEqual(code, 0, err)
            self.assertEqual(opened, [out_path.resolve().as_uri()])
            ace.webbrowser.open = lambda uri: False
            code, out, err = self.run_cli([str(self.repo), str(out_path), "--open"])
            self.assertEqual(code, 2)
            self.assertIn(str(out_path), err)
            self.assertIn("Prompts: 3", out)  # the page is written before the browser is asked
        finally:
            ace.webbrowser.open = original
        self.assertEqual(len(ace.page_credits(out_path)), 3)

    def test_relative_output_path_reads_the_ledger_beside_it(self):
        self.populate()
        cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            code, _, err = self.run_cli([str(self.repo), "out.html"])
            self.assertEqual(code, 0, err)
            (self.claude_root / "p" / "s.jsonl").unlink()
            code, out, err = self.run_cli([str(self.repo), "out.html"])
            self.assertEqual(code, 0, err)
            self.assertIn("Prompts: 3", out)
            self.assertIn(str(self.tmp / "out.html"), out)
        finally:
            os.chdir(cwd)

    def test_output_path_that_is_a_directory_rejected_cleanly(self):
        self.populate()
        out_dir = self.tmp / "out.html"
        out_dir.mkdir()
        code, _, err = self.run_cli([str(self.repo), str(out_dir)])
        self.assertEqual(code, 2)
        self.assertIn(str(out_dir), err)

    def test_unreadable_output_page_rejected_cleanly(self):
        self.populate()
        out_path = self.tmp / "out.html"
        self.generate(out_path)
        before = out_path.read_bytes()
        out_path.chmod(0)
        try:
            code, _, err = self.run_cli([str(self.repo), str(out_path)])
        finally:
            out_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        self.assertEqual(code, 2)
        self.assertIn(str(out_path), err)
        self.assertEqual(out_path.read_bytes(), before)

    def test_printed_count_deck_articles_and_ledger_agree(self):
        self.populate()
        out_path = self.tmp / "out.html"
        self.generate(out_path)
        (self.claude_root / "p" / "s.jsonl").unlink()
        write_jsonl(
            self.codex_root / "sessions" / "2026" / "rollout-1.jsonl",
            [cxmeta(str(self.repo)), cxuser("codex prompt", ts=T1), cxagent("codex reply", ts=T2), cxuser("later", ts=T3)],
        )
        out, page = self.generate(out_path)
        count = int(out.split("Prompts: ")[1].split()[0])
        self.assertEqual(count, 4)
        self.assertEqual(page.count("<article "), count)
        self.assertIn(f'<p class="deck">{count} prompts', page)
        self.assertIn(f"\n  {LOGIN}: {count}\n", out)
        self.assertEqual(len(ledger_rows(self.tmp / f"sourcery.{LOGIN}.jsonl")) - 1, count)
        self.assertEqual(len(ace.page_credits(out_path)), count)

    def test_kept_and_fresh_sharing_a_timestamp_both_survive_in_stable_order(self):
        write_jsonl(
            self.claude_root / "p" / "s.jsonl",
            [cu("claude at t", ts=T0, cwd=str(self.repo)), ca([{"type": "text", "text": "r"}], cwd=str(self.repo))],
        )
        out_path = self.tmp / "out.html"
        self.generate(out_path)
        (self.claude_root / "p" / "s.jsonl").unlink()
        write_jsonl(
            self.codex_root / "sessions" / "2026" / "rollout-1.jsonl",
            [cxmeta(str(self.repo)), cxuser("codex at t", ts=T0), cxagent("codex reply", ts=T1)],
        )
        out, page = self.generate(out_path)
        self.assertIn("Prompts: 2", out)
        self.assertLess(page.index("claude at t"), page.index("codex at t"))
        _, again = self.generate(out_path)
        self.assertEqual(again, page)

    def test_antigravity_exchanges_reach_the_page(self):
        self.populate()
        ws = self.repo.as_uri()
        write_ag(
            self.antigravity_root,
            "conv-1",
            agconv([
                aguser("antigravity prompt", T3, workspace=ws),
                agnotify(T3, "2026-03-01T10:15:01.000Z", "2026-03-01T10:15:02.000Z", "antigravity reply"),
            ]),
        )
        out_path = self.tmp / "out.html"
        out, page = self.generate(out_path)
        self.assertIn("Prompts: 4", out)
        for text in ("antigravity prompt", "antigravity reply", 'class="exchange antigravity"', "gemini-3-pro-high"):
            self.assertIn(text, page)
        order = [page.index(t) for t in ("claude prompt", "codex prompt", "copilot prompt", "antigravity prompt")]
        self.assertEqual(order, sorted(order))

    def test_prompt_the_application_later_emptied_survives_on_the_page(self):
        # Replicata: a run while the conversation was intact; then the
        # application compacts it, emptying that prompt's step but keeping the
        # file. Expectata: the page keeps the prompt and its reply — the store
        # no longer holds their words, so the emptied steps are no holdings.
        ws = self.repo.as_uri()
        write_ag(self.antigravity_root, "conv-1", agconv([
            aguser("early prompt", T0, workspace=ws),
            agnotify(T0, T1, T2, "early reply"),
        ]))
        out_path = self.tmp / "out.html"
        self.generate(out_path)
        write_ag(self.antigravity_root, "conv-1", agconv([
            aguser(None, T0, status=5),
            agstep(82, T0, None, None, model=None, status=5),
            aguser("later prompt", T3, workspace=ws),
        ]))
        out, page = self.generate(out_path)
        for text in ("early prompt", "early reply", "later prompt"):
            self.assertIn(text, page)
        self.assertIn("Prompts: 2", out)

    def test_version_and_help_exit_zero(self):
        code, out, err = self.run_cli(["--version"])
        self.assertEqual((code, out, err), (0, f"{ace.VERSION}\n", ""))
        code, out, _ = self.run_cli(["--help"])
        self.assertEqual(code, 0)
        self.assertIn("REPODIR", out)

    def test_minimum_python_version_enforced_at_parse_time(self):
        # The interpreter's parser is the version check: on Python < 3.10 the
        # script's match statements are a SyntaxError before anything runs, so
        # no stale interpreter can half-run it. (An in-file sys.version_info
        # check could never fire — the file doesn't parse where it would
        # matter.) If this qual goes red, the last 3.10-only syntax left the
        # file and old interpreters would fail somewhere arbitrary instead.
        source = Path(ace.__file__).read_text(encoding="utf-8")
        with self.assertRaises(SyntaxError):
            ast.parse(source, feature_version=(3, 9))
        ast.parse(source, feature_version=(3, 10))

    # ------------------------------------------------------------- ledgers

    def ledger(self, login=LOGIN):
        return self.tmp / f"sourcery.{login}.jsonl"

    def origin(self, url):
        """Give the repo one remote, origin, at url."""
        git = self.repo / ".git"
        git.mkdir(exist_ok=True)
        (git / "config").write_text(f'[remote "origin"]\n\turl = {url}\n', encoding="utf-8")

    def refused(self, argv):
        """Run the CLI expecting a refusal: exit 2, nothing on stdout, and
        every file beside the page left byte-identical, none added or
        removed. Returns the error text."""
        return self.refused_by(self.run_cli, argv)

    def refused_by(self, cli, argv):
        """refused() for the CLI `cli` runs: sourcery's, or adopt.py's."""
        before = {path.name: path.read_bytes() for path in self.tmp.iterdir() if path.is_file()}
        code, out, err = cli(argv)
        self.assertEqual((code, out), (2, ""), err)
        self.assertEqual({path.name: path.read_bytes() for path in self.tmp.iterdir() if path.is_file()}, before)
        return err

    def theirs(self):
        """mister-person's exchange: a Codex prompt an hour before the
        populated stores' first."""
        return exchange(timestamp=utc("2026-03-01T09:00:00.000Z"), provider="Codex", model="gpt-5.4",
                        session="lg1", prompt="logan prompt", reply="logan reply")

    def test_run_writes_the_runners_ledger_beside_the_page(self):
        # Replicata: a run by the login dreeves, each provider's store
        # holding one prompt. Expectata: beside the page,
        # sourcery.dreeves.jsonl: its first line the header naming the
        # ledger format, 1, and the project, then one line per exchange in
        # freeze() form, in the page's order, the file ending in a newline;
        # stdout naming the ledger, then the page, as written. Resultata
        # (v5.5.2): no ledger; the page carried a snapshot of its exchanges
        # instead.
        self.populate()
        out_path = self.tmp / "out.html"
        out, _ = self.generate(out_path)
        header, *rows = ledger_rows(self.ledger())
        self.assertEqual(header, ledger_header(self.repo.name))
        exchanges, _ = ace.collect(self.repo, ace.discover_roots(self.env))
        self.assertEqual(rows, [jsonable(e) for e in ace.weave(exchanges)])
        self.assertEqual([row["prompt"] for row in rows], ["claude prompt", "codex prompt", "copilot prompt"])
        self.assertEqual(re.findall(r"^Written: (.*)$", out, re.MULTILINE), [str(self.ledger()), str(out_path)])

    def test_page_holds_no_snapshot_and_credits_each_row_invisibly(self):
        # Replicata: a run. Expectata: the page holds one script, its own,
        # and no data block; each article records the login whose ledger
        # holds its row, and its meta line shows that human's display name.
        # Resultata (v5.5.2): a second script, the JSON snapshot of every
        # exchange, and no human named.
        self.populate()
        _, page = self.generate(self.tmp / "out.html")
        self.assertEqual(count_scripts(page), 1)
        self.assertNotIn("application/json", page)
        self.assertEqual(len(re.findall(r'<article class="exchange \w+" id="p\d" data-login="dreeves">', page)), 3)
        self.assertEqual(page.count('<span class="human">dreev</span>'), 3)

    def test_ledger_not_page_remembers_what_the_stores_lost(self):
        # Replicata: a run; then the page is deleted and the Claude session
        # pruned from the store. Expectata: the next run restores the pruned
        # exchange from the ledger. Resultata (v5.5.2): the exchange gone,
        # the page having been its only copy.
        self.populate()
        out_path = self.tmp / "out.html"
        self.generate(out_path)
        out_path.unlink()
        (self.claude_root / "p" / "s.jsonl").unlink()
        out, page = self.generate(out_path)
        for text in ("claude prompt", "claude reply", "codex prompt", "copilot prompt"):
            self.assertIn(text, page)
        self.assertIn("Prompts: 3", out)

    def test_page_rendered_from_every_ledger_and_only_the_runners_written(self):
        # Replicata: beside the page lies another human's ledger,
        # sourcery.mister-person.jsonl, holding a Codex exchange an hour
        # older than the stores' prompts, and dreeves runs. Expectata: the
        # page shows both humans' prompts in time order, each credited to
        # its login and named by its display name; stdout counts the prompts
        # per human, by login; the other ledger is never written, and
        # dreeves's ledger holds dreeves's rows alone. Resultata (v5.5.2):
        # the other ledger ignored, and the page dreeves's alone.
        self.populate()
        other = self.ledger("mister-person")
        # Its header's keys written in another order, so a rewrite could
        # not match it.
        other.write_text(
            ledger_text(self.repo.name, [self.theirs()], {"repo": self.repo.name, "ledger": 1}), encoding="utf-8"
        )
        before = (other.stat().st_ino, other.read_bytes())
        out_path = self.tmp / "out.html"
        out, page = self.generate(out_path)
        self.assertIn("Prompts: 4\n  dreeves: 3\n  mister-person: 1\nPrompts deleted: 0\n", out)
        self.assertEqual(re.findall(r"^Written: (.*)$", out, re.MULTILINE), [str(self.ledger()), str(out_path)])
        self.assertEqual((other.stat().st_ino, other.read_bytes()), before)
        self.assertEqual(
            re.findall(
                r'data-login="([a-z-]+)">.*?<pre class="prompt">([a-z ]+)</pre>.*?<span class="human">(\w+)</span>',
                page,
                re.DOTALL,
            ),
            [
                ("mister-person", "logan prompt", "logan"),
                ("dreeves", "claude prompt", "dreev"),
                ("dreeves", "codex prompt", "dreev"),
                ("dreeves", "copilot prompt", "dreev"),
            ],
        )
        self.assertEqual(
            [row["prompt"] for row in ledger_rows(self.ledger())[1:]], ["claude prompt", "codex prompt", "copilot prompt"]
        )

    def test_runner_without_prompts_writes_a_ledger_holding_its_header_alone(self):
        # Replicata: dreeves runs with empty stores beside mister-person's
        # ledger. Expectata: the page shows mister-person's prompt; dreeves's
        # ledger is written all the same, holding its header alone; stdout
        # counts dreeves at zero. Resultata (v5.5.2): "No prompts found".
        self.ledger("mister-person").write_text(ledger_text(self.repo.name, [self.theirs()]), encoding="utf-8")
        out, page = self.generate(self.tmp / "out.html")
        self.assertIn("logan prompt", page)
        self.assertIn("Prompts: 1\n  dreeves: 0\n  mister-person: 1\n", out)
        self.assertEqual(ledger_rows(self.ledger()), [ledger_header(self.repo.name)])

    def test_ledger_rows_in_any_order_render_in_time_order(self):
        # Replicata: ledgers whose rows lie out of time order, as a git
        # merge keeping both sides' lines can leave them. Expectata: the
        # page shows every row in time order; the runner's rewritten ledger
        # is in time order; the other ledger stays as it lies. Resultata
        # (v5.5.2): ledgers ignored.
        mine = [exchange(timestamp=utc(T3), prompt="mine late"), exchange(timestamp=utc(T0), prompt="mine early")]
        their = [
            exchange(timestamp=utc(T2), provider="Codex", prompt="their late"),
            exchange(timestamp=utc(T1), provider="Codex", prompt="their early"),
        ]
        self.ledger().write_text(ledger_text(self.repo.name, mine), encoding="utf-8")
        self.ledger("mister-person").write_text(ledger_text(self.repo.name, their), encoding="utf-8")
        before = self.ledger("mister-person").read_bytes()
        _, page = self.generate(self.tmp / "out.html")
        order = [page.index(f">{text}<") for text in ("mine early", "their early", "their late", "mine late")]
        self.assertEqual(order, sorted(order))
        self.assertEqual([row["prompt"] for row in ledger_rows(self.ledger())[1:]], ["mine early", "mine late"])
        self.assertEqual(self.ledger("mister-person").read_bytes(), before)

    def test_row_doubled_in_a_ledger_renders_once_whoever_runs(self):
        # Replicata: alice's ledger holds one row twice, beside another, as
        # an editor's "accept both" on a git conflict can leave it; dreeves
        # runs, then alice runs, on the same files, no store holding
        # anything. Expectata: each run shows the doubled row once and
        # counts alice's prompts as 2, and both write the same page.
        # Resultata (v6.0.0 before the fix): dreeves's run showed the row
        # twice and counted alice at 3; alice's run, weaving her own rows,
        # showed it once and counted 2.
        rows = [
            exchange(timestamp=utc("2026-03-01T09:00:00.000Z"), provider="Codex", model="gpt-5.4", session="al1",
                     prompt="alice prompt", reply="alice reply"),
            exchange(timestamp=utc(T3), provider="Codex", model="gpt-5.4", session="al1", prompt="alice two"),
        ]
        header, first, second = ledger_text(self.repo.name, rows).removesuffix("\n").split("\n")
        self.ledger("alice").write_text("\n".join([header, first, first, second]) + "\n", encoding="utf-8")
        out_path = self.tmp / "out.html"
        runs = {}
        for login in (LOGIN, "alice"):
            ace.gh_user = gh_says(f"{login}\n")
            runs[login] = self.generate(out_path)
        for login, (out, page) in runs.items():
            with self.subTest(runner=login):
                self.assertIn("Prompts: 2\n  alice: 2\n  dreeves: 0\n", out)
                self.assertEqual(page.count('<pre class="prompt">alice prompt</pre>'), 1)
        self.assertEqual(runs[LOGIN][1], runs["alice"][1])

    def test_rows_alike_but_for_the_session_in_a_ledger_render_once_whoever_runs(self):
        # Replicata: alice's ledger holds one Codex prompt twice, its rows
        # alike but for the session, as when one of her machines read the
        # prompt from its session's transcript and another from a fork that
        # replayed the session's records into a new one, and a merge kept
        # both lines; beside them, another prompt. dreeves runs, then alice
        # runs, on the same files, no store holding anything. Expectata:
        # each run shows the prompt once and counts alice's prompts as 2,
        # and both write the same page. Resultata (v6.0.0): as expected;
        # this guards read_ledger's weaving of every ledger it reads, which
        # collapses such rows as weave collapses a fork's replayed copies,
        # against reading other humans' ledgers unwoven: dreeves's run then
        # showed the prompt twice and counted alice at 3. (Rows identical
        # outright collapse even unwoven, read_ledger keying each row by its
        # exchange.)
        first = exchange(timestamp=utc("2026-03-01T09:00:00.000Z"), provider="Codex", model="gpt-5.4", session="al1",
                         prompt="alice prompt", reply="alice reply")
        forked = dataclasses.replace(first, session="al2")
        other = exchange(timestamp=utc(T3), provider="Codex", model="gpt-5.4", session="al1", prompt="alice two")
        self.ledger("alice").write_text(ledger_text(self.repo.name, [first, forked, other]), encoding="utf-8")
        out_path = self.tmp / "out.html"
        runs = {}
        for login in (LOGIN, "alice"):
            ace.gh_user = gh_says(f"{login}\n")
            runs[login] = self.generate(out_path)
        for login, (out, page) in runs.items():
            with self.subTest(runner=login):
                self.assertIn("Prompts: 2\n  alice: 2\n  dreeves: 0\n", out)
                self.assertEqual(page.count('<pre class="prompt">alice prompt</pre>'), 1)
        self.assertEqual(runs[LOGIN][1], runs["alice"][1])

    def test_two_readings_of_one_prompt_in_a_ledger_refused_citing_both_lines(self):
        # Replicata: a ledger holding two rows of one Codex prompt, alike in
        # agent, time, session, and words, as a git merge keeping both
        # sides' lines (merge=union) leaves them when runs on two machines
        # read the prompt differently; the rows differ in one other field: the
        # reply, the model, the effort, the images, the working time, the
        # wall-clock span, the ballots, the lines added, or the lines
        # deleted. They lie at lines 2 and 4, another prompt's row between
        # them, in dreeves's ledger (the runner's) or mister-person's, and
        # no store holds the session. Expectata: exit 2, the refusal naming
        # the prompt by agent, time, and session, citing both rows by ledger
        # and line, and saying to delete the stale line, then rerun, or to
        # rerun on the machine whose transcripts hold that session; nothing
        # written. Resultata (v6.0.0): exit 0, the prompt shown twice.
        held = self.theirs()
        between = exchange(timestamp=utc("2026-03-01T09:30:00.000Z"), provider="Codex", session="lg1", prompt="between")
        changes = (
            ("reply", "logan reply, then more"), ("model", "gpt-5.5"), ("effort", "high"),
            ("images", ("data:image/png;base64,AA==",)), ("elapsed", 12.5), ("wall", 99.0),
            ("ballots", (ace.Ballot("Q?", ("a", "b"), ("a",)),)), ("added", 3), ("deleted", 2),
        )
        out_path = self.tmp / "out.html"
        for login in (LOGIN, "mister-person"):
            for field, value in changes:
                reading = dataclasses.replace(held, **{field: value})
                self.ledger(login).write_text(ledger_text(self.repo.name, [held, between, reading]), encoding="utf-8")
                with self.subTest(login=login, field=field):
                    err = self.refused([str(self.repo), str(out_path)])
                    self.assertIn(
                        f"Rogationis Codex {held.timestamp.isoformat()} (sessio lg1) duae lectiones sunt:\n"
                        f"  {self.ledger(login)}:2\n  {self.ledger(login)}:4\n"
                        "Lineam obsoletam dele, deinde iterum curre; aut iterum curre in machina cuius "
                        "transcripta illam sessionem tenent.\n",
                        err,
                    )
            self.ledger(login).unlink()

    def test_two_readings_cite_the_first_line_holding_a_doubled_row(self):
        # Replicata: dreeves's ledger holds a Codex row at lines 2 and 3,
        # doubled, as an editor's "accept both" on a git conflict can leave
        # a row, and at line 4 a second reading of that prompt, its reply
        # longer; no store holds the session. Expectata:
        # exit 2, the refusal citing the doubled row by the first line
        # holding it, 2, and the second reading by its line, 4; nothing
        # written. Resultata (v6.0.0): as expected; this guards a row's
        # citation, the first line holding it, as read_ledger says, against
        # citing a later line holding the same row.
        held = self.theirs()
        reading = dataclasses.replace(held, reply="logan reply, then more")
        header, doubled, other = ledger_text(self.repo.name, [held, reading]).removesuffix("\n").split("\n")
        self.ledger().write_text("\n".join([header, doubled, doubled, other]) + "\n", encoding="utf-8")
        err = self.refused([str(self.repo), str(self.tmp / "out.html")])
        self.assertIn(f" duae lectiones sunt:\n  {self.ledger()}:2\n  {self.ledger()}:4\n", err)

    def test_rerun_on_the_machine_holding_the_session_heals_two_readings(self):
        # Replicata: dreeves's ledger holds two readings of the Claude Code
        # prompt his stores hold, their replies cut short at two points (as
        # runs on two machines while the reply arrived leave them, once a
        # merge keeps both sides' lines), and a prompt the stores lost.
        # Expectata: exit 0; the stores' reading replaces both readings, so
        # the page shows the prompt once, with the stores' reply, and the
        # ledger holds it once; the lost prompt is kept; no prompt is
        # reported deleted. Resultata (v6.0.0): as expected; this guards the
        # healing rerun against the check reading the ledger as it lies,
        # before the stores' readings replace its rows.
        self.populate()
        lost = exchange(timestamp=utc("2026-03-01T08:00:00.000Z"), session="cs0", prompt="lost prompt")
        cut = [exchange(session="cs1", prompt="claude prompt", reply=reply) for reply in ("claude", "claude rep")]
        self.ledger().write_text(ledger_text(self.repo.name, [lost, *cut]), encoding="utf-8")
        out, page = self.generate(self.tmp / "out.html")
        self.assertIn(f"Prompts: 4\n  {LOGIN}: 4\nPrompts deleted: 0\n", out)
        self.assertEqual(page.count('<pre class="prompt">claude prompt</pre>'), 1)
        self.assertIn("<p>claude reply</p>", page)
        self.assertEqual(
            [(row["prompt"], row["reply"]) for row in ledger_rows(self.ledger())[1:]],
            [("lost prompt", "r"), ("claude prompt", "claude reply"), ("codex prompt", "codex reply"),
             ("copilot prompt", "copilot reply")],
        )

    def test_two_readings_of_one_prompt_in_the_transcripts_refused_citing_both_files(self):
        # Replicata: two Codex transcripts of one session (one id) hold one
        # prompt, alike in time and words, one of them with the reply cut
        # short; no ledger exists. Expectata: exit 2, the refusal naming the
        # prompt by agent, time, and session and citing each reading where
        # it was read, its transcript's file, there being no ledger line;
        # nothing written. Resultata (v6.0.0): exit 0, the prompt shown
        # twice, and both readings written to the ledger.
        cwd = str(self.repo)
        first, second = (
            write_jsonl(
                self.codex_root / "sessions" / "2026" / name,
                [cxmeta(cwd, sid="cx1"), cxuser("codex prompt", ts=T1), cxagent(reply, ts=T2)],
            )
            for name, reply in (("rollout-1.jsonl", "codex reply"), ("rollout-2.jsonl", "codex re"))
        )
        err = self.refused([str(self.repo), str(self.tmp / "out.html")])
        self.assertIn(
            f"Rogationis Codex {utc(T1).isoformat()} (sessio cx1) duae lectiones sunt:\n  {first}\n  {second}\n", err
        )

    def test_two_readings_of_one_prompt_in_the_transcripts_refused_with_advice_that_fits(self):
        # Replicata: as in the qual before, two Codex transcripts of one
        # session hold one prompt, alike in time and words, one of them with
        # the reply cut short; no ledger exists. Expectata: exit 2, the
        # refusal citing both files, then saying that both readings were
        # read from this machine's transcripts, which disagree about the
        # prompt, and that if one file is a stale copy of the other, the
        # copy is to be moved out of the transcript store, then rerun; if
        # not, that sourcery does not decide which reading to believe, the
        # case to be examined and the script updated; nothing written.
        # Resultata (v6.0.0): the advice for two readings in a ledger, to
        # delete the stale line or to rerun on the machine whose
        # transcripts hold the session, neither of which can apply: no
        # ledger line holds either reading, and this is that machine.
        cwd = str(self.repo)
        first, second = (
            write_jsonl(
                self.codex_root / "sessions" / "2026" / name,
                [cxmeta(cwd, sid="cx1"), cxuser("codex prompt", ts=T1), cxagent(reply, ts=T2)],
            )
            for name, reply in (("rollout-1.jsonl", "codex reply"), ("rollout-2.jsonl", "codex re"))
        )
        err = self.refused([str(self.repo), str(self.tmp / "out.html")])
        self.assertIn(
            f" duae lectiones sunt:\n  {first}\n  {second}\n"
            "Ambae lectiones e transcriptis huius machinae lectae sunt, quae de rogatione dissentiunt. "
            "Si alter fasciculus alterius exemplar obsoletum est, exemplar e reposito transcriptorum alio move, "
            "deinde iterum curre. Sin minus, ut cum unus fasciculus bis citatur, utri lectioni credendum sit "
            "sourcery non decernit: casus inspiciendus, scriptum renovandum est.\nNihil scriptum est.",
            err,
        )
        self.assertNotIn("Lineam obsoletam dele", err)

    def test_two_readings_of_one_prompt_in_one_transcript_refused_with_advice_that_fits(self):
        # Replicata: one Claude Code transcript holding a prompt's user
        # record twice, verbatim, the reply after the second, so the first
        # reads with no reply; no ledger exists. A Claude Code pair always
        # comes from one file: a session in two files is refused before
        # any two readings are sought. Expectata: exit 2, the refusal
        # citing that one file twice, then the transcripts' advice: if one
        # file is a stale copy of the other, the copy is to be moved out of
        # the transcript store, then rerun; if not, as when one file is
        # cited twice, sourcery does not decide which reading to believe:
        # the case is to be examined and the script updated; nothing
        # written. Resultata (v6.0.0): the advice spoke only of one file a
        # stale copy of another, which cannot apply to one file holding
        # both readings.
        cwd = str(self.repo)
        transcript = write_jsonl(
            self.claude_root / "p" / "s.jsonl",
            [cu("claude prompt", ts=T0, cwd=cwd), cu("claude prompt", ts=T0, cwd=cwd),
             ca([{"type": "text", "text": "claude reply"}], cwd=cwd)],
        )
        err = self.refused([str(self.repo), str(self.tmp / "out.html")])
        self.assertIn(
            f"Rogationis Claude Code {utc(T0).isoformat()} (sessio cs1) duae lectiones sunt:\n"
            f"  {transcript}\n  {transcript}\n"
            "Ambae lectiones e transcriptis huius machinae lectae sunt, quae de rogatione dissentiunt. "
            "Si alter fasciculus alterius exemplar obsoletum est, exemplar e reposito transcriptorum alio move, "
            "deinde iterum curre. Sin minus, ut cum unus fasciculus bis citatur, utri lectioni credendum sit "
            "sourcery non decernit: casus inspiciendus, scriptum renovandum est.\nNihil scriptum est.",
            err,
        )

    def test_one_transcript_read_twice_is_no_two_readings(self):
        # Replicata: a run whose stores hold each of two transcripts twice:
        # the Claude Code store listed in AI_CHAT_CLAUDE_ROOTS beside a
        # symlink to it, and a Codex transcript copied whole into a second
        # file. Expectata: exit 0, each prompt shown and counted once: rows
        # alike but for the file they were read from are one reading, as
        # weave collapses them, never two. Resultata (v6.0.0): as expected;
        # this guards the check of the stores' own rows for two readings,
        # made apart from the ledgers', against checking them unwoven: each
        # such prompt was then refused as two readings, its one transcript
        # cited twice.
        self.populate()
        alias = self.tmp / "alias"
        alias.symlink_to(self.claude_root)
        self.env["AI_CHAT_CLAUDE_ROOTS"] = os.pathsep.join([str(self.claude_root), str(alias)])
        rollout = self.codex_root / "sessions" / "2026" / "rollout-1.jsonl"
        shutil.copy(rollout, rollout.with_name("rollout-1-copy.jsonl"))
        out, page = self.generate(self.tmp / "out.html")
        self.assertIn(f"Prompts: 3\n  {LOGIN}: 3\n", out)
        for text in ("claude prompt", "codex prompt"):
            self.assertEqual(page.count(f'<pre class="prompt">{text}</pre>'), 1)

    def test_rows_of_one_time_in_other_sessions_or_words_are_no_two_readings_of_one_prompt(self):
        # Replicata: dreeves's ledger holds a Codex prompt, then a row of the
        # same agent and time in another session (a forked session whose
        # reply differs), and one in the same session whose words differ;
        # no store holds them. Expectata: exit 0, all three shown and
        # counted: rows unlike in session or in words are no two readings of
        # one prompt. Resultata (v6.0.0): as expected; this guards the check's
        # key, agent, time, session, and words, against narrowing.
        held = self.theirs()
        forked = dataclasses.replace(held, session="lg2", reply="another reply")
        reworded = dataclasses.replace(held, prompt="logan prompt, reworded", reply="a third reply")
        self.ledger().write_text(ledger_text(self.repo.name, [held, forked, reworded]), encoding="utf-8")
        out, page = self.generate(self.tmp / "out.html")
        self.assertIn(f"Prompts: 3\n  {LOGIN}: 3\nPrompts deleted: 0\n", out)
        for text in ("logan reply", "another reply", "a third reply"):
            self.assertIn(f"<p>{text}</p>", page)

    def test_rows_of_one_session_and_words_at_another_time_or_by_another_agent_are_no_two_readings(self):
        # Replicata: dreeves's ledger holds a Codex prompt; a row alike in
        # agent, session, and words five minutes later, as when the human
        # types the same words twice in one session; and a row alike in
        # time, session, and words but of another agent, Claude Code; the
        # three replies differ, and no store holds them. Expectata: exit 0,
        # all three shown and counted, and the ledger holding all three as
        # they were, in time order: rows unlike in time or in agent are no
        # two readings of one prompt. Resultata (v6.0.0): as expected; this
        # guards the check's key, agent, time, session, and words, against
        # losing the time or the agent.
        held = self.theirs()
        again = dataclasses.replace(held, timestamp=utc("2026-03-01T09:05:00.000Z"), reply="again reply")
        other = dataclasses.replace(held, provider="Claude Code", model="claude-opus-4-8", reply="claude reply")
        self.ledger().write_text(ledger_text(self.repo.name, [held, again, other]), encoding="utf-8")
        out, page = self.generate(self.tmp / "out.html")
        self.assertIn(f"Prompts: 3\n  {LOGIN}: 3\nPrompts deleted: 0\n", out)
        for text in ("logan reply", "again reply", "claude reply"):
            self.assertIn(f"<p>{text}</p>", page)
        self.assertEqual(ledger_rows(self.ledger())[1:], [jsonable(e) for e in (other, held, again)])

    def test_two_prompts_of_one_codex_time_and_session_kept_whatever_else_differs(self):
        # Replicata: dreeves's ledger holds two Codex rows of one time and
        # one session whose words differ, and with them the model, the
        # reply (the second's empty), the effort, the working time, and the
        # wall-clock span, the shape of a pair crashla's sourcery 5 page
        # holds; no store holds them any longer. Expectata: exit 0, both
        # shown and counted, and the ledger holding both as they were.
        # Resultata (v6.0.0): as expected; this guards the check's key
        # against losing the words, which would refuse such a ledger on
        # every run, with no store left to heal it by a rerun.
        first = exchange(timestamp=utc("2026-02-21T04:18:45.833Z"), provider="Codex", model="gpt-5.4", session="cr1",
                         prompt="first words", reply="first reply", effort="high", elapsed=3.0, wall=4.0)
        second = dataclasses.replace(first, model="gpt-5.5", prompt="second words", reply="", effort="", elapsed=0.0,
                                     wall=0.0)
        self.ledger().write_text(ledger_text(self.repo.name, [first, second]), encoding="utf-8")
        out, page = self.generate(self.tmp / "out.html")
        self.assertIn(f"Prompts: 2\n  {LOGIN}: 2\nPrompts deleted: 0\n", out)
        for text in ("first words", "second words"):
            self.assertIn(f'<pre class="prompt">{text}</pre>', page)
        self.assertEqual(ledger_rows(self.ledger())[1:], [jsonable(first), jsonable(second)])

    def test_unparseable_ledger_refused_saying_to_keep_both_sides_lines(self):
        # Replicata: a ledger a git merge left conflict markers in, or one
        # otherwise unreadable as a header line then one exchange per line:
        # a line that is no JSON object, a blank line, a header missing a
        # key or carrying an extra one or none at all, a row with a field
        # unknown or missing or a timestamp unreadable, an empty file; the
        # runner's ledger, or another human's. Expectata: exit 2 citing the
        # ledger at the line, saying that a git conflict is resolved by
        # keeping both sides' lines, and nothing written. Resultata
        # (v5.5.2): ledgers ignored.
        self.populate()
        out_path = self.tmp / "out.html"
        self.generate(out_path)
        mine = self.ledger().read_text(encoding="utf-8")
        header, first, second, third = mine.removesuffix("\n").split("\n")
        row = json.loads(first)
        cases = [
            (LOGIN, [header, first, "<<<<<<< HEAD", second, "=======", third, ">>>>>>> theirs"], 3),
            (LOGIN, [header, "{not json", second, third], 2),
            (LOGIN, [header, first, "", second, third], 3),
            (LOGIN, [header, "[1, 2]", second, third], 2),
            (LOGIN, [header, json.dumps({**row, "author": "someone"}), second, third], 2),
            (LOGIN, [header, json.dumps({k: v for k, v in row.items() if k != "wall"}), second, third], 2),
            (LOGIN, [header, first, json.dumps({**row, "timestamp": "yesterday"}), third], 3),
            (LOGIN, [json.dumps({"ledger": 1}), first, second, third], 1),
            (LOGIN, [json.dumps({"ledger": 1, "repo": self.repo.name, "login": LOGIN}), first], 1),
            (LOGIN, [json.dumps({"ledger": 1, "repo": 5}), first], 1),
            (LOGIN, [first, second, third], 1),
            (LOGIN, None, 1),
            ("mister-person", [header, "<<<<<<< HEAD", first, "=======", ">>>>>>> theirs"], 2),
        ]
        for login, lines, number in cases:
            text = "" if lines is None else "\n".join(lines) + "\n"
            self.ledger(login).write_text(text, encoding="utf-8")
            with self.subTest(login=login, lines=lines):
                err = self.refused([str(self.repo), str(out_path)])
                self.assertIn(f"{self.ledger(login)}:{number}\n", err)
                self.assertIn("utriusque partis lineas servando", err)
            self.ledger().write_text(mine, encoding="utf-8")
            self.ledger("mister-person").unlink(missing_ok=True)

    def test_unparseable_ledger_refusal_says_git_keeps_both_sides_lines_given_the_union_attribute(self):
        # Replicata: the runner's ledger holding git conflict markers, or
        # another human's, or a ledger damaged otherwise (a line that is no
        # JSON). Expectata: exit 2, the refusal saying, after how to resolve
        # a conflict by keeping both sides' lines, that with the line
        # "sourcery.*.jsonl merge=union" added to the attributes file git
        # reads for all the human's repositories (normally
        # ~/.config/git/attributes) git keeps both sides' lines by itself,
        # but that such a merge can leave the header line twice, the one to
        # keep on the first line being the one naming this project, 'repo';
        # nothing written. Resultata (v6.0.0): no word of git's attributes
        # file. Resultata (v6.0.0 before the fix): the attributes file named
        # as ~/.config/git/attributes alone, which git reads only when
        # core.attributesFile and $XDG_CONFIG_HOME are unset
        # (gitattributes(5)), and no word of a header line left twice.
        self.populate()
        out_path = self.tmp / "out.html"
        self.generate(out_path)
        mine = self.ledger().read_text(encoding="utf-8")
        header, first, second, third = mine.removesuffix("\n").split("\n")
        for login, lines in (
            (LOGIN, [header, first, "<<<<<<< HEAD", second, "=======", third, ">>>>>>> theirs"]),
            ("mister-person", [header, "<<<<<<< HEAD", first, "=======", ">>>>>>> theirs"]),
            (LOGIN, [header, "{not json", second, third]),
        ):
            self.ledger(login).write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.subTest(login=login, lines=lines):
                err = self.refused([str(self.repo), str(out_path)])
                self.assertIn(
                    "conflictum solve utriusque partis lineas servando (lineam capitis semel), deinde iterum curre.\n"
                    "Linea 'sourcery.*.jsonl merge=union' in fasciculo attributorum quem git omnibus repositoriis "
                    "tuis legit (plerumque ~/.config/git/attributes) addita, git sponte utriusque partis lineas "
                    "servat; sed fusio talis lineam capitis bis relinquere potest: eam quae 'repo' nominat in prima "
                    "linea serva, alteram dele, deinde iterum curre.\nNihil scriptum est.",
                    err,
                )
            self.ledger().write_text(mine, encoding="utf-8")
            self.ledger("mister-person").unlink(missing_ok=True)

    def test_unreadable_ledger_refused_cleanly(self):
        # Replicata: beside the page lies a file named like a ledger that is
        # no UTF-8 text, or a directory so named. Expectata: exit 2 naming
        # it, nothing written. Resultata (v5.5.2): ledgers ignored.
        self.populate()
        out_path = self.tmp / "out.html"
        self.ledger().write_bytes(b'{"ledger": 1, "repo": "repo"}\n\xff\xfe\n')
        err = self.refused([str(self.repo), str(out_path)])
        self.assertIn(str(self.ledger()), err)
        self.ledger().unlink()
        self.ledger().mkdir()
        err = self.refused([str(self.repo), str(out_path)])
        self.assertIn(str(self.ledger()), err)

    def test_ledger_of_another_project_refused(self):
        # Replicata: beside the page lies a ledger whose header names
        # another project (that project's page shares the directory, or
        # this checkout's directory is named otherwise than the one the
        # ledger was written in): the runner's own, or another human's.
        # Expectata: exit 2 naming both projects, the ledger, and this
        # checkout's directory; nothing written. Resultata (v5.5.2): ledgers
        # ignored. Resultata (v6.0.0 before the fix): the directory unnamed,
        # and no word that a project is named by its checkout's directory.
        self.populate()
        out_path = self.tmp / "out.html"
        for login in (LOGIN, "mister-person"):
            self.ledger(login).write_text(ledger_text("elsewhere", [self.theirs()]), encoding="utf-8")
            with self.subTest(login=login):
                err = self.refused([str(self.repo), str(out_path)])
                self.assertIn(str(self.ledger(login)), err)
                self.assertIn("'elsewhere'", err)
                self.assertIn(f"'{self.repo.name}'", err)
                self.assertIn(str(self.repo), err)
            self.ledger(login).unlink()

    def test_ledger_names_its_project_by_its_public_homes_last_segment(self):
        # Replicata: a run in a checkout whose directory, repo, is named
        # otherwise than its repository, as beemblog is for
        # github.com/dreeves/blog: origin is that repository, given by ssh
        # or by https. Expectata: the ledger's first line names the project
        # blog, the last segment of the public home; the page's title and
        # heading still name the directory, repo. Resultata (v6.0.0): the
        # first line named the directory, repo.
        self.populate()
        out_path = self.tmp / "out.html"
        for url in ("git@github.com:dreeves/blog.git", "https://github.com/dreeves/blog"):
            self.origin(url)
            with self.subTest(url=url):
                _, page = self.generate(out_path)
                self.assertEqual(ledger_rows(self.ledger())[0], ledger_header("blog"))
                self.assertIn("<title>repo</title>", page)
                self.assertIn("<h1>repo</h1>", page)

    def test_ledgers_naming_the_repository_read_in_a_checkout_named_otherwise(self):
        # Replicata: beside the page lie dreeves's and mister-person's
        # ledgers, their first lines naming blog, as runs in checkouts of
        # github.com/dreeves/blog named blog wrote them, dreeves's holding a
        # prompt the stores lost; dreeves runs in a checkout of that
        # repository named repo. Expectata: exit 0; the page shows both
        # humans' prompts, the lost one among them; dreeves's ledger still
        # names blog; mister-person's is left as it was. Resultata (v6.0.0):
        # refused as another project's ledger, the directory to be renamed
        # blog.
        self.populate()
        self.origin("git@github.com:dreeves/blog.git")
        lost = exchange(timestamp=utc("2026-03-01T08:00:00.000Z"), session="cs0", prompt="lost prompt")
        self.ledger().write_text(ledger_text("blog", [lost]), encoding="utf-8")
        self.ledger("mister-person").write_text(ledger_text("blog", [self.theirs()]), encoding="utf-8")
        before = self.ledger("mister-person").read_bytes()
        out, page = self.generate(self.tmp / "out.html")
        self.assertIn("Prompts: 5\n  dreeves: 4\n  mister-person: 1\nPrompts deleted: 0\n", out)
        for text in ("lost prompt", "logan prompt", "claude prompt"):
            self.assertIn(f'<pre class="prompt">{text}</pre>', page)
        self.assertEqual(ledger_rows(self.ledger())[0], ledger_header("blog"))
        self.assertEqual(self.ledger("mister-person").read_bytes(), before)

    def test_checkout_without_a_public_home_names_its_project_by_its_directory(self):
        # Replicata: a run in a checkout named repo whose origin names no
        # public home (a git:// mirror of github.com/facebook/codemod, an
        # ssh host serving no web page), or with no remote, or no .git at
        # all; then beside the page a ledger whose first line names codemod.
        # Expectata: the run's ledger names the directory, repo; the codemod
        # ledger is refused as another project's, nothing written.
        # Resultata (v6.0.0): as expected; this guards the directory's name
        # as the project's when repo_remote finds no public home, against a
        # name read off the raw remote URL.
        self.populate()
        out_path = self.tmp / "out.html"
        configs = (
            '[remote "origin"]\n\turl = git://github.com/facebook/codemod.git\n',
            '[remote "origin"]\n\turl = ssh://dreeves@mpdev.mooo.com/var/dev/codemod\n',
            "[core]\n\tbare = false\n",
            None,
        )
        for config in configs:
            shutil.rmtree(self.repo / ".git", ignore_errors=True)
            if config is not None:
                (self.repo / ".git").mkdir()
                (self.repo / ".git" / "config").write_text(config, encoding="utf-8")
            with self.subTest(config=config):
                self.generate(out_path)
                self.assertEqual(ledger_rows(self.ledger())[0], ledger_header("repo"))
                self.ledger("mister-person").write_text(ledger_text("codemod", [self.theirs()]), encoding="utf-8")
                err = self.refused([str(self.repo), str(out_path)])
                self.assertIn("'codemod', non 'repo'", err)
                self.ledger("mister-person").unlink()

    def test_ledger_naming_the_checkouts_directory_refused_while_it_has_a_public_home(self):
        # Replicata: a checkout named repo whose origin is
        # github.com/dreeves/blog; beside the page, a ledger whose first
        # line names repo, the directory, as only a checkout without a
        # public home names its project: dreeves's own, or mister-person's.
        # Expectata: exit 2, naming 'repo', non 'blog', this checkout, and
        # the ledger, and saying that a project is named by the last part of
        # the path of its public home's URL, and by its directory only when
        # it has no public home; nothing written. Resultata (v6.0.0): as
        # expected; this guards the directory's name as the project's only
        # without a public home, against a check accepting either name, and
        # the naming rule against dropping out of the refusal.
        self.populate()
        self.origin("git@github.com:dreeves/blog.git")
        out_path = self.tmp / "out.html"
        for login in (LOGIN, "mister-person"):
            self.ledger(login).write_text(ledger_text(self.repo.name, [self.theirs()]), encoding="utf-8")
            with self.subTest(login=login):
                err = self.refused([str(self.repo), str(out_path)])
                self.assertIn(
                    f"Tabula ad aliud inceptum pertinet: 'repo', non 'blog' ({self.repo}): {self.ledger(login)}\n", err
                )
                self.assertIn(
                    "Inceptum nominatur ultima parte viae URL sedis suae publicae (remoti 'origin'), "
                    "aut, si sedem publicam non habet, nomine directorii sui.\n",
                    err,
                )
            self.ledger(login).unlink()

    def test_ledgers_of_a_repository_renamed_on_github_refused_until_their_first_lines_name_it(self):
        # Replicata: github.com/dreeves/blog is renamed newblog on GitHub,
        # and origin now names newblog; dreeves's and mister-person's
        # ledgers' first lines still name blog. Expectata: each ledger
        # refused in turn, exit 2, naming the project its first line names,
        # this checkout's, this checkout's directory, and the ledger, and
        # saying that a repository renamed on GitHub has its ledgers refused
        # until each ledger's first line names the new repository, and that
        # if the ledger is this project's and the repository is now named
        # newblog, 'newblog' is to be written in place of 'blog' in each
        # ledger's first line; nothing written. With both first lines so
        # edited, the run succeeds. Resultata (v6.0.0): the ledgers compared
        # with the directory's name, repo, and the directory to be renamed
        # blog. Resultata (v6.0.0 before the fix): the first lines to be
        # edited "if this checkout is that project", as true of a checkout
        # whose origin still names the old repository (see the next qual).
        self.populate()
        self.origin("https://github.com/dreeves/newblog.git")
        lost = exchange(timestamp=utc("2026-03-01T08:00:00.000Z"), session="cs0", prompt="lost prompt")
        self.ledger().write_text(ledger_text("blog", [lost]), encoding="utf-8")
        self.ledger("mister-person").write_text(ledger_text("blog", [self.theirs()]), encoding="utf-8")
        out_path = self.tmp / "out.html"
        for login in (LOGIN, "mister-person"):
            with self.subTest(login=login):
                err = self.refused([str(self.repo), str(out_path)])
                self.assertIn(
                    f"Tabula ad aliud inceptum pertinet: 'blog', non 'newblog' ({self.repo}): {self.ledger(login)}\n",
                    err,
                )
                self.assertIn(
                    "Repositorio in GitHub renominato, sourcery tabulas eius recusat donec prima linea "
                    "cuiusque tabulae novum repositorium nominet.\n"
                    "Si tabula ad hoc inceptum pertinet: si repositorium nunc 'newblog' nominatur, in prima "
                    "linea cuiusque tabulae pro 'blog' scribe 'newblog'; ",
                    err,
                )
            first, rest = self.ledger(login).read_text(encoding="utf-8").split("\n", 1)
            self.ledger(login).write_text(first.replace('"blog"', '"newblog"') + "\n" + rest, encoding="utf-8")
        out, page = self.generate(out_path)
        self.assertIn("Prompts: 5\n  dreeves: 4\n  mister-person: 1\n", out)
        self.assertIn('<pre class="prompt">lost prompt</pre>', page)

    def test_another_projects_ledger_refusal_names_the_public_home_and_the_fix_for_each_cause(self):
        # Replicata: beside the page lies mister-person's ledger, its first
        # line naming another project than this checkout's, for one of
        # three causes: (a) github.com/dreeves/blog was renamed newblog on
        # GitHub and the ledger names newblog, but this checkout's origin
        # still names blog, as a clone keeps working while GitHub redirects
        # the old URL; (b) the repository has no public home (origin
        # ssh://dreeves@mpdev.mooo.com/var/dev/mp) and the ledger names mp,
        # mister-person's checkout's directory, this one being named repo;
        # (c) the ledger is in fact another project's, elsewhere's, whose
        # page shares this page's directory. Expectata: exit 2, the refusal
        # naming, after the ledger, this checkout's public home as sourcery
        # found it ('' when none); then, after the mandated sentence on
        # repositories renamed on GitHub, saying: if the ledger is this
        # project's, write this checkout's project name in place of the
        # ledger's in each first line if the repository is now so named,
        # and if not, fix this checkout, setting origin's URL to the
        # repository's home, or, without a public home, renaming the
        # directory as the ledger names the project; if the ledger is in
        # fact another project's, choose another output path; then rerun;
        # nothing written. Resultata (v6.0.0): no public home named, and the
        # one advice, "if this checkout is that project", to write this
        # checkout's project name into each first line: in (a) undoing the
        # rename ("pro 'newblog' scribe 'blog'"), in (b) making
        # mister-person's checkout refuse in turn ("pro 'mp' scribe
        # 'repo'"), in (c) no advice at all.
        self.populate()
        out_path = self.tmp / "out.html"
        for url, theirs, ours, home in (
            ("git@github.com:dreeves/blog.git", "newblog", "blog", "https://github.com/dreeves/blog"),
            ("ssh://dreeves@mpdev.mooo.com/var/dev/mp", "mp", "repo", ""),
            ("https://github.com/dreeves/repo", "elsewhere", "repo", "https://github.com/dreeves/repo"),
        ):
            self.origin(url)
            self.ledger("mister-person").write_text(ledger_text(theirs, [self.theirs()]), encoding="utf-8")
            with self.subTest(url=url):
                err = self.refused([str(self.repo), str(out_path)])
                self.assertIn(
                    f"Tabula ad aliud inceptum pertinet: {theirs!r}, non {ours!r} ({self.repo}): "
                    f"{self.ledger('mister-person')}\n"
                    f"Sedes publica huius directorii (ex remoto 'origin'): {home!r}\n",
                    err,
                )
                self.assertIn(
                    "cuiusque tabulae novum repositorium nominet.\n"
                    f"Si tabula ad hoc inceptum pertinet: si repositorium nunc {ours!r} nominatur, in prima "
                    f"linea cuiusque tabulae pro {theirs!r} scribe {ours!r}; sin minus, hic corrige: "
                    "URL ipsius 'origin' ad sedem repositorii constitue, aut, si sedem publicam non habet, "
                    f"directorium renomina {theirs!r}. Si vero re vera ad aliud inceptum pertinet, "
                    "aliam viam output elige.\n"
                    "Deinde iterum curre.\nNihil scriptum est.",
                    err,
                )

    def test_ledger_naming_the_project_in_other_capitals_refused(self):
        # Replicata: a checkout named repo whose origin is
        # github.com/dreeves/TagTime; beside the page, the runner's ledger,
        # or mister-person's, its first line naming tagtime, as a run in a
        # checkout cloned by that URL in lowercase names the project (GitHub
        # ignores case in repository names). Expectata: exit 2, refused as
        # another project's ledger, 'tagtime', non 'TagTime'; nothing
        # written. Resultata (v6.0.0): as expected; this guards project
        # names compared exactly, capitals and all, as decided, against a
        # comparison ignoring case.
        self.populate()
        self.origin("https://github.com/dreeves/TagTime.git")
        out_path = self.tmp / "out.html"
        for login in (LOGIN, "mister-person"):
            self.ledger(login).write_text(ledger_text("tagtime", [self.theirs()]), encoding="utf-8")
            with self.subTest(login=login):
                err = self.refused([str(self.repo), str(out_path)])
                self.assertIn(
                    f"Tabula ad aliud inceptum pertinet: 'tagtime', non 'TagTime' ({self.repo}): {self.ledger(login)}\n",
                    err,
                )
            self.ledger(login).unlink()

    def test_public_home_naming_no_repository_refused_naming_it(self):
        # Replicata: a run, and an adoption of a page sourcery 5 wrote, in a
        # checkout whose origin's URL leaves a public home naming no
        # repository: a host alone (https://github.com), or a path ending in
        # "/", as an origin URL ending in "//" or "/.git" leaves it, in https
        # or scp form. Expectata: exit 2, the refusal naming that home as
        # sourcery read it from origin and this checkout, and saying to set
        # origin's URL to the repository's home, then rerun; nothing
        # written. Resultata (v6.0.0): an AssertionError traceback.
        self.populate()
        out_path, old = self.tmp / "out.html", self.tmp / "old.html"
        old.write_text(five_page(self.repo, [exchange()]), encoding="utf-8")
        for url, home in (
            ("https://github.com", "https://github.com"),
            ("https://github.com/dreeves//", "https://github.com/dreeves/"),
            ("https://github.com/dreeves/.git", "https://github.com/dreeves/"),
            ("git@github.com:dreeves/.git", "https://github.com/dreeves/"),
            ("https://github.com/dreeves/blog//", "https://github.com/dreeves/blog/"),
        ):
            self.origin(url)
            for cli, argv in (
                (self.run_cli, [str(self.repo), str(out_path)]),
                (self.adopt_cli, [str(self.repo), str(old), str(out_path), LOGIN]),
            ):
                with self.subTest(url=url, cli=cli.__name__):
                    err = self.refused_by(cli, argv)
                    self.assertIn(
                        f"Sedes publica ex remoto 'origin' lecta nullum repositorium nominat: {home!r} ({self.repo})\n"
                        "URL ipsius 'origin' ad sedem repositorii constitue, deinde iterum curre.",
                        err,
                    )

    def test_prompt_held_by_two_ledgers_refused_naming_both(self):
        # Replicata: dreeves's and mister-person's ledgers both hold a Codex
        # prompt of one time (one ledger copied as the other, say), which no
        # store holds. Expectata: exit 2 naming both ledgers and the
        # prompt's agent and time; nothing written. Two rows of one ledger
        # sharing a time (a forked session whose replies differ) are no
        # refusal. Resultata (v5.5.2): ledgers ignored.
        held = self.theirs()
        for login in (LOGIN, "mister-person"):
            self.ledger(login).write_text(ledger_text(self.repo.name, [held]), encoding="utf-8")
        out_path = self.tmp / "out.html"
        err = self.refused([str(self.repo), str(out_path)])
        self.assertIn(str(self.ledger()), err)
        self.assertIn(str(self.ledger("mister-person")), err)
        self.assertIn(f"Codex {held.timestamp.isoformat()}", err)
        self.ledger("mister-person").unlink()
        fork = dataclasses.replace(held, session="lg2", reply="another reply")
        self.ledger().write_text(ledger_text(self.repo.name, [held, fork]), encoding="utf-8")
        out, _ = self.generate(out_path)
        self.assertIn("Prompts: 2", out)

    def test_prompt_held_by_two_other_humans_ledgers_refused_naming_both(self):
        # Replicata: alice's and bob's ledgers both hold a Codex prompt of
        # one time, which no store holds, and dreeves runs, his own ledger
        # holding neither. Expectata: exit 2 naming both ledgers and the
        # prompt's agent and time; nothing written. Resultata with the check
        # confined to pairs that include the runner's ledger: exit 0, the
        # prompt shown twice, credited to each of them.
        self.populate()
        held = self.theirs()
        for login in ("alice", "bob"):
            self.ledger(login).write_text(ledger_text(self.repo.name, [held]), encoding="utf-8")
        err = self.refused([str(self.repo), str(self.tmp / "out.html")])
        self.assertIn(str(self.ledger("alice")), err)
        self.assertIn(str(self.ledger("bob")), err)
        self.assertIn(f"Codex {held.timestamp.isoformat()}", err)

    def test_runners_store_holding_another_humans_prompt_refused_naming_them(self):
        # Replicata: mister-person's ledger holds a Claude Code prompt whose
        # record dreeves's store holds too: as a typed prompt, or as machine
        # text the parser drops (an interrupt marker). Expectata: exit 2
        # naming mister-person and the prompt's agent and time; nothing
        # written. Resultata (v5.5.2): ledgers ignored, and the prompt shown
        # as dreeves's.
        theirs = exchange(timestamp=utc(T0), session="lg1", prompt="logan prompt")
        self.ledger("mister-person").write_text(ledger_text(self.repo.name, [theirs]), encoding="utf-8")
        for text in ("claude prompt", "[Request interrupted by user]"):
            write_jsonl(self.claude_root / "p" / "s.jsonl", [cu(text, ts=T0, cwd=str(self.repo))])
            with self.subTest(text=text):
                err = self.refused([str(self.repo), str(self.tmp / "out.html")])
                self.assertIn("mister-person", err)
                self.assertIn(f"Claude Code {utc(T0).isoformat()}", err)

    def test_runners_store_holding_other_humans_prompts_refused_listing_every_one(self):
        # Replicata: mister-person's ledger holds two Claude Code prompts
        # whose records dreeves's store holds too, as when a transcript was
        # copied from one human's machine to the other's, in either
        # direction. Expectata: exit 2 listing, under mister-person's
        # ledger with its count, both prompts by agent and time, and naming
        # the runner; nothing written. Resultata (v6.0.0 before the fix):
        # the first prompt alone named.
        theirs = [
            exchange(timestamp=utc(T0), session="lg1", prompt="logan prompt"),
            exchange(timestamp=utc(T2), session="lg1", prompt="logan two"),
        ]
        self.ledger("mister-person").write_text(ledger_text(self.repo.name, theirs), encoding="utf-8")
        write_jsonl(
            self.claude_root / "p" / "s.jsonl",
            [
                cu("logan prompt", ts=T0, cwd=str(self.repo), session="lg1"),
                ca([{"type": "text", "text": "r"}], ts=T1, cwd=str(self.repo), session="lg1"),
                cu("logan two", ts=T2, cwd=str(self.repo), session="lg1"),
            ],
        )
        err = self.refused([str(self.repo), str(self.tmp / "out.html")])
        self.assertIn(
            f"\n  {self.ledger('mister-person')}: 2"
            f"\n    Claude Code {utc(T0).isoformat()}\n    Claude Code {utc(T2).isoformat()}\n",
            err,
        )
        self.assertIn(f"({LOGIN})", err)

    def test_page_crediting_a_row_its_ledger_lacks_refused(self):
        # Replicata: a page written beside dreeves's and mister-person's
        # ledgers; then a ledger the page credits goes missing:
        # mister-person's deleted (or never committed); dreeves's renamed,
        # the stores no longer holding its rows; or dreeves's deleted while
        # the stores still hold them. Expectata: exit 2 naming the ledger
        # the page credits and a row it lacks, by agent and time; nothing
        # written. Resultata (v5.5.2): ledgers unknown; rows the stores lost
        # would vanish from the page.
        self.populate()
        theirs = self.theirs()
        self.ledger("mister-person").write_text(ledger_text(self.repo.name, [theirs]), encoding="utf-8")
        out_path = self.tmp / "out.html"
        self.generate(out_path)
        argv = [str(self.repo), str(out_path)]
        saved = {login: self.ledger(login).read_bytes() for login in (LOGIN, "mister-person")}

        self.ledger("mister-person").unlink()
        err = self.refused(argv)
        self.assertIn(str(self.ledger("mister-person")), err)
        self.assertIn(f"Codex {theirs.timestamp.isoformat()}", err)
        self.ledger("mister-person").write_bytes(saved["mister-person"])

        self.ledger().rename(self.ledger("someone"))
        stores = self.tmp / "stores"
        stores.mkdir()
        for root in (self.claude_root, self.codex_root, self.vscode_root):
            root.rename(stores / root.name)
        err = self.refused(argv)
        self.assertIn(str(self.ledger()), err)
        self.assertIn(f"Claude Code {utc(T0).isoformat()}", err)
        for root in (self.claude_root, self.codex_root, self.vscode_root):
            (stores / root.name).rename(root)
        self.ledger("someone").unlink()

        err = self.refused(argv)
        self.assertIn(str(self.ledger()), err)
        self.assertIn(f"Claude Code {utc(T0).isoformat()}", err)

    def test_page_crediting_rows_its_ledgers_lack_refused_listing_every_one(self):
        # Replicata: a page written beside dreeves's ledger (three rows) and
        # mister-person's (one); then both ledgers go missing (deleted in
        # one checkout, say), dreeves's stores still holding his rows.
        # Expectata: exit 2 listing, under each missing ledger's path with
        # its count, every row the page credits to it, by agent and time,
        # and saying how version control brings back a deleted ledger;
        # nothing written. Resultata (v6.0.0 before the fix): the first
        # lacking row alone named, and deleting the page offered with no
        # word that it drops every human's credits; taken, it turned the
        # missing ledger into a silent loss for its owner.
        self.populate()
        theirs = self.theirs()
        self.ledger("mister-person").write_text(ledger_text(self.repo.name, [theirs]), encoding="utf-8")
        out_path = self.tmp / "out.html"
        self.generate(out_path)
        for login in (LOGIN, "mister-person"):
            self.ledger(login).unlink()
        err = self.refused([str(self.repo), str(out_path)])
        mine = "".join(
            f"\n    {provider} {utc(when).isoformat()}"
            for provider, when in (("Claude Code", T0), ("Codex", T1), ("Copilot Chat", T2))
        )
        self.assertIn(f"\n  {self.ledger()}: 3{mine}\n", err)
        self.assertIn(f"\n  {self.ledger('mister-person')}: 1\n    Codex {theirs.timestamp.isoformat()}\n", err)
        self.assertIn("git log --diff-filter=D -- ", err)

    def test_page_sourcery_5_wrote_refused_pointing_to_adopt(self):
        # Replicata: the output path holds a page sourcery 5.5.2 wrote: its
        # articles credit no one, and its exchanges ride in an embedded
        # snapshot; or a page some of whose articles credit no one.
        # Expectata: exit 2 naming the page and the adopt.py command that
        # imports it, its LOGIN left for the human to give (the login of
        # whoever's sourcery 5 wrote the page, perhaps not the runner's);
        # nothing written, no ledger created. Resultata (v5.5.2): the page
        # read back and rewritten. Resultata (v6.0.0 before the fix):
        # adopt.py named with no word on whose login it takes, and a second
        # human, so refused, adopted the first human's page as his own.
        self.populate()
        out_path = self.tmp / "out.html"
        current = ace.render(self.repo, {LOGIN: [exchange(), exchange(timestamp=utc(T1), prompt="b")]})
        five = five_page(self.repo, [exchange(), exchange(timestamp=utc(T1), prompt="b")])
        self.assertEqual((count_scripts(five), five.count("data-login")), (2, 0))
        partly = current.replace(' data-login="dreeves"', "", 1)
        for page in (five, partly):
            out_path.write_text(page, encoding="utf-8")
            with self.subTest(page=page[-300:]):
                err = self.refused([str(self.repo), str(out_path)])
                self.assertIn(str(out_path), err)
                self.assertIn(f"\n  python3 adopt.py REPODIR {out_path} {out_path} LOGIN\n", err)

    def test_file_named_like_a_ledger_for_no_lowercase_login_refused(self):
        # Replicata: beside the page, a file named sourcery.*.jsonl, ignoring
        # case, that is no lowercase login's ledger: its login in capitals
        # (GitHub logins ignore case, and so do macOS file names, so it
        # would be the lowercase login's ledger too), its own prefix or
        # suffix in capitals, or a middle that is no login. Expectata: exit
        # 2 naming the file; nothing written. Files that merely share the
        # prefix or the suffix are no ledgers, and pass. Resultata (v5.5.2):
        # ledgers ignored.
        self.populate()
        out_path = self.tmp / "out.html"
        for name in (
            "sourcery.DReeves.jsonl", "Sourcery.someone.jsonl", "sourcery.someone.JSONL",
            "sourcery.foo.bar.jsonl", "sourcery.-x.jsonl", "sourcery.x-.jsonl", "sourcery.dree--ves.jsonl",
            "sourcery.dree_ves.jsonl", "sourcery..jsonl", f"sourcery.{'x' * 40}.jsonl",
        ):
            path = self.tmp / name
            path.write_text(ledger_text(self.repo.name, []), encoding="utf-8")
            with self.subTest(name=name):
                err = self.refused([str(self.repo), str(out_path)])
                self.assertIn(str(path), err)
            path.unlink()
        for name in ("sourcery.html", "sourcery.jsonl", "notes.jsonl", f"sourcery.{LOGIN}.jsonl.orig"):
            (self.tmp / name).write_text("no ledger", encoding="utf-8")
        out, _ = self.generate(out_path)
        self.assertIn(f"Prompts: 3\n  {LOGIN}: 3\n", out)

    def refusing_render(self):
        """A stand-in for render that refuses, as no input makes the real
        one refuse today. Returns the real one."""
        render = ace.render

        def refusing(*args, **kwargs):
            raise ace.UserError("render refused")

        ace.render = refusing
        return render

    def test_refusal_found_while_rendering_writes_no_ledger(self):
        # Replicata: a first run, then a rerun after the stores gained a
        # prompt, each refused as its page is rendered (render replaced by
        # a stand-in that refuses, as no input makes the real one refuse
        # today), every check made before rendering passing. Expectata:
        # exit 2 with the stand-in's refusal, and every file beside the page
        # as it was: no ledger written or rewritten, no page. Resultata
        # (v6.0.0): as expected; this guards the page's rendering before
        # anything is written, against writing the runner's ledger first.
        # Resultata (v6.0.0 before the fix): the qual made the repo's
        # remotes refuse, which they now do before anything is read, so it
        # no longer guarded the order its name states.
        self.populate()
        out_path = self.tmp / "out.html"
        argv = [str(self.repo), str(out_path)]
        render = self.refusing_render()
        try:
            self.assertIn("render refused", self.refused(argv))
            ace.render = render
            self.generate(out_path)
            write_jsonl(self.claude_root / "p" / "later.jsonl", [cu("later prompt", ts=T3, cwd=str(self.repo), session="cs9")])
            self.refusing_render()
            self.assertIn("render refused", self.refused(argv))
        finally:
            ace.render = render

    def test_failed_ledger_write_leaves_the_page_as_it_was(self):
        # Replicata: a run; then the stores gain a prompt, and the rerun's
        # write of the runner's ledger fails (os.replace refusing the
        # ledger's path, as on a full disk). Expectata: the failure
        # surfaces, and the page is left as it was, never crediting a
        # prompt its ledger lacks: once writing works again, the next run
        # succeeds, showing the new prompt. Resultata (v6.0.0): as expected;
        # this guards the write order, the runner's ledger before the page,
        # against writing the page first, which left the page crediting a
        # prompt the ledger lacked, so that every later run was refused.
        self.populate()
        out_path = self.tmp / "out.html"
        self.generate(out_path)
        page = out_path.read_bytes()
        write_jsonl(self.claude_root / "p" / "later.jsonl", [cu("later prompt", ts=T3, cwd=str(self.repo), session="cs9")])
        replace = ace.os.replace

        def failing(source, target):
            if Path(target) == self.ledger():
                raise OSError(28, "No space left on device", str(target))
            return replace(source, target)

        ace.os.replace = failing
        try:
            with self.assertRaises(OSError):
                self.run_cli([str(self.repo), str(out_path)])
        finally:
            ace.os.replace = replace
        self.assertEqual(out_path.read_bytes(), page)
        out, page = self.generate(out_path)
        self.assertIn(f"Prompts: 4\n  {LOGIN}: 4\n", out)
        self.assertIn('<pre class="prompt">later prompt</pre>', page)

    def test_run_without_a_usable_gh_login_writes_nothing(self):
        # Replicata: a run where gh is missing, or not logged in.
        # Expectata: exit 2, the error saying to run "gh auth login", and
        # neither page nor ledger written. Resultata (v5.5.2): the page
        # written, crediting no one.
        self.populate()
        for seam in (gh_missing, gh_says("", 1, 'could not find key "user"\n')):
            ace.gh_user = seam
            with self.subTest(seam=seam):
                err = self.refused([str(self.repo), str(self.tmp / "out.html")])
                self.assertIn("gh auth login", err)

    # ------------------------------------------------------------ adopt.py

    def adopt_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = adopt.run(argv)
        return code, out.getvalue(), err.getvalue()

    def adopted(self, page, output, login=LOGIN):
        """Adopt page into the ledgers beside output as login, expecting
        success. Returns stdout."""
        code, out, err = self.adopt_cli([str(self.repo), str(page), str(output), login])
        self.assertEqual(code, 0, err)
        return out

    def test_adopt_moves_a_sourcery_5_pages_snapshot_into_the_logins_ledger(self):
        # Replicata: out.html is a page sourcery 5.5.2 wrote, its snapshot
        # holding a seeded claude.ai exchange and a Claude Code exchange; it
        # is adopted in place as dreeves. Expectata: exit 0; beside it,
        # sourcery.dreeves.jsonl holds the header naming the ledger format,
        # 1, and the project, then the snapshot's exchanges in freeze()
        # form, in time order; out.html is the page sourcery 6 renders from
        # that ledger, crediting every row to dreeves, its snapshot gone;
        # stdout names the ledger, then the page, as written, and counts the
        # prompts the page held, those the ledger gained, and those each
        # human's ledger holds. Resultata (v6.0.0 before adopt.py): sourcery
        # 6 refused the page, and nothing could bring it past the refusal.
        rows = [
            exchange(provider="claude.ai", model="claude-opus-5-5", session="chat-1", prompt="chat prompt"),
            exchange(timestamp=utc(T1), prompt="claude prompt", reply="claude reply"),
        ]
        out_path = self.tmp / "out.html"
        out_path.write_text(five_page(self.repo, rows), encoding="utf-8")
        out = self.adopted(out_path, out_path)
        header, *lines = ledger_rows(self.ledger())
        self.assertEqual(header, ledger_header(self.repo.name))
        self.assertEqual(lines, [jsonable(e) for e in rows])
        self.assertEqual(out_path.read_text(encoding="utf-8"), ace.render(self.repo, {LOGIN: rows}))
        self.assertEqual(ace.page_credits(out_path), {(LOGIN, ace.holding(e)) for e in rows})
        self.assertEqual(re.findall(r"^Scriptum: (.*)$", out, re.MULTILINE), [str(self.ledger()), str(out_path)])
        self.assertIn(f"Rogationes paginae: 2; tabulae additae: 2\nRogationes: 2\n  {LOGIN}: 2\n", out)

    def test_adopted_page_then_sourcery_merges_the_stores(self):
        # Replicata: a page sourcery 5 wrote holds a prompt the stores have
        # since lost and a superseded reading of the Claude prompt they
        # still hold; it is adopted in place as dreeves, then sourcery runs
        # as dreeves. Expectata: the run exits 0; the page shows the lost
        # prompt, kept from the ledger, the store's reading of the Claude
        # prompt and not the superseded one, and the stores' other prompts;
        # stdout counts them all as dreeves's. Resultata (v6.0.0 before
        # adopt.py): the run refused the page, pointing to adopt.py.
        self.populate()
        lost = exchange(timestamp=utc("2026-03-01T09:00:00.000Z"), prompt="lost prompt", reply="lost reply")
        superseded = exchange(prompt="superseded reading", reply="superseded reply")
        out_path = self.tmp / "out.html"
        out_path.write_text(five_page(self.repo, [lost, superseded]), encoding="utf-8")
        self.adopted(out_path, out_path)
        out, page = self.generate(out_path)
        self.assertIn(f"Prompts: 4\n  {LOGIN}: 4\nPrompts deleted: 0\n", out)
        for text in ("lost prompt", "lost reply", "claude prompt", "codex prompt", "copilot prompt"):
            self.assertIn(text, page)
        self.assertNotIn("superseded", page)

    def fold_tallybee(self):
        """tallybee's migration: out.html is dreeves's page and logan.html
        mister-person's, both written by sourcery 5; out.html is adopted in
        place as dreeves, then logan.html is folded into out.html as
        mister-person. Checks that the second adoption leaves dreeves's
        ledger and logan.html as they were. Returns both pages' paths."""
        out_path, logan = self.tmp / "out.html", self.tmp / "logan.html"
        out_path.write_text(
            five_page(self.repo, [exchange(prompt="dreev one"), exchange(timestamp=utc(T2), prompt="dreev two")]),
            encoding="utf-8",
        )
        theirs = [self.theirs(), exchange(timestamp=utc(T3), provider="Codex", session="lg1", prompt="logan two")]
        logan.write_text(five_page(self.repo, theirs, version="5.5.0"), encoding="utf-8")
        before = logan.read_bytes()
        self.adopted(out_path, out_path)
        mine = (self.ledger().stat().st_ino, self.ledger().read_bytes())
        self.adopted(logan, out_path, "mister-person")
        self.assertEqual((self.ledger().stat().st_ino, self.ledger().read_bytes()), mine)
        self.assertEqual(logan.read_bytes(), before)
        return out_path, logan

    def test_second_humans_page_folded_into_their_own_ledger(self):
        # Replicata: tallybee's migration (see fold_tallybee), then a
        # sourcery run as dreeves with empty stores. Expectata: both
        # adoptions exit 0; mister-person's rows land in
        # sourcery.mister-person.jsonl and dreeves's ledger is untouched by
        # the second adoption; logan.html is left as it was; out.html, as
        # adopted and as rerun, shows both humans' prompts in time order,
        # each credited to its login and named by its display name; the run
        # counts the prompts per human. Resultata (v6.0.0 before adopt.py):
        # sourcery 6 refused both pages, and nothing could bring them past
        # the refusal.
        out_path, _ = self.fold_tallybee()
        self.assertEqual(
            [row["prompt"] for row in ledger_rows(self.ledger("mister-person"))[1:]], ["logan prompt", "logan two"]
        )
        folded = out_path.read_text(encoding="utf-8")
        out, rerun = self.generate(out_path)
        self.assertIn("Prompts: 4\n  dreeves: 2\n  mister-person: 2\nPrompts deleted: 0\n", out)
        for page in (folded, rerun):
            self.assertEqual(
                re.findall(
                    r'data-login="([a-z-]+)">.*?<pre class="prompt">([a-z ]+)</pre>.*?<span class="human">(\w+)</span>',
                    page,
                    re.DOTALL,
                ),
                [
                    ("mister-person", "logan prompt", "logan"),
                    ("dreeves", "dreev one", "dreev"),
                    ("dreeves", "dreev two", "dreev"),
                    ("mister-person", "logan two", "logan"),
                ],
            )

    def test_adoption_names_whom_the_pages_prompts_are_credited_to(self):
        # Replicata: a page sourcery 5 wrote, adopted as mister-person (as a
        # second human might adopt the first's page as himself), and the
        # same page adopted beside another output as dreeves. Expectata:
        # each adoption's stdout names the human the page's prompts are now
        # credited to, by display name and login, so a wrong login shows at
        # once. Resultata (v6.0.0 before the fix): prompts counted by login
        # alone, nothing naming whom the page's prompts went to.
        old = self.tmp / "old.html"
        old.write_text(five_page(self.repo, [exchange()]), encoding="utf-8")
        for login, named in (("mister-person", "logan (mister-person)"), (LOGIN, "dreev (dreeves)")):
            output = self.tmp / login / "out.html"
            output.parent.mkdir()
            with self.subTest(login=login):
                self.assertIn(f": {named}\n", self.adopted(old, output, login))

    def test_adopting_again_changes_nothing(self):
        # Replicata: after tallybee's migration (see fold_tallybee) and a
        # sourcery run, logan.html is folded into out.html as mister-person
        # again; then out.html, now a page sourcery 6 wrote, is adopted in
        # place as dreeves again. Expectata: the first exits 0, every file
        # byte-identical, stdout counting no prompt added; the second is
        # refused naming the page, which carries no snapshot, and saying a
        # page sourcery 6 wrote needs only sourcery.py; every file
        # byte-identical. Resultata (v6.0.0 before adopt.py): no adopt.py.
        out_path, logan = self.fold_tallybee()
        self.generate(out_path)
        before = {path.name: path.read_bytes() for path in self.tmp.iterdir() if path.is_file()}
        out = self.adopted(logan, out_path, "mister-person")
        self.assertIn("Rogationes paginae: 2; tabulae additae: 0\n", out)
        self.assertEqual({path.name: path.read_bytes() for path in self.tmp.iterdir() if path.is_file()}, before)
        err = self.refused_by(self.adopt_cli, [str(self.repo), str(out_path), str(out_path), LOGIN])
        self.assertIn(str(out_path), err)
        self.assertIn("sourcery.py", err)

    def test_adoption_keeps_the_ledgers_rows_and_adds_only_holdings_it_lacks(self):
        # Replicata: dreeves's ledger holds a reading of a Claude prompt at
        # T0 and a Codex prompt at T1; old.html, a page sourcery 5 wrote,
        # holds an older reading of the T0 prompt and a Claude prompt at T2;
        # old.html is adopted into the ledgers beside out.html as dreeves.
        # Expectata: the ledger holds its own two rows as they were, then
        # old.html's T2 row; old.html's T0 row, a prompt the ledger holds
        # already, gives way to the ledger's reading, as a ledger's row
        # gives way to a store's in a sourcery run; stdout counts 2 prompts
        # on the page, 1 added. Resultata (v6.0.0 before adopt.py): no
        # adopt.py.
        mine = [
            exchange(prompt="newer reading"),
            exchange(timestamp=utc(T1), provider="Codex", session="cx1", prompt="codex prompt"),
        ]
        self.ledger().write_text(ledger_text(self.repo.name, mine), encoding="utf-8")
        added = exchange(timestamp=utc(T2), prompt="later prompt")
        old = self.tmp / "old.html"
        old.write_text(five_page(self.repo, [exchange(prompt="older reading"), added]), encoding="utf-8")
        out = self.adopted(old, self.tmp / "out.html")
        self.assertEqual(ledger_rows(self.ledger())[1:], [jsonable(e) for e in [*mine, added]])
        self.assertIn("Rogationes paginae: 2; tabulae additae: 1\n", out)

    def test_page_holding_a_prompt_another_ledger_holds_refused_naming_its_owner(self):
        # Replicata: mister-person's ledger holds a Codex prompt; old.html,
        # a page sourcery 5 wrote, holds that prompt (its agent and time)
        # beside a prompt of its own, and is adopted as dreeves. Expectata:
        # exit 2 naming mister-person's ledger and the prompt's agent and
        # time; nothing written. Adopted as mister-person, the page is
        # theirs: exit 0, the prompt their ledger held staying as it was and
        # the other added. Resultata (v6.0.0 before adopt.py): no adopt.py.
        theirs = self.theirs()
        self.ledger("mister-person").write_text(ledger_text(self.repo.name, [theirs]), encoding="utf-8")
        old = self.tmp / "old.html"
        old.write_text(five_page(self.repo, [theirs, exchange(prompt="own prompt")]), encoding="utf-8")
        out_path = self.tmp / "out.html"
        err = self.refused_by(self.adopt_cli, [str(self.repo), str(old), str(out_path), LOGIN])
        self.assertIn(str(self.ledger("mister-person")), err)
        self.assertIn(f"Codex {theirs.timestamp.isoformat()}", err)
        out = self.adopted(old, out_path, "mister-person")
        self.assertIn("Rogationes paginae: 2; tabulae additae: 1\n", out)

    def test_adoption_refuses_two_readings_of_one_prompt_citing_each(self):
        # Replicata: old.html, a page sourcery 5 wrote, adopted beside
        # out.html as dreeves, where (a) mister-person's ledger holds two
        # readings of one Codex prompt at lines 2 and 3, as a merge keeping
        # both sides' lines leaves them, or (b) the snapshot itself holds
        # two readings of one Claude Code prompt, as sourcery 5 wrote them
        # when two transcripts disagreed. Expectata: exit 2, the refusal
        # naming the prompt by agent, time, and session, citing each reading
        # where adoption read it, a ledger's row by the ledger's line, the
        # snapshot's by the page, and saying to delete the stale line (the
        # ledger's, or the snapshot's in the page), then rerun; nothing
        # written. Resultata (v6.0.0): exit 0, the ledger and the page
        # written, the page showing the prompt twice, a page sourcery then
        # refused.
        held = self.theirs()
        own = exchange(prompt="own prompt")
        old = self.tmp / "old.html"
        argv = [str(self.repo), str(old), str(self.tmp / "out.html"), LOGIN]
        advice = "Lineam obsoletam dele (tabulae, aut memoriae paginae), deinde iterum curre.\nNihil scriptum est."
        old.write_text(five_page(self.repo, [own]), encoding="utf-8")
        reading = dataclasses.replace(held, reply="logan reply, then more")
        self.ledger("mister-person").write_text(ledger_text(self.repo.name, [held, reading]), encoding="utf-8")
        err = self.refused_by(self.adopt_cli, argv)
        self.assertIn(
            f"Rogationis Codex {held.timestamp.isoformat()} (sessio lg1) duae lectiones sunt:\n"
            f"  {self.ledger('mister-person')}:2\n  {self.ledger('mister-person')}:3\n{advice}",
            err,
        )
        self.ledger("mister-person").unlink()
        old.write_text(five_page(self.repo, [own, dataclasses.replace(own, reply="r, then more")]), encoding="utf-8")
        err = self.refused_by(self.adopt_cli, argv)
        self.assertIn(
            f"Rogationis Claude Code {own.timestamp.isoformat()} (sessio s) duae lectiones sunt:\n"
            f"  {old}\n  {old}\n{advice}",
            err,
        )

    def test_page_of_another_project_refused(self):
        # Replicata: a page sourcery 5 wrote, its snapshot naming another
        # project (elsewhere), is adopted for this repo. Expectata: exit 2
        # naming both projects, the page, and this checkout's directory, the
        # one to rename if it is that project after all; nothing written.
        # Resultata (v6.0.0 before adopt.py): no adopt.py. Resultata (v6.0.0
        # before the fix): the directory unnamed.
        old = self.tmp / "old.html"
        old.write_text(five_page(self.repo, [exchange()], name="elsewhere"), encoding="utf-8")
        err = self.refused_by(self.adopt_cli, [str(self.repo), str(old), str(self.tmp / "out.html"), LOGIN])
        for text in (str(old), "'elsewhere'", f"'{self.repo.name}'", str(self.repo)):
            self.assertIn(text, err)

    def test_adoption_checks_the_snapshot_by_directory_and_writes_the_project_name(self):
        # Replicata: a checkout named repo whose origin is
        # github.com/dreeves/blog. old.html, a page sourcery 5 wrote there,
        # its snapshot naming blog, the repository, is adopted beside
        # out.html as dreeves; then out.html, a page sourcery 5 wrote, its
        # snapshot naming repo, the directory, as sourcery 5 named projects,
        # is adopted in place as dreeves; then sourcery runs as dreeves.
        # Expectata: the first adoption refused as another project's page,
        # naming both projects; nothing written. The second exits 0, the
        # ledger's first line naming the project blog, not the directory;
        # the run reads that ledger and exits 0. Resultata (v6.0.0): the
        # ledger's first line named the directory, repo.
        self.origin("git@github.com:dreeves/blog.git")
        old, out_path = self.tmp / "old.html", self.tmp / "out.html"
        old.write_text(five_page(self.repo, [exchange()], name="blog"), encoding="utf-8")
        err = self.refused_by(self.adopt_cli, [str(self.repo), str(old), str(out_path), LOGIN])
        self.assertIn("'blog', non 'repo'", err)
        old.unlink()
        out_path.write_text(five_page(self.repo, [exchange()]), encoding="utf-8")
        self.adopted(out_path, out_path)
        self.assertEqual(ledger_rows(self.ledger())[0], ledger_header("blog"))
        out, _ = self.generate(out_path)
        self.assertIn(f"Prompts: 1\n  {LOGIN}: 1\nPrompts deleted: 0\n", out)

    def test_adoption_reads_ledgers_naming_the_project_in_a_checkout_named_otherwise(self):
        # Replicata: a checkout named repo whose origin is
        # github.com/dreeves/blog; beside out.html, a page sourcery 5 wrote
        # there, its snapshot naming repo, lies mister-person's ledger, its
        # first line naming blog; out.html is adopted in place as dreeves.
        # Expectata: exit 0; mister-person's ledger left as it was; the page
        # shows both humans' prompts, and stdout counts one prompt for each.
        # Resultata (v6.0.0): as expected; this guards adoption's reading of
        # the ledgers by the project's name, against reading them by the
        # directory's.
        self.origin("git@github.com:dreeves/blog.git")
        self.ledger("mister-person").write_text(ledger_text("blog", [self.theirs()]), encoding="utf-8")
        before = self.ledger("mister-person").read_bytes()
        out_path = self.tmp / "out.html"
        out_path.write_text(five_page(self.repo, [exchange()]), encoding="utf-8")
        out = self.adopted(out_path, out_path)
        self.assertIn(f"Rogationes: 2\n  {LOGIN}: 1\n  mister-person: 1", out)
        self.assertEqual(self.ledger("mister-person").read_bytes(), before)
        page = out_path.read_text(encoding="utf-8")
        for text in ("p", "logan prompt"):
            self.assertIn(f'<pre class="prompt">{text}</pre>', page)

    def test_page_without_exactly_one_snapshot_refused(self):
        # Replicata: adopted as dreeves: a page sourcery 6 wrote, carrying no
        # snapshot; a page carrying two snapshots; a file that is no
        # sourcery page. Expectata: exit 2 naming the page and saying a page
        # sourcery 6 wrote needs only sourcery.py; nothing written.
        # Resultata (v6.0.0 before adopt.py): no adopt.py.
        five = five_page(self.repo, [exchange()])
        block = re.search(r'<script type="application/json" id="snapshot">\n.*?\n</script>\n', five, re.DOTALL)[0]
        old = self.tmp / "old.html"
        for text in (ace.render(self.repo, {LOGIN: [exchange()]}), five.replace(block, block * 2), "no sourcery page"):
            old.write_text(text, encoding="utf-8")
            with self.subTest(text=text[-200:]):
                err = self.refused_by(self.adopt_cli, [str(self.repo), str(old), str(self.tmp / "out.html"), LOGIN])
                self.assertIn(str(old), err)
                self.assertIn("sourcery.py", err)

    def test_unreadable_page_refused_cleanly(self):
        # Replicata: adopted as dreeves: a page path where no file is, a
        # directory, a file that is no UTF-8 text. Expectata: exit 2 naming
        # the page; nothing written. Resultata (v6.0.0 before adopt.py): no
        # adopt.py.
        old = self.tmp / "old.html"
        argv = [str(self.repo), str(old), str(self.tmp / "out.html"), LOGIN]
        self.assertIn(str(old), self.refused_by(self.adopt_cli, argv))
        old.mkdir()
        self.assertIn(str(old), self.refused_by(self.adopt_cli, argv))
        old.rmdir()
        old.write_bytes(five_page(self.repo, [exchange()]).encode("utf-8") + b"\xff\xfe")
        self.assertIn(str(old), self.refused_by(self.adopt_cli, argv))

    def test_damaged_snapshot_refused(self):
        # Replicata: adopted as dreeves, a page sourcery 5 wrote whose
        # snapshot is damaged: no JSON; no JSON object; its repo key
        # missing; a key extra; its exchanges no list; a row with a field
        # unknown; a row whose timestamp is unreadable; a row that is no
        # object. Expectata: exit 2 naming the page; nothing written.
        # Resultata (v6.0.0 before adopt.py): no adopt.py.
        five = five_page(self.repo, [exchange()])
        row = ace.freeze(exchange())
        name = json.dumps(self.repo.name)
        old = self.tmp / "old.html"
        for snapshot in (
            "{not json",
            "[1, 2]",
            '{"sourcery": "5.5.2", "exchanges": []}',
            f'{{"sourcery": "5.5.2", "repo": {name}, "exchanges": [], "author": "someone"}}',
            f'{{"sourcery": "5.5.2", "repo": {name}, "exchanges": {{}}}}',
            f'{{"sourcery": "5.5.2", "repo": {name}, "exchanges": [{json.dumps({**row, "author": "someone"})}]}}',
            f'{{"sourcery": "5.5.2", "repo": {name}, "exchanges": [{json.dumps({**row, "timestamp": "yesterday"})}]}}',
            f'{{"sourcery": "5.5.2", "repo": {name}, "exchanges": ["just text"]}}',
            f'{{"sourcery": "5.5.2", "repo": {name}, "exchanges": [5]}}',
        ):
            old.write_text(with_snapshot(five, snapshot), encoding="utf-8")
            with self.subTest(snapshot=snapshot):
                err = self.refused_by(self.adopt_cli, [str(self.repo), str(old), str(self.tmp / "out.html"), LOGIN])
                self.assertIn(str(old), err)

    def test_output_page_whose_rows_would_be_lost_refused(self):
        # Replicata: old.html, a page sourcery 5 wrote, is adopted into an
        # output path holding a page adoption would replace without its
        # rows: another page sourcery 5 wrote, whose snapshot is not the one
        # adopted; a file that is no sourcery page; a page sourcery 6 wrote
        # crediting a row to mister-person, whose ledger is missing (deleted,
        # or never committed). Expectata: exit 2 naming the output page, or
        # the missing ledger; nothing written. Resultata (v6.0.0 before
        # adopt.py): no adopt.py.
        old = self.tmp / "old.html"
        old.write_text(five_page(self.repo, [exchange(timestamp=utc(T1), prompt="old prompt")]), encoding="utf-8")
        out_path = self.tmp / "out.html"
        argv = [str(self.repo), str(old), str(out_path), LOGIN]
        for text in (five_page(self.repo, [exchange()]), "no sourcery page"):
            out_path.write_text(text, encoding="utf-8")
            with self.subTest(text=text[-200:]):
                self.assertIn(str(out_path), self.refused_by(self.adopt_cli, argv))
        out_path.write_text(ace.render(self.repo, {"mister-person": [self.theirs()]}), encoding="utf-8")
        self.assertIn(str(self.ledger("mister-person")), self.refused_by(self.adopt_cli, argv))

    def test_ledger_lands_beside_the_output_page(self):
        # Replicata: old.html, a page sourcery 5 wrote, is adopted as dreeves
        # into an output page in a subdirectory, pub/sourcery.html, where
        # molecall keeps its page. Expectata: exit 0; the ledger is
        # pub/sourcery.dreeves.jsonl, beside the output page, which is
        # written there; old.html is left as it was. Resultata (v6.0.0
        # before adopt.py): no adopt.py.
        old = self.tmp / "old.html"
        old.write_text(five_page(self.repo, [exchange()]), encoding="utf-8")
        before = old.read_bytes()
        pub = self.tmp / "pub"
        pub.mkdir()
        self.adopted(old, pub / "sourcery.html")
        self.assertEqual(sorted(path.name for path in pub.iterdir()), ["sourcery.dreeves.jsonl", "sourcery.html"])
        self.assertFalse(self.ledger().exists())
        self.assertEqual(old.read_bytes(), before)

    def test_login_given_in_capitals_names_the_lowercase_ledger_and_no_login_refused(self):
        # Replicata: a page sourcery 5 wrote, adopted under a string that is
        # no GitHub login, or as "Mister-Person" (GitHub keeps the capitals
        # a human registered with, but ignores case). Expectata: no login is
        # refused, exit 2, quoting it, nothing written; capitals name the
        # lowercase login's ledger, sourcery.mister-person.jsonl, as sourcery
        # lowercases the login gh reports. Resultata (v6.0.0 before
        # adopt.py): no adopt.py.
        old, out_path = self.tmp / "old.html", self.tmp / "out.html"
        old.write_text(five_page(self.repo, [exchange()]), encoding="utf-8")
        for login in ("", "dree ves", "-x", "x-", "dree--ves", "dree_ves", "../x", "x" * 40):
            with self.subTest(login=login):
                err = self.refused_by(self.adopt_cli, [str(self.repo), str(old), str(out_path), login])
                self.assertIn(repr(login), err)
        self.adopted(old, out_path, "Mister-Person")
        self.assertEqual([path.name for path in self.tmp.glob("sourcery.*.jsonl")], ["sourcery.mister-person.jsonl"])

    def test_adopt_usage_refused_and_nothing_written(self):
        # Replicata: adopt.py run with too few or too many arguments, or
        # naming a project directory that does not exist. Expectata: exit 2
        # with the usage text, or naming the missing directory; nothing
        # written. Resultata (v6.0.0 before adopt.py): no adopt.py.
        old, out_path = self.tmp / "old.html", self.tmp / "out.html"
        old.write_text(five_page(self.repo, [exchange()]), encoding="utf-8")
        for argv in ([], [str(self.repo), str(old), str(out_path)], [str(self.repo), str(old), str(out_path), LOGIN, "x"]):
            with self.subTest(argv=argv):
                self.assertIn("adopt.py REPODIR", self.refused_by(self.adopt_cli, argv))
        absent = self.tmp / "absent"
        self.assertIn(str(absent), self.refused_by(self.adopt_cli, [str(absent), str(old), str(out_path), LOGIN]))

    def test_refusal_found_while_rendering_writes_nothing_on_adoption(self):
        # Replicata: a page sourcery 5 wrote, adopted in place as dreeves,
        # refused as the new page is rendered (render replaced by a stand-in
        # that refuses, as no input makes the real one refuse today), every
        # check made before rendering passing. Expectata: exit 2 with the
        # stand-in's refusal, with neither ledger nor page written.
        # Resultata (v6.0.0): as expected; this guards the page's rendering
        # before anything is written, against writing the ledger first.
        # Resultata (v6.0.0 before the fix): the qual made the repo's
        # remotes refuse, which they now do before anything is read, so it
        # no longer guarded the order its name states.
        out_path = self.tmp / "out.html"
        out_path.write_text(five_page(self.repo, [exchange()]), encoding="utf-8")
        render = self.refusing_render()
        try:
            err = self.refused_by(self.adopt_cli, [str(self.repo), str(out_path), str(out_path), LOGIN])
        finally:
            ace.render = render
        self.assertIn("render refused", err)

    def test_adoption_refuses_a_home_under_a_remote_not_named_origin(self):
        # Replicata: a page sourcery 5 wrote, adopted in place as dreeves,
        # in a checkout whose one remote, efme, names its public home, no
        # remote being named origin. Expectata: exit 2 naming efme and
        # origin, as sourcery refuses such a checkout, with neither ledger
        # nor page written. Resultata (v6.0.0): as expected; this guards
        # adopt.py against reading such a home as none, which would name
        # the project after the checkout's directory and write a ledger and
        # a page in a checkout sourcery refuses. The scenario is the one
        # test_refusal_found_while_rendering_writes_nothing_on_adoption
        # used before its rewrite, kept as its own qual.
        git = self.repo / ".git"
        git.mkdir()
        (git / "config").write_text('[remote "efme"]\n\turl = git@github.com:beeminder/efme.git\n', encoding="utf-8")
        out_path = self.tmp / "out.html"
        out_path.write_text(five_page(self.repo, [exchange()]), encoding="utf-8")
        err = self.refused_by(self.adopt_cli, [str(self.repo), str(out_path), str(out_path), LOGIN])
        self.assertIn("efme (https://github.com/beeminder/efme)", err)
        self.assertIn("'origin'", err)

    def test_failed_ledger_write_leaves_the_page_as_it_was_on_adoption(self):
        # Replicata: a page sourcery 5 wrote, adopted in place as dreeves,
        # the ledger's write failing (os.replace refusing the ledger's path,
        # as on a full disk). Expectata: the failure surfaces, and the page
        # is left as it was, its snapshot intact, so that once writing works
        # again, adopting it succeeds. Resultata (v6.0.0): as expected; this
        # guards the write order, the ledger before the page, against
        # writing the page first, which replaced the page sourcery 5 wrote,
        # and with it its snapshot, the only copy of its rows.
        out_path = self.tmp / "out.html"
        out_path.write_text(five_page(self.repo, [exchange()]), encoding="utf-8")
        page = out_path.read_bytes()
        argv = [str(self.repo), str(out_path), str(out_path), LOGIN]
        replace = ace.os.replace

        def failing(source, target):
            if Path(target) == self.ledger():
                raise OSError(28, "No space left on device", str(target))
            return replace(source, target)

        ace.os.replace = failing
        try:
            with self.assertRaises(OSError):
                self.adopt_cli(argv)
        finally:
            ace.os.replace = replace
        self.assertEqual(out_path.read_bytes(), page)
        self.assertFalse(self.ledger().exists())
        self.assertIn("Rogationes paginae: 1; tabulae additae: 1\n", self.adopted(out_path, out_path))


class IdentityQuals(unittest.TestCase):
    def setUp(self):
        self.gh_user = ace.gh_user

    def tearDown(self):
        ace.gh_user = self.gh_user

    def test_login_is_ghs_saved_github_login_lowercased(self):
        # Replicata: gh's saved login for github.com is "Mister-Person"
        # (GitHub keeps the capitals a human registered with but ignores
        # case), or all lowercase, or one letter, or 39 characters.
        # Expectata: the login, lowercased, so a human's ledger has one file
        # name on every file system. Resultata (v5.5.2): no identity.
        for said, login in (
            ("Mister-Person\n", "mister-person"), ("dreeves\n", "dreeves"), ("a\n", "a"),
            ("A1-b2-C3\n", "a1-b2-c3"), ("x" * 39 + "\n", "x" * 39),
        ):
            ace.gh_user = gh_says(said)
            with self.subTest(said=said):
                self.assertEqual(ace.github_login(), login)

    def test_identity_lookup_is_one_local_gh_config_call(self):
        # Replicata: the seam's own body, run with subprocess.run replaced
        # by a recorder. Expectata: one call, of "gh config get user -h
        # github.com", which reads gh's saved configuration (never "gh api",
        # so no network). Resultata (v5.5.2): no identity.
        calls = []
        original = ace.subprocess.run

        def recorder(args, **options):
            calls.append(tuple(args))
            return subprocess.CompletedProcess(args, 0, "dreeves\n", "")

        ace.subprocess.run = recorder
        try:
            self.assertEqual(ace.github_login(), "dreeves")
        finally:
            ace.subprocess.run = original
        self.assertEqual(calls, [("gh", "config", "get", "user", "-h", "github.com")])

    def test_real_seam_reads_the_gh_on_path(self):
        # Replicata: the seam itself, unreplaced, with PATH holding nothing
        # but a stand-in executable named gh (quals never call the real
        # gh): asked for gh's saved login, it prints one in capitals; or
        # prints bytes that are no UTF-8; or exits 1, as gh does when no one
        # is signed in; or PATH holds no gh at all. Expectata: the login,
        # lowercased; each of the others a UserError saying to run "gh auth
        # login". Resultata with the seam's output not captured, or captured
        # undecoded: an AttributeError or TypeError traceback on every real
        # run, while every qual that replaces the seam stayed green.
        stand_in = Path(tempfile.mkdtemp()).resolve()
        gh = stand_in / "gh"
        asked = '#!/bin/sh\n[ "$*" = "config get user -h github.com" ] || exit 9\n'
        path = os.environ["PATH"]
        os.environ["PATH"] = str(stand_in)
        try:
            gh.write_text(asked + "printf 'Mister-Person\\n'\n", encoding="utf-8")
            gh.chmod(0o755)
            self.assertEqual(ace.github_login(), "mister-person")
            refusals = []
            for said in ("printf '\\377dreeves\\n'\n", "printf 'could not find key \"user\"\\n' >&2; exit 1\n"):
                gh.write_text(asked + said, encoding="utf-8")
                with self.assertRaises(ace.UserError, msg=said) as ctx:
                    ace.github_login()
                refusals.append(str(ctx.exception))
            gh.unlink()
            with self.assertRaises(ace.UserError, msg="no gh") as ctx:
                ace.github_login()
            refusals.append(str(ctx.exception))
        finally:
            os.environ["PATH"] = path
        for refusal in refusals:
            self.assertIn("gh auth login", refusal)

    def test_unusable_gh_identity_refused_saying_to_run_gh_auth_login(self):
        # Replicata: gh not installed or not runnable; gh run but failing,
        # as when not logged in; or gh's saved login empty or no GitHub
        # login (letters, digits, and single hyphens, 1 to 39 characters,
        # no hyphen at either end). Expectata: a UserError telling the human
        # to run "gh auth login". Resultata (v5.5.2): no identity.
        def unrunnable():
            raise PermissionError(13, "Permission denied", "gh")

        seams = [gh_missing, unrunnable, gh_says("", 1, 'could not find key "user"\n'), gh_says("dreeves\n", 1)]
        seams += [
            gh_says(said)
            for said in (
                "", "\n", "-dreeves\n", "dreeves-\n", "dree--ves\n", "x" * 40 + "\n", "dree ves\n",
                "dreeves \n", " dreeves\n", "dreeves\n\n", "dree_ves\n", "dréeves\n", "dree.ves\n", "../x\n",
            )
        ]
        for seam in seams:
            ace.gh_user = seam
            with self.subTest(seam=seam):
                with self.assertRaises(ace.UserError) as ctx:
                    ace.github_login()
                self.assertIn("gh auth login", str(ctx.exception))


class RemoteQuals(Fixture):
    def test_web_url_userinfo_is_not_part_of_the_home(self):
        # Replicata: a remote URL carrying a username, or a hosting token, in
        # front of the host, as several real configs do. Expectata: the home
        # is host plus path, exactly as the scp form already yields it, and
        # any http scheme is upgraded. Resultata before: the username, or the
        # token, was rendered into the published masthead link.
        cases = [
            ("https://dreeves@github.com/dreeves/yardsign.git", "https://github.com/dreeves/yardsign"),
            ("https://2b7aeb35-385a@api.glitch.com/iwill/git", "https://api.glitch.com/iwill/git"),
            ("http://github.com/bsoule/namer.git", "https://github.com/bsoule/namer"),
            ("https://github.com/a/b/", "https://github.com/a/b"),
            ("https://github.com/a/b@2.git", "https://github.com/a/b@2"),  # "@" in the path is not userinfo
            ("https://github.com/dreeves/crashla", "https://github.com/dreeves/crashla"),
            ("git@github.com:dreeves/crashla.git", "https://github.com/dreeves/crashla"),
            ("ssh://git@github.com/dreeves/bid.git", "https://github.com/dreeves/bid"),
            ("git://github.com/facebook/codemod.git", ""),
            ("ssh://dreeves@mpdev.mooo.com/var/dev/mp", ""),
            ("", ""),
        ]
        for url, expected in cases:
            self.assertEqual(ace.https_remote(url), expected, url)

    def test_origin_url_forms_normalized(self):
        git = self.repo / ".git"
        git.mkdir()
        cases = [
            ('[remote "origin"]\n\turl = git@github.com:dreeves/crashla.git\n', "https://github.com/dreeves/crashla"),
            ('[core]\n\tbare = false\n[remote "origin"]\n\turl = https://github.com/dreeves/road.git\n', "https://github.com/dreeves/road"),
            ('[remote "origin"]\n\turl = ssh://git@github.com/dreeves/bid.git\n', "https://github.com/dreeves/bid"),
            # Extra remotes beside origin are ordinary: origin is the home.
            (
                '[remote "origin"]\n\turl = git@github.com:dreeves/tagtime.git\n'
                '[remote "slycoder"]\n\turl = git@github.com:slycoder/tagtime.git\n',
                "https://github.com/dreeves/tagtime",
            ),
        ]
        for config, expected in cases:
            (git / "config").write_text(config, encoding="utf-8")
            self.assertEqual(ace.repo_remote(self.repo), expected)

    def test_public_home_under_another_name_fails_loudly(self):
        # Replicata: the project's only remote is named something other than
        # origin (one repo here was, until it was renamed), so its GitHub home
        # is plainly there under another name. Expectata: a loud refusal
        # naming the remote and its home. Resultata before: the masthead
        # quietly showed the local directory, as though the project had no
        # public home at all.
        git = self.repo / ".git"
        git.mkdir()
        (git / "config").write_text(
            '[remote "efme"]\n\turl = https://github.com/beeminder/efme.git\n'
            '[branch "main"]\n\tremote = efme\n',
            encoding="utf-8",
        )
        with self.assertRaises(ace.UserError) as caught:
            ace.repo_remote(self.repo)
        message = str(caught.exception)
        self.assertIn("nullum remotum nomine 'origin'".lower(), message.lower())
        self.assertIn("efme (https://github.com/beeminder/efme)", message)

    def test_refusal_never_claims_an_origin_is_absent(self):
        # Replicata: origin exists but points where no web page lives (a git
        # protocol mirror, or a push-only entry), while another remote does
        # hold the public home. Expectata: still a refusal, since the home is
        # visible and the masthead would otherwise print a local path — but a
        # message that is true, and advice that works when origin exists.
        # Resultata before: it announced that no remote was named origin, and
        # advised a rename that the version-control tool would reject.
        git = self.repo / ".git"
        git.mkdir()
        for config in (
            '[remote "origin"]\n\turl = git://github.com/facebook/codemod.git\n'
            '[remote "gh"]\n\turl = https://github.com/facebook/codemod.git\n',
            '[remote "origin"]\n\tpushurl = git@github.com:facebook/codemod.git\n'
            '[remote "gh"]\n\turl = https://github.com/facebook/codemod.git\n',
        ):
            (git / "config").write_text(config, encoding="utf-8")
            with self.assertRaises(ace.UserError) as caught:
                ace.repo_remote(self.repo)
            message = str(caught.exception)
            self.assertIn("gh (https://github.com/facebook/codemod)", message)
            for lie in ("sed nullum remotum", "muta (git"):
                self.assertNotIn(lie, message)

    def test_first_url_of_a_remote_wins(self):
        # A remote may carry several URLs; the version-control tool reports
        # the first, so the masthead must name the same one.
        git = self.repo / ".git"
        git.mkdir()
        (git / "config").write_text(
            '[remote "origin"]\n\turl = https://github.com/a/first.git\n'
            "\turl = https://github.com/a/second.git\n",
            encoding="utf-8",
        )
        self.assertEqual(ace.repo_remote(self.repo), "https://github.com/a/first")

    def test_remote_without_a_public_home_keeps_the_local_path(self):
        # Remotes whose URL names no browsable https home (a git:// mirror, an
        # ssh host that serves no web page) are not a hidden home: nothing is
        # being concealed, so the local path stands and nothing is raised.
        git = self.repo / ".git"
        git.mkdir()
        for config in (
            '[remote "origin"]\n\turl = git://github.com/facebook/codemod.git\n',
            '[remote "origin"]\n\turl = ssh://dreeves@mpdev.mooo.com/var/dev/mp\n',
            '[remote "backup"]\n\turl = ssh://kibotzer.com/home/dyang/plweb.git\n',
            "[core]\n\tbare = false\n",
        ):
            (git / "config").write_text(config, encoding="utf-8")
            self.assertEqual(ace.repo_remote(self.repo), "", config)

    def test_no_git_directory_means_no_link(self):
        self.assertEqual(ace.repo_remote(self.repo), "")

    def test_project_name_is_the_public_homes_last_segment_else_the_directorys_name(self):
        # Replicata: the project name of a checkout whose directory is named
        # repo, given the public home repo_remote found there: a repository
        # (blog, beemblog's repository), one named in capitals (TagTime,
        # tagtime's), one under nested groups, one whose name holds a dot or
        # an "@", or no public home at all (""). Expectata: the last path
        # segment of the home, exactly as written, or the directory's name
        # when there is no home; a home whose path names no repository (no
        # path at all, or one ending in "/") refused, naming the home and
        # the checkout. Resultata (v6.0.0): no project_name; a ledger named
        # its project by the directory alone. Resultata (v6.0.0 before the
        # fix): a home whose path names no repository failed an assertion.
        for remote, expected in (
            ("https://github.com/dreeves/blog", "blog"),
            ("https://github.com/dreeves/TagTime", "TagTime"),
            ("https://gitlab.com/group/subgroup/project", "project"),
            ("https://github.com/beeminder/beeminder.github.io", "beeminder.github.io"),
            ("https://github.com/a/b@2", "b@2"),
            ("", "repo"),
        ):
            with self.subTest(remote=remote):
                self.assertEqual(ace.project_name(self.repo, remote), expected)
        for remote in ("https://github.com", "https://github.com/", "https://github.com/dreeves/"):
            with self.subTest(remote=remote):
                with self.assertRaises(ace.UserError) as caught:
                    ace.project_name(self.repo, remote)
                self.assertIn(f"{remote!r} ({self.repo})", str(caught.exception))


# ---------------------------------------------------------------- Antigravity

def varint(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def pbv(number, value):
    """A protobuf varint field."""
    return varint(number << 3) + varint(value)


def pbb(number, value):
    """A protobuf length-delimited field holding bytes or UTF-8 text."""
    data = value.encode("utf-8") if isinstance(value, str) else value
    return varint(number << 3 | 2) + varint(len(data)) + data


def pbm(number, *parts):
    """A protobuf length-delimited field holding a nested message."""
    return pbb(number, b"".join(parts))


def agstamp(number, iso):
    when = utc(iso)
    return pbm(number, pbv(1, int(when.timestamp())), pbv(2, when.microsecond * 1000))


def agmeta(created, start=None, end=None, model=1008, turn="turn-1"):
    parts = [agstamp(1, created)]
    if start is not None:
        parts += [agstamp(6, start), agstamp(7, end)]
    if model is not None:
        parts.append(pbv(11, model))
    parts.append(pbb(12, turn))
    return pbm(5, *parts)


AG_ARTIFACT = "file:///Users/someone/.gemini/antigravity/brain/conv-1/implementation_plan.md"


def aguser(text, created, workspace=None, status=3, click=None, extra=b""):
    """A type-14 step as the store writes it: text is the typed prompt (None
    for none), workspace the context URI, click an artifact action carrying
    a comment (an empty comment is a bare click), extra raw payload bytes."""
    payload = []
    if text is not None:
        payload += [pbb(2, text), pbm(3, pbb(1, text))]
    if workspace is not None:
        payload.append(pbm(4, pbm(2, pbb(4, "javascript"), pbv(5, 17), pbb(13, workspace))))
    if click is not None:
        payload.append(pbm(7, pbb(1, AG_ARTIFACT), pbb(5, click), pbv(7, 1)))
    payload.append(pbv(8, 1))
    return pbm(2, pbv(1, 14), pbv(4, status), agmeta(created, model=None), pbm(19, *payload, extra))


def agplanner(created, start, end, text=None, thinking=None, tool=False, model=1008, turn="turn-1"):
    payload = []
    if text is not None:
        payload += [pbb(1, text), pbb(8, text)]
    if thinking is not None:
        payload.append(pbb(3, thinking))
    if tool:
        payload.append(pbm(7, pbb(1, "call-1"), pbb(2, "list_dir"), pbb(3, "{}")))
    return pbm(2, pbv(1, 15), pbv(4, 3), agmeta(created, start, end, model, turn), pbm(20, *payload))


def agnotify(created, start, end, text, model=1008, turn="turn-1"):
    payload = [pbb(1, AG_ARTIFACT), pbb(2, text), pbv(3, 1), pbb(5, "confidence justification")]
    return pbm(2, pbv(1, 82), pbv(4, 3), agmeta(created, start, end, model, turn), pbm(94, *payload))


def agstep(kind, created, start, end, model=1008, payload=b"", status=3, turn="turn-1"):
    """A machine step of the given type with an opaque payload."""
    return pbm(2, pbv(1, kind), pbv(4, status), agmeta(created, start, end, model, turn), payload)


def agconv(steps, cid="conv-1", created=T0):
    return b"".join([pbb(1, "owner-1"), *steps, pbb(6, cid), pbm(7, agstamp(2, created), pbb(3, "x"))])


# The quals never touch the application's real key: fixtures are sealed with
# this one, and the CLI quals hand it to sourcery in place of the finder.
AG_TEST_KEY = b"q" * 32


def agseal(plain, key=AG_TEST_KEY):
    """Seal bytes the way the store does — nonce, ciphertext, tag — through
    the same system cipher sourcery opens them with."""
    import ctypes
    import ctypes.util

    seal = ctypes.CDLL(ctypes.util.find_library("System")).CCCryptorGCMOneshotEncrypt
    seal.restype = ctypes.c_int32
    seal.argtypes = [
        ctypes.c_uint32, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t,
        ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p,
        ctypes.c_char_p, ctypes.c_size_t,
    ]
    nonce = os.urandom(12)
    out = ctypes.create_string_buffer(len(plain))
    tag = ctypes.create_string_buffer(16)
    assert seal(0, key, 32, nonce, 12, None, 0, plain, len(plain), out, tag, 16) == 0
    return nonce + out.raw + tag.raw


def write_ag(root, cid, plain):
    path = root / "conversations" / f"{cid}.pb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(agseal(plain))
    return path


class AntigravityQuals(Fixture):
    def parse(self, plain, repo=None):
        path = write_ag(self.tmp / "ag", "conv-1", plain)
        return ace.antigravity_exchanges(path, repo or self.repo, AG_TEST_KEY)

    def test_key_is_read_from_the_installed_binary_by_digest(self):
        # Replicata: a binary holding many letter runs, one of them the key.
        # Expectata: the finder returns exactly that window; with a digest
        # matching nothing, or no binary at all, it refuses loudly.
        key = b"ThirtyTwoLettersOfLocalStoreKeyx"
        assert len(key) == 32
        binary = self.tmp / "bin" / "language_server_fake"
        binary.parent.mkdir()
        binary.write_bytes(b"\x00\x01" * 500 + b"prefixLetters" + key + b"MoreLettersAfterwards\x00" + b"z" * 40 + b"\x7f")
        finder = ace.antigravity_key.__wrapped__
        original = ace.ANTIGRAVITY_BIN, ace.ANTIGRAVITY_KEY_DIGEST
        try:
            ace.ANTIGRAVITY_BIN, ace.ANTIGRAVITY_KEY_DIGEST = binary.parent, hashlib.sha256(key).hexdigest()
            self.assertEqual(finder(), key)
            ace.ANTIGRAVITY_KEY_DIGEST = hashlib.sha256(b"another").hexdigest()
            with self.assertRaises(ace.UserError):
                finder()
            ace.ANTIGRAVITY_BIN = self.tmp / "absent"
            with self.assertRaises(ace.UserError):
                finder()
        finally:
            ace.ANTIGRAVITY_BIN, ace.ANTIGRAVITY_KEY_DIGEST = original

    def test_key_literal_appears_nowhere_in_the_sources(self):
        # The key belongs to the application; only its digest is recorded.
        for name in ("sourcery.py", "adopt.py", "quals.py", "README.md"):
            source = (Path(ace.__file__).parent / name).read_bytes()
            for run in re.finditer(rb"[A-Za-z]{32,}", source):
                letters = run.group()
                for at in range(len(letters) - 31):
                    self.assertNotEqual(hashlib.sha256(letters[at:at + 32]).hexdigest(), ace.ANTIGRAVITY_KEY_DIGEST, name)

    def test_unknown_model_enum_is_shown_not_hidden(self):
        # Replicata: a reply stamped with a model enum the table does not
        # name. Expectata: the exchange still renders, its model reading as
        # an unknown model with the number — the gap is visible, the page
        # is not refused.
        steps = [aguser("p", T0, workspace=self.repo.as_uri()), agnotify(T0, T1, T2, "r", model=1012)]
        exchanges, _ = self.parse(agconv(steps))
        self.assertEqual(exchanges[0].model, "unknown model 1012")

    def test_typed_prompts_and_visible_replies_extracted(self):
        # Replicata: a typed prompt; a planner step with thinking and a tool
        # call; a planner step with visible text; a notify-user message; a
        # second prompt. Expectata: two exchanges; the first's reply is the
        # visible planner text and the notify message joined, its model named
        # from the enum, elapsed the sum of the agent steps' spans, wall from
        # the prompt to the last reply; the thinking and tool call absent.
        ws = self.repo.as_uri()
        steps = [
            aguser("first prompt", T0, workspace=ws),
            agplanner(T0, "2026-03-01T10:00:01.000Z", "2026-03-01T10:00:04.000Z", thinking="**Thinking**\nhidden", tool=True),
            agplanner(T0, "2026-03-01T10:00:05.000Z", "2026-03-01T10:00:07.000Z", text="Visible **answer**."),
            agnotify(T0, "2026-03-01T10:00:08.000Z", "2026-03-01T10:00:09.000Z", "Done; please review."),
            aguser("second prompt", T1, workspace=ws),
        ]
        exchanges, holdings = self.parse(agconv(steps))
        self.assertEqual([e.prompt for e in exchanges], ["first prompt", "second prompt"])
        first = exchanges[0]
        self.assertEqual(first.reply, "Visible **answer**.\n\nDone; please review.")
        self.assertNotIn("hidden", first.reply)
        self.assertEqual((first.provider, first.model, first.session), ("Antigravity", "gemini-3-pro-high", "conv-1"))
        self.assertEqual(first.timestamp, utc(T0))
        self.assertEqual((first.elapsed, first.wall), (6.0, 9.0))
        self.assertEqual(exchanges[1].reply, "")
        self.assertEqual(holdings, {("Antigravity", utc(T0)), ("Antigravity", utc(T1))})

    def test_machine_steps_carry_time_but_never_words(self):
        # Replicata: between a prompt and its reply, an ephemeral injection, a
        # history injection, a code action, a file view, a terminal run, a
        # progress summary and a browser subtask, each spanning two seconds.
        # Expectata: none contributes reply text; every span counts as work.
        ws = self.repo.as_uri()
        steps = [aguser("p", T0, workspace=ws)]
        second = 1
        for kind in (90, 98, 5, 8, 21, 81, 85):
            start, end = f"2026-03-01T10:00:{second:02d}.000Z", f"2026-03-01T10:00:{second + 2:02d}.000Z"
            steps.append(agstep(kind, T0, start, end, payload=pbm(103, pbb(1, "The following is an <EPHEMERAL_MESSAGE>"))))
            second += 3
        steps.append(agnotify(T0, "2026-03-01T10:00:30.000Z", "2026-03-01T10:00:31.000Z", "reply"))
        exchanges, _ = self.parse(agconv(steps))
        self.assertEqual(exchanges[0].reply, "reply")
        self.assertEqual(exchanges[0].elapsed, 15.0)

    def test_cancelled_and_failed_agent_steps_still_speak_and_take_time(self):
        # The real store marks agent steps 6 (cancelled) and 7 (failed) with
        # their payloads intact; their words reached the human and their
        # spans were spent.
        ws = self.repo.as_uri()
        steps = [
            aguser("p", T0, workspace=ws),
            agstep(5, T0, "2026-03-01T10:00:01.000Z", "2026-03-01T10:00:03.000Z", status=7),
            agnotify(T0, "2026-03-01T10:00:04.000Z", "2026-03-01T10:00:05.000Z", "stopped midway"),
        ]
        steps[2] = steps[2].replace(pbv(4, 3), pbv(4, 6), 1)
        exchanges, _ = self.parse(agconv(steps))
        self.assertEqual((exchanges[0].reply, exchanges[0].elapsed), ("stopped midway", 3.0))

    def test_conversation_attributed_by_the_one_workspace_its_prompts_name(self):
        ws = self.repo.as_uri()
        exchanges, holdings = self.parse(agconv([aguser("a", T0, workspace=ws), aguser("b", T1), aguser("c", T2, workspace=ws)]))
        self.assertEqual([e.prompt for e in exchanges], ["a", "b", "c"])
        self.assertEqual(len(holdings), 3)
        elsewhere = (self.tmp / "elsewhere").as_uri()
        self.assertEqual(self.parse(agconv([aguser("x", T0, workspace=elsewhere), aguser("y", T1)])), ([], set()))
        self.assertEqual(self.parse(agconv([aguser("x", T0)])), ([], set()))

    def test_two_workspaces_in_one_conversation_fail_loudly(self):
        steps = [aguser("a", T0, workspace=self.repo.as_uri()), aguser("b", T1, workspace=(self.tmp / "other").as_uri())]
        with self.assertRaises(ace.UserError):
            self.parse(agconv(steps))

    def test_trimmed_steps_are_neither_prompts_nor_holdings(self):
        # Replicata: the application compacted the conversation, leaving its
        # early steps with status 5 and no content, then a live prompt.
        # Expectata: the live prompt alone, and a holding for it alone — a
        # ledger copy of the emptied prompt must survive the merge, since the
        # store no longer holds its words.
        ws = self.repo.as_uri()
        steps = [
            aguser(None, T0, status=5),
            agstep(15, T0, None, None, model=None, status=5),
            aguser("live", T1, workspace=ws),
        ]
        exchanges, holdings = self.parse(agconv(steps))
        self.assertEqual([e.prompt for e in exchanges], ["live"])
        self.assertEqual(holdings, {("Antigravity", utc(T1))})

    def test_artifact_click_is_a_holding_but_a_comment_is_a_prompt(self):
        ws = self.repo.as_uri()
        steps = [
            aguser("typed", T0, workspace=ws),
            aguser(None, T1, workspace=ws, click=""),
            aguser(None, T2, workspace=ws, click="please also handle mobile"),
        ]
        exchanges, holdings = self.parse(agconv(steps))
        self.assertEqual([e.prompt for e in exchanges], ["typed", "please also handle mobile"])
        self.assertEqual(holdings, {("Antigravity", utc(T0)), ("Antigravity", utc(T1)), ("Antigravity", utc(T2))})

    def test_unknown_shapes_fail_loudly(self):
        ws = self.repo.as_uri()
        cases = {
            "step type": [aguser("p", T0, workspace=ws), agstep(999, T0, T1, T2)],
            "prompt field": [aguser("p", T0, workspace=ws, extra=pbb(99, "an attachment?"))],
            "empty live prompt": [aguser(None, T0, workspace=ws)],
            "step status": [aguser("p", T0, workspace=ws, status=9)],
        }
        for label, steps in cases.items():
            with self.assertRaises(ace.UserError, msg=label) as caught:
                self.parse(agconv(steps))
            self.assertIn("conv-1", str(caught.exception), label)
            self.assertIn(label.split()[-1] if label == "step type" else "", str(caught.exception))

    def test_unopenable_transcript_fails_loudly(self):
        path = write_ag(self.tmp / "ag", "conv-1", agconv([aguser("p", T0, workspace=self.repo.as_uri())]))
        blob = bytearray(path.read_bytes())
        blob[-1] ^= 1
        path.write_bytes(bytes(blob))
        with self.assertRaises(ace.UserError) as caught:
            ace.antigravity_exchanges(path, self.repo, AG_TEST_KEY)
        self.assertIn("conv-1.pb", str(caught.exception))
        path.write_bytes(agseal(b"\xff\xff\xff"))
        with self.assertRaises(ace.UserError):
            ace.antigravity_exchanges(path, self.repo, AG_TEST_KEY)
        # A different key than the one the file was sealed with is the same
        # refusal: the file is unreadable as far as this tool can tell.
        path.write_bytes(agseal(agconv([aguser("p", T0, workspace=self.repo.as_uri())]), key=b"k" * 32))
        with self.assertRaises(ace.UserError):
            ace.antigravity_exchanges(path, self.repo, AG_TEST_KEY)

    def test_provider_color_defined_in_both_modes(self):
        self.assertEqual(ace.CSS.count("--antigravity:"), 2)
        self.assertIn(".exchange.antigravity { --provider: var(--antigravity); }", ace.CSS)
        self.assertEqual(ace.PROVIDER_SLUGS["Antigravity"], "antigravity")


# ------------------------------------------------------------------- ledgers

MAXIMAL = dict(
    timestamp=utc("2026-03-01T10:00:00.123456Z"),
    provider="Codex",
    model="gpt-5.4",
    session="cx1",
    prompt='<p>&amp; "quoted"\n\ttab and trailing blank\n\n',
    reply="**bold** `code` </script> <!--<script>alert(1)</script>-->",
    source=Path("/store/rollout.jsonl"),
    effort="xhigh",
    images=("data:image/png;base64,AA==",),
    elapsed=0.1 + 0.2,
    wall=1234.5678,
    ballots=(ace.Ballot("Which?", ("a", "b"), ("b",)), ace.Ballot("Typed?", ("x",), ())),
    added=1234,
    deleted=5,
)


def maximal_exchange(**overrides):
    return exchange(**{**MAXIMAL, **overrides})


def count_scripts(page):
    class Tally(HTMLParser):
        scripts = 0

        def handle_starttag(self, tag, attrs):
            if tag == "script":
                self.scripts += 1

    tally = Tally()
    tally.feed(page)
    return tally.scripts


class LedgerQuals(Fixture):
    def test_freeze_thaw_round_trips_every_field(self):
        given = maximal_exchange()
        thawed = ace.thaw(json.loads(json.dumps(ace.freeze(given))), self.tmp / "page.html")
        self.assertEqual(thawed.source, self.tmp / "page.html")
        restored = dataclasses.replace(thawed, source=given.source)
        self.assertEqual(restored, given)
        self.assertEqual(hash(restored), hash(given))
        self.assertEqual(ace.weave([given, thawed]), [given])

    def test_freeze_omits_the_store_path(self):
        self.assertNotIn("source", ace.freeze(maximal_exchange()))

    def test_claude_ai_exchange_thaws(self):
        # Replicata: a ledger seeded with a claude.ai chat exchange, the
        # only way such an exchange enters a page. Expectata: it thaws back
        # exactly. Resultata (v5.5.0): ValueError, claude.ai being no known
        # provider.
        given = maximal_exchange(provider="claude.ai", model="claude-opus-5-5", effort="max")
        thawed = ace.thaw(json.loads(json.dumps(ace.freeze(given))), given.source)
        self.assertEqual(thawed, given)

    def test_thaw_rejects_unknown_and_missing_fields(self):
        frozen = ace.freeze(maximal_exchange())
        page = self.tmp / "page.html"
        with self.assertRaises(ValueError):
            ace.thaw({**frozen, "author": "someone"}, page)
        for missing in ("prompt", "effort", "wall", "images", "ballots"):
            with self.assertRaises(ValueError, msg=missing):
                ace.thaw({k: v for k, v in frozen.items() if k != missing}, page)

    def test_thaw_rejects_wrong_types_and_ranges(self):
        # A hand-edited ledger row must not thaw into a guess: every field's
        # JSON type and range is checked before an Exchange is built.
        frozen = ace.freeze(maximal_exchange())
        page = self.tmp / "page.html"
        for field, value in (
            ("images", "abc"),
            ("images", [1]),
            ("elapsed", "5"),
            ("elapsed", True),
            ("elapsed", float("nan")),
            ("elapsed", -1),
            ("wall", float("inf")),
            ("added", 1.5),
            ("added", -1),
            ("provider", "Foo"),
            ("prompt", None),
            ("session", 123),
            ("timestamp", 1772576233307),
            ("ballots", "ab"),
            ("ballots", [{"question": "q", "options": ["a"], "picked": [], "extra": 1}]),
            ("ballots", [{"question": 5, "options": [], "picked": []}]),
            ("ballots", [{"question": "q", "options": "ab", "picked": []}]),
        ):
            with self.assertRaises(ValueError, msg=(field, value)):
                ace.thaw({**frozen, field: value}, page)

    def test_bare_tool_result_is_no_holding(self):
        # Tool plumbing that carries no typing was never rendered, so it must
        # never purge a ledger copy that merely shares its millisecond.
        claude = write_jsonl(
            self.tmp / "claude" / "p" / "s.jsonl",
            [
                cu("typed", ts=T0, cwd=str(self.repo)),
                ca([{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}], ts=T1, cwd=str(self.repo)),
                cu(
                    [{"type": "tool_result", "tool_use_id": "t1", "content": "file contents"}],
                    ts=T2, cwd=str(self.repo), toolUseResult={"type": "text", "file": {}},
                ),
                ca([{"type": "text", "text": "reply"}], ts=T3, cwd=str(self.repo)),
            ],
        )
        exchanges, holdings = ace.claude_exchanges(claude, self.repo)
        self.assertEqual([e.prompt for e in exchanges], ["typed"])
        self.assertEqual(holdings, {("Claude Code", utc(T0))})

    def test_page_holds_no_data_block_and_ledger_text_round_trips(self):
        # Replicata: the maximal exchange, its reply holding a closing
        # script tag and a comment around a script. Expectata: its page
        # holds one script element, the page's own, with the reply's markup
        # inert; its ledger text gives the exchange back. Resultata
        # (v5.5.2): a second script, the JSON snapshot.
        given = maximal_exchange()
        page = ace.render(self.repo, {LOGIN: [given]}, "")
        self.assertNotIn("<script>alert(1)", page)
        self.assertEqual(count_scripts(page), 1)
        header, row = ace.ledger_text(self.repo.name, [given]).removesuffix("\n").split("\n")
        self.assertEqual(json.loads(header), ledger_header(self.repo.name))
        self.assertEqual(ace.thaw(json.loads(row), given.source), given)

    def test_ledger_text_is_a_header_then_one_exchange_per_line(self):
        # Replicata: a ledger's text for two exchanges whose prompts hold
        # line breaks. Expectata: the header line, then one line per
        # exchange, each its freeze() form as json.dumps writes it with
        # non-ASCII characters kept, every line ending in a newline.
        # Resultata (v5.5.2): no ledgers; the snapshot was one JSON value.
        first = maximal_exchange()
        second = maximal_exchange(timestamp=utc(T1), prompt="second\nline   é")
        text = ace.ledger_text(self.repo.name, [first, second])
        self.assertTrue(text.endswith("\n"))
        lines = text.removesuffix("\n").split("\n")
        self.assertEqual(
            lines,
            [
                json.dumps(ledger_header(self.repo.name)),
                json.dumps(ace.freeze(first), ensure_ascii=False),
                json.dumps(ace.freeze(second), ensure_ascii=False),
            ],
        )
        self.assertEqual(json.loads(lines[2])["prompt"], second.prompt)

    def test_parsers_report_holdings_for_records_they_drop(self):
        # A record the store still holds is reported even when the parser
        # drops it as machine text, so a ledger's stale copy of it gets purged.
        claude = write_jsonl(
            self.tmp / "claude" / "p" / "s.jsonl",
            [
                cu(
                    "<task-notification>done</task-notification>",
                    ts=T0,
                    cwd=str(self.repo),
                    origin={"kind": "task-notification"},
                )
            ],
        )
        self.assertEqual(ace.claude_exchanges(claude, self.repo), ([], {("Claude Code", utc(T0))}))
        codex = write_jsonl(
            self.tmp / "codex" / "r.jsonl",
            [
                cxmeta(str(self.repo)),
                cxuser("The following is the Codex agent history added since your last message.", ts=T1),
            ],
        )
        self.assertEqual(ace.codex_exchanges(codex, self.repo), ([], {("Codex", utc(T1))}))
        vscode = self.tmp / "chatSessions" / "s.json"
        vscode.parent.mkdir(parents=True)
        click = vsreq("@agent Try Again", [], confirmation="Try Again", ts=int(utc(T2).timestamp() * 1000))
        vscode.write_text(json.dumps(vssession([click])), encoding="utf-8")
        self.assertEqual(
            ace.vscode_exchanges(vscode, (self.repo,), self.repo), ([], {("Copilot Chat", utc(T2))})
        )

    def test_parsers_hold_only_prompt_records_inside_the_repo(self):
        elsewhere = str(self.tmp / "elsewhere")
        claude = write_jsonl(
            self.tmp / "claude" / "p" / "s.jsonl",
            [
                cu("outside", ts=T0, cwd=elsewhere),
                cu("inside", ts=T1, cwd=str(self.repo)),
                ca([{"type": "text", "text": "reply"}], ts=T2, cwd=str(self.repo)),
            ],
        )
        exchanges, holdings = ace.claude_exchanges(claude, self.repo)
        self.assertEqual([e.prompt for e in exchanges], ["inside"])
        self.assertEqual(holdings, {("Claude Code", utc(T1))})
        codex = write_jsonl(
            self.tmp / "codex" / "r.jsonl", [cxmeta(elsewhere), cxuser("outside", ts=T0)]
        )
        self.assertEqual(ace.codex_exchanges(codex, self.repo), ([], set()))
        vscode = self.tmp / "chatSessions" / "s.json"
        vscode.parent.mkdir(parents=True)
        vscode.write_text(json.dumps(vssession([vsreq("outside", [])])), encoding="utf-8")
        self.assertEqual(ace.vscode_exchanges(vscode, (Path(elsewhere),), self.repo), ([], set()))


class ParseTimeQuals(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(ace.parse_time(T0), utc(T0))
        self.assertEqual(ace.parse_time("2026-03-01T10:00:00+02:00").utcoffset(), dt.timedelta(0))
        self.assertEqual(
            ace.parse_time(1772576233307),
            dt.datetime.fromtimestamp(1772576233.307, tz=dt.timezone.utc),
        )
        self.assertEqual(
            ace.parse_time(1772576233),
            dt.datetime.fromtimestamp(1772576233, tz=dt.timezone.utc),
        )
        naive = ace.parse_time("2026-03-01T10:00:00")
        self.assertEqual(naive, utc(T0))
        for bad in (None, "yesterday", [], {}):
            with self.assertRaises(ace.UserError):
                ace.parse_time(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
