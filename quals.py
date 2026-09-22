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
import json
import os
import re
import shutil
import stat
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

import sourcery as ace
import unrender

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
            [exchange(prompt="a<b>&c\n  indented", reply="<script>alert(1)</script>")],
        )
        self.assertIn("a&lt;b&gt;&amp;c\n  indented", page)
        self.assertNotIn("<script>alert(1)", page)
        self.assertIn("<details>", page)
        self.assertNotIn("<details open", page)
        prompt_at = page.index("a&lt;b&gt;")
        details_at = page.index("<details>")
        self.assertLess(prompt_at, details_at)

    def test_meta_line_carries_time_provider_model(self):
        page = ace.render(self.repo, [exchange()])
        local = utc(T0).astimezone().strftime("%H:%M")
        summary = page[page.index("<summary") : page.index("</summary>")]
        self.assertIn(local, summary)
        self.assertIn("Claude Code", summary)
        self.assertIn("claude-opus-4-8", summary)
        self.assertNotIn("(", summary)

    def test_effort_shown_in_parens_after_model(self):
        page = ace.render(self.repo, [exchange(effort="xhigh")])
        summary = page[page.index("<summary") : page.index("</summary>")]
        self.assertIn("claude-opus-4-8", summary)
        self.assertIn("(xhigh)", summary)
        self.assertLess(summary.index("claude-opus-4-8"), summary.index("(xhigh)"))

    def test_one_day_header_per_local_day_with_weekday(self):
        page = ace.render(
            self.repo,
            [exchange(prompt="x"), exchange(timestamp=utc(T1), prompt="y")],
        )
        self.assertEqual(page.count('class="day"'), 1)
        local = utc(T0).astimezone().date()
        self.assertIn(f"{local.isoformat()} {ace.WEEKDAYS[local.weekday()]}<", page)

    def test_thought_duration_shown_when_known(self):
        page = ace.render(self.repo, [exchange(elapsed=327.0)])
        summary = page[page.index("<summary") : page.index("</summary>")]
        self.assertIn("thought for 5m27s", summary)
        page = ace.render(self.repo, [exchange()])
        self.assertNotIn("thought for", page)

    def test_provider_identity_stripe_class_and_chip(self):
        page = ace.render(
            self.repo,
            [
                exchange(),
                exchange(timestamp=utc(T1), provider="Codex", prompt="b"),
                exchange(timestamp=utc(T2), provider="Copilot Chat", prompt="c"),
            ],
        )
        # Each exchange wears its provider's class so the CSS edge stripe and
        # meta-line chip can color by agent.
        for slug in ("claude", "codex", "copilot"):
            self.assertIn(f'<article class="exchange {slug}"', page)
        # The chip is a mark beside the agent name — the name itself stays in
        # ink (text never wears mark colors), one chip per exchange.
        self.assertEqual(page.count('<span class="chip"></span>'), 3)
        self.assertLess(page.index('class="chip"'), page.index('class="agent"'))

    def test_unknown_provider_fails_loudly(self):
        with self.assertRaises(KeyError):
            ace.render(self.repo, [exchange(provider="Quantum")])

    def test_autogenerated_warning_comment_after_doctype(self):
        # The warning must follow the doctype: a comment before it would
        # throw browsers into quirks mode.
        page = ace.render(self.repo, [exchange()])
        self.assertTrue(page.startswith("<!doctype html>\n<!-- "), page[:60])
        self.assertLess(page.index("-->"), page.index("<html"))

    def test_wall_clock_shown_only_when_beyond_thought(self):
        page = ace.render(self.repo, [exchange(elapsed=327.0, wall=540.0)])
        summary = page[page.index("<summary") : page.index("</summary>")]
        self.assertIn("thought for 5m27s · 9m wall-clock time", summary)
        # A wall span matching the working time at display precision adds
        # nothing and stays off,
        page = ace.render(self.repo, [exchange(elapsed=327.0, wall=327.4)])
        self.assertNotIn("wall-clock time", page)
        # as does one the working time exceeds (activity credited off records
        # later than the last reply),
        page = ace.render(self.repo, [exchange(elapsed=327.0, wall=300.0)])
        self.assertNotIn("wall-clock time", page)
        # and one on an exchange with no working time shown at all.
        page = ace.render(self.repo, [exchange(wall=540.0)])
        self.assertNotIn("thought for", page)
        self.assertNotIn("wall-clock time", page)

    def test_diffstat_shown_with_ratio_blocks(self):
        page = ace.render(self.repo, [exchange(added=1234, deleted=45, prompt="hi")])
        self.assertIn("+1,234 −45", page)
        self.assertEqual(page.count('<span class="add"></span>'), 4)  # round(5·1234/1279)
        self.assertEqual(page.count('<span class="del"></span>'), 1)
        self.assertEqual(page.count('<span class="nil"></span>'), 0)
        # The stat precedes the prompt so it floats beside the prompt's top.
        self.assertLess(page.index('class="diffstat"'), page.index('class="prompt"'))
        # A prompt that touched no code gets no diffstat at all.
        page = ace.render(self.repo, [exchange()])
        self.assertNotIn('<div class="diffstat">', page)

    def test_diffstat_tiny_and_onesided_ratios(self):
        page = ace.render(self.repo, [exchange(added=1, deleted=1)])
        self.assertEqual(page.count('<span class="add"></span>'), 1)
        self.assertEqual(page.count('<span class="del"></span>'), 1)
        self.assertEqual(page.count('<span class="nil"></span>'), 3)
        page = ace.render(self.repo, [exchange(added=0, deleted=7)])
        self.assertIn("+0 −7", page)
        self.assertEqual(page.count('<span class="add"></span>'), 0)
        self.assertEqual(page.count('<span class="del"></span>'), 5)
        # A nonzero side always gets at least one block.
        page = ace.render(self.repo, [exchange(added=1, deleted=999)])
        self.assertEqual(page.count('<span class="add"></span>'), 1)
        self.assertEqual(page.count('<span class="del"></span>'), 4)
        # An even split renders symmetrically, remainder unfilled.
        page = ace.render(self.repo, [exchange(added=10, deleted=10)])
        self.assertEqual(page.count('<span class="add"></span>'), 2)
        self.assertEqual(page.count('<span class="del"></span>'), 2)
        self.assertEqual(page.count('<span class="nil"></span>'), 1)

    def test_deck_totals_shown_when_any_lines_counted(self):
        page = ace.render(
            self.repo,
            [exchange(added=2, deleted=1), exchange(timestamp=utc(T1), prompt="b", added=3)],
        )
        local = utc(T0).astimezone().date().isoformat()
        deck = f"2 prompts · {local} · +5 −1"
        self.assertIn(f'<p class="deck">{deck}</p>', page)
        self.assertIn(f'<meta name="description" content="{deck}">', page)
        page = ace.render(self.repo, [exchange()])
        self.assertNotIn("+0 −0", page)

    def test_minimap_sliver_per_prompt_linked_and_scaled(self):
        page = ace.render(
            self.repo,
            [
                exchange(added=100, deleted=0),
                exchange(timestamp=utc(T1), prompt="b", added=25, deleted=4),
                exchange(timestamp=utc(T2), prompt="c"),
            ],
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
        page = ace.render(self.repo, [exchange(images=("data:image/png;base64,QUJD",))])
        img_at = page.index('src="data:image/png;base64,QUJD"')
        self.assertLess(page.index('class="prompt"'), img_at)
        self.assertLess(img_at, page.index("<details>"))

    def test_generator_attribution_links_home(self):
        page = ace.render(self.repo, [exchange()])
        self.assertIn(
            '<a href="https://github.com/beeminder/sourcery">generated by sourcery</a>',
            page,
        )

    def test_remote_link_replaces_directory_when_known(self):
        page = ace.render(self.repo, [exchange()], remote="https://github.com/dreeves/crashla")
        self.assertIn('<a href="https://github.com/dreeves/crashla">github.com/dreeves/crashla</a>', page)
        self.assertNotIn(str(self.repo), page)
        page = ace.render(self.repo, [exchange()])
        self.assertIn(str(self.repo), page)
        self.assertNotIn("github.com/dreeves/crashla", page)

    def test_expand_controls_and_permalink_anchors(self):
        page = ace.render(
            self.repo,
            [exchange(prompt="a"), exchange(timestamp=utc(T1), prompt="b")],
        )
        self.assertIn('data-omnia="open"', page)
        self.assertIn('data-omnia="close"', page)
        self.assertIn("<script>", page)
        self.assertIn('id="p1"', page)
        self.assertIn('href="#p2"', page)

    def test_empty_reply_marked(self):
        page = ace.render(self.repo, [exchange(reply="")])
        self.assertIn('class="reply machine empty"', page)

    def test_only_final_empty_reply_marked_still_generating(self):
        page = ace.render(
            self.repo,
            [exchange(prompt="a", reply=""), exchange(timestamp=utc(T1), prompt="b", reply="")],
        )
        self.assertEqual(page.count("Response still generating when this transcript was captured"), 1)
        self.assertIn("No response.", page)
        self.assertLess(page.index("No response."), page.index("Response still generating"))
        page = ace.render(self.repo, [exchange(prompt="a", reply=""), exchange(timestamp=utc(T1), prompt="b", reply="done")])
        self.assertNotIn("Response still generating", page)

    def test_machine_prose_only_inside_machine_containers(self):
        ballot = ace.Ballot("Q?", ("A label", "B label"), ("A label",))
        page = ace.render(
            self.repo,
            [exchange(prompt="typed words", reply="agent words", ballots=(ballot,))],
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
        page = ace.render(self.repo, [exchange(prompt="typed words", ballots=(ballot,))])
        article = page[page.index("<article") : page.index("</article>")]
        self.assertLess(article.index('<div class="ballot machine">'), article.index("<details>"))

    def test_click_only_answer_renders_without_prompt_block(self):
        ballot = ace.Ballot("Q?", ("Delete it", "Keep it"), ("Delete it",))
        page = ace.render(self.repo, [exchange(prompt="", ballots=(ballot,))])
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
            [exchange(prompt="a"), exchange(timestamp=utc("2026-03-05T10:00:00.000Z"), prompt="b")],
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
            self.repo, [exchange(prompt="a"), exchange(timestamp=utc(T1), prompt="b")]
        )
        self.assertEqual(page.count('class="daymark"'), 1)

    def test_progress_rail_revealed_and_driven_by_script(self):
        # The rail appears only once the reader has actually set off: the
        # stylesheet ships the bar transparent and the marks offscreen, and
        # the script shows them past 300px of scroll, resizing the bar on
        # scroll and resize and after any disclosure opens or closes (which
        # reflows the whole page under the rail).
        page = ace.render(self.repo, [exchange()])
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

    def tearDown(self):
        ace.antigravity_key = self.finder

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

    def test_existing_output_without_snapshot_rejected(self):
        # Replicata: the output path already holds a file carrying no
        # snapshot — a page from before snapshots, or not a sourcery page at
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

    def test_pruned_session_kept_from_page(self):
        # Replicata: run, then the Claude session file vanishes from the store
        # (pruned, or this is another machine). Expectata: the next run keeps
        # that exchange from the page. Resultata before snapshots: gone.
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

    def test_empty_stores_regenerate_from_page_alone(self):
        # A new machine: no transcript roots exist at all, only the page.
        self.populate()
        out_path = self.tmp / "out.html"
        _, first = self.generate(out_path)
        for root in (self.claude_root, self.codex_root, self.vscode_root):
            shutil.rmtree(root)
        _, again = self.generate(out_path)
        self.assertEqual(again, first)

    def test_store_record_supersedes_page_copy(self):
        # The store is the truth for a record it still holds: the record's
        # text changes under the same timestamp, so the page copy is replaced.
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
        # store still holds the record, so the page's copy is purged even
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
        # file; the page holds one copy; then the original file is pruned.
        # Expectata: the fork's live record supersedes the page copy and the
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

    def test_snapshot_omits_store_paths(self):
        self.populate()
        out_path = self.tmp / "out.html"
        _, page = self.generate(out_path)
        block = ace.SNAPSHOT.search(page).group(1)
        for root in (self.claude_root, self.codex_root, self.vscode_root):
            self.assertNotIn(str(root), block)
        self.assertNotIn('"source"', block)

    def test_malformed_snapshot_rejected_and_page_untouched(self):
        self.populate()
        out_path = self.tmp / "out.html"
        _, page = self.generate(out_path)
        block = ace.SNAPSHOT.search(page).group(1)
        data = json.loads(block)
        data["exchanges"][0]["author"] = "someone"
        foreign = json.dumps(data).replace("<", "\\u003c")
        for bad in ("{not json", foreign):
            broken = page.replace(block, bad)
            self.assertNotEqual(broken, page)
            out_path.write_text(broken, encoding="utf-8")
            code, _, err = self.run_cli([str(self.repo), str(out_path)])
            self.assertEqual(code, 2)
            self.assertIn(str(out_path), err)
            self.assertEqual(out_path.read_text(encoding="utf-8"), broken)

    def test_prompt_quoting_the_snapshot_opener_does_not_confuse_the_reader(self):
        # Replicata: a typed prompt is the snapshot opener line itself plus a
        # closing tag. Expectata: the page still holds exactly one block and
        # gives the prompt back byte-exact. Resultata with an unescaped block:
        # two openers, and the reader refuses the page.
        text = ace.SNAPSHOT_OPEN + "\n{}\n</script>"
        write_jsonl(self.claude_root / "p" / "s.jsonl", [cu(text, ts=T0, cwd=str(self.repo))])
        out_path = self.tmp / "out.html"
        self.generate(out_path)
        _, page = self.generate(out_path)
        self.assertEqual(len(ace.SNAPSHOT.findall(page)), 1)
        self.assertEqual([e.prompt for e in ace.inherit(out_path, self.repo)], [text])

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

    def test_flagship_legacy_page_imported_then_merged_with_partial_store(self):
        # Replicata: a page from before snapshots, rendered from all three
        # providers; then the Claude session is pruned, the Copilot request's
        # text is revised in place under the same timestamp, and Codex is
        # intact plus one new prompt. unrender.py imports the page, then
        # sourcery.py merges. Expectata: the pruned Claude prompt is kept from
        # the page, the Copilot page copy is superseded by the store's reading,
        # the Codex prompt renders once under its store session, the new Codex
        # prompt is added, all in order; a rerun is byte-identical.
        self.populate()
        out_path = self.tmp / "out.html"
        _, page = self.generate(out_path)
        out_path.write_text(legacy(page), encoding="utf-8")
        (self.claude_root / "p" / "s.jsonl").unlink()
        revised = vsreq("revised copilot prompt", [md("revised copilot reply")], ts=int(utc(T2).timestamp() * 1000))
        (self.vscode_root / "workspaceStorage" / "h1" / "chatSessions" / "a.json").write_text(
            json.dumps(vssession([revised])), encoding="utf-8"
        )
        write_jsonl(
            self.codex_root / "sessions" / "2026" / "rollout-1.jsonl",
            [
                cxmeta(str(self.repo)),
                cxuser("codex prompt", ts=T1),
                cxagent("codex reply", ts=T2),
                cxuser("new codex prompt", ts=T3),
                cxagent("new codex reply", ts="2026-03-01T10:20:00.000Z"),
            ],
        )
        self.assertEqual(self.run_cli([str(self.repo), str(out_path)])[0], 2)  # a legacy page is refused
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = unrender.run([str(self.repo), str(out_path)])
        self.assertEqual(code, 0, err.getvalue())
        self.assertEqual({e.session for e in ace.inherit(out_path, self.repo)}, {"unrendered"})
        out, merged = self.generate(out_path)
        self.assertIn("Prompts: 4", out)
        for text in (
            "claude prompt", "claude reply", "codex prompt", "new codex prompt",
            "revised copilot prompt", "revised copilot reply",
        ):
            self.assertIn(text, merged)
        self.assertNotIn('<pre class="prompt">copilot prompt</pre>', merged)
        self.assertEqual(merged.count('<pre class="prompt">codex prompt</pre>'), 1)
        order = [
            merged.index(f'<pre class="prompt">{text}</pre>')
            for text in ("claude prompt", "codex prompt", "revised copilot prompt", "new codex prompt")
        ]
        self.assertEqual(order, sorted(order))
        sessions = {e.prompt: e.session for e in ace.inherit(out_path, self.repo)}
        self.assertEqual((sessions["codex prompt"], sessions["claude prompt"]), ("cx1", "unrendered"))
        _, again = self.generate(out_path)
        self.assertEqual(again, merged)

    def test_every_json_escape_survives_page_round_trip(self):
        text = (
            'quote " backslash \\ slash / bs \b ff \f nl \n cr \r tab \t nul \x00 del \x7f '
            "ls   ps   lit \\u003c amp & lt < gt > tag </script> cmt <!-- --> cdata ]]> "
            "astral \U0001F41D combining é crlf \r\n opener " + ace.SNAPSHOT_OPEN
        )
        write_jsonl(
            self.claude_root / "p" / "s.jsonl",
            [cu(text, ts=T0, cwd=str(self.repo)), ca([{"type": "text", "text": text}], cwd=str(self.repo))],
        )
        out_path = self.tmp / "out.html"
        self.generate(out_path)
        (self.claude_root / "p" / "s.jsonl").unlink()
        _, page = self.generate(out_path)
        self.assertEqual([(e.prompt, e.reply) for e in ace.inherit(out_path, self.repo)], [(text, text)])
        self.assertEqual(len(ace.SNAPSHOT.findall(page)), 1)
        self.assertEqual(count_scripts(page), 2)

    def test_tool_result_at_same_millisecond_elsewhere_never_purges_page_copy(self):
        # Replicata: the page holds a Claude prompt at T0 from session A; A is
        # pruned; session B, still in the store, has a bare tool_result record
        # at T0 (tool plumbing, no typing). Expectata: the page copy is kept —
        # a record that could never have been rendered is no holding.
        # Resultata before the fix: purged, and nothing replaced it.
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
        # Replicata: the page holds a prompt with its reply; the store's copy
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

    def test_snapshot_missing_field_rejected_and_page_untouched(self):
        # A snapshot another schema wrote: a row lacking a field, even one the
        # dataclass would default, is refused rather than thawed into a guess.
        self.populate()
        out_path = self.tmp / "out.html"
        _, page = self.generate(out_path)
        block = ace.SNAPSHOT.search(page).group(1)
        data = json.loads(block)
        for row in data["exchanges"]:
            del row["effort"]
        broken = page.replace(block, json.dumps(data).replace("<", "\\u003c"))
        out_path.write_text(broken, encoding="utf-8")
        code, _, err = self.run_cli([str(self.repo), str(out_path)])
        self.assertEqual(code, 2)
        self.assertIn(str(out_path), err)
        self.assertEqual(out_path.read_text(encoding="utf-8"), broken)

    def test_snapshot_version_key_is_provenance_not_a_gate(self):
        self.populate()
        out_path = self.tmp / "out.html"
        _, page = self.generate(out_path)
        block = ace.SNAPSHOT.search(page).group(1)
        self.assertEqual(json.loads(block)["sourcery"], ace.VERSION)
        out_path.write_text(
            page.replace(block, block.replace(json.dumps(ace.VERSION), '"0.0.0"', 1)), encoding="utf-8"
        )
        code, out, err = self.run_cli([str(self.repo), str(out_path)])
        self.assertEqual(code, 0, err)
        self.assertIn("Prompts: 3", out)

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
        self.assertEqual(len(ace.SNAPSHOT.findall(out_path.read_text(encoding="utf-8"))), 1)

    def test_relative_output_path_inherits_previous_page(self):
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

    def test_printed_count_deck_articles_and_snapshot_agree(self):
        self.populate()
        out_path = self.tmp / "out.html"
        self.generate(out_path)
        (self.claude_root / "p" / "s.jsonl").unlink()
        write_jsonl(
            self.codex_root / "sessions" / "2026" / "rollout-1.jsonl",
            [cxmeta(str(self.repo)), cxuser("codex prompt", ts=T1), cxagent("codex reply", ts=T2), cxuser("later", ts=T3)],
        )
        out, page = self.generate(out_path)
        count = int(out.split("Prompts: ")[1])
        self.assertEqual(count, 4)
        self.assertEqual(page.count("<article "), count)
        self.assertIn(f'<p class="deck">{count} prompts', page)
        self.assertEqual(len(json.loads(ace.SNAPSHOT.search(page).group(1))["exchanges"]), count)
        self.assertEqual(len(ace.inherit(out_path, self.repo)), count)

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
        self.assertEqual((code, out, err), (0, "5.5.0\n", ""))
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
        for name in ("sourcery.py", "quals.py", "unrender.py", "README.md"):
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
        # page copy of the emptied prompt must survive the merge, since the
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


# ------------------------------------------------------------------ snapshot

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


class SnapshotQuals(Fixture):
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

    def test_thaw_rejects_unknown_and_missing_fields(self):
        frozen = ace.freeze(maximal_exchange())
        page = self.tmp / "page.html"
        with self.assertRaises(ValueError):
            ace.thaw({**frozen, "author": "someone"}, page)
        for missing in ("prompt", "effort", "wall", "images", "ballots"):
            with self.assertRaises(ValueError, msg=missing):
                ace.thaw({k: v for k, v in frozen.items() if k != missing}, page)

    def test_thaw_rejects_wrong_types_and_ranges(self):
        # A hand-edited snapshot must not thaw into a guess: every field's
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
        # never purge a page copy that merely shares its millisecond.
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

    def test_snapshot_block_is_inert_json_without_angle_brackets(self):
        given = maximal_exchange()
        page = ace.render(self.repo, [given], "")
        found = ace.SNAPSHOT.findall(page)
        self.assertEqual(len(found), 1)
        self.assertNotIn("<", found[0])
        data = json.loads(found[0])
        self.assertEqual(data["repo"], self.repo.name)
        self.assertEqual(data["sourcery"], ace.VERSION)
        thawed = [ace.thaw(f, given.source) for f in data["exchanges"]]
        self.assertEqual(thawed, [given])
        self.assertNotIn("<script>alert(1)", page)
        self.assertEqual(count_scripts(page), 2)

    def test_snapshot_one_exchange_per_line_last_in_body(self):
        first = maximal_exchange()
        second = maximal_exchange(timestamp=utc(T1), prompt="second")
        page = ace.render(self.repo, [first, second], "")
        lines = ace.SNAPSHOT.search(page).group(1).split("\n")
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[0].endswith("["))
        self.assertEqual(json.loads(lines[1].rstrip(","))["prompt"], first.prompt)
        self.assertEqual(json.loads(lines[2])["prompt"], "second")
        self.assertEqual(lines[3], "]}")
        opener = page.index(ace.SNAPSHOT_OPEN)
        self.assertGreater(opener, page.rindex("</article>"))
        self.assertGreater(opener, page.index("<script>"))
        self.assertLess(opener, page.index("</body>"))

    def test_parsers_report_holdings_for_records_they_drop(self):
        # A record the store still holds is reported even when the parser
        # drops it as machine text, so a page's stale copy of it gets purged.
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


# ------------------------------------------------------------------ unrender

SNAPSHOT_BLOCK = re.compile(re.escape(ace.SNAPSHOT_OPEN) + r"\n.*?\n</script>\n", re.DOTALL)


def legacy(page: str) -> str:
    """The page as sourcery wrote it before snapshots: the block removed."""
    assert page.count(ace.SNAPSHOT_OPEN) == 1, page[-400:]
    return SNAPSHOT_BLOCK.sub("", page)


# A reply in the canonical form unrender writes back, exercising every
# construct markdown_html emits: a heading per level it distinguishes,
# paragraphs with hard line breaks, strong, code spans, links, bullets,
# numbered items, blockquotes, fenced code with and without a language,
# a fence holding a fence line, a code span holding backticks, raw HTML
# in prose, and the quoting characters html.escape rewrites.
REPLY_EVERYTHING = "\n\n".join(
    [
        "# Caput primum",
        'Paragraphus **fortis** et `codex` et [nexus](https://example.com/?a=1&b=2) et <b>non-tag</b>.'
        "\nLinea secunda \"citata\" 'apostrophus' & signum.",
        "## Caput secundum",
        "### Tertium",
        "#### Quartum",
        "##### Quintum",
        "- primum\n- `secundum` cum **forti**\n- [nexus **fortis**](mailto:x@y.z)",
        "1. unum\n2. duo\n3. tres",
        "> citatum\n> alterum citatum",
        '```python\nprint("salve")\nx = 1 < 2 & 3\n```',
        "```\nsine lingua\n```",
        "````\n```\nsaeptum intus\n````",
        "`` `intus` `` et ``` `` ``` finis.",
    ]
)

# The same constructs spelled the ways markdown_html also accepts; the page
# is identical, so unrender can only give back the canonical spelling.
REPLY_VARIANT = "\n\n".join(
    [
        "__sublineatum__ et `x`",
        "* stella\n+ plus",
        "3) tres\n7. septem",
        "~~~js\nx\n~~~",
        "## cauda ##",
        "###### sextum",
    ]
)
REPLY_VARIANT_CANONICAL = "\n\n".join(
    [
        "**sublineatum** et `x`",
        "- stella\n- plus",
        "1. tres\n2. septem",
        "```js\nx\n```",
        "## cauda",
        "##### sextum",
    ]
)


def maximal() -> list:
    """Four exchanges covering every field the page shows, on three days,
    from all three providers."""
    return [
        exchange(
            timestamp=utc("2026-03-01T10:00:00.123456Z"),
            effort="xhigh",
            prompt="a<b>&c\n  indented\ttab\n\n",
            reply=REPLY_EVERYTHING,
            images=("data:image/png;base64,AA==", "data:image/jpeg;base64,/9j/\"q"),
            elapsed=327.0,
            wall=3600.0,
            ballots=(
                ace.Ballot("Quid & quo?", ("A & B", "C"), ("A & B",)),
                ace.Ballot("Plura?", ("X", "Y", "Z"), ("X", "Z")),
                ace.Ballot("Scriptum?", ("P", "Q"), ()),
            ),
            added=1234,
            deleted=5678,
        ),
        exchange(
            timestamp=utc("2026-03-01T10:05:00.000001Z"),
            provider="Codex",
            model="gpt-5.6",
            prompt="secunda",
            reply="planum",
            elapsed=12.6,
            wall=12.9,
            deleted=3,
        ),
        exchange(
            timestamp=utc("2026-03-02T12:00:00Z"),
            provider="Copilot Chat",
            model="",
            prompt="",
            reply="",
            ballots=(ace.Ballot("Solum scriptum?", ("P",), ()),),
            elapsed=0.3,
        ),
        exchange(
            timestamp=utc("2026-03-03T23:59:59.999999Z"),
            model="m",
            prompt="ultima",
            reply="",
            elapsed=5.0,
            wall=5.0,
        ),
    ]


class UnrenderQuals(Fixture):
    def setUp(self):
        super().setUp()
        self.page = self.repo / "sourcery.html"

    def render(self, exchanges) -> str:
        """The page as a legacy sourcery rendered it: no snapshot block."""
        return legacy(ace.render(self.repo, exchanges, ace.repo_remote(self.repo)))

    def write(self, page: str) -> None:
        self.page.write_bytes(page.encode("utf-8"))

    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = unrender.run(argv)
        return code, out.getvalue(), err.getvalue()

    def refuse(self, page: str, repo=None) -> str:
        """Run the CLI on page, expect refusal, prove nothing was written,
        and return the error text."""
        self.write(page)
        before = sorted(p.name for p in self.page.parent.iterdir())
        code, out, err = self.run_cli([str(repo or self.repo), str(self.page)])
        self.assertEqual((code, out), (2, ""), err)
        self.assertTrue(err.startswith("Error:\n"), err)
        self.assertEqual(self.page.read_bytes(), page.encode("utf-8"))
        self.assertEqual(sorted(p.name for p in self.page.parent.iterdir()), before)
        return err

    def expected(self, originals) -> list:
        """What unrender can recover: session and source replaced, elapsed
        at display precision, wall only when the page showed it."""
        recovered = dict(session="unrendered", source=self.page)
        e1, e2, e3, e4 = originals
        return [
            dataclasses.replace(e1, **recovered),
            dataclasses.replace(e2, **recovered, elapsed=13.0, wall=0.0),
            dataclasses.replace(e3, **recovered, elapsed=0.0),
            dataclasses.replace(e4, **recovered, wall=0.0),
        ]

    def test_maximal_page_recovers_every_shown_field(self):
        originals = maximal()
        page = self.render(originals)
        self.write(page)
        got, rendered = unrender.recover(self.repo, self.page)
        self.assertEqual(got, self.expected(originals))
        self.assertEqual(legacy(rendered), page)
        self.assertEqual(self.render(got), page)

    def test_cli_rewrites_page_through_render_and_reports_count(self):
        originals = maximal()
        self.write(self.render(originals))
        code, out, err = self.run_cli([str(self.repo), str(self.page)])
        self.assertEqual((code, err), (0, ""), out)
        self.assertIn("4", out)
        self.assertIn(str(self.page), out)
        rewritten = self.page.read_text(encoding="utf-8")
        self.assertIn(ace.SNAPSHOT_OPEN, rewritten)
        self.assertEqual(
            rewritten, ace.render(self.repo, self.expected(originals), ace.repo_remote(self.repo))
        )
        # What gets written is exactly the rendering the proof passed on.
        self.assertEqual(ace.inherit(self.page, self.repo), self.expected(originals))

    def test_variant_markdown_spellings_canonicalize_page_identically(self):
        original = exchange(reply=REPLY_VARIANT)
        page = self.render([original])
        self.write(page)
        got, _ = unrender.recover(self.repo, self.page)
        self.assertEqual(got[0].reply, REPLY_VARIANT_CANONICAL)
        self.assertEqual(self.render(got), page)

    def test_carriage_return_in_legacy_prompt_survives_unrender(self):
        # Replicata: a legacy page whose prompt holds a lone CR and a CRLF, as
        # one real page does (a reply's line breaks are canonical already).
        # Expectata: the prompt is recovered byte-exact and the rewritten
        # page's snapshot gives the same bytes back. Resultata before the fix:
        # every CR silently became LF, through universal-newlines reading.
        original = exchange(prompt="one\r\ntwo\rthree", reply="a\r\nb")
        page = self.render([original])
        self.assertIn("\r", page)
        self.write(page)
        got, _ = unrender.recover(self.repo, self.page)
        self.assertEqual(got[0].prompt, original.prompt)
        self.assertEqual(ace.markdown_html(got[0].reply), ace.markdown_html(original.reply))
        code, _, err = self.run_cli([str(self.repo), str(self.page)])
        self.assertEqual(code, 0, err)
        self.assertEqual([e.prompt for e in ace.inherit(self.page, self.repo)], [original.prompt])

    def test_page_whose_public_home_is_hidden_refused(self):
        # unrender renders through the same masthead, so a project whose
        # public home hides under another remote name is refused here too,
        # before the page it was asked to import is touched.
        page = self.render([exchange()])
        self.write(page)
        git = self.repo / ".git"
        git.mkdir()
        (git / "config").write_text(
            '[remote "elsewhere"]\n\turl = git@github.com:beeminder/efme.git\n', encoding="utf-8"
        )
        code, out, err = self.run_cli([str(self.repo), str(self.page)])
        self.assertEqual((code, out), (2, ""))
        self.assertIn("elsewhere (https://github.com/beeminder/efme)", err)
        self.assertEqual(self.page.read_bytes(), page.encode("utf-8"))

    def test_antigravity_exchange_round_trips(self):
        original = exchange(provider="Antigravity", model="gemini-3-pro-high", reply="Visible **answer**.")
        page = self.render([original])
        self.write(page)
        got, _ = unrender.recover(self.repo, self.page)
        self.assertEqual(got, [dataclasses.replace(original, session="unrendered", source=self.page)])

    def test_non_utf8_page_refused(self):
        page = self.render([exchange()]).encode("utf-8") + b"\xff\xfe"
        self.page.write_bytes(page)
        code, out, err = self.run_cli([str(self.repo), str(self.page)])
        self.assertEqual((code, out), (2, ""))
        self.assertIn(str(self.page), err)
        self.assertEqual(self.page.read_bytes(), page)

    def test_newer_stylesheet_outside_main_tolerated(self):
        # Only the <main> region must round-trip: head, styles and the header
        # comment may legitimately differ between sourcery versions.
        page = self.render([exchange()])
        stale = page.replace("<style>", "<style>/* older stylesheet */ body{color:red}", 1)
        self.assertNotEqual(stale, page)
        self.write(stale)
        code, out, err = self.run_cli([str(self.repo), str(self.page)])
        self.assertEqual((code, err), (0, ""), out)
        expected = dataclasses.replace(exchange(), session="unrendered", source=self.page)
        self.assertEqual(
            self.page.read_text(encoding="utf-8"),
            ace.render(self.repo, [expected], ace.repo_remote(self.repo)),
        )

    def test_page_whose_subtitle_differs_from_repo_today_refused(self):
        # The masthead's where-line sits inside <main>: a page rendered when
        # the repo had a remote it no longer has is refused, nothing written.
        page = ace.render(self.repo, [exchange()], "https://github.com/x/y")
        err = self.refuse(legacy(page))
        self.assertIn("<main>", err)

    def test_rewrite_then_sourcery_with_empty_stores_byte_identical(self):
        self.write(self.render(maximal()))
        code, _, err = self.run_cli([str(self.repo), str(self.page)])
        self.assertEqual(code, 0, err)
        first = self.page.read_bytes()
        env = {
            "AI_CHAT_CLAUDE_ROOTS": str(self.tmp / "nc"),
            "AI_CHAT_CODEX_ROOTS": str(self.tmp / "nx"),
            "AI_CHAT_VSCODE_USER_ROOTS": str(self.tmp / "nv"),
            "AI_CHAT_ANTIGRAVITY_ROOTS": str(self.tmp / "na"),
        }
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = ace.run([str(self.repo), str(self.page)], env)
        self.assertEqual(code, 0, err.getvalue())
        self.assertIn("Prompts: 4", out.getvalue())
        self.assertEqual(self.page.read_bytes(), first)

    def test_snapshot_page_refused(self):
        page = ace.render(self.repo, [exchange()], ace.repo_remote(self.repo))
        self.assertIn(ace.SNAPSHOT_OPEN, page)
        err = self.refuse(page)
        self.assertIn("snapshot", err)

    def test_title_basename_mismatch_refused(self):
        other = self.tmp / "alius"
        other.mkdir()
        err = self.refuse(self.render([exchange()]), repo=other)
        self.assertIn("repo", err)
        self.assertIn("alius", err)

    def test_tampered_deck_count_refused(self):
        page = self.render([exchange(), exchange(timestamp=utc("2026-03-01T11:00:00Z"))])
        self.assertIn("2 prompts", page)
        self.refuse(page.replace("2 prompts", "3 prompts"))

    def test_tampered_deck_totals_refused(self):
        page = self.render([exchange(added=10, deleted=2)])
        self.assertIn("+10 −2", page)
        self.refuse(page.replace("+10 −2</p>", "+11 −2</p>"))

    def test_unknown_span_class_refused(self):
        page = self.render([exchange()])
        err = self.refuse(
            page.replace('<span class="chip"></span>', '<span class="chip"></span> <span class="arcanum">x</span>')
        )
        self.assertIn("arcanum", err)

    def test_unexpected_element_in_reply_refused(self):
        page = self.render([exchange(reply="r")])
        err = self.refuse(page.replace("<p>r</p>", "<table><tr><td>r</td></tr></table>"))
        self.assertIn("<table>", err)

    def test_unknown_inline_tag_in_reply_refused(self):
        page = self.render([exchange(reply="r")])
        err = self.refuse(page.replace("<p>r</p>", "<p><em>r</em></p>"))
        self.assertIn("<em>", err)

    def test_non_sourcery_html_refused(self):
        self.refuse("<!doctype html>\n<html><head><title>repo</title></head><body>salve</body></html>\n")

    def test_markup_without_provider_class_refused(self):
        # The article shape sourcery wrote before provider classes existed.
        page = self.render([exchange()])
        err = self.refuse(page.replace('<article class="exchange claude" id="p1">', '<article class="exchange" id="p1">'))
        self.assertIn('<article class="exchange" id="p1">', err)

    def test_day_header_without_id_refused(self):
        page = self.render([exchange()])
        day = utc(T0).astimezone().date().isoformat()
        self.assertIn(f'<h2 class="day" id="d{day}">', page)
        self.refuse(page.replace(f'<h2 class="day" id="d{day}">', '<h2 class="day">'))

    def test_inexact_round_trip_refused(self):
        # Every span is well-formed but the displayed clock time is wrong.
        page = self.render([exchange()])
        local = utc(T0).astimezone().strftime("%H:%M")
        wrong = "23:59" if local != "23:59" else "00:00"
        self.refuse(page.replace(f">{local}</time>", f">{wrong}</time>"))

    def test_unrecognized_duration_refused(self):
        page = self.render([exchange(elapsed=30.0)])
        self.assertIn("thought for 30s", page)
        self.refuse(page.replace("thought for 30s", "thought for 90s"))

    def test_missing_page_refused(self):
        code, out, err = self.run_cli([str(self.repo), str(self.page)])
        self.assertEqual((code, out), (2, ""))
        self.assertIn(str(self.page), err)

    def test_missing_repo_refused(self):
        self.write(self.render([exchange()]))
        code, out, err = self.run_cli([str(self.tmp / "nusquam"), str(self.page)])
        self.assertEqual((code, out), (2, ""))
        self.assertIn("nusquam", err)

    def test_wrong_argument_count_refused(self):
        self.write(self.render([exchange()]))
        for argv in ([], [str(self.repo)], [str(self.repo), str(self.page), "--open"]):
            code, out, err = self.run_cli(argv)
            self.assertEqual((code, out), (2, ""), argv)
            self.assertIn("unrender.py", err)
        self.assertEqual(self.page.read_bytes(), self.render([exchange()]).encode("utf-8"))



if __name__ == "__main__":
    unittest.main(verbosity=2)
