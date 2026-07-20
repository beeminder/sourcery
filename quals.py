#!/usr/bin/env python3
"""Quals for sourcery.py. Run: python3 quals.py

Each qual replicates a transcript shape observed in the real stores (Claude
Code, Codex, VS Code / Copilot Chat), states the expected extraction, and
lets unittest report what happened instead.
"""

import ast
import contextlib
import datetime as dt
import io
import json
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

import sourcery as ace

T0 = "2026-03-01T10:00:00.000Z"
T1 = "2026-03-01T10:05:00.000Z"
T2 = "2026-03-01T10:10:00.000Z"


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
        got = ace.claude_exchanges(self.path([cu(text, cwd=str(self.repo))]), self.repo)
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
        got = ace.claude_exchanges(self.path([cu(content, cwd=str(self.repo))]), self.repo)
        self.assertEqual([e.prompt for e in got], ["real prompt, exactly  this"])

    def test_wrapper_only_message_yields_nothing(self):
        content = [{"type": "text", "text": "<ide_opened_file>x</ide_opened_file>"}]
        got = ace.claude_exchanges(self.path([cu(content, cwd=str(self.repo))]), self.repo)
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
        self.assertEqual(ace.claude_exchanges(self.path(records), self.repo), [])

    def test_reply_joins_text_blocks_across_streamed_records(self):
        cwd = str(self.repo)
        records = [
            cu("go", cwd=cwd),
            ca([{"type": "thinking", "thinking": "hmm"}], cwd=cwd, mid="mA"),
            ca([{"type": "text", "text": "Part one."}], cwd=cwd, mid="mA"),
            ca([{"type": "tool_use", "id": "t", "name": "Bash", "input": {}}], cwd=cwd, mid="mA"),
            ca([{"type": "text", "text": "Part two."}], ts=T2, cwd=cwd, mid="mB", effort="xhigh"),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)
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
        got = ace.claude_exchanges(self.path(records), self.repo)
        self.assertEqual(
            [(e.prompt, e.images) for e in got],
            [("look at this", ("data:image/png;base64,AAA",)), ("", ("data:image/png;base64,AAA",))],
        )

    def test_unrecognized_image_form_fails_loudly(self):
        content = [{"type": "image", "source": {"type": "url", "url": "https://x"}}]
        with self.assertRaises(ace.UserError):
            ace.claude_exchanges(self.path([cu(content, cwd=str(self.repo))]), self.repo)

    def test_synthetic_harness_notices_dropped(self):
        cwd = str(self.repo)
        records = [
            cu("go", cwd=cwd),
            ca([{"type": "text", "text": "API Error: 401"}], cwd=cwd, model="<synthetic>"),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)
        self.assertEqual(
            [(e.prompt, e.reply, e.model, e.elapsed) for e in got],
            [("go", "", "", 0.0)],
        )

    def test_identical_consecutive_prompts_are_two_human_acts(self):
        cwd = str(self.repo)
        records = [cu("retry", ts=T0, cwd=cwd), cu("retry", ts=T1, cwd=cwd)]
        got = ace.claude_exchanges(self.path(records), self.repo)
        self.assertEqual([(e.prompt, e.reply) for e in got], [("retry", ""), ("retry", "")])

    def test_cwd_outside_repo_excluded_subdir_included(self):
        records = [
            cu("outside", cwd="/somewhere/else"),
            cu("inside", ts=T1, cwd=str(self.repo / "sub" / "dir")),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)
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
        got = ace.claude_exchanges(self.path(records), self.repo)
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
        got = ace.claude_exchanges(self.path(records), self.repo)
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
        got = ace.claude_exchanges(self.path(records), self.repo)
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
        got = ace.claude_exchanges(self.path(records), self.repo)
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
        got = ace.claude_exchanges(self.path(records), self.repo)
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
        got = ace.claude_exchanges(self.path(records), self.repo)
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
        got = ace.claude_exchanges(self.path(records), self.repo)
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
        got = ace.claude_exchanges(self.path(records), self.repo)
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
        got = ace.claude_exchanges(self.path(records), self.repo)
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
        got = ace.claude_exchanges(self.path(records), self.repo)
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
            ace.claude_exchanges(self.path(records), self.repo)

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
        got = ace.claude_exchanges(self.path(records), self.repo)
        self.assertEqual(
            [(e.prompt, e.reply) for e in got],
            [("start the work", "Working."), ("wait, also do X", "Doing X.")],
        )
        bad = [queued("hologram-mode", [{"type": "text", "text": "x"}], T0)]
        with self.assertRaises(ace.UserError):
            ace.claude_exchanges(self.path(bad), self.repo)

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
        got = ace.claude_exchanges(self.path(records), self.repo)
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
        got = ace.claude_exchanges(self.path(records), self.repo)
        self.assertEqual([e.prompt for e in got], ["typed with origin", "typed without origin"])
        with self.assertRaises(ace.UserError):
            ace.claude_exchanges(
                self.path([cu("x", cwd=cwd, origin={"kind": "hologram"})]), self.repo
            )

    def test_interrupt_markers_dropped_but_typed_text_around_them_kept(self):
        cwd = str(self.repo)
        records = [
            cu("[Request interrupted by user]", cwd=cwd),
            cu("[Request interrupted by user for tool use]", ts=T1, cwd=cwd),
            cu("[Request interrupted by user] but i typed this", ts=T2, cwd=cwd),
        ]
        got = ace.claude_exchanges(self.path(records), self.repo)
        self.assertEqual([e.prompt for e in got], ["[Request interrupted by user] but i typed this"])

    def test_slash_command_wrapper_recovered_as_typed_command(self):
        wrapped = "<command-message>insights</command-message>\n<command-name>/insights</command-name>"
        got = ace.claude_exchanges(self.path([cu(wrapped, cwd=str(self.repo))]), self.repo)
        self.assertEqual([e.prompt for e in got], ["/insights"])

    def test_malformed_slash_command_wrapper_fails_loudly(self):
        wrapped = "<command-message>insights</command-message>\n<command-name>/different</command-name>"
        with self.assertRaises(ace.UserError):
            ace.claude_exchanges(self.path([cu(wrapped, cwd=str(self.repo))]), self.repo)

    def test_partial_slash_command_wrapper_fails_loudly(self):
        wrapped = "<command-name>/insights</command-name>"
        with self.assertRaises(ace.UserError):
            ace.claude_exchanges(self.path([cu(wrapped, cwd=str(self.repo))]), self.repo)

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
        got = ace.claude_exchanges(self.path(records), self.repo)
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
        got = ace.claude_exchanges(self.path(records), self.repo)
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
        got = ace.claude_exchanges(self.path(records), self.repo)
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
        got = ace.claude_exchanges(self.path(records), self.repo)
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
        got = ace.claude_exchanges(self.path(records), self.repo)
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
        self.assertEqual(ace.claude_exchanges(self.path(records), self.repo), [])

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
        got = ace.claude_exchanges(self.path(records), self.repo)
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
            ace.claude_exchanges(self.path(records), self.repo)

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
            ace.claude_exchanges(self.path(records), self.repo)

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
        got = ace.claude_exchanges(self.path(records), self.repo)
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
        got = ace.claude_exchanges(self.path(records), self.repo)
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
        got = ace.claude_exchanges(self.path(records), self.repo)
        self.assertEqual(got[0].ballots[0].picked, ("Tutorial", "Sandbox"))
        self.assertEqual(got[0].prompt, "")

    def test_missing_timestamp_fails_loudly(self):
        path = self.path([cu("go", ts=None, cwd=str(self.repo))])
        with self.assertRaises(ace.UserError):
            ace.claude_exchanges(path, self.repo)

    def test_live_capture_tolerates_truncated_final_line_only(self):
        path = self.tmp / "claude" / "p1" / "live.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        good = json.dumps(cu("hi", cwd=str(self.repo)))
        path.write_text(good + '\n{"type": "user", "mess', encoding="utf-8")
        got = ace.claude_exchanges(path, self.repo)
        self.assertEqual([e.prompt for e in got], ["hi"])
        path.write_text('not json\n' + good + "\n", encoding="utf-8")
        with self.assertRaises(ace.UserError):
            ace.claude_exchanges(path, self.repo)

    def test_malformed_jsonl_cites_file_and_line(self):
        path = self.tmp / "claude" / "p1" / "bad.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"type": "system"}\nnot json\n', encoding="utf-8")
        with self.assertRaises(ace.UserError) as ctx:
            ace.claude_exchanges(path, self.repo)
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
        got = ace.codex_exchanges(self.path(records), self.repo)
        self.assertEqual([e.prompt for e in got], ["sure, let's see how it looks"])
        self.assertEqual(got[0].reply, "Looks fine.")
        self.assertEqual(got[0].model, "gpt-5.3-codex")
        self.assertEqual(got[0].effort, "xhigh")
        self.assertEqual(got[0].provider, "Codex")
        self.assertEqual(got[0].elapsed, 300.0)  # user at T0, agent_message at T1

    def test_plain_message_kept_verbatim(self):
        records = [cxmeta(str(self.repo)), cxuser("fix the  bug\nplease")]
        got = ace.codex_exchanges(self.path(records), self.repo)
        self.assertEqual([e.prompt for e in got], ["fix the  bug\nplease"])

    def test_identical_consecutive_prompts_are_two_human_acts(self):
        records = [
            cxmeta(str(self.repo)),
            cxuser("retry", ts=T0),
            cxuser("retry", ts=T1),
            cxagent("done", ts=T2),
        ]
        got = ace.codex_exchanges(self.path(records), self.repo)
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
        got = ace.codex_exchanges(self.path(records), self.repo)
        self.assertEqual([(e.prompt, e.reply) for e in got], [("go", "done")])

    def test_machine_sessions_skipped(self):
        subagent = [cxmeta(str(self.repo), source={"subagent": {"other": "guardian"}}), cxuser("audit")]
        titler = [cxmeta(str(self.repo), source="exec"), cxuser("Generate a concise UI title")]
        self.assertEqual(ace.codex_exchanges(self.path(subagent, "a.jsonl"), self.repo), [])
        self.assertEqual(ace.codex_exchanges(self.path(titler, "b.jsonl"), self.repo), [])

    def test_unknown_source_fails_loudly(self):
        path = self.path([cxmeta(str(self.repo), source="quantum"), cxuser("hi")])
        with self.assertRaises(ace.UserError) as ctx:
            ace.codex_exchanges(path, self.repo)
        self.assertIn("quantum", str(ctx.exception))

    def test_cwd_outside_repo_excluded(self):
        records = [cxmeta("/elsewhere"), cxuser("hi")]
        self.assertEqual(ace.codex_exchanges(self.path(records), self.repo), [])

    def test_agent_history_handoff_dropped(self):
        records = [
            cxmeta(str(self.repo)),
            cxuser("The following is the Codex agent history added since your last message."),
            cxuser("real question", ts=T1),
        ]
        got = ace.codex_exchanges(self.path(records), self.repo)
        self.assertEqual([e.prompt for e in got], ["real question"])

    def test_pasted_image_data_uris_pass_through(self):
        records = [
            cxmeta(str(self.repo)),
            cxuser("see image", images=["data:image/png;base64,QUJD"]),
        ]
        got = ace.codex_exchanges(self.path(records), self.repo)
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
        got = ace.codex_exchanges(self.path(records), self.repo)
        self.assertEqual([(e.prompt, e.elapsed) for e in got], [("q1", 300.0), ("q2", 60.0)])

    def test_elapsed_excludes_tool_output_gaps(self):
        records = [
            cxmeta(str(self.repo)),
            cxuser("go", ts=T0),
            {"type": "response_item", "timestamp": T1, "payload": {"type": "function_call", "name": "shell"}},
            {"type": "response_item", "timestamp": T2, "payload": {"type": "function_call_output", "output": "x"}},
            cxagent("done", ts="2026-03-01T10:15:00.000Z"),
        ]
        got = ace.codex_exchanges(self.path(records), self.repo)
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
        got = ace.codex_exchanges(self.path(records), self.repo)
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
        got = ace.codex_exchanges(self.path(records), self.repo)
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
        got = ace.codex_exchanges(self.path(records), self.repo)
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
        got = ace.codex_exchanges(self.path(records), self.repo)
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
        got = ace.codex_exchanges(self.path(records), self.repo)
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
        got = ace.codex_exchanges(self.path(records), self.repo)
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
        got = ace.codex_exchanges(self.path(records), self.repo)
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
            ace.codex_exchanges(self.path(records), self.repo)
        self.assertIn("codex_tally", str(ctx.exception))

    def test_unknown_patch_change_form_fails_loudly(self):
        changes = {str(self.repo / "a.js"): {"type": "transmogrify"}}
        records = [cxmeta(str(self.repo)), cxuser("go"), cxpatch(changes)]
        with self.assertRaises(ace.UserError) as ctx:
            ace.codex_exchanges(self.path(records), self.repo)
        self.assertIn("codex_tally", str(ctx.exception))

    def test_terminal_fix_template_dropped_even_in_codex(self):
        records = [
            cxmeta(str(self.repo)),
            cxuser("I get the following error. Please fix the error. Completed code only, no commentary.\n\n$ x\nboom"),
        ]
        self.assertEqual(ace.codex_exchanges(self.path(records), self.repo), [])


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
        got = ace.vscode_exchanges(path, (self.repo,), self.repo)
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
        got = ace.vscode_exchanges(path, (self.repo,), self.repo)
        self.assertEqual([(e.reply, e.model) for e in got], [("done", "gpt-5.4")])

    def test_malformed_or_duplicate_auto_mode_resolution_fails_loudly(self):
        malformed = {"kind": "autoModeResolution"}
        valid = {"kind": "autoModeResolution", "resolvedModel": "gpt-5.4"}
        for response in ([malformed], [valid, valid]):
            with self.subTest(response=response):
                path = self.write_session(vssession([vsreq("q", response, model="copilot/auto")]))
                with self.assertRaises(ace.UserError) as ctx:
                    ace.vscode_exchanges(path, (self.repo,), self.repo)
                self.assertIn("auto-mode", str(ctx.exception))

    def test_inline_reference_name_spliced_into_reply(self):
        response = [
            md("see "),
            {"kind": "inlineReference", "inlineReference": {"name": "foo.py", "location": {}}},
            md(" for details"),
        ]
        path = self.write_session(vssession([vsreq("q", response)]))
        got = ace.vscode_exchanges(path, (self.repo,), self.repo)
        self.assertEqual(got[0].reply, "see foo.py for details")

    def test_unknown_response_kind_fails_loudly(self):
        path = self.write_session(vssession([vsreq("q", [{"kind": "hologram"}])]))
        with self.assertRaises(ace.UserError) as ctx:
            ace.vscode_exchanges(path, (self.repo,), self.repo)
        self.assertIn("hologram", str(ctx.exception))
        self.assertIn(str(path), str(ctx.exception))

    def test_unknown_version_fails_loudly(self):
        path = self.write_session(vssession([vsreq("q", [])], version=2))
        with self.assertRaises(ace.UserError):
            ace.vscode_exchanges(path, (self.repo,), self.repo)

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
        got = ace.vscode_exchanges(path, (self.repo,), self.repo)
        self.assertEqual([e.prompt for e in got], ["first", "second"])
        self.assertEqual(got[1].reply, "late reply")

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
        got = ace.vscode_exchanges(path, (self.repo,), self.repo)
        self.assertEqual(
            [e.prompt for e in got],
            ["can you fix this:\n\n$ ./x.py\nTraceback", "(general reminder about AGENTS.md)"],
        )

    def test_elapsed_taken_from_result_timings(self):
        path = self.write_session(vssession([vsreq("q", [md("r")], elapsed_ms=327000)]))
        got = ace.vscode_exchanges(path, (self.repo,), self.repo)
        self.assertEqual(got[0].elapsed, 327.0)

    def test_elapsed_suppressed_when_turn_paused_for_confirmation(self):
        response = [md("r"), {"kind": "confirmation"}]
        path = self.write_session(vssession([vsreq("q", response, elapsed_ms=41000000)]))
        got = ace.vscode_exchanges(path, (self.repo,), self.repo)
        self.assertEqual(got[0].elapsed, 0.0)

    def test_pasted_image_bytes_rebuilt_file_attachments_ignored(self):
        variables = [
            {"kind": "file", "id": "f", "name": "x.py", "value": {"path": "/x.py"}},
            {"kind": "image", "id": "i", "name": "Pasted Image", "mimeType": "image/png",
             "isPasted": True, "value": {"0": 65, "1": 66, "2": 67}},
        ]
        path = self.write_session(vssession([vsreq("q", [], variables=variables)]))
        got = ace.vscode_exchanges(path, (self.repo,), self.repo)
        self.assertEqual(got[0].images, ("data:image/png;base64,QUJD",))

    def test_workspace_outside_repo_excluded_and_empty_session_ok(self):
        path = self.write_session(vssession([vsreq("q", [])]))
        other = self.tmp / "other"
        other.mkdir()
        self.assertEqual(ace.vscode_exchanges(path, (other,), self.repo), [])
        empty = self.write_session(vssession([]), "e.json")
        self.assertEqual(ace.vscode_exchanges(empty, (self.repo,), self.repo), [])

    def test_straddling_multiroot_workspace_fails_loudly(self):
        other = self.tmp / "other"
        other.mkdir()
        path = self.write_session(vssession([vsreq("q", [])]))
        with self.assertRaises(ace.UserError):
            ace.vscode_exchanges(path, (self.repo, other), self.repo)

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
        for root in (self.claude_root, self.codex_root, self.vscode_root):
            root.mkdir()
        self.env = {
            "AI_CHAT_CLAUDE_ROOTS": str(self.claude_root),
            "AI_CHAT_CODEX_ROOTS": str(self.codex_root),
            "AI_CHAT_VSCODE_USER_ROOTS": str(self.vscode_root),
        }

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

    def test_existing_output_overwritten(self):
        self.populate()
        out_path = self.tmp / "out.html"
        out_path.write_text("stale generation", encoding="utf-8")
        code, _, err = self.run_cli([str(self.repo), str(out_path)])
        self.assertEqual(code, 0, err)
        page = out_path.read_text(encoding="utf-8")
        self.assertNotIn("stale generation", page)
        self.assertIn("claude prompt", page)

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

    def test_version_and_help_exit_zero(self):
        code, out, err = self.run_cli(["--version"])
        self.assertEqual((code, out, err), (0, "5.1.0\n", ""))
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
    def test_origin_url_forms_normalized(self):
        git = self.repo / ".git"
        git.mkdir()
        cases = [
            ('[remote "origin"]\n\turl = git@github.com:dreeves/crashla.git\n', "https://github.com/dreeves/crashla"),
            ('[core]\n\tbare = false\n[remote "origin"]\n\turl = https://github.com/dreeves/road.git\n', "https://github.com/dreeves/road"),
            ('[remote "origin"]\n\turl = ssh://git@github.com/dreeves/bid.git\n', "https://github.com/dreeves/bid"),
            ('[remote "upstream"]\n\turl = git@github.com:other/x.git\n', ""),
        ]
        for config, expected in cases:
            (git / "config").write_text(config, encoding="utf-8")
            self.assertEqual(ace.repo_remote(self.repo), expected)

    def test_no_git_directory_means_no_link(self):
        self.assertEqual(ace.repo_remote(self.repo), "")


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
