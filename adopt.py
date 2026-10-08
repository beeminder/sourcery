#!/usr/bin/env python3
"""Adopt a page sourcery 5 wrote into a sourcery 6 ledger.

Sourcery 5 kept its memory of the dialog in the page itself: its
"snapshot", every exchange the page shows, embedded as JSON at the page's
end. Sourcery 6 keeps that memory in ledgers beside the page instead (see
sourcery.py for "ledger", "login", "credit" and "holding"), and refuses a
page sourcery 5 wrote. This one-time importer moves such a page's snapshot
into one human's ledger beside the output page, then renders the output
page anew from every ledger beside it, as sourcery 6 would, so sourcery 6
runs on from there.

Usage:
    python3 adopt.py REPODIR PAGE.html OUTPUT.html LOGIN

PAGE.html is the page sourcery 5 wrote. OUTPUT.html is the page sourcery 6
writes, usually PAGE.html itself; the ledger is sourcery.LOGIN.jsonl beside
it, LOGIN being the GitHub login of the human whose prompts PAGE.html shows,
lowercased as sourcery lowercases the login gh reports. PAGE.html is
rewritten only when it is OUTPUT.html.

The page's exchanges join whatever that ledger holds already. A prompt the
ledger holds (the same holding) keeps the ledger's reading, as a ledger's
row gives way to a store's in a sourcery run, so adopting a page twice
changes nothing. Refused, with nothing written: a page whose snapshot names
another repo, a page without exactly one snapshot, a page holding a prompt
another human's ledger holds, an output page adoption would replace
without its rows (one sourcery 6 did not write, unless it is PAGE.html, or
one crediting a row its ledger lacks), and two readings of one prompt (see
sourcery's check_readings) among the rows adoption would write and render,
so it never writes a page sourcery would refuse.

A snapshot names its repo as sourcery 5 named it, by the checkout
directory's name, so it is checked against this checkout's directory; the
ledger's first line names the repo by its project name, as sourcery 6 does.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Sequence

import sourcery
from sourcery import Exchange, UserError, holding

# A page sourcery 5 wrote ends with its snapshot: this script element
# holding {"sourcery": version, "repo": name, "exchanges": [...]}, every
# exchange in freeze() form on its own line, every "<" escaped, so no prompt
# can close the element early.
SNAPSHOT_OPEN = '<script type="application/json" id="snapshot">'
SNAPSHOT = re.compile(re.escape(SNAPSHOT_OPEN) + r"\n(.*?)\n</script>", re.DOTALL)
# What a snapshot must look like: each key's JSON type.
SNAPSHOT_SHAPE = {"sourcery": "str", "repo": "str", "exchanges": "list"}

# TODO: The usage text: adopt.py takes the project directory, the page
# sourcery 5 wrote, the output page sourcery 6 is to write (usually that
# same page), beside which the ledger is kept, and the GitHub login of the
# human whose prompts the page shows.
USAGE = (
    "Usus:\n"
    "  python3 adopt.py REPODIR PAGE.html OUTPUT.html LOGIN\n\n"
    "PAGE.html    pagina a sourcery 5 scripta, memoriam (snapshot) ferens\n"
    "OUTPUT.html  pagina a sourcery 6 scribenda, plerumque eadem; tabula iuxta eam servatur\n"
    "LOGIN        nomen GitHub hominis cuius rogationes pagina ostendit"
)


def adopted_login(text: str) -> str:
    """The login text names, lowercased, as sourcery lowercases the login gh
    reports; anything that is no GitHub login is refused."""
    if not sourcery.GITHUB_LOGIN.fullmatch(text):
        # TODO: Says the login given is no GitHub login (letters, digits,
        # and single hyphens, 1 to 39 characters, no hyphen at either end),
        # quoting it, then the usage text.
        raise UserError(f"Nomen GitHub invalidum: {text!r}\n\n{USAGE}")
    return text.lower()


def snapshot(page: Path, repo: Path) -> list[Exchange]:
    """Every exchange the snapshot of a page sourcery 5 wrote holds, each
    thawed and checked as sourcery checks a ledger's rows."""
    try:
        # Bytes, not text: universal newlines would rewrite a carriage return
        # someone typed, and the page must give back exactly what it holds.
        text = page.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        # TODO: Says the page to adopt could not be read as a UTF-8 file (it
        # may be missing, a directory, unreadable, or not text), gives the
        # reason, and that nothing was written.
        raise UserError(f"Pagina legi non potest: {page}\n{exc}\nNihil scriptum est.") from exc
    found = SNAPSHOT.findall(text)
    if len(found) != 1:
        # TODO: Says the page carries not one snapshot but this many, so it
        # is no page sourcery 5 wrote and there is nothing to adopt: a page
        # sourcery 6 wrote needs no adopting (run sourcery.py on it), and
        # any other file is no sourcery page. Nothing was written.
        raise UserError(
            f"Pagina non unam memoriam (snapshot) fert, sed {len(found)}: {page}\n"
            "Pagina a sourcery 6 scripta adoptione non eget: curre sourcery.py. "
            "Alius fasciculus pagina sourcery non est.\nNihil scriptum est."
        )
    try:
        data = sourcery.json_object(found[0])
        shape = {key: type(value).__name__ for key, value in data.items()}
        if shape != SNAPSHOT_SHAPE:
            raise ValueError(f"{shape} != {SNAPSHOT_SHAPE}")
        exchanges = [sourcery.thaw(frozen, page) for frozen in data["exchanges"]]
    except (ValueError, TypeError, UserError) as exc:
        # TODO: Says the page's snapshot is malformed, gives the reason, and
        # that nothing was written.
        raise UserError(f"Memoria paginae corrupta est: {page}\n{exc}\nNihil scriptum est.") from exc
    if data["repo"] != repo.name:
        # TODO: Says the page's snapshot belongs to another project, naming
        # both, and nothing was written. A snapshot names its project by the
        # name of the directory sourcery ran in: if this checkout's
        # directory (named) is that same project after all, rename the
        # directory to the snapshot's project name, then rerun.
        raise UserError(
            f"Memoria paginae ad aliud inceptum pertinet: {data['repo']!r}, non {repo.name!r}: {page}\n"
            f"Memoria inceptum suum nomine directorii nominat. Si {repo} idem inceptum est, "
            f"directorium renomina {data['repo']!r}, deinde iterum curre.\n"
            "Nihil scriptum est."
        )
    return exchanges


def run(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if len(args) != 4:
            raise UserError(USAGE)
        repo = sourcery.canonical_repo(Path(args[0]).expanduser())
        remote = sourcery.repo_remote(repo)
        project = sourcery.project_name(repo, remote)
        page, output = Path(args[1]).expanduser(), Path(args[2]).expanduser()
        login = adopted_login(args[3])
        rows = snapshot(page, repo)
        # The output page is replaced below, so every row it shows must
        # survive: PAGE.html's rows are the snapshot adopted here, and any
        # other page there must be one sourcery 6 wrote, whose credits the
        # ledgers hold, as sourcery itself checks before replacing it.
        credits = set() if output.exists() and output.samefile(page) else sourcery.page_credits(output)
        ledgers, citations = sourcery.read_ledgers(output, repo, remote)
        sourcery.check_ledgers(output, login, ledgers, credits, set())
        owners = {holding(exchange): other for other, kept in ledgers.items() for exchange in kept}
        claimed = sorted(h for h in map(holding, rows) if owners.get(h, login) != login)
        if claimed:
            (provider, when), owner = claimed[0], owners[claimed[0]]
            # TODO: Says the page to adopt holds a prompt (agent and time)
            # that another human's ledger holds already, naming that human's
            # login and ledger: each prompt belongs to one human, so the
            # page cannot be adopted as the login given (named). Perhaps the
            # page is that other human's, or the login given is wrong.
            # Nothing was written.
            raise UserError(
                f"Pagina rogationem {provider} {when.isoformat()} tenet, quam tabula {owner} iam tenet: "
                f"{sourcery.ledger_path(output, owner)}\n"
                f"Rogatio unius hominis est: pagina ut {login} adoptari non potest. "
                "Fortasse pagina illius hominis est, aut nomen datum erratum.\nNihil scriptum est."
            )
        # The ledger is the truth for every prompt it holds and the page for
        # the rest, as a store is for a ledger in a sourcery run: a page row
        # whose holding the ledger holds gives way to the ledger's reading,
        # which a sourcery 6 run or an earlier adoption gave it.
        mine = ledgers.get(login, [])
        held = {holding(exchange) for exchange in mine}
        added = [exchange for exchange in rows if holding(exchange) not in held]
        ledgers = dict(sorted({**ledgers, login: sourcery.weave(mine + added)}.items()))
        # Two readings of one prompt are refused as sourcery refuses them,
        # among the rows adoption would write and render. A row the page
        # gave has no ledger line yet, so it is cited by the page.
        # TODO: Says to delete the stale reading's line, a ledger's line as
        # cited, or, for a reading cited by the page, its line in the page's
        # snapshot, then rerun.
        sourcery.check_readings(
            ledgers,
            {**citations, **{exchange: str(exchange.source) for exchange in added}},
            "Lineam obsoletam dele (tabulae, aut memoriae paginae), deinde iterum curre.",
        )
        # The page is rendered before anything is written, so a refusal found
        # while rendering it writes nothing.
        rendered = sourcery.render(repo, ledgers, remote)
        ledger = sourcery.ledger_path(output, login)
        sourcery.write_output(ledger, sourcery.ledger_text(project, ledgers[login]))
        sourcery.write_output(output, rendered)
        # TODO: Reports success: each path written (the ledger, then the
        # page); the human the adopted page's prompts are now credited to,
        # by display name and login, so a wrong login shows at once; how
        # many prompts the adopted page held, and how many of them the
        # ledger gained; then how many prompts the page now shows, and how
        # many of them each human's ledger holds, by login.
        print(
            f"Scriptum: {ledger.resolve()}\nScriptum: {output.resolve()}\n"
            f"Rogationes paginae tribuuntur: {sourcery.display_name(login)} ({login})\n"
            f"Rogationes paginae: {len(rows)}; tabulae additae: {len(added)}\n"
            f"Rogationes: {sum(map(len, ledgers.values()))}"
            + "".join(f"\n  {who}: {len(kept)}" for who, kept in ledgers.items())
        )
        return 0
    except UserError as exc:
        print(f"Error:\n{exc}", file=sys.stderr)
        return 2


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
