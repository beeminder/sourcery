#!/usr/bin/env python3
"""Import a legacy sourcery page back into Exchange records.

Transcript stores get pruned, so for many repos the rendered page is the
only copy of its early prompts. Newer sourcery embeds a JSON snapshot of
every exchange in the page it writes and merges it back on the next run;
pages generated before that carry none. This one-time importer parses such
a page back into Exchange records — the exact inverse of sourcery.render()
for every field the page shows — and rewrites the page through render(),
which adds the snapshot. Nothing is written unless the recovered exchanges
re-render the page's <main> region byte for byte.

Usage:
    python3 unrender.py REPODIR PAGE.html

Jargon: a "legacy page" is a sourcery page without a snapshot block; the
"skeleton" is the page's fixed frame (doctype, title, masthead, main);
"tiling" is matching a pattern back to back over a string so that any gap
is markup this importer does not understand. Every recovered exchange
carries the session id "unrendered", a sentinel no transcript store can
produce.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path
from typing import Sequence

import sourcery
from sourcery import Ballot, Exchange, UserError

UNRENDERED = "unrendered"
PROVIDERS = {slug: name for name, slug in sourcery.PROVIDER_SLUGS.items()}

# TODO: The usage line: the script takes the project directory and the
# page to import, nothing else.
USAGE = "Usus:\n  python3 unrender.py REPODIR PAGE.html"


def anonymous(pattern: re.Pattern[str]) -> str:
    """The pattern's source with its named groups made non-capturing, so it
    can be embedded (and repeated) inside a larger pattern."""
    return re.sub(r"\(\?P<\w+>", "(?:", pattern.pattern)


# --------------------------------------------------------- page-shape patterns
# Each pattern mirrors one f-string in sourcery.render() (or markdown_html /
# markdown_inline) character for character. Text the renderer escaped can
# hold no "<" and, where quote=True, no '"', which is what makes [^<]* and
# [^"]* exact delimiters.

COUNT = r"\d{1,3}(?:,\d{3})*"  # an integer as f"{n:,}" writes it
DAY = r"\d{4}-\d\d-\d\d"
WEEKDAY = "|".join(sourcery.WEEKDAYS)
SLUG = "|".join(re.escape(slug) for slug in sourcery.PROVIDER_SLUGS.values())

PAGE = re.compile(
    r"\A<!doctype html>\n.*?^<title>(?P<title>[^<]*)</title>$.*?"
    r'(?P<main>\n<main>\n<header class="masthead">\n.*?\n</header>\n(?P<body>.*?)\n</main>\n)',
    re.DOTALL | re.MULTILINE,
)
CHUNK = re.compile(
    rf'(?:<h2 class="day" id="d(?P<day>{DAY})"><time datetime="(?P=day)">(?P=day) (?:{WEEKDAY})</time></h2>'
    rf'|<article class="exchange (?P<slug>{SLUG})" id="p(?P<number>\d+)">(?P<inner>.*?)</article>)'
    r"(?:\n|\Z)",
    re.DOTALL,
)
OPTION = re.compile(
    r'\n<div class="option picked">✓ (?P<picked>[^<]*)</div>'
    r'|\n<div class="option">· (?P<unpicked>[^<]*)</div>'
)
BALLOT = re.compile(
    r'\n<div class="ballot machine">\n<div class="ballot-question">(?P<question>[^<]*)</div>'
    rf"(?P<options>(?:{anonymous(OPTION)})*)\n</div>"
)
IMAGE = re.compile(r'<img class="attachment" src="(?P<uri>[^"]*)" alt="">')
DURATION = re.compile(r"(?:(?P<h>\d+)h)?(?:(?P<m>\d+)m)?(?:(?P<s>\d+)s)?")
# The two empty-reply notes render() writes, verbatim.
EMPTY_REPLY = r"No response\.|Response still generating when this transcript was captured"
# An article's inside, in three pieces so that a piece unrender does not
# understand is reported by itself: what precedes the disclosure, the
# disclosure line, and the reply — empty when the page shows one of the two
# empty-reply notes instead.
FRAME = re.compile(
    r"(?P<head>.*?)\n<details>\n<summary>(?P<summary>.*?)</summary>\n"
    rf'(?:<div class="reply machine empty"><p>(?:{EMPTY_REPLY})</p>'
    r'|<div class="reply machine">)(?P<reply>.*?)</div>\n</details>\n',
    re.DOTALL,
)
HEAD = re.compile(
    rf'(?:\n<div class="diffstat">\+(?P<added>{COUNT}) −(?P<deleted>{COUNT})'
    r' <span class="blocks" aria-hidden="true">(?:<span class="(?:add|del|nil)"></span>)*</span></div>)?'
    rf"(?P<ballots>(?:{anonymous(BALLOT)})*)"
    r'(?:\n<pre class="prompt">(?P<prompt>[^<]*)</pre>)?'
    rf'(?:\n<div class="attachments">(?P<attachments>(?:{anonymous(IMAGE)})*)</div>)?'
)
SUMMARY = re.compile(
    r'<a class="anchor" href="#p\d+"><time datetime="(?P<when>[^"]*)">\d\d:\d\d</time></a>'
    r' <span class="chip"></span> <span class="agent">[^<]*</span>'
    r'(?: <span class="model">(?P<model>[^<]*)</span>)?'
    r'(?: <span class="effort">\((?P<effort>[^<]*)\)</span>)?'
    rf'(?: <span class="elapsed">thought for (?P<elapsed>{anonymous(DURATION)})'
    rf"(?: · (?P<wall>{anonymous(DURATION)}) wall-clock time)?</span>)?"
)

# One block of markdown_html output; the group that matched names the kind.
BLOCK = re.compile(
    r'(?:<pre class="code"><code(?: data-language="(?P<language>[^"]*)")?>(?P<code>[^<]*)</code></pre>'
    r"|<h(?P<level>[2-6])>(?P<heading>.*?)</h(?P=level)>"
    r"|<(?P<list>ul|ol)>(?P<items>(?:<li>.*?</li>)*)</(?P=list)>"
    r"|<blockquote><p>(?P<quote>.*?)</p></blockquote>"
    r"|<p>(?P<paragraph>.*?)</p>)"
    r"(?:\n|\Z)"
)
ITEM = re.compile(r"<li>(?P<item>.*?)</li>")
CODE_SPAN = re.compile(r"<code>(?P<code>[^<]*)</code>")
LINK = re.compile(r'<a href="(?P<url>[^"]*)">(?P<text>.*?)</a>')
STRONG = re.compile(r"<strong>(?P<text>.*?)</strong>")


# ------------------------------------------------------------------- parsing


# TODO: The hint every markup refusal ends with: the page's markup is older
# (or newer) than unrender understands, and if the transcript store is
# intact, plain sourcery.py regenerates the page.
UNKNOWN_HINT = (
    "Forma paginae vetustior (vel recentior) est quam unrender intellegit.\n"
    "Si repositum transcriptorum integrum est, sourcery.py simpliciter curre."
)
# TODO: The place an article-level refusal names: "in article N".
ARTICLE = "articulo p{}"
# TODO: The place a page-level refusal names: "in the page body".
BODY = "corpore paginae"


def unknown_markup(where: str, excerpt: str) -> UserError:
    # TODO: Says markup unrender does not recognize was found at the named
    # place and shows the start of it, then the hint above.
    return UserError(f"Forma ignota in {where}: {excerpt[:400]!r}\n{UNKNOWN_HINT}")


def tiled(pattern: re.Pattern[str], text: str, where: str) -> list[re.Match[str]]:
    """Matches of pattern laid back to back over all of text."""
    matches: list[re.Match[str]] = []
    position = 0
    while position < len(text):
        match = pattern.match(text, position)
        if match is None:
            raise unknown_markup(where, text[position:])
        matches.append(match)
        position = match.end()
    return matches


def backticks(code: str, floor: int) -> str:
    """A backtick fence longer than any run inside code, so the fence can
    only close where the inverse puts it; at least floor long."""
    longest = max((len(run) for run in re.findall(r"`+", code)), default=0)
    return "`" * max(floor, longest + 1)


def unrender_inline(fragment: str, where: str) -> str:
    """Inverse of markdown_inline for one rendered line."""
    text = CODE_SPAN.sub(lambda m: backticks(m["code"], 1) + m["code"] + backticks(m["code"], 1), fragment)
    text = LINK.sub(r"[\g<text>](\g<url>)", text)
    text = STRONG.sub(r"**\g<text>**", text)
    if "<" in text:
        raise unknown_markup(where, text[text.index("<") :])
    return html.unescape(text)


def unrender_block(match: re.Match[str], where: str) -> str:
    """Inverse of one markdown_html block, in the spelling unrender
    canonically writes back (dash bullets, dotted numbers, ** for strong,
    backtick fences)."""
    match match.lastgroup:
        case "code":
            fence = backticks(match["code"], 3)
            language = html.unescape(match["language"] or "")
            return f"{fence}{language}\n{html.unescape(match['code'])}\n{fence}"
        case "heading":
            return "#" * (int(match["level"]) - 1) + " " + unrender_inline(match["heading"], where)
        case "items":
            marker = "-" if match["list"] == "ul" else "{}."
            return "\n".join(
                f"{marker.format(i)} {unrender_inline(item['item'], where)}"
                for i, item in enumerate(tiled(ITEM, match["items"], where), 1)
            )
        case "quote":
            return "\n".join(f"> {unrender_inline(part, where)}" for part in match["quote"].split("<br>"))
        case "paragraph":
            return "\n".join(unrender_inline(part, where) for part in match["paragraph"].split("<br>"))
        case other:
            raise AssertionError(other)


def unrender_markdown(rendered: str, where: str) -> str:
    """Inverse of markdown_html, proven: the result renders back to rendered."""
    text = "\n\n".join(unrender_block(block, where) for block in tiled(BLOCK, rendered, where))
    again = sourcery.markdown_html(text)
    if again != rendered:
        # TODO: Says the reply in this article cannot be reversed exactly:
        # the markdown recovered from it renders to something else, and
        # shows where the two first differ.
        raise UserError(
            f"Responsum in {where} exacte reverti non potest: markdown_html(reply) idem non reddit.\n"
            f"Pagina:     {differing_excerpt(rendered, again)!r}\n"
            f"Regenerata: {differing_excerpt(again, rendered)!r}"
        )
    return text


def differing_excerpt(text: str, other: str, width: int = 120) -> str:
    """text from the first character where it differs from other."""
    start = next((i for i, (a, b) in enumerate(zip(text, other)) if a != b), min(len(text), len(other)))
    return text[start : start + width]


def seconds(text: str | None, where: str) -> float:
    """Inverse of elapsed_text; 0.0 when the page shows no duration."""
    if text is None:
        return 0.0
    match = DURATION.fullmatch(text)
    assert match is not None, text
    # A duration sourcery would never write re-renders differently, which the
    # <main> proof in recover() reports.
    return float(sum(int(match[unit] or 0) * scale for unit, scale in (("h", 3600), ("m", 60), ("s", 1))))


def unrender_ballot(match: re.Match[str], where: str) -> Ballot:
    options = tiled(OPTION, match["options"], where)
    return Ballot(
        question=html.unescape(match["question"]),
        options=tuple(html.unescape(option[option.lastgroup]) for option in options),
        picked=tuple(html.unescape(option["picked"]) for option in options if option["picked"] is not None),
    )


def piece(pattern: re.Pattern[str], text: str, where: str) -> re.Match[str]:
    """pattern matched over all of text, or the text reported as unknown."""
    match = pattern.fullmatch(text)
    if match is None:
        raise unknown_markup(where, text)
    return match


def unrender_article(number: int, slug: str, inner: str, source: Path) -> Exchange:
    where = ARTICLE.format(number)
    frame = piece(FRAME, inner, where)
    head = piece(HEAD, frame["head"], where)
    summary = piece(SUMMARY, frame["summary"], where)
    return Exchange(
        timestamp=sourcery.parse_time(summary["when"]),
        provider=PROVIDERS[slug],
        model=html.unescape(summary["model"] or ""),
        session=UNRENDERED,
        prompt=html.unescape(head["prompt"] or ""),
        reply=unrender_markdown(frame["reply"], where),
        source=source,
        effort=html.unescape(summary["effort"] or ""),
        images=tuple(html.unescape(image["uri"]) for image in tiled(IMAGE, head["attachments"] or "", where)),
        elapsed=seconds(summary["elapsed"], where),
        wall=seconds(summary["wall"], where),
        ballots=tuple(unrender_ballot(ballot, where) for ballot in tiled(BALLOT, head["ballots"], where)),
        added=int((head["added"] or "0").replace(",", "")),
        deleted=int((head["deleted"] or "0").replace(",", "")),
    )


def skeleton(page: str, source: Path) -> re.Match[str]:
    """The page's frame: title, deck figures, and the <main> region."""
    match = PAGE.search(page)
    if match is None:
        # TODO: Says the file is not a sourcery page as unrender knows them
        # (sourcery may not have generated it at all), then the hint above.
        raise UserError(
            f"Pagina sourcery non agnoscitur (fortasse sourcery eam non generavit): {source}\n"
            f"{UNKNOWN_HINT}"
        )
    return match


def recover(repo: Path, source: Path) -> tuple[list[Exchange], str]:
    """Every exchange the page at source shows, proven exact: the list
    re-renders the page's <main> region byte for byte. Returns the list and
    that proven re-rendering, which is what gets written."""
    if not source.is_file():
        # TODO: Says the page to import was not found.
        raise UserError(f"Pagina non inventa: {source}")
    try:
        # Bytes, not text: universal newlines would rewrite a carriage return
        # someone typed, and the proof must compare what the file holds.
        page = source.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        # TODO: Says the page could not be read as a UTF-8 file, with the
        # reason.
        raise UserError(f"Pagina legi non potest: {source}\n{exc}") from exc
    if sourcery.SNAPSHOT_OPEN in page:
        # TODO: Says the page already carries a snapshot, so there is
        # nothing to import: plain sourcery.py reads it back by itself.
        raise UserError(f"Pagina iam snapshot fert: {source}\nNihil importandum: sourcery.py simpliciter curre.")
    frame = skeleton(page, source)
    title = html.unescape(frame["title"])
    if title != repo.name:
        # TODO: Says the page's title names a different project than the
        # directory given, shows both, and asks for the page's own project
        # directory — the guard against importing into the wrong repo.
        raise UserError(
            f"Titulus paginae directorio non congruit: pagina {title!r}, directorium {repo.name!r} ({repo}).\n"
            "Da directorium incepti cuius haec pagina est."
        )
    chunks = tiled(CHUNK, frame["body"], BODY)
    # Article numbering, the masthead's counts and totals, the minimap: all
    # derived from the exchanges by render(), so the one proof below covers
    # every one of them.
    exchanges = [
        unrender_article(int(chunk["number"]), chunk["slug"], chunk["inner"], source)
        for chunk in chunks
        if chunk["number"] is not None
    ]
    rendered = sourcery.render(repo, exchanges, sourcery.repo_remote(repo))
    again = skeleton(rendered, source)
    if again["main"] != frame["main"]:
        # TODO: Says the recovered exchanges do not regenerate the page's
        # <main> region exactly, and shows both sides from the first
        # differing character.
        raise UserError(
            "Rogationes receptae paginam exacte non regenerant; regio <main> differt.\n"
            f"Pagina:     {differing_excerpt(frame['main'], again['main'])!r}\n"
            f"Regenerata: {differing_excerpt(again['main'], frame['main'])!r}"
        )
    return exchanges, rendered


# ----------------------------------------------------------------------- exit


def run(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if len(args) != 2:
            raise UserError(USAGE)
        repo = sourcery.canonical_repo(Path(args[0]))
        source = Path(args[1]).resolve()
        exchanges, page = recover(repo, source)
        sourcery.write_output(source, page)
        # TODO: Reports success with the rewritten page's path and the
        # number of prompts recovered from it.
        print(f"Scriptum: {source}\nRogationes receptae: {len(exchanges)}")
        return 0
    except UserError as exc:
        print(f"Error:\n{exc}", file=sys.stderr)
        return 2


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
