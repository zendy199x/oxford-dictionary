#!/usr/bin/env python3
"""Build Oxford 3000/5000 full-word JSON from the PDF word lists."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader


PDFS = ("Oxford 3000.pdf", "Oxford 5000.pdf")
DEFAULT_FULL_WORD = "full-word.json"
DEFAULT_OUTPUT = "oxford-3000-5000-full-word.json"
DEFAULT_REPORT = "oxford-merge-report.json"
OXFORD_BASE = "https://www.oxfordlearnersdictionaries.com/definition/english"

LEVEL_RE = re.compile(r"\b[ABC][12]\b")
POS_TOKENS = sorted(
    (
        "indefinite article",
        "definite article",
        "infinitive marker",
        "linking verb",
        "auxiliary v.",
        "modal v.",
        "ordinal number",
        "number",
        "adj./adv.",
        "det./pron.",
        "conj./adv.",
        "n.",
        "v.",
        "adj.",
        "adv.",
        "prep.",
        "conj.",
        "det.",
        "pron.",
        "exclam.",
    ),
    key=len,
    reverse=True,
)
POS_TOKEN_RE = re.compile("|".join(re.escape(token) for token in POS_TOKENS))
SCAN_TOKEN_RE = re.compile(
    r"indefinite article|definite article|infinitive marker|linking verb|"
    r"auxiliary v\.|modal v\.|ordinal number|adj\./adv\.|det\./pron\.|"
    r"conj\./adv\.|number|n\.|v\.|adj\.|adv\.|prep\.|conj\.|det\.|"
    r"pron\.|exclam\.|[ABC][12]"
)

POS_MAP = {
    "n.": "noun",
    "v.": "verb",
    "adj.": "adjective",
    "adv.": "adverb",
    "prep.": "preposition",
    "conj.": "conjunction",
    "det.": "determiner",
    "pron.": "pronoun",
    "exclam.": "exclamation",
    "number": "number",
    "modal v.": "modal verb",
    "auxiliary v.": "auxiliary verb",
    "ordinal number": "ordinal number",
    "indefinite article": "indefinite article",
    "definite article": "definite article",
    "infinitive marker": "infinitive marker",
    "linking verb": "linking verb",
}

COMBINED_POS = {
    "adj./adv.": ("adj.", "adv."),
    "det./pron.": ("det.", "pron."),
    "conj./adv.": ("conj.", "adv."),
}

COMPATIBLE_TYPES = {
    "verb": {"verb", "modal verb", "auxiliary verb", "linking verb"},
    "number": {"number", "ordinal number"},
}

OXFORD_POS_MAP = {
    "adjective": "adjective",
    "adverb": "adverb",
    "auxiliary verb": "auxiliary verb",
    "conjunction": "conjunction",
    "definite article": "definite article",
    "determiner": "determiner",
    "exclamation": "exclamation",
    "indefinite article": "indefinite article",
    "infinitive marker": "infinitive marker",
    "linking verb": "linking verb",
    "modal verb": "modal verb",
    "noun": "noun",
    "number": "number",
    "ordinal number": "ordinal number",
    "preposition": "preposition",
    "pronoun": "pronoun",
    "verb": "verb",
}


@dataclass(frozen=True)
class ParsedEntry:
    word: str
    type: str
    level: str
    source_pdf: str
    source_head: str
    source_line: str
    order: int

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.word, self.type, self.level)


def canonical_word(value: str) -> str:
    """Normalize PDF-only sense labels while keeping the dictionary word."""
    value = " ".join(value.replace("\xa0", " ").split())
    value = value.split(",", 1)[0].strip()
    value = re.sub(r"\s*\([^)]*\)\s*", " ", value).strip()
    value = re.sub(r"\s+\d+$", "", value).strip()
    value = re.sub(r"(?<=[A-Za-z])\d+$", "", value).strip()
    return " ".join(value.split())


def json_word_key(value: str) -> str:
    return value.strip().casefold()


def is_compatible_type(requested: str, candidate: str) -> bool:
    if requested == candidate:
        return True
    return candidate in COMPATIBLE_TYPES.get(requested, set())


def better_type(requested: str, candidate: str) -> str:
    if requested != candidate and is_compatible_type(requested, candidate):
        return candidate
    return requested


def first_pos_after_head(line: str) -> re.Match[str] | None:
    for match in POS_TOKEN_RE.finditer(line):
        if match.start() == 0:
            continue
        if line[: match.start()].strip(" ,"):
            return match
    return None


def starts_with_pos_continuation(line: str) -> bool:
    match = POS_TOKEN_RE.match(line)
    if not match:
        return False
    # "number n. A1" is the word "number", not a continuation line.
    if match.group(0) == "number":
        rest = line[match.end() :]
        if re.match(r"\s+(n\.|v\.|adj\.|adv\.|prep\.|conj\.|det\.|pron\.|exclam\.)", rest):
            return False
    return True


def extract_pdf_lines(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    text = text.replace("\xa0", " ")
    text = re.sub(r"© Oxford University Press\s*\d+\s*/\s*\d+\s*", "\n", text)
    text = re.sub(r"The Oxford (?:3000|5000)™", "\n", text)
    text = re.sub(r"(?<=[ABC][12])(?=[A-Za-z0-9])", "\n", text)

    raw_lines: list[str] = []
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if line and (first_pos_after_head(line) or starts_with_pos_continuation(line)):
            raw_lines.append(line)

    merged: list[str] = []
    buffer = ""
    for line in raw_lines:
        if not buffer:
            buffer = line
            continue
        if starts_with_pos_continuation(line) or not LEVEL_RE.search(buffer):
            buffer = f"{buffer.rstrip(' /')} {line}"
        else:
            merged.append(buffer)
            buffer = line
    if buffer:
        merged.append(buffer)
    return merged


def expand_pos(token: str) -> tuple[str, ...]:
    return COMBINED_POS.get(token, (token,))


def parse_pdf_line(line: str, source_pdf: str, order_start: int) -> list[ParsedEntry]:
    match = first_pos_after_head(line)
    if not match:
        return []

    source_head = line[: match.start()].strip()
    word = canonical_word(source_head)
    rest = line[match.start() :]
    pending: list[str] = []
    parsed: list[ParsedEntry] = []
    order = order_start

    for token in SCAN_TOKEN_RE.findall(rest):
        if LEVEL_RE.fullmatch(token):
            for pos_token in pending:
                parsed.append(
                    ParsedEntry(
                        word=word,
                        type=POS_MAP[pos_token],
                        level=token,
                        source_pdf=source_pdf,
                        source_head=source_head,
                        source_line=line,
                        order=order,
                    )
                )
                order += 1
            pending = []
        else:
            pending.extend(expand_pos(token))

    return parsed


def parse_pdfs(root: Path) -> tuple[list[ParsedEntry], dict[str, Any]]:
    parsed: list[ParsedEntry] = []
    report: dict[str, Any] = {"pdfs": {}, "duplicates": []}

    for pdf_name in PDFS:
        lines = extract_pdf_lines(root / pdf_name)
        before = len(parsed)
        for line in lines:
            parsed.extend(parse_pdf_line(line, pdf_name, len(parsed)))
        report["pdfs"][pdf_name] = {
            "lines": len(lines),
            "items": len(parsed) - before,
        }

    grouped: dict[tuple[str, str, str], list[ParsedEntry]] = defaultdict(list)
    for item in parsed:
        grouped[item.key].append(item)

    unique: list[ParsedEntry] = []
    for item in parsed:
        if grouped[item.key][0] == item:
            unique.append(item)

    report["raw_items"] = len(parsed)
    report["unique_items"] = len(unique)
    report["duplicates"] = [
        {
            "word": key[0],
            "type": key[1],
            "level": key[2],
            "sources": [
                {
                    "pdf": duplicate.source_pdf,
                    "head": duplicate.source_head,
                    "line": duplicate.source_line,
                }
                for duplicate in duplicates
            ],
        }
        for key, duplicates in grouped.items()
        if len(duplicates) > 1
    ]
    return unique, report


def load_full_word(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    indexes: dict[str, Any] = {
        "exact": defaultdict(list),
        "word_type": defaultdict(list),
        "word_level": defaultdict(list),
        "word": defaultdict(list),
    }
    for item in data:
        value = item["value"]
        word = json_word_key(value.get("word", ""))
        item_type = value.get("type", "")
        level = value.get("level", "")
        indexes["exact"][(word, item_type, level)].append(item)
        indexes["word_type"][(word, item_type)].append(item)
        indexes["word_level"][(word, level)].append(item)
        indexes["word"][word].append(item)
    return data, indexes


def clone_item(item: dict[str, Any], level: str | None = None, item_type: str | None = None) -> dict[str, Any]:
    cloned = copy.deepcopy(item)
    if level is not None:
        cloned["value"]["level"] = level
    if item_type is not None:
        cloned["value"]["type"] = item_type
    return cloned


def make_empty_item(word: str, item_type: str, level: str, href: str = "") -> dict[str, Any]:
    return {
        "id": -1,
        "value": {
            "word": word,
            "href": href,
            "type": item_type,
            "level": level,
            "us": {"mp3": "", "ogg": ""},
            "uk": {"mp3": "", "ogg": ""},
            "phonetics": {"us": "", "uk": ""},
            "examples": [],
        },
    }


def has_invalid_media(item: dict[str, Any]) -> bool:
    value = item.get("value", {})
    for side in ("us", "uk"):
        media = value.get(side, {})
        for kind in ("mp3", "ogg"):
            url = media.get(kind, "")
            if isinstance(url, str) and "undefined" in url:
                return True
    return False


def normalize_text(value: str) -> str:
    value = " ".join(value.split())
    value = value.replace(" .", ".").replace(" ,", ",").replace(" ;", ";").replace(" :", ":")
    value = value.replace("( ", "(").replace(" )", ")")
    return value.strip()


def slug_for_word(word: str) -> str:
    slug = word.casefold()
    slug = slug.replace("\u2019", "-").replace("'", "-")
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"[^a-z0-9-]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def pos_text_to_types(pos_text: str) -> list[str]:
    cleaned = pos_text.casefold()
    cleaned = cleaned.replace("/", ",")
    cleaned = cleaned.replace(" and ", ",")
    cleaned = re.sub(r"\s*,\s*", ",", cleaned)
    types: list[str] = []
    for part in cleaned.split(","):
        part = normalize_text(part)
        if part in OXFORD_POS_MAP:
            types.append(OXFORD_POS_MAP[part])
    return types


def cefr_levels_in(element: Any) -> set[str]:
    levels: set[str] = set()
    for descendant in element.find_all(True):
        for class_name in descendant.get("class", []):
            match = re.search(r"ox(?:3k|5k)sym_([abc][12])", class_name)
            if match:
                levels.add(match.group(1).upper())
    return levels


def entry_phonetics(entry: Any) -> dict[str, str]:
    uk = entry.select_one(".phons_br .phon")
    us = entry.select_one(".phons_n_am .phon")
    all_phons = [normalize_text(node.get_text(" ", strip=True)) for node in entry.select(".phon")]
    return {
        "us": normalize_text(us.get_text(" ", strip=True)) if us else (all_phons[-1] if all_phons else ""),
        "uk": normalize_text(uk.get_text(" ", strip=True)) if uk else (all_phons[0] if all_phons else ""),
    }


def entry_audio(entry: Any) -> tuple[dict[str, str], dict[str, str]]:
    us = {"mp3": "", "ogg": ""}
    uk = {"mp3": "", "ogg": ""}
    for node in entry.select(".sound.audio_play_button"):
        mp3 = node.get("data-src-mp3") or ""
        ogg = node.get("data-src-ogg") or ""
        classes = " ".join(node.get("class", []))
        target: dict[str, str] | None = None
        if "/us_pron" in mp3 or "pron-us" in classes or "us" in classes:
            target = us
        elif "/uk_pron" in mp3 or "pron-uk" in classes or "uk" in classes:
            target = uk
        if target is not None:
            target["mp3"] = mp3
            target["ogg"] = ogg
    return us, uk


def examples_for_level(entry: Any, level: str) -> list[str]:
    examples: list[str] = []
    for sense in entry.select(".sense"):
        if level in cefr_levels_in(sense):
            examples.extend(normalize_text(node.get_text(" ", strip=True)) for node in sense.select(".x"))
    if not examples:
        examples = [normalize_text(node.get_text(" ", strip=True)) for node in entry.select(".x")]

    deduped: list[str] = []
    seen: set[str] = set()
    for example in examples:
        if example and example not in seen:
            seen.add(example)
            deduped.append(example)
    return deduped[:12]


def parse_entry_from_html(
    html: str,
    final_url: str,
    requested_word: str,
    requested_type: str,
    requested_level: str,
) -> dict[str, Any] | None:
    soup = BeautifulSoup(html, "lxml")
    requested_key = json_word_key(requested_word)

    best: tuple[int, dict[str, Any]] | None = None
    for entry in soup.select(".entry"):
        headword_node = entry.select_one(".webtop .headword")
        pos_node = entry.select_one(".webtop .pos")
        if not headword_node or not pos_node:
            continue

        headword = normalize_text(headword_node.get_text(" ", strip=True))
        headword_key = json_word_key(canonical_word(headword))
        exact_word_match = headword_key == requested_key
        singular_headword_match = requested_key == f"{headword_key}s"
        if not exact_word_match and not singular_headword_match:
            continue

        entry_types = pos_text_to_types(pos_node.get_text(" ", strip=True))
        compatible = [item_type for item_type in entry_types if is_compatible_type(requested_type, item_type)]
        exact_type = requested_type in entry_types
        if not compatible and not exact_type:
            continue

        resolved_type = requested_type if exact_type else better_type(requested_type, compatible[0])
        entry_levels = cefr_levels_in(entry)
        level_score = 2 if requested_level in entry_levels else 0
        type_score = 2 if exact_type else 1
        examples = examples_for_level(entry, requested_level)
        example_score = 1 if examples else 0
        score = level_score + type_score + example_score
        us, uk = entry_audio(entry)

        item = {
            "id": -1,
                "value": {
                    "word": headword if exact_word_match else requested_word,
                    "href": final_url,
                    "type": resolved_type,
                    "level": requested_level,
                "us": us,
                "uk": uk,
                "phonetics": entry_phonetics(entry),
                "examples": examples,
            },
        }
        if best is None or score > best[0]:
            best = (score, item)

    return best[1] if best else None


class OxfordCrawler:
    def __init__(self, timeout: float, delay: float) -> None:
        self.timeout = timeout
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                )
            }
        )
        self.cache: dict[str, tuple[int, str, str]] = {}
        self.errors: list[dict[str, Any]] = []

    def get(self, url: str) -> tuple[int, str, str]:
        if url in self.cache:
            return self.cache[url]
        try:
            response = self.session.get(url, timeout=self.timeout)
            result = (response.status_code, response.url, response.text)
        except requests.RequestException as exc:
            self.errors.append({"url": url, "error": str(exc)})
            result = (0, url, "")
        self.cache[url] = result
        if self.delay:
            time.sleep(self.delay)
        return result

    def crawl(
        self,
        parsed: ParsedEntry,
        indexes: dict[str, Any],
        suffix_limit: int,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        candidates = candidate_urls(parsed, indexes, suffix_limit)
        for url in candidates:
            status, final_url, html = self.get(url)
            if status != 200 or not html:
                continue
            item = parse_entry_from_html(html, final_url, parsed.word, parsed.type, parsed.level)
            if item is not None:
                return item, candidates
        return None, candidates


def candidate_urls(parsed: ParsedEntry, indexes: dict[str, Any], suffix_limit: int) -> list[str]:
    urls: list[str] = []
    word_key = json_word_key(parsed.word)
    for item in indexes["word"].get(word_key, []):
        href = item["value"].get("href")
        if href:
            urls.append(href)

    slug = quote(slug_for_word(parsed.word))
    slugs: list[str] = []
    if slug:
        slugs.append(slug)
        if slug.endswith("s") and not slug.endswith("ss"):
            slugs.append(slug[:-1])
    for slug_item in slugs:
        urls.append(f"{OXFORD_BASE}/{slug_item}")
        for suffix in range(1, suffix_limit + 1):
            urls.append(f"{OXFORD_BASE}/{slug_item}_{suffix}")

    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def local_exact_match(parsed: ParsedEntry, indexes: dict[str, Any]) -> dict[str, Any] | None:
    matches = indexes["exact"].get((json_word_key(parsed.word), parsed.type, parsed.level), [])
    if len(matches) == 1:
        return clone_item(matches[0])
    return None


def local_loose_word_type(parsed: ParsedEntry, indexes: dict[str, Any]) -> dict[str, Any] | None:
    matches = indexes["word_type"].get((json_word_key(parsed.word), parsed.type), [])
    if len(matches) == 1:
        return clone_item(matches[0], level=parsed.level)
    return None


def local_compatible_word_level(
    parsed: ParsedEntry, indexes: dict[str, Any]
) -> tuple[dict[str, Any], str] | None:
    matches = indexes["word_level"].get((json_word_key(parsed.word), parsed.level), [])
    compatible = [
        item
        for item in matches
        if is_compatible_type(parsed.type, item["value"].get("type", ""))
    ]
    if len(compatible) == 1:
        item_type = compatible[0]["value"].get("type", parsed.type)
        return clone_item(compatible[0], level=parsed.level, item_type=item_type), item_type
    return None


def local_single_word_level_correction(
    parsed: ParsedEntry,
    indexes: dict[str, Any],
    emitted_keys: set[tuple[str, str, str]],
) -> tuple[dict[str, Any], str] | None:
    matches = indexes["word_level"].get((json_word_key(parsed.word), parsed.level), [])
    if len(matches) == 1:
        # A single Oxford/full-word entry for the same word and level is a clear
        # canonical-type correction for PDF extraction typos such as "seldom n.".
        item_type = matches[0]["value"].get("type", parsed.type)
        final_key = (canonical_word(matches[0]["value"].get("word", parsed.word)), item_type, parsed.level)
        if final_key in emitted_keys:
            return None
        return clone_item(matches[0], level=parsed.level, item_type=item_type), item_type
    return None


def word_type_has_multiple_pdf_levels(parsed: ParsedEntry, levels_by_word_type: dict[tuple[str, str], set[str]]) -> bool:
    return len(levels_by_word_type[(parsed.word, parsed.type)]) > 1


def should_crawl_before_loose(
    parsed: ParsedEntry,
    exact_item: dict[str, Any] | None,
    levels_by_word_type: dict[tuple[str, str], set[str]],
) -> bool:
    if exact_item is not None:
        return False
    return word_type_has_multiple_pdf_levels(parsed, levels_by_word_type)


def resolve_items(
    parsed_entries: list[ParsedEntry],
    indexes: dict[str, Any],
    crawler: OxfordCrawler,
    suffix_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    levels_by_word_type: dict[tuple[str, str], set[str]] = defaultdict(set)
    for parsed in parsed_entries:
        levels_by_word_type[(parsed.word, parsed.type)].add(parsed.level)

    output: list[dict[str, Any]] = []
    emitted_keys: set[tuple[str, str, str]] = set()
    report: dict[str, Any] = {
        "matches": {
            "exact": 0,
            "loose_word_type": 0,
            "compatible_word_level": 0,
            "crawled": 0,
            "placeholder_unresolved": 0,
        },
        "level_overrides": [],
        "type_corrections": [],
        "crawled": [],
        "unresolved": [],
    }

    for parsed in parsed_entries:
        exact_item = local_exact_match(parsed, indexes)
        item: dict[str, Any] | None = None
        last_tried_urls: list[str] = []
        source = ""

        if exact_item is not None:
            item = exact_item
            source = "exact"
            report["matches"]["exact"] += 1
            if has_invalid_media(item):
                refreshed, tried_urls = crawler.crawl(parsed, indexes, suffix_limit)
                last_tried_urls = tried_urls
                if refreshed is not None:
                    item = refreshed
                    source = "crawled"
                    report["matches"]["exact"] -= 1
                    report["matches"]["crawled"] += 1
                    report["crawled"].append(
                        {
                            "word": parsed.word,
                            "type": parsed.type,
                            "level": parsed.level,
                            "href": item["value"].get("href", ""),
                            "reason": "invalid_local_media",
                        }
                    )
        elif should_crawl_before_loose(parsed, exact_item, levels_by_word_type):
            item, tried_urls = crawler.crawl(parsed, indexes, suffix_limit)
            last_tried_urls = tried_urls
            if item is not None:
                source = "crawled"
                report["matches"]["crawled"] += 1
                report["crawled"].append(
                    {
                        "word": parsed.word,
                        "type": parsed.type,
                        "level": parsed.level,
                        "href": item["value"].get("href", ""),
                        "reason": "multi_level_word_type",
                    }
                )

        if item is None:
            loose = local_loose_word_type(parsed, indexes)
            if loose is not None:
                original_level = loose["value"].get("level", "")
                item = loose
                source = "loose_word_type"
                report["matches"]["loose_word_type"] += 1
                if original_level != parsed.level:
                    report["level_overrides"].append(
                        {
                            "word": parsed.word,
                            "type": parsed.type,
                            "from": original_level,
                            "to": parsed.level,
                        }
                    )
                if has_invalid_media(item):
                    refreshed, tried_urls = crawler.crawl(parsed, indexes, suffix_limit)
                    last_tried_urls = tried_urls
                    if refreshed is not None:
                        item = refreshed
                        source = "crawled"
                        report["matches"]["loose_word_type"] -= 1
                        report["matches"]["crawled"] += 1
                        report["crawled"].append(
                            {
                                "word": parsed.word,
                                "type": parsed.type,
                                "level": parsed.level,
                                "href": item["value"].get("href", ""),
                                "reason": "invalid_local_media",
                            }
                        )

        if item is None:
            compatible = local_compatible_word_level(parsed, indexes)
            if compatible is not None:
                item, resolved_type = compatible
                source = "compatible_word_level"
                report["matches"]["compatible_word_level"] += 1
                if resolved_type != parsed.type:
                    report["type_corrections"].append(
                        {
                            "word": parsed.word,
                            "level": parsed.level,
                            "from": parsed.type,
                            "to": resolved_type,
                            "reason": "single_local_word_level_candidate",
                        }
                    )

        if item is None:
            item, tried_urls = crawler.crawl(parsed, indexes, suffix_limit)
            last_tried_urls = tried_urls
            if item is not None:
                source = "crawled"
                report["matches"]["crawled"] += 1
                if item["value"].get("type") != parsed.type:
                    report["type_corrections"].append(
                        {
                            "word": parsed.word,
                            "level": parsed.level,
                            "from": parsed.type,
                            "to": item["value"].get("type", ""),
                            "reason": "crawler_canonical_type",
                        }
                    )
                report["crawled"].append(
                    {
                        "word": parsed.word,
                        "type": parsed.type,
                        "level": parsed.level,
                        "href": item["value"].get("href", ""),
                        "reason": "missing_local_match",
                    }
                )

        if item is None:
            correction = local_single_word_level_correction(parsed, indexes, emitted_keys)
            if correction is not None:
                item, resolved_type = correction
                source = "compatible_word_level"
                report["matches"]["compatible_word_level"] += 1
                if resolved_type != parsed.type:
                    report["type_corrections"].append(
                        {
                            "word": parsed.word,
                            "level": parsed.level,
                            "from": parsed.type,
                            "to": resolved_type,
                            "reason": "single_local_word_level_candidate_after_crawl",
                        }
                    )

        if item is None:
            href = candidate_urls(parsed, indexes, suffix_limit)[0] if parsed.word else ""
            item = make_empty_item(parsed.word, parsed.type, parsed.level, href)
            source = "placeholder_unresolved"
            report["matches"]["placeholder_unresolved"] += 1
            if not last_tried_urls:
                last_tried_urls = candidate_urls(parsed, indexes, suffix_limit)
            report["unresolved"].append(unresolved_report_item(parsed, last_tried_urls))

        item["id"] = len(output)
        item["value"]["word"] = canonical_word(item["value"].get("word") or parsed.word)
        item["value"]["level"] = parsed.level
        item.setdefault("_source", source)
        output.append(item)
        emitted_keys.add(
            (item["value"].get("word", ""), item["value"].get("type", ""), item["value"].get("level", ""))
        )

    for item in output:
        item.pop("_source", None)

    report["crawl_errors"] = crawler.errors
    return output, report


def unresolved_report_item(parsed: ParsedEntry, tried_urls: list[str]) -> dict[str, Any]:
    return {
        "word": parsed.word,
        "type": parsed.type,
        "level": parsed.level,
        "source_pdf": parsed.source_pdf,
        "source_head": parsed.source_head,
        "source_line": parsed.source_line,
        "tried_urls": tried_urls,
    }


def validate_output(items: list[dict[str, Any]]) -> dict[str, Any]:
    required = ("word", "href", "type", "level", "us", "uk", "phonetics", "examples")
    missing_required: list[dict[str, Any]] = []
    keys: dict[tuple[str, str, str], int] = {}
    duplicates: list[dict[str, Any]] = []

    for item in items:
        value = item.get("value", {})
        for field in required:
            if field not in value:
                missing_required.append({"id": item.get("id"), "field": field})
        key = (value.get("word", ""), value.get("type", ""), value.get("level", ""))
        if key in keys:
            duplicates.append({"first_id": keys[key], "second_id": item.get("id"), "key": key})
        else:
            keys[key] = item.get("id", -1)

    return {
        "items": len(items),
        "missing_required": missing_required,
        "duplicate_word_type_level": duplicates,
    }


def build(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    parsed_entries, parse_report = parse_pdfs(root)
    _, indexes = load_full_word(root / args.full_word)
    crawler = OxfordCrawler(timeout=args.timeout, delay=args.delay)
    output, resolve_report = resolve_items(parsed_entries, indexes, crawler, args.suffix_limit)

    report = {
        "input": {
            "pdfs": list(PDFS),
            "full_word": args.full_word,
        },
        "parsed": parse_report,
        "resolved": resolve_report,
        "validation": validate_output(output),
    }

    output_path = root / args.output
    report_path = root / args.report
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=4)
        handle.write("\n")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=4)
        handle.write("\n")

    print(f"Wrote {len(output)} items to {output_path.name}")
    print(f"Wrote report to {report_path.name}")
    print(json.dumps(report["resolved"]["matches"], ensure_ascii=False, sort_keys=True))
    print(json.dumps(report["validation"], ensure_ascii=False, sort_keys=True))
    if report["resolved"]["unresolved"]:
        print(f"Unresolved entries: {len(report['resolved']['unresolved'])}", file=sys.stderr)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--full-word", default=DEFAULT_FULL_WORD, help="Existing full-word JSON")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Generated full-word JSON")
    parser.add_argument("--report", default=DEFAULT_REPORT, help="Generated merge report JSON")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between uncached HTTP requests")
    parser.add_argument("--suffix-limit", type=int, default=8, help="Maximum Oxford URL suffix to try")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(build(parse_args()))
