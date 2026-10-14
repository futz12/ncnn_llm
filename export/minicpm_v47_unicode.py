#!/usr/bin/env python3
"""Export the installed Hugging Face tokenizers backend's NFC data.

The Rust normalizer and Python's unicodedata can use *different* Unicode
versions. In particular, tokenizers 0.23.3 does not recognize several marks
that Python Unicode 15.0 recognizes. Never generate these tables using
unicodedata.normalize(), decomposition(), or combining() alone.

This module probes every Unicode scalar with tokenizers.normalizers.NFD.
Combining-class membership, equality and ordering are also probed through
that backend. Python's combining() supplies only the conventional numeric
labels for already-proven equivalence classes; missing or inconsistent
labels cause an error. NFC composition pairs are verified with backend NFC.

Records contain decimal integers, without a header or footer:
  D codepoint length fully_decomposed_codepoints...
  C codepoint canonical_combining_class
  P starter combining composed

Hangul is omitted and must be handled algorithmically. C records need only
cover NFD-invariant scalars: consumers must fully decompose before looking
up classes. Absent D/P entries mean no decomposition/composition; absent C
entries have class zero. The export needs tokenizers, but the consumer needs
no Unicode library. Normal batches contain at most 4,096 scalar probes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import random
import sys
from typing import TextIO
import unicodedata

import tokenizers
from tokenizers.normalizers import NFC, NFD


SBASE, LBASE, VBASE, TBASE = 0xAC00, 0x1100, 0x1161, 0x11A7
LCOUNT, VCOUNT, TCOUNT = 19, 21, 28
NCOUNT = VCOUNT * TCOUNT
SCOUNT = LCOUNT * NCOUNT


@dataclass
class NfcTables:
    decompositions: dict[int, tuple[int, ...]]
    combining_classes: dict[int, int]
    compositions: dict[tuple[int, int], int]
    backend_version: str

    def records(self):
        for cp, decomposition in sorted(self.decompositions.items()):
            yield "D {} {} {}\n".format(
                cp, len(decomposition), " ".join(map(str, decomposition))
            )
        for cp, combining_class in sorted(self.combining_classes.items()):
            yield f"C {cp} {combining_class}\n"
        for (starter, combining), composed in sorted(self.compositions.items()):
            yield f"P {starter} {combining} {composed}\n"

    def summary(self):
        digest = hashlib.sha256()
        size = 0
        for record in self.records():
            data = record.encode("ascii")
            digest.update(data)
            size += len(data)
        return {
            "tokenizers_version": self.backend_version,
            "class_label_unicodedata_version": unicodedata.unidata_version,
            "decompositions": len(self.decompositions),
            "nonzero_classes": len(self.combining_classes),
            "distinct_nonzero_classes": len(set(self.combining_classes.values())),
            "compositions": len(self.compositions),
            "record_bytes": size,
            "records_sha256": digest.hexdigest(),
        }


def _scalar_chunks(batch_size):
    for first in range(0, 0x110000, batch_size):
        yield [
            cp for cp in range(first, min(first + batch_size, 0x110000))
            if cp != 0 and not 0xD800 <= cp <= 0xDFFF
        ]


def _normalize_batch(normalizer, strings):
    """NUL is a verified class-zero separator, never present in probes."""
    if not strings:
        return []
    results = normalizer.normalize_str("\0".join(strings)).split("\0")
    if len(results) != len(strings):
        raise RuntimeError("The normalizer changed a probe separator")
    return results


def derive_nfc_tables(batch_size: int = 4096) -> NfcTables:
    """Read the backend's canonical normalization behavior, bounded in RAM."""
    if not 1 <= batch_size <= 65536:
        raise ValueError("batch_size must be between 1 and 65536")
    nfd, nfc = NFD(), NFC()
    for normalizer in (nfd, nfc):
        if normalizer.normalize_str("\u0345\0\u0334") != "\u0345\0\u0334":
            raise RuntimeError("NUL is not a canonical-ordering barrier")
    if nfd.normalize_str("\u0345\u0334") != "\u0334\u0345":
        raise RuntimeError("The combining-class probe anchors changed")

    decompositions = {}
    nonzero = []
    for scalars in _scalar_chunks(batch_size):
        # Hangul's 11,172 decompositions are represented by arithmetic.
        scalars = [cp for cp in scalars if not SBASE <= cp < SBASE + SCOUNT]
        decomposed = _normalize_batch(nfd, [chr(cp) for cp in scalars])
        invariant = []
        for cp, result in zip(scalars, decomposed):
            if result == chr(cp):
                invariant.append(cp)
            else:
                decompositions[cp] = tuple(map(ord, result))

        # A class-zero scalar blocks movement between these two marks.
        # Any nonzero scalar allows the class-1 mark to move leftward.
        probes = ["\u0345" + chr(cp) + "\u0334" for cp in invariant]
        for cp, probe, result in zip(invariant, probes, _normalize_batch(nfd, probes)):
            if probe != result:
                nonzero.append(cp)

    # Sort solely with backend canonical ordering. Stable reverse-pair
    # probes distinguish class equality from strict increasing order.
    ordered = list(map(ord, nfd.normalize_str("".join(map(chr, nonzero)))))
    if Counter(ordered) != Counter(nonzero):
        raise RuntimeError("Class probes were not NFD-invariant")
    groups = []
    for cp in ordered:
        if not groups:
            groups.append([cp])
            continue
        previous = groups[-1][-1]
        reverse = chr(cp) + chr(previous)
        normalized = nfd.normalize_str(reverse)
        if normalized == reverse:
            groups[-1].append(cp)
        elif normalized == chr(previous) + chr(cp):
            groups.append([cp])
        else:
            raise RuntimeError("Unexpected canonical-class comparison")

    classes = {}
    last_label = 0
    for group in groups:
        # Numeric labels are incidental; NFC uses only equality, ordering,
        # and zero. Require at least one known representative per group.
        labels = {unicodedata.combining(chr(cp)) for cp in group} - {0}
        if len(labels) != 1:
            raise RuntimeError(
                "Cannot label backend combining class containing U+{:04X}; "
                "Python Unicode {} is incompatible".format(group[0], unicodedata.unidata_version)
            )
        label = labels.pop()
        if label <= last_label:
            raise RuntimeError("Python class labels disagree with backend ordering")
        last_label = label
        classes.update((cp, label) for cp in group)

    compositions = {}
    for cp, decomposition in decompositions.items():
        if len(decomposition) < 2 or nfc.normalize_str(chr(cp)) != chr(cp):
            # Includes canonical singletons and composition exclusions.
            continue
        starter = nfc.normalize_str("".join(map(chr, decomposition[:-1])))
        combining = chr(decomposition[-1])
        if len(starter) != 1 or nfc.normalize_str(starter + combining) != chr(cp):
            raise RuntimeError(f"Cannot derive canonical composition for U+{cp:04X}")
        pair = (ord(starter), ord(combining))
        if pair in compositions and compositions[pair] != cp:
            raise RuntimeError("Conflicting canonical compositions")
        compositions[pair] = cp

    for decomposition in decompositions.values():
        if any(cp in decompositions or SBASE <= cp < SBASE + SCOUNT for cp in decomposition):
            raise RuntimeError("Backend returned a nonrecursive decomposition")
    return NfcTables(decompositions, classes, compositions, tokenizers.__version__)


def write_nfc_records(stream: TextIO, tables: NfcTables | None = None) -> NfcTables:
    """Append D/C/P records to an existing text artifact and return tables."""
    if tables is None:
        tables = derive_nfc_tables()
    stream.writelines(tables.records())
    return tables


def normalize_with_tables(text: str, tables: NfcTables) -> str:
    """Small reference consumer used by export verification, not runtime."""
    decomposed = []
    for char in text:
        cp = ord(char)
        index = cp - SBASE
        if 0 <= index < SCOUNT:
            decomposed.extend((LBASE + index // NCOUNT, VBASE + index % NCOUNT // TCOUNT))
            if index % TCOUNT:
                decomposed.append(TBASE + index % TCOUNT)
        else:
            decomposed.extend(tables.decompositions.get(cp, (cp,)))

    # Stable sort each nonstarter run. Leading nonstarters form a run too.
    run = 0
    for index in range(len(decomposed) + 1):
        if index == len(decomposed) or tables.combining_classes.get(decomposed[index], 0) == 0:
            decomposed[run:index] = sorted(
                decomposed[run:index], key=lambda cp: tables.combining_classes.get(cp, 0)
            )
            run = index + 1

    composed = []
    starter_index = None
    last_class = 0
    for cp in decomposed:
        combining_class = tables.combining_classes.get(cp, 0)
        replacement = None
        if starter_index is not None and (last_class == 0 or last_class < combining_class):
            starter = composed[starter_index]
            if LBASE <= starter < LBASE + LCOUNT and VBASE <= cp < VBASE + VCOUNT:
                replacement = SBASE + ((starter - LBASE) * VCOUNT + cp - VBASE) * TCOUNT
            elif (SBASE <= starter < SBASE + SCOUNT and
                  (starter - SBASE) % TCOUNT == 0 and TBASE < cp < TBASE + TCOUNT):
                replacement = starter + cp - TBASE
            else:
                replacement = tables.compositions.get((starter, cp))
        if replacement is not None:
            composed[starter_index] = replacement
            # A consumed combining mark does not update the blocking class.
        else:
            if combining_class == 0:
                starter_index = len(composed)
            composed.append(cp)
            last_class = combining_class
    return "".join(map(chr, composed))


def verify_nfc_tables(tables: NfcTables, batch_size: int = 4096) -> dict[str, int]:
    """Check all scalars, all class-order pairs, and seeded mixed strings."""
    nfc = NFC()
    checked = 0

    def check(strings):
        nonlocal checked
        expected = _normalize_batch(nfc, strings)
        for text, result in zip(strings, expected):
            actual = normalize_with_tables(text, tables)
            if actual != result:
                raise AssertionError(
                    f"NFC mismatch: {text.encode('unicode_escape')!r}: "
                    f"{actual.encode('unicode_escape')!r} != {result.encode('unicode_escape')!r}"
                )
        checked += len(strings)

    if normalize_with_tables("\0", tables) != nfc.normalize_str("\0"):
        raise AssertionError("NUL mismatch")
    checked += 1
    for scalars in _scalar_chunks(batch_size):
        check([chr(cp) for cp in scalars])
    scalar_checks = checked

    representatives = {}
    for cp, label in tables.combining_classes.items():
        representatives.setdefault(label, cp)
    strings = []
    for cp in tables.combining_classes:
        for representative in representatives.values():
            for a, b in ((cp, representative), (representative, cp)):
                strings.append("A" + chr(a) + chr(b))
                if len(strings) >= batch_size:
                    check(strings)
                    strings.clear()
    check(strings)
    class_checks = checked - scalar_checks

    randomizer = random.Random(0x4D43504D)
    pool = list(tables.decompositions) + list(tables.combining_classes)
    pool += list(range(LBASE, LBASE + LCOUNT)) + list(range(VBASE, VBASE + VCOUNT))
    pool += list(range(TBASE + 1, TBASE + TCOUNT)) + list(range(SBASE, SBASE + SCOUNT))
    # Include newer Unicode scalars deliberately treated as class zero by
    # older tokenizers backends, plus ASCII controls and noncharacters.
    pool += [0x7FD, 0x898, 0x11938, 0x16FF0, 0x1E4EC, 0x10FFFF, 10, 13, 32, 65, 122]
    strings = ["", "\u0301\u0323", "A\u030A\u0301", "A\u0305\u0301",
               "\u1100\u1161\u11A8", "A\u0345\u07FD\u0334", "\U00011938"]
    for _ in range(20000):
        strings.append("".join(chr(randomizer.choice(pool)) for _ in range(randomizer.randrange(1, 17))))
        if len(strings) >= batch_size:
            check(strings)
            strings.clear()
    check(strings)
    return {"scalar_checks": scalar_checks, "class_pair_checks": class_checks,
            "mixed_string_checks": checked - scalar_checks - class_checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Write D/C/P records here; default: stdout")
    parser.add_argument("--verify", action="store_true", help="Check exhaustive scalar and mixed NFC parity")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    tables = derive_nfc_tables(args.batch_size)
    summary = tables.summary()
    if args.verify:
        summary["verification"] = verify_nfc_tables(tables, args.batch_size)
    if args.output:
        with open(args.output, "w", encoding="ascii", newline="\n") as stream:
            write_nfc_records(stream, tables)
    else:
        write_nfc_records(sys.stdout, tables)
    print(json.dumps(summary, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    main()
