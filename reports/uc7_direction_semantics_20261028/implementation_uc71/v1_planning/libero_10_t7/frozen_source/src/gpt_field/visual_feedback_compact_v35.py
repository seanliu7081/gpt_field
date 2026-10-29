"""Lossless, readable factoring of one current-observation feedback snapshot.

This is a request presentation transform, never a controller/validator input.
Every value in the supplied (possibly already V21/V32-presented) snapshot is
retained. Tables remove repeated dictionary keys; named shared facts remove
exact repeated subtrees. No rounding, omitted adverse evidence, cross-observation
references, inferred equivalence or new permission is involved.
"""
from __future__ import annotations

import copy
import re
from collections import Counter

from .astra import _canonical

VERSION = 'current-feedback-exact-shared-facts-v35-v1'
_REF = '$same_v35'
_TABLE = '$table_v35'
_WITH = '$with_v35'
_RESERVED = frozenset((_REF, _TABLE, _WITH, 'presentation_v35'))
SEMANTICS = (
    'Exact current-observation presentation. $same_v35 names the identical value '
    'in shared_facts; $with_v35 starts with the named common-fields dictionary '
    'and replaces/adds every different_values key. No keys are inferred. '
    'References never point to another observation. $table_v35 '
    'pairs each row with named columns; absent_cells means an absent key, not null. '
    'Expand references/tables when reading any current constraint, candidate or '
    'numerical value. Prospective/debug snapshots retain their original status; '
    'sharing grants no motion, identity, clearance or budget.'
)


def _reserved(value):
    if isinstance(value, dict):
        return bool(_RESERVED.intersection(value)) or any(_reserved(v) for v in value.values())
    return isinstance(value, list) and any(_reserved(v) for v in value)


def _tables(value):
    if isinstance(value, dict):
        return {key: _tables(item) for key, item in value.items()}
    if not isinstance(value, list):
        return copy.deepcopy(value)
    rows = [_tables(item) for item in value]
    if len(rows) < 3 or not all(isinstance(row, dict) and row and not _RESERVED.intersection(row) for row in rows):
        return rows
    keys = set().union(*(row.keys() for row in rows))
    # Similar records retain readable named columns; unrelated heterogeneous
    # structures stay ordinary objects. Unknown extension keys are not dropped.
    if any(len(row) < .7 * len(keys) for row in rows):
        return rows
    columns = sorted(keys)
    table = {'columns': columns, 'rows': [[row.get(key) for key in columns] for row in rows]}
    missing = [[i, j] for i, row in enumerate(rows)
               for j, key in enumerate(columns) if key not in row]
    if missing:
        table['absent_cells'] = missing
    encoded = {_TABLE: table}
    return encoded if len(_canonical(encoded)) + 64 < len(_canonical(rows)) else rows


def _name(path, serial):
    # Names tell the reader which fact is shared; no hashes or opaque payloads.
    names = [re.sub(r'[^a-zA-Z0-9_]+', '_', str(k)).strip('_') for k in path if isinstance(k, str)]
    stem = (names[-1] if names else 'value')[-28:] or 'value'
    return f'{stem}_{serial}'


def _common_fields(value):
    """Factor exact repeated fields in same-key record families, no semantics."""
    groups = {}

    def collect(item, path=()):
        if isinstance(item, dict):
            if len(item) >= 5 and path:
                groups.setdefault(tuple(sorted(item)), []).append((item, path))
            for key, child in item.items():
                collect(child, (*path, key))
        elif isinstance(item, list):
            for i, child in enumerate(item):
                collect(child, (*path, i))

    collect(value)
    shared, choices = {}, {}
    for columns, records in groups.items():
        if len(records) < 2:
            continue
        base = {}
        for key in columns:
            values = Counter(_canonical(item[key]) for item, _ in records)
            best, count = values.most_common(1)[0]
            if count >= 2:
                base[key] = next(item[key] for item, _ in records if _canonical(item[key]) == best)
        if not base:
            continue
        name = 'common_' + _name(records[0][1], len(shared) + 1)
        selected, saved = [], 0
        for item, _ in records:
            different = {key: child for key, child in item.items()
                         if key not in base or _canonical(child) != _canonical(base[key])}
            packed = {_WITH: name, 'different_values': different}
            saving = len(_canonical(item)) - len(_canonical(packed))
            if saving > 32:
                selected.append((item, different))
                saved += saving
        if len(selected) >= 2 and saved > len(_canonical(base)) + len(_canonical(name)) + 128:
            shared[name] = copy.deepcopy(base)
            for item, different in selected:
                choices[id(item)] = (name, different)

    def encode(item):
        if isinstance(item, dict):
            if id(item) in choices:
                name, different = choices[id(item)]
                return {_WITH: name, 'different_values': {key: encode(child) for key, child in different.items()}}
            return {key: encode(child) for key, child in item.items()}
        if isinstance(item, list):
            return [encode(child) for child in item]
        return copy.deepcopy(item)

    # Definitions remain literal here. Applying common templates inside their
    # own definitions could create cycles for recursive record shapes.
    return encode(value), shared


def compact_current_feedback(value):
    """Return a detached, exact packet only when it saves serialized bytes.

    Idempotent for this presentation. A source containing a reserved literal
    key is left intact instead of interpreting or silently escaping user data.
    JSON-compatible feedback is expected, just as in the subscription client.
    """
    if not isinstance(value, dict) or _reserved(value):
        return copy.deepcopy(value)
    transformed, common = _common_fields(value)
    transformed, common = _tables(transformed), _tables(common)
    counts, first_path = Counter(), {}

    def collect(item, path=()):
        if isinstance(item, (dict, list, str)):
            text = _canonical(item)
            if len(text) >= 80:
                counts[text] += 1
                first_path.setdefault(text, path)
        if isinstance(item, dict):
            for key, child in item.items():
                collect(child, (*path, key))
        elif isinstance(item, list):
            for index, child in enumerate(item):
                collect(child, (*path, index))

    collect(transformed)
    collect(common, ('common_fields',))
    candidates = {text for text, count in counts.items() if count > 1}
    shared, names = {}, {}

    def encode(item, *, root=False):
        text = _canonical(item) if isinstance(item, (dict, list, str)) else None
        if not root and text in candidates:
            if text not in names:
                name = 'same_' + _name(first_path[text], len(names) + 1)
                names[text] = name
                shared[name] = encode(item, root=True)
            return {_REF: names[text]}
        if isinstance(item, dict):
            return {key: encode(child) for key, child in item.items()}
        if isinstance(item, list):
            return [encode(child) for child in item]
        return copy.deepcopy(item)

    current = encode(transformed, root=True)
    common_encoded = {name: encode(item, root=True) for name, item in common.items()}
    shared.update(common_encoded)

    def uses(item, counter):
        if isinstance(item, dict):
            if set(item) == {_REF}:
                counter[item[_REF]] += 1
            elif set(item) == {_WITH, 'different_values'}:
                counter[item[_WITH]] += 1
                uses(item['different_values'], counter)
            else:
                for child in item.values():
                    uses(child, counter)
        elif isinstance(item, list):
            for child in item:
                uses(child, counter)

    def inline(item, removals):
        if isinstance(item, dict):
            if set(item) == {_REF} and item[_REF] in removals:
                return inline(shared[item[_REF]], removals)
            return {key: inline(child, removals) for key, child in item.items()}
        if isinstance(item, list):
            return [inline(child, removals) for child in item]
        return item

    # Common dictionaries nested in a chosen parent may disappear. Do not keep
    # orphan definitions (nor their descendants) merely because they were found.
    reachable = set()

    def mark(item):
        found = Counter()
        uses(item, found)
        for name in found:
            if name not in reachable:
                reachable.add(name)
                mark(shared[name])

    mark(current)
    shared = {name: item for name, item in shared.items() if name in reachable}

    # A repeated parent may subsume repeated children. Keep only definitions
    # that still reduce total packet size, rather than leaving one-use aliases.
    while shared:
        references = Counter()
        uses(current, references)
        for item in shared.values():
            uses(item, references)
        removals = {name for name, item in shared.items()
                    if name not in common and (references[name] - 1) * len(_canonical(item)) <=
                    references[name] * len(_canonical({_REF: name})) + len(_canonical(name)) + 2}
        if not removals:
            break
        current = inline(current, removals)
        survivors = {name: inline(item, removals) for name, item in shared.items() if name not in removals}
        shared = survivors

    packet = {'presentation_v35': VERSION, 'semantics': SEMANTICS,
              'shared_facts': shared, 'current': current}
    return packet if len(_canonical(packet)) + 64 < len(_canonical(value)) else copy.deepcopy(value)


def expand_current_feedback(value):
    """Strict detached inverse for produced packets; no external references.

    Marker-free original snapshots pass through. A literal source that already
    uses ``presentation_v35 == VERSION`` is deliberately reserved: compaction
    leaves it unchanged, but this decoder treats that marker as a packet claim
    and rejects malformed packets instead of guessing a different meaning.
    """
    if not isinstance(value, dict) or value.get('presentation_v35') != VERSION:
        return copy.deepcopy(value)
    if set(value) != {'presentation_v35', 'semantics', 'shared_facts', 'current'}:
        raise ValueError('Invalid V35 feedback packet keys')
    shared = value['shared_facts']
    if not isinstance(shared, dict):
        raise ValueError('Invalid V35 shared facts')

    def decode(item, stack=()):
        if isinstance(item, dict):
            if set(item) == {_REF}:
                name = item[_REF]
                if not isinstance(name, str) or name not in shared:
                    raise ValueError('V35 reference requires a same-observation shared fact')
                if name in stack:
                    raise ValueError('Cyclic V35 shared fact')
                return decode(shared[name], (*stack, name))
            if set(item) == {_WITH, 'different_values'}:
                base = decode({_REF: item[_WITH]}, stack)
                different = decode(item['different_values'], stack)
                if not isinstance(base, dict) or not isinstance(different, dict):
                    raise ValueError('Invalid V35 common-fields dictionary')
                base.update(different)
                return base
            if set(item) == {_TABLE}:
                table = decode(item[_TABLE], stack)
                if not isinstance(table, dict) or not {'columns', 'rows'} <= set(table) <= {'columns', 'rows', 'absent_cells'}:
                    raise ValueError('Invalid V35 table')
                columns, rows = table['columns'], table['rows']
                if (not isinstance(columns, list) or not all(isinstance(key, str) for key in columns)
                        or len(set(columns)) != len(columns) or not isinstance(rows, list)
                        or any(not isinstance(row, list) or len(row) != len(columns) for row in rows)):
                    raise ValueError('Invalid V35 table shape')
                cells = table.get('absent_cells', [])
                if not isinstance(cells, list) or any(
                        not isinstance(cell, list) or len(cell) != 2
                        or any(type(index) is not int for index in cell)
                        or not 0 <= cell[0] < len(rows) or not 0 <= cell[1] < len(columns)
                        for cell in cells):
                    raise ValueError('Invalid V35 absent cells')
                absent = {tuple(cell) for cell in cells}
                return [decode({key: copy.deepcopy(row[j]) for j, key in enumerate(columns)
                                if (i, j) not in absent}, stack)
                        for i, row in enumerate(rows)]
            return {key: decode(child, stack) for key, child in item.items()}
        if isinstance(item, list):
            return [decode(child, stack) for child in item]
        return copy.deepcopy(item)

    return decode(value['current'])
