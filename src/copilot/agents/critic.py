"""Critic (step 68): the notebook reviewed against the standards before it runs.

Deterministic on purpose (ADR-004). Each rule is a check over the notebook's
syntax tree, tied to the catalog standard it enforces, and reports the line
in the notebook file - so a finding is the same every run, costs no tokens
and can be fed back to the Generator verbatim. Standards a program cannot
judge (is this the right natural key?) are left to execution (steps 69-71)
and to the human reviewing the pull request; an LLM reviewer is added only
if the agent eval (step 78) shows these rules miss real problems.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass

STANDARD = "standard:notebook."
COLLECT = STANDARD + "no-unbounded-collect-or-topandas"
SCHEMA = STANDARD + "explicit-schemas-and-column-lists"
IDEMPOTENT = STANDARD + "writes-are-idempotent"
CONFIG = STANDARD + "configuration-comes-from-metadata-never-literals"
SECRETS = STANDARD + "no-credentials-in-code"

MAX_TAKE = 1000
"""A literal `take(n)` / `head(n)` up to this size is a deliberate, bounded sample."""
_PATH = re.compile(r"^(?:(?:s3a?|abfss?|wasbs?|gs|dbfs|hdfs|file)://|dbfs:/|/mnt/|/dbfs/)")
_SECRET_NAME = re.compile(
    r"(password|passwd|secret|token|api_?key|access_?key|account_?key|conn(?:ection)?_?str)",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|AccountKey=|password=", re.IGNORECASE
)
_SELECT_STAR = re.compile(r"\bselect\s+\*", re.IGNORECASE)


@dataclass(frozen=True)
class Finding:
    rule: str
    standard: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"line {self.line}: {self.message} ({self.standard})"


@dataclass(frozen=True)
class Review:
    findings: tuple[Finding, ...]

    @property
    def passed(self) -> bool:
        return not self.findings


def review(source: str) -> Review:
    """Every standards violation the rules can see in a notebook source."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return Review(
            (Finding("syntax", "python", exc.lineno or 0, f"not valid Python: {exc.msg}"),)
        )
    found = sorted(set(_findings(tree)), key=lambda f: (f.line, f.rule))
    return Review(tuple(found))


def _findings(tree: ast.AST) -> Iterator[Finding]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield from _call(node)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield from _literal(node)
        elif isinstance(node, ast.Assign | ast.AnnAssign | ast.keyword):
            yield from _assignment(node)


def _call(node: ast.Call) -> Iterator[Finding]:
    func = node.func
    if not isinstance(func, ast.Attribute):
        return
    name, line = func.attr, node.lineno
    if name in {"collect", "toPandas"} and not _is_limited(func.value):
        yield Finding(
            "unbounded-collect", COLLECT, line, f"{name}() pulls the whole dataset to the driver"
        )
    if name in {"take", "head"} and not _small_literal(node):
        yield Finding(
            "unbounded-collect",
            COLLECT,
            line,
            f"{name}() needs a literal limit of at most {MAX_TAKE}",
        )
    if name == "select" and any(_is_str(a, "*") for a in node.args):
        yield Finding("select-star", SCHEMA, line, 'select("*") - name the columns')
    if name == "mode" and any(_is_str(a, "append") for a in node.args):
        yield Finding("blind-append", IDEMPOTENT, line, 'mode("append") duplicates on re-run')
    for kw in node.keywords:
        if kw.arg == "mode" and _is_str(kw.value, "append"):
            yield Finding("blind-append", IDEMPOTENT, line, 'mode="append" duplicates on re-run')


def _literal(node: ast.Constant) -> Iterator[Finding]:
    text = str(node.value)
    if _PATH.match(text.strip()):
        yield Finding("hardcoded-path", CONFIG, node.lineno, "a storage path as a literal")
    if _SECRET_VALUE.search(text):
        yield Finding("credential", SECRETS, node.lineno, "a literal that looks like a credential")
    if _SELECT_STAR.search(text):
        yield Finding("select-star", SCHEMA, node.lineno, "SELECT * in SQL - name the columns")


def _assignment(node: ast.Assign | ast.AnnAssign | ast.keyword) -> Iterator[Finding]:
    value: ast.expr | None
    if isinstance(node, ast.keyword):
        names, value = [node.arg or ""], node.value
    elif isinstance(node, ast.Assign):
        names, value = [_name(t) for t in node.targets], node.value
    else:
        names, value = [_name(node.target)], node.value
    if not (isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value):
        return
    for name in names:
        if _SECRET_NAME.search(name):
            yield Finding(
                "credential", SECRETS, value.lineno, f"'{name}' is set to a literal - resolve it"
            )


def _name(target: ast.expr) -> str:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def _is_limited(receiver: ast.expr) -> bool:
    return (
        isinstance(receiver, ast.Call)
        and isinstance(receiver.func, ast.Attribute)
        and receiver.func.attr == "limit"
    )


def _small_literal(node: ast.Call) -> bool:
    arg = node.args[0] if node.args else next((k.value for k in node.keywords), None)
    return (
        isinstance(arg, ast.Constant)
        and isinstance(arg.value, int)
        and not isinstance(arg.value, bool)
        and 0 < arg.value <= MAX_TAKE
    )


def _is_str(node: ast.expr, value: str) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value == value
