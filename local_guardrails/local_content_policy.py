import json
import re
import unicodedata
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Pattern, Sequence

import yaml
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.utils import GenericGuardrailAPIInputs


PUBLIC_REJECTION_MESSAGE = "Request rejected by content policy."
_CLAUSE_SEPARATOR = re.compile(r"[。！？!?]+")
_BYPASS_SEPARATOR = r"[\s*._\-·•~～]{0,3}"
_TEXT_PART_TYPES = frozenset({"text", "input_text"})


class _PolicyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class KeywordRuleConfig(_PolicyModel):
    rule_id: str
    category: str
    severity: Literal["low", "medium", "high"] = "high"
    keywords: tuple[str, ...]
    separator_insensitive: bool = False


class RegexRuleConfig(_PolicyModel):
    rule_id: str
    category: str
    severity: Literal["low", "medium", "high"] = "high"
    pattern: str


class ConditionalRuleConfig(_PolicyModel):
    rule_id: str
    category: str
    severity: Literal["low", "medium", "high"] = "high"
    left_set: str
    right_set: str
    max_gap: int = Field(ge=0, le=200)


class TermConfig(_PolicyModel):
    term: str
    pattern: str | None = None


class PolicyConfig(_PolicyModel):
    guardrail_internal_type: str
    allow_patterns: tuple[str, ...] = ()
    term_sets: Mapping[str, tuple[str | TermConfig, ...]] = Field(default_factory=dict)
    keyword_rules: tuple[KeywordRuleConfig, ...] = ()
    regex_rules: tuple[RegexRuleConfig, ...] = ()
    conditional_rules: tuple[ConditionalRuleConfig, ...] = ()


@dataclass(frozen=True, slots=True)
class Detection:
    rule_id: str
    category: str
    severity: str
    matched_keyword: str | None
    matched_pattern: str | None


@dataclass(frozen=True, slots=True)
class _CompiledKeyword:
    keyword: str
    pattern: Pattern[str]


@dataclass(frozen=True, slots=True)
class _CompiledKeywordRule:
    rule_id: str
    category: str
    severity: str
    keywords: tuple[_CompiledKeyword, ...]


@dataclass(frozen=True, slots=True)
class _CompiledRegexRule:
    rule_id: str
    category: str
    severity: str
    pattern: Pattern[str]


@dataclass(frozen=True, slots=True)
class _CompiledTerm:
    literal: str | None
    pattern: Pattern[str] | None
    ascii_boundary: bool


@dataclass(frozen=True, slots=True)
class _CompiledTermSet:
    name: str
    terms: tuple[_CompiledTerm, ...]


@dataclass(frozen=True, slots=True)
class _TermMatch:
    value: str
    start_index: int
    end_index: int

    def start(self) -> int:
        return self.start_index

    def end(self) -> int:
        return self.end_index

    def group(self, _group: int = 0) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class _CompiledConditionalRule:
    rule_id: str
    category: str
    severity: str
    left: _CompiledTermSet
    right: _CompiledTermSet
    max_gap: int


class _PublicPolicyViolation(HTTPException):
    type = "invalid_request_error"
    param = None

    def __init__(self) -> None:
        super().__init__(status_code=400, detail=PUBLIC_REJECTION_MESSAGE)


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Cf"
        and not "\ufe00" <= character <= "\ufe0f"
        and not "\U000e0100" <= character <= "\U000e01ef"
    )


def _term_pattern(term: str, separator_insensitive: bool = False) -> str:
    normalized = _normalize(term).strip()
    if not normalized:
        raise ValueError("policy terms must not be empty")
    if separator_insensitive:
        compact = tuple(character for character in normalized if not character.isspace())
        body = _BYPASS_SEPARATOR.join(re.escape(character) for character in compact)
    else:
        body = r"\s+".join(re.escape(part) for part in normalized.split())
    if re.fullmatch(r"[a-z0-9 ]+", normalized):
        return rf"(?<![a-z0-9_])(?:{body})(?![a-z0-9_])"
    return body


def _compile_term_set(name: str, terms: Sequence[str | TermConfig]) -> _CompiledTermSet:
    if not terms:
        raise ValueError(f"term set '{name}' must not be empty")

    compiled_terms: list[_CompiledTerm] = []
    for configured_term in terms:
        if isinstance(configured_term, TermConfig) and configured_term.pattern is not None:
            compiled_terms.append(
                _CompiledTerm(
                    literal=None,
                    pattern=re.compile(configured_term.pattern, re.IGNORECASE),
                    ascii_boundary=False,
                )
            )
            continue
        raw_term = configured_term if isinstance(configured_term, str) else configured_term.term
        normalized = _normalize(raw_term).strip()
        if not normalized:
            raise ValueError("policy terms must not be empty")
        if any(character.isspace() for character in normalized):
            compiled_terms.append(
                _CompiledTerm(
                    literal=None,
                    pattern=re.compile(_term_pattern(normalized), re.IGNORECASE),
                    ascii_boundary=False,
                )
            )
            continue
        compiled_terms.append(
            _CompiledTerm(
                literal=normalized,
                pattern=None,
                ascii_boundary=re.fullmatch(r"[a-z0-9]+", normalized) is not None,
            )
        )
    return _CompiledTermSet(name=name, terms=tuple(compiled_terms))


def _is_ascii_word_character(character: str) -> bool:
    return character == "_" or "a" <= character <= "z" or "0" <= character <= "9"


def _find_term_set_matches(term_set: _CompiledTermSet, text: str) -> tuple[_TermMatch, ...]:
    matches: list[_TermMatch] = []
    for term in term_set.terms:
        if term.literal is None:
            if term.pattern is None:
                continue
            matches.extend(
                _TermMatch(match.group(0), match.start(), match.end()) for match in term.pattern.finditer(text)
            )
            continue
        start = text.find(term.literal)
        while start >= 0:
            end = start + len(term.literal)
            left_ok = start == 0 or not _is_ascii_word_character(text[start - 1])
            right_ok = end == len(text) or not _is_ascii_word_character(text[end])
            if not term.ascii_boundary or left_ok and right_ok:
                matches.append(_TermMatch(text[start:end], start, end))
            start = text.find(term.literal, end)
    return tuple(sorted(matches, key=lambda match: (match.start_index, -match.end_index)))


def _find_term_set_matches_by_clause(
    term_set: _CompiledTermSet,
    text: str,
    separator_ends: Sequence[int],
) -> dict[int, tuple[_TermMatch, ...]]:
    grouped: dict[int, list[_TermMatch]] = {}
    for match in _find_term_set_matches(term_set, text):
        clause_index = bisect_right(separator_ends, match.start_index)
        grouped.setdefault(clause_index, []).append(match)
    return {clause_index: tuple(matches) for clause_index, matches in grouped.items()}


def _extract_text_parts(content: object) -> tuple[str, ...]:
    if isinstance(content, str):
        return (content,) if content else ()
    if not isinstance(content, list):
        return ()
    return tuple(
        text
        for part in content
        if isinstance(part, dict)
        and part.get("type") in _TEXT_PART_TYPES
        and isinstance((text := part.get("text")), str)
        and text
    )


def _iter_user_text(inputs: GenericGuardrailAPIInputs) -> tuple[str, ...]:
    structured_messages = inputs.get("structured_messages")
    if isinstance(structured_messages, list):
        if not structured_messages:
            return ()
        current_message = structured_messages[-1]
        if not isinstance(current_message, dict) or str(current_message.get("role") or "").lower() != "user":
            return ()
        parts = _extract_text_parts(current_message.get("content"))
        return ("\n".join(parts),) if parts else ()
    texts = inputs.get("texts")
    if not isinstance(texts, list) or not texts:
        return ()
    current_text = texts[-1]
    return (current_text,) if isinstance(current_text, str) and current_text else ()


def _nearest_non_overlapping_match(
    left_matches: Sequence[_TermMatch],
    right_matches: Sequence[_TermMatch],
    max_gap: int,
) -> tuple[_TermMatch, _TermMatch] | None:
    left_index = 0
    right_index = 0
    while left_index < len(left_matches) and right_index < len(right_matches):
        left = left_matches[left_index]
        right = right_matches[right_index]
        if left.end() <= right.start():
            if right.start() - left.end() <= max_gap:
                return left, right
            left_index += 1
            continue
        if right.end() <= left.start():
            if left.start() - right.end() <= max_gap:
                return left, right
            right_index += 1
            continue
        if left.end() <= right.end():
            left_index += 1
        else:
            right_index += 1
    return None


def _request_id(request_data: Mapping[str, object]) -> str:
    direct = request_data.get("litellm_call_id")
    if isinstance(direct, str) and direct:
        return direct
    for metadata_key in ("metadata", "litellm_metadata"):
        metadata = request_data.get(metadata_key)
        if not isinstance(metadata, dict):
            continue
        for key in ("request_id", "litellm_call_id", "trace_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value
    return "unknown"


class LocalPolicyMatcher:
    def __init__(self, config: PolicyConfig) -> None:
        rule_ids = (
            tuple(rule.rule_id for rule in config.keyword_rules)
            + tuple(rule.rule_id for rule in config.regex_rules)
            + tuple(rule.rule_id for rule in config.conditional_rules)
        )
        if len(rule_ids) != len(frozenset(rule_ids)):
            raise ValueError("policy rule_id values must be unique")
        self.internal_type = config.guardrail_internal_type
        self.allow_patterns = tuple(re.compile(pattern, re.IGNORECASE) for pattern in config.allow_patterns)
        self.keyword_rules = tuple(
            _CompiledKeywordRule(
                rule_id=rule.rule_id,
                category=rule.category,
                severity=rule.severity,
                keywords=tuple(
                    _CompiledKeyword(
                        keyword=_normalize(keyword),
                        pattern=re.compile(
                            rf"^(?:{_term_pattern(keyword, rule.separator_insensitive)})"
                            r"(?:[!！?？。.,，~～啊呀吧呢嘛啦喽]*)$",
                            re.IGNORECASE,
                        ),
                    )
                    for keyword in rule.keywords
                ),
            )
            for rule in config.keyword_rules
        )
        self.regex_rules = tuple(
            _CompiledRegexRule(
                rule_id=rule.rule_id,
                category=rule.category,
                severity=rule.severity,
                pattern=re.compile(rule.pattern, re.IGNORECASE),
            )
            for rule in config.regex_rules
        )
        term_sets = {name: _compile_term_set(name, terms) for name, terms in config.term_sets.items()}
        referenced_term_sets = {
            term_set for rule in config.conditional_rules for term_set in (rule.left_set, rule.right_set)
        }
        missing_term_sets = referenced_term_sets - term_sets.keys()
        if missing_term_sets:
            missing = ", ".join(sorted(missing_term_sets))
            raise ValueError(f"conditional rules reference missing term sets: {missing}")
        self.conditional_rules = tuple(
            _CompiledConditionalRule(
                rule_id=rule.rule_id,
                category=rule.category,
                severity=rule.severity,
                left=term_sets[rule.left_set],
                right=term_sets[rule.right_set],
                max_gap=rule.max_gap,
            )
            for rule in config.conditional_rules
        )

    @classmethod
    def from_file(cls, policy_file: str) -> "LocalPolicyMatcher":
        with Path(policy_file).open("r", encoding="utf-8") as handle:
            raw: object = yaml.safe_load(handle)
        return cls(PolicyConfig.model_validate(raw))

    def detect(self, text: str) -> Detection | None:
        normalized = _normalize(text).strip()
        if not normalized or any(pattern.fullmatch(normalized) for pattern in self.allow_patterns):
            return None
        regex_detection = self._detect_regex(normalized)
        if regex_detection is not None:
            return regex_detection
        keyword_detection = self._detect_keyword(normalized)
        if keyword_detection is not None:
            return keyword_detection
        return self._detect_conditional(normalized)

    def _detect_keyword(self, text: str) -> Detection | None:
        for rule in self.keyword_rules:
            for keyword in rule.keywords:
                if keyword.pattern.search(text):
                    return Detection(
                        rule_id=rule.rule_id,
                        category=rule.category,
                        severity=rule.severity,
                        matched_keyword=keyword.keyword,
                        matched_pattern=None,
                    )
        return None

    def _detect_regex(self, text: str) -> Detection | None:
        for rule in self.regex_rules:
            match = rule.pattern.search(text)
            if match is not None:
                return Detection(
                    rule_id=rule.rule_id,
                    category=rule.category,
                    severity=rule.severity,
                    matched_keyword=None,
                    matched_pattern=rule.rule_id,
                )
        return None

    def _detect_conditional(self, text: str) -> Detection | None:
        if not self.conditional_rules:
            return None
        separator_ends = tuple(match.end() for match in _CLAUSE_SEPARATOR.finditer(text))
        matches_by_set: dict[str, dict[int, tuple[_TermMatch, ...]]] = {}
        for rule in self.conditional_rules:
            left_by_clause = matches_by_set.get(rule.left.name)
            if left_by_clause is None:
                left_by_clause = _find_term_set_matches_by_clause(rule.left, text, separator_ends)
                matches_by_set[rule.left.name] = left_by_clause
            if not left_by_clause:
                continue
            right_by_clause = matches_by_set.get(rule.right.name)
            if right_by_clause is None:
                right_by_clause = _find_term_set_matches_by_clause(rule.right, text, separator_ends)
                matches_by_set[rule.right.name] = right_by_clause
            if not right_by_clause:
                continue
            for clause_index, left_matches in left_by_clause.items():
                right_matches = right_by_clause.get(clause_index)
                if not right_matches:
                    continue
                nearest = _nearest_non_overlapping_match(left_matches, right_matches, rule.max_gap)
                if nearest is not None:
                    left_match, right_match = nearest
                    return Detection(
                        rule_id=rule.rule_id,
                        category=rule.category,
                        severity=rule.severity,
                        matched_keyword=f"{left_match.group(0)} + {right_match.group(0)}",
                        matched_pattern=f"{rule.left.name}<={rule.max_gap}=>{rule.right.name}",
                    )
        return None


class LocalContentPolicyGuardrail(CustomGuardrail):
    def __init__(
        self,
        guardrail_name: str | None = None,
        event_hook: GuardrailEventHooks | list[GuardrailEventHooks] | Mode | None = None,
        default_on: bool = False,
        policy_file: str | None = None,
        **_kwargs: object,
    ) -> None:
        if policy_file is None:
            raise ValueError("policy_file is required")
        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=[GuardrailEventHooks.pre_call],
            event_hook=event_hook,
            default_on=default_on,
        )
        self.matcher = LocalPolicyMatcher.from_file(policy_file)

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: object | None = None,
    ) -> GenericGuardrailAPIInputs:
        if input_type != "request":
            return inputs
        for text in _iter_user_text(inputs):
            detection = self.matcher.detect(text)
            if detection is None:
                continue
            payload = {
                "guardrail_internal_type": self.matcher.internal_type,
                "rule_id": detection.rule_id,
                "category": detection.category,
                "matched_keyword": detection.matched_keyword,
                "matched_pattern": detection.matched_pattern,
                "severity": detection.severity,
                "request_id": _request_id(request_data),
            }
            verbose_proxy_logger.warning(
                "内容策略拦截 %s",
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            )
            raise _PublicPolicyViolation()
        return inputs
