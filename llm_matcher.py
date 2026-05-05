from __future__ import annotations

import json
import re
from dataclasses import dataclass

from config import LLMConfig, llm_config


@dataclass(frozen=True)
class LLMDecision:
    match: bool
    confidence: float
    reason: str
    raw: str


class LLMMatcher:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or llm_config()
        if not self.config.api_key:
            raise RuntimeError(
                "Missing LLM API key. Put one in `.env` as LLM_API_KEY=... "
                "or OPENAI_API_KEY=..., then rerun."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install with `python3 -m pip install -r requirements.txt`.") from exc
        kwargs = {"api_key": self.config.api_key}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        self.client = OpenAI(timeout=self.config.timeout_seconds, max_retries=1, **kwargs)

    def decide(self, prompt: str) -> LLMDecision:
        votes: list[LLMDecision] = []
        rounds = max(1, self.config.self_consistency)
        for _ in range(rounds):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    temperature=self.config.temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an entity resolution judge. Reply only as compact JSON "
                                'with keys "match", "confidence", and "reason".'
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                raw = response.choices[0].message.content or ""
                votes.append(parse_decision(raw))
            except Exception as exc:
                votes.append(LLMDecision(False, 0.0, f"LLM request failed: {exc}", ""))

        positives = [vote for vote in votes if vote.match]
        negatives = [vote for vote in votes if not vote.match]
        chosen = positives if len(positives) >= len(negatives) else negatives
        confidence = sum(vote.confidence for vote in chosen) / len(chosen)
        return LLMDecision(
            match=len(positives) >= len(negatives),
            confidence=confidence,
            reason=" | ".join(vote.reason for vote in chosen if vote.reason)[:500],
            raw="\n".join(vote.raw for vote in votes),
        )


def parse_decision(raw: str) -> LLMDecision:
    text = raw.strip()
    try:
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        data = json.loads(text)
        return LLMDecision(
            match=bool(data.get("match")),
            confidence=float(data.get("confidence", 0.5)),
            reason=str(data.get("reason", "")),
            raw=raw,
        )
    except Exception:
        lowered = text.lower()
        is_match = "yes" in lowered or '"match": true' in lowered or "same entity" in lowered
        is_nonmatch = "no" in lowered or '"match": false' in lowered or "not the same" in lowered
        if is_match and not is_nonmatch:
            return LLMDecision(True, 0.5, text[:200], raw)
        return LLMDecision(False, 0.5, text[:200], raw)
