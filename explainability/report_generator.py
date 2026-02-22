"""
Report Generator — produces structured JSON + human-readable summaries from
DetectionResult objects, optionally enriched with SHAP explanations.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger


# ──────────────────────────────────────────────────── Helpers ─────────────────

_RISK_COLOUR = {
    "LOW": "🟢",
    "MEDIUM": "🟡",
    "HIGH": "🟠",
    "CRITICAL": "🔴",
}

_PATTERN_LABELS: Dict[str, str] = {
    "wash_trading": "Wash Trading",
    "flash_loan_flash_loan": "Flash Loan Exploit",
    "flash_loan_mev_sandwich": "MEV Sandwich Attack",
    "market_manipulation_pump_dump": "Pump-and-Dump Scheme",
    "market_manipulation_coordinated_buy": "Coordinated Buying",
    "coordinated_wallets": "Coordinated Multi-Wallet Activity",
}

_RECOMMENDATIONS: Dict[str, List[str]] = {
    "wash_trading": [
        "Report suspected wash trading to exchange compliance teams.",
        "Analyse counterparty wallets for shared funding source.",
    ],
    "flash_loan": [
        "Notify the affected DeFi protocol's security team.",
        "Check whether protocol invariants were violated in the same block.",
    ],
    "market_manipulation": [
        "Cross-reference coordinated wallets with known bad-actor lists.",
        "Assess price impact on liquidity pools during the flagged window.",
    ],
    "coordinated_wallets": [
        "Trace common funding sources across the detected wallet cluster.",
        "Apply cluster-wide risk escalation.",
    ],
    "HIGH": [
        "Flag for immediate manual review.",
        "Cross-reference with sanctions screening databases (OFAC, EU).",
    ],
    "CRITICAL": [
        "Escalate to the security operations centre immediately.",
        "Consider temporary rate-limiting or blocking at the infrastructure level.",
    ],
}


def _label(pattern: str) -> str:
    return _PATTERN_LABELS.get(pattern, pattern.replace("_", " ").title())


def _collect_recommendations(risk_level: str, patterns: List[str]) -> List[str]:
    seen: set = set()
    recs: List[str] = []

    def _add(items: List[str]) -> None:
        for r in items:
            if r not in seen:
                seen.add(r)
                recs.append(r)

    if risk_level in ("HIGH", "CRITICAL"):
        _add(_RECOMMENDATIONS.get(risk_level, []))

    for p in patterns:
        for key in _RECOMMENDATIONS:
            if key in p:
                _add(_RECOMMENDATIONS[key])

    if not recs:
        recs.append("Continue standard monitoring — no immediate action required.")

    return recs


# ──────────────────────────────────────────────────────── Main class ───────────

class ReportGenerator:
    """
    Generates structured anomaly reports from detection results.

    Usage::

        gen = ReportGenerator()
        report = gen.generate(
            identifier="0xabcdef...",
            anomaly_score=0.87,
            risk_level="HIGH",
            detected_patterns=["wash_trading"],
            shap_features=[...],          # optional
            raw_metadata={...},           # optional extra context
        )
        print(gen.to_text(report))
        print(gen.to_json(report))
    """

    def generate(
        self,
        identifier: str,
        anomaly_score: float,
        risk_level: str,
        detected_patterns: Optional[List[str]] = None,
        shap_features: Optional[List[Dict[str, Any]]] = None,
        raw_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a complete anomaly report dictionary.

        Args:
            identifier:        Transaction hash or wallet address.
            anomaly_score:     Float in [0, 1] — higher = more anomalous.
            risk_level:        One of LOW / MEDIUM / HIGH / CRITICAL.
            detected_patterns: List of pattern keys from the detection engine.
            shap_features:     Top-k SHAP feature dicts (from SHAPExplainer).
            raw_metadata:      Optional extra key/value pairs to embed.

        Returns:
            Structured report dict.
        """
        patterns = detected_patterns or []
        features = shap_features or []
        risk_level = risk_level.upper()

        report: Dict[str, Any] = {
            "identifier": identifier,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "anomaly_score": round(float(anomaly_score), 4),
            "risk_level": risk_level,
            "risk_icon": _RISK_COLOUR.get(risk_level, "⚪"),
            "detected_patterns": [
                {"key": p, "label": _label(p)} for p in patterns
            ],
            "summary": self._narrative(risk_level, patterns, features),
            "top_features": features[:5],
            "recommendations": _collect_recommendations(risk_level, patterns),
        }

        if raw_metadata:
            report["metadata"] = raw_metadata

        logger.debug(
            f"Report generated for {identifier[:12]}… "
            f"| score={anomaly_score:.3f} | risk={risk_level} "
            f"| patterns={len(patterns)}"
        )
        return report

    # ─────────────────────────────────────────── Narrative summary ────────────

    @staticmethod
    def _narrative(
        risk_level: str,
        patterns: List[str],
        features: List[Dict[str, Any]],
    ) -> str:
        """One-sentence plain-language summary."""
        if not patterns and risk_level == "LOW":
            return (
                "No significant anomalous behaviour detected. "
                "Standard monitoring applies."
            )

        pattern_labels = [_label(p) for p in patterns[:3]]
        feature_desc = (
            features[0]["feature"].replace("_", " ")
            if features
            else "transaction pattern"
        )

        parts = ", ".join(pattern_labels) if pattern_labels else "anomalous behaviour"
        return (
            f"This {risk_level.lower()}-risk entity exhibits signs of {parts}, "
            f"driven primarily by elevated {feature_desc}."
        )

    # ─────────────────────────────────────────────── Output formatters ────────

    @staticmethod
    def to_json(report: Dict[str, Any], indent: int = 2) -> str:
        """Serialise the report to a JSON string."""
        return json.dumps(report, indent=indent, default=str)

    @staticmethod
    def to_text(report: Dict[str, Any]) -> str:
        """Render the report as a human-readable plain-text block."""
        icon = report.get("risk_icon", "⚪")
        lines: List[str] = [
            "=" * 60,
            f"  ANOMALY REPORT  {icon} {report['risk_level']}",
            "=" * 60,
            f"  Identifier  : {report['identifier']}",
            f"  Score       : {report['anomaly_score']:.4f}",
            f"  Generated   : {report['generated_at']}",
            "",
            "  SUMMARY",
            f"  {report['summary']}",
            "",
        ]

        patterns = report.get("detected_patterns", [])
        if patterns:
            lines.append("  DETECTED PATTERNS")
            for p in patterns:
                lines.append(f"    • {p['label']}")
            lines.append("")

        features = report.get("top_features", [])
        if features:
            lines.append("  TOP CONTRIBUTING FEATURES")
            for f in features:
                direction = "↑" if f.get("contribution") == "increases_risk" else "↓"
                lines.append(
                    f"    {direction} {f['feature']:<30} "
                    f"val={f.get('feature_value', 0):.4f}  "
                    f"shap={f.get('shap_value', 0):+.4f}"
                )
            lines.append("")

        recs = report.get("recommendations", [])
        if recs:
            lines.append("  RECOMMENDATIONS")
            for r in recs:
                lines.append(f"    → {r}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)
