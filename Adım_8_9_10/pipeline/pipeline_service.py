from __future__ import annotations

"""
PipelineService — Score Cache + Fallback Pipeline
=================================================
Hibrit strateji:
  1. TransactionID varsa → parquet cache'de ara
  2. Cache'de bulunursa → skorları doğrudan döndür
  3. Bulunamazsa (yeni ID) → Adım 3-7 pipeline'ını tetikle, skoru üret

Tasarım prensipleri:
  - Her adım opsiyonel import: pipeline kurulu değilse sadece cache çalışır
  - Tek satır input: Dict[str, Any] — ham transaction alanları
  - Çıktı her zaman aynı schema: PipelineResult
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Çıktı şeması
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    transaction_id: Optional[Any]
    fraud_score: float
    rule_adjusted_score: float
    context_adjusted_score: float
    column_score_mean: Optional[float]
    column_score_max: Optional[float]
    multivariate_score: Optional[float]
    entity_score: Optional[float]
    temporal_score: Optional[float]
    rules_triggered: str
    rule_flags: str
    rule_severity: str
    rule_score: float
    rule_audit_trail: str               # JSON string
    source: str                         # "cache" | "pipeline"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "TransactionID": self.transaction_id,
            "fraud_score": self.fraud_score,
            "rule_adjusted_score": self.rule_adjusted_score,
            "context_adjusted_score": self.context_adjusted_score,
            "column_score_mean": self.column_score_mean,
            "column_score_max": self.column_score_max,
            "multivariate_score": self.multivariate_score,
            "entity_score": self.entity_score,
            "temporal_score": self.temporal_score,
            "rules_triggered": self.rules_triggered,
            "rule_flags": self.rule_flags,
            "rule_severity": self.rule_severity,
            "rule_score": self.rule_score,
            "rule_audit_trail": self.rule_audit_trail,
            "source": self.source,
            **self.extra,
        }


# ---------------------------------------------------------------------------
# Score Cache
# ---------------------------------------------------------------------------
class ScoreCache:
    """
    Adım 7 parquet çıktısını okur, TransactionID bazlı lookup sağlar.
    Startup'ta bir kez yüklenir (Singleton olarak Container'a inject edilir).
    """

    SCORE_COLS = [
        "TransactionID",
        "fraud_score",
        "rule_adjusted_score",
        "context_adjusted_score",
        "column_score_mean",
        "column_score_max",
        "multivariate_score",
        "entity_score",
        "temporal_score",
        "rules_triggered",
        "rule_flags",
        "rule_severity",
        "rule_score",
        "rule_audit_trail",
        # feature kolonları da (narrative için)
        "tx_hour", "tx_is_night", "tx_is_weekend", "TransactionAmt",
        "card1_velocity_1d", "card1_amt_zscore",
        "ctx_card4_fraud_lift", "ctx_card6_fraud_lift",
        "ctx_pemaildomain_fraud_lift", "ctx_productcd_fraud_lift",
    ]

    def __init__(self, parquet_path: str) -> None:
        self.parquet_path = parquet_path
        self._index: Optional[Dict[Any, int]] = None
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> None:
        path = Path(self.parquet_path)
        if not path.exists():
            logger.warning("Cache parquet not found: %s — cache disabled.", self.parquet_path)
            return

        # Sadece ilgili kolonları yükle (590k satır × tüm kolonlar çok ağır)
        available_cols = pd.read_parquet(path, columns=["TransactionID"]).columns.tolist()
        load_cols = [c for c in self.SCORE_COLS if c in pd.read_parquet(path, columns=self.SCORE_COLS[:1]).columns or True]

        try:
            self._df = pd.read_parquet(path, columns=[c for c in self.SCORE_COLS if c in self._get_cols(path)])
        except Exception:
            self._df = pd.read_parquet(path)

        if "TransactionID" in self._df.columns:
            self._index = {v: i for i, v in enumerate(self._df["TransactionID"])}
            logger.info("ScoreCache loaded: %d rows from %s", len(self._df), path)
        else:
            logger.warning("TransactionID column not found in cache — cache disabled.")

    def _get_cols(self, path: Path) -> List[str]:
        sample = pd.read_parquet(path, columns=["TransactionID"])
        all_cols = pd.read_parquet(path).columns.tolist()
        return [c for c in self.SCORE_COLS if c in all_cols]

    @property
    def loaded(self) -> bool:
        return self._df is not None and self._index is not None

    def lookup(self, transaction_id: Any) -> Optional[Dict[str, Any]]:
        """TransactionID ile cache'de ara. Bulamazsa None döndür."""
        if not self.loaded:
            return None
        idx = self._index.get(transaction_id)
        if idx is None:
            return None
        return self._df.iloc[idx].to_dict()


# ---------------------------------------------------------------------------
# Pipeline Runner (Adım 3-7)
# ---------------------------------------------------------------------------
class PipelineRunner:
    """
    Ham transaction verisi geldiğinde Adım 3-7'yi sırayla çalıştırır.

    Import'lar lazy yapılır — step*.py dosyaları yoksa ImportError yerine
    uyarı verir ve None döndürür.
    """

    def __init__(self, rules_path: str, entity_cols: Optional[List[str]] = None) -> None:
        self.rules_path = rules_path
        self.entity_cols = entity_cols or ["card1", "card2", "addr1"]

    def run(self, transaction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Tek bir transaction dict'ini Adım 3-7'den geçirir.
        Başarısızlık durumunda None döndürür.
        """
        try:
            df = pd.DataFrame([transaction])
            df = self._run_step3(df)
            df = self._run_step4(df)
            df = self._run_step5(df)
            df = self._run_step6(df)
            df = self._run_step7(df)
            return df.iloc[0].to_dict()
        except Exception as e:
            logger.error("Pipeline failed for transaction %s: %s",
                         transaction.get("TransactionID"), e)
            return None

    def _run_step3(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            from pipeline.step3_features import run_feature_engineering
            return run_feature_engineering(
                df,
                entity_cols=self.entity_cols,
                target_col=None,        # inference: target yok
            )
        except ImportError:
            logger.warning("step3_features.py bulunamadı — feature engineering atlandı.")
            return df

    def _run_step4(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            from pipeline.step4_anomaly import run_anomaly_detection
            df_scored, _ = run_anomaly_detection(
                df,
                entity_cols=self.entity_cols,
                target_col=None,
                verbose=False,
            )
            return df_scored
        except ImportError:
            logger.warning("step4_anomaly.py bulunamadı — anomaly detection atlandı.")
            return df

    def _run_step5(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            from pipeline.step5_scoring import aggregate_scores
            df_final, _ = aggregate_scores(df, target_col=None)
            return df_final
        except ImportError:
            logger.warning("step5_scoring.py bulunamadı — score aggregation atlandı.")
            return df

    def _run_step6(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            from pipeline.step6_context import run_context_adjustment
            df_adj, _ = run_context_adjustment(df, target_col=None)
            return df_adj
        except ImportError:
            logger.warning("step6_context.py bulunamadı — context adjustment atlandı.")
            return df

    def _run_step7(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            from rule_engine import run_rule_engine
            score_col = "context_adjusted_score" if "context_adjusted_score" in df.columns else "fraud_score"
            df_rules, _ = run_rule_engine(
                df,
                rules_path=self.rules_path,
                score_col=score_col,
                target_col=None,
                save_outputs=False,     # inference'ta kaydetme
            )
            return df_rules
        except ImportError:
            logger.warning("rule_engine.py bulunamadı — rule engine atlandı.")
            return df


# ---------------------------------------------------------------------------
# PipelineService — hibrit orkestratör
# ---------------------------------------------------------------------------
class PipelineService:
    """
    Hibrit score cache + fallback pipeline servisi.

    Akış:
    1. TransactionID cache'de var mı? → Evet: cache'den döndür
    2. Hayır: PipelineRunner ile Adım 3-7 çalıştır
    3. Pipeline da başarısız olursa: ham skoru 0.0 ile döndür

    Container'a Singleton olarak inject edilir.
    """

    def __init__(self, cache: ScoreCache, runner: PipelineRunner) -> None:
        self.cache = cache
        self.runner = runner

    def score(self, transaction: Dict[str, Any]) -> PipelineResult:
        tx_id = transaction.get("TransactionID")

        # ── 1. Cache lookup ───────────────────────────────────────────────
        if tx_id is not None:
            cached = self.cache.lookup(tx_id)
            if cached is not None:
                logger.debug("Cache hit: TransactionID=%s", tx_id)
                return self._from_row(cached, source="cache")

        # ── 2. Pipeline fallback ──────────────────────────────────────────
        logger.info("Cache miss for TransactionID=%s — running pipeline.", tx_id)
        result_row = self.runner.run(transaction)

        if result_row is not None:
            return self._from_row(result_row, source="pipeline")

        # ── 3. Tam başarısızlık — ham transaction değerlerini kullan ────────
        logger.warning("Pipeline failed for TransactionID=%s — using raw transaction values.", tx_id)
        return PipelineResult(
            transaction_id=tx_id,
            fraud_score=0.0,
            rule_adjusted_score=0.0,
            context_adjusted_score=0.0,
            column_score_mean=None,
            column_score_max=None,
            multivariate_score=None,
            entity_score=None,
            temporal_score=None,
            rules_triggered="",
            rule_flags="",
            rule_severity="NONE",
            rule_score=0.0,
            rule_audit_trail="[]",
            source="fallback",
        )

    def _from_row(self, row: Dict[str, Any], source: str) -> PipelineResult:
        def _f(key: str) -> Optional[float]:
            v = row.get(key)
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        # rule_adjusted_score → ana fraud_score olarak kullan
        rule_score = _f("rule_adjusted_score")
        fraud_score = rule_score or _f("fraud_score") or 0.0

        # Scalar olmayan değerleri temizle
        extra = {
            k: v for k, v in row.items()
            if k.startswith("tx_") or k.startswith("ctx_") or k.startswith("card1_")
        }

        return PipelineResult(
            transaction_id=row.get("TransactionID"),
            fraud_score=fraud_score,
            rule_adjusted_score=rule_score or 0.0,
            context_adjusted_score=_f("context_adjusted_score") or 0.0,
            column_score_mean=_f("column_score_mean"),
            column_score_max=_f("column_score_max"),
            multivariate_score=_f("multivariate_score"),
            entity_score=_f("entity_score"),
            temporal_score=_f("temporal_score"),
            rules_triggered=str(row.get("rules_triggered", "")),
            rule_flags=str(row.get("rule_flags", "")),
            rule_severity=str(row.get("rule_severity", "NONE")),
            rule_score=_f("rule_score") or 0.0,
            rule_audit_trail=str(row.get("rule_audit_trail", "[]")),
            source=source,
            extra=extra,
        )