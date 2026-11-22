from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import uuid
import numpy as np

from src.core.contracts import (
    EpisodicMemoryRecord,
    MemoryFrictionConfig,
)


class EpistemicIsolationError(PermissionError):
    """Raised whenever an agent attempts to read or mutate another agent's private episodic memory."""


@dataclass(frozen=True)
class RetrievalResult:
    """Audit output of a single stress-modulated episodic RAG retrieval."""
    agent_id: str
    records: List[EpisodicMemoryRecord]
    top_k_used: int
    noise_sigma_used: float
    friction_enabled: bool
    total_candidate_pool: int


class DeterministicHashEmbedder:
    """
    Zero-VRAM, deterministic L2-normalized character/word trigram projection embedder
    in R^{embedding_dim}. Ensures identical semantic retrieval across CPU CI and 4 GB
    GPU workstations without competing for CUDA VRAM with the white-box LLM.
    Optionally wraps SentenceTransformer on CPU if explicitly requested and installed.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        model_name: str = "BAAI/bge-small-en-v1.5",
        use_neural_backend: bool = False,
    ) -> None:
        self.embedding_dim = int(embedding_dim)
        self.model_name = model_name
        self._neural_model: Any = None
        if use_neural_backend:
            try:
                from sentence_transformers import SentenceTransformer

                self._neural_model = SentenceTransformer(model_name, device="cpu")
            except Exception:
                self._neural_model = None

    def embed(self, text: str) -> np.ndarray:
        if self._neural_model is not None:
            vec = self._neural_model.encode(
                text, normalize_embeddings=True, show_progress_bar=False
            )
            arr = np.asarray(vec, dtype=np.float32)
            if arr.shape[0] == self.embedding_dim:
                return arr

        vec = np.zeros((self.embedding_dim,), dtype=np.float32)
        cleaned = (text or "").lower().strip()
        if not cleaned:
            return vec

        tokens = cleaned.split()
        features: List[str] = list(tokens)
        for tok in tokens:
            if len(tok) >= 3:
                for i in range(len(tok) - 2):
                    features.append(tok[i : i + 3])
        for i in range(len(tokens) - 1):
            features.append(f"{tokens[i]}_{tokens[i + 1]}")

        for feat in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            h_val = int.from_bytes(digest, byteorder="little", signed=False)
            bucket = h_val % self.embedding_dim
            sign = 1.0 if ((h_val >> 63) & 1) == 0 else -1.0
            vec[bucket] += sign

        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm
        return vec


class ScopedMemoryHandle:
    """Strictly isolates an agent's read/write operations to `owner_id`."""

    def __init__(self, store: "EpisodicMemoryTMS", owner_id: str) -> None:
        self._store = store
        self.owner_id = owner_id

    def _assert_owner(self, requested_agent_id: str) -> None:
        if requested_agent_id != self.owner_id:
            raise EpistemicIsolationError(
                f"Epistemic Isolation Violation: Handle '{self.owner_id}' cannot access '{requested_agent_id}'."
            )

    def write_episode(
        self,
        agent_id: str,
        tick: int,
        event_type: str,
        summary_text: str,
        salience_score: float = 1.0,
    ) -> EpisodicMemoryRecord:
        self._assert_owner(agent_id)
        return self._store._insert_episode(
            agent_id=self.owner_id,
            tick=tick,
            event_type=event_type,
            summary_text=summary_text,
            salience_score=salience_score,
        )

    def retrieve(
        self,
        agent_id: str,
        query_text: str,
        current_tick: int,
        stress_index: float,
        rng: np.random.Generator,
    ) -> RetrievalResult:
        self._assert_owner(agent_id)
        return self._store._retrieve_with_friction(
            agent_id=self.owner_id,
            query_text=query_text,
            current_tick=current_tick,
            stress_index=stress_index,
            rng=rng,
        )

    def update_peer_reliability(
        self,
        agent_id: str,
        peer_id: str,
        completed_signal: float,
        tick: int,
    ) -> float:
        self._assert_owner(agent_id)
        return self._store._update_peer_directory(
            owner_id=self.owner_id,
            peer_id=peer_id,
            completed_signal=completed_signal,
            tick=tick,
        )

    def get_top_peers_summary(
        self, agent_id: str, max_peers: int = 3
    ) -> List[Tuple[str, float, int]]:
        self._assert_owner(agent_id)
        return self._store._get_peer_directory_summary(
            owner_id=self.owner_id, max_peers=max_peers
        )


class EpisodicMemoryTMS:
    """
    Factor B (Domain-General Cognitive Bottleneck): Episodic Memory Friction &
    Transactive Memory System (TMS) SQLite WAL Store.
    """

    def __init__(
        self,
        config: MemoryFrictionConfig,
        db_path: str = ":memory:",
        use_neural_embeddings: bool = False,
    ) -> None:
        self.config = config
        self.db_path = db_path
        self.embedder = DeterministicHashEmbedder(
            embedding_dim=config.embedding_dim,
            model_name=config.embedding_model_name,
            use_neural_backend=use_neural_embeddings,
        )
        self._shared_conn: Optional[sqlite3.Connection] = (
            sqlite3.connect(":memory:") if db_path == ":memory:" else None
        )
        if self._shared_conn is not None:
            self._shared_conn.row_factory = sqlite3.Row
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _close_if_not_shared(self, conn: sqlite3.Connection) -> None:
        if self._shared_conn is None:
            conn.close()

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    mem_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    tick INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    salience_score REAL NOT NULL,
                    embedding_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_ep_agent_tick
                    ON episodic_memory (agent_id, tick DESC);

                CREATE TABLE IF NOT EXISTS tms_peer_directory (
                    owner_id TEXT NOT NULL,
                    peer_id TEXT NOT NULL,
                    alpha_obs REAL NOT NULL DEFAULT 2.0,
                    beta_obs REAL NOT NULL DEFAULT 2.0,
                    interaction_count INTEGER NOT NULL DEFAULT 0,
                    last_tick INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (owner_id, peer_id)
                );
                """
            )
            conn.commit()
        finally:
            self._close_if_not_shared(conn)

    def __deepcopy__(self, memo: Dict[int, Any]) -> "EpisodicMemoryTMS":
        """
        Safely clones `EpisodicMemoryTMS` (including its `:memory:` SQLite database)
        via `conn.iterdump()` so paired exogenous-windfall shadow worlds can run
        without triggering `TypeError: cannot pickle 'sqlite3.Connection' object`.
        """
        cls = self.__class__
        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj

        new_obj.config = copy.deepcopy(self.config, memo)
        new_obj.db_path = ":memory:"
        # Embedder is stateless/read-only; share safely without duplicating model weights
        new_obj.embedder = self.embedder

        src_conn = self._connect()
        try:
            src_conn.commit()
            dump_sql = "\n".join(src_conn.iterdump())
        finally:
            self._close_if_not_shared(src_conn)

        new_conn = sqlite3.connect(":memory:")
        new_conn.row_factory = sqlite3.Row
        new_conn.executescript(dump_sql)
        new_conn.commit()
        new_obj._shared_conn = new_conn
        return new_obj

    def get_scoped_handle(self, owner_id: str) -> ScopedMemoryHandle:
        return ScopedMemoryHandle(self, owner_id)

    def compute_effective_top_k_and_sigma(
        self, stress_index: float
    ) -> Tuple[int, float]:
        s = max(0.0, min(1.0, float(stress_index)))
        if not self.config.enabled:
            return (int(self.config.top_k_calm), 0.0)

        k_max = int(self.config.top_k_calm)
        k_min = int(self.config.top_k_stressed)
        eff_k = max(k_min, int(math.floor(k_max - (k_max - k_min) * s)))
        eff_sigma = round(float(self.config.similarity_noise_sigma_max) * s, 6)
        return (eff_k, eff_sigma)

    def _insert_episode(
        self,
        agent_id: str,
        tick: int,
        event_type: str,
        summary_text: str,
        salience_score: float = 1.0,
    ) -> EpisodicMemoryRecord:
        emb = self.embedder.embed(summary_text)
        emb_list = [round(float(x), 6) for x in emb.tolist()]
        mem_id = f"ep_{tick:04d}_{uuid.uuid4().hex[:8]}"
        sal = max(0.05, float(salience_score))

        record = EpisodicMemoryRecord(
            mem_id=mem_id,
            agent_id=agent_id,
            tick=tick,
            event_type=event_type,  # type: ignore[arg-type]
            summary_text=summary_text,
            salience_score=sal,
            embedding=emb_list,
        )

        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO episodic_memory (
                    mem_id, agent_id, tick, event_type, summary_text, salience_score, embedding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    record.mem_id,
                    record.agent_id,
                    record.tick,
                    record.event_type,
                    record.summary_text,
                    record.salience_score,
                    json.dumps(emb_list),
                ),
            )
            cap = int(self.config.max_episodic_capacity)
            conn.execute(
                """
                DELETE FROM episodic_memory
                WHERE agent_id = ? AND mem_id NOT IN (
                    SELECT mem_id FROM episodic_memory
                    WHERE agent_id = ?
                    ORDER BY tick DESC, salience_score DESC
                    LIMIT ?
                );
                """,
                (agent_id, agent_id, cap),
            )
            conn.commit()
        finally:
            self._close_if_not_shared(conn)
        return record

    def _retrieve_with_friction(
        self,
        agent_id: str,
        query_text: str,
        current_tick: int,
        stress_index: float,
        rng: np.random.Generator,
    ) -> RetrievalResult:
        eff_k, eff_sigma = self.compute_effective_top_k_and_sigma(stress_index)
        q_vec = self.embedder.embed(query_text)

        if self.config.enabled and eff_sigma > 1e-8:
            emb_noise = rng.normal(
                0.0,
                eff_sigma / math.sqrt(float(self.config.embedding_dim)),
                size=q_vec.shape,
            ).astype(np.float32)
            q_noisy = q_vec + emb_noise
            q_norm = float(np.linalg.norm(q_noisy))
            if q_norm > 1e-8:
                q_vec = q_noisy / q_norm

        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT mem_id, agent_id, tick, event_type, summary_text, salience_score, embedding_json
                FROM episodic_memory
                WHERE agent_id = ?
                ORDER BY tick DESC;
                """,
                (agent_id,),
            ).fetchall()
        finally:
            self._close_if_not_shared(conn)

        if not rows:
            return RetrievalResult(
                agent_id=agent_id,
                records=[],
                top_k_used=eff_k,
                noise_sigma_used=eff_sigma,
                friction_enabled=self.config.enabled,
                total_candidate_pool=0,
            )

        w_sal = float(self.config.recency_salience_weight)
        eff_w_sal = (
            min(0.75, w_sal + 0.35 * max(0.0, min(1.0, stress_index)))
            if self.config.enabled
            else w_sal
        )

        scored: List[Tuple[float, EpisodicMemoryRecord]] = []
        for r in rows:
            emb_list = json.loads(r["embedding_json"])
            m_vec = np.asarray(emb_list, dtype=np.float32)
            cos_sim = float(np.dot(q_vec, m_vec))
            delta_t = max(0, current_tick - int(r["tick"]))
            sal_recency = float(r["salience_score"]) * math.exp(-0.15 * delta_t)

            score_noise = (
                float(rng.normal(0.0, eff_sigma))
                if (self.config.enabled and eff_sigma > 1e-8)
                else 0.0
            )
            total_score = (
                (1.0 - eff_w_sal) * cos_sim
                + eff_w_sal * sal_recency
                + score_noise
            )
            rec = EpisodicMemoryRecord(
                mem_id=r["mem_id"],
                agent_id=r["agent_id"],
                tick=int(r["tick"]),
                event_type=r["event_type"],
                summary_text=r["summary_text"],
                salience_score=float(r["salience_score"]),
                embedding=None,
            )
            scored.append((total_score, rec))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [rec for _, rec in scored[:eff_k]]

        return RetrievalResult(
            agent_id=agent_id,
            records=selected,
            top_k_used=eff_k,
            noise_sigma_used=eff_sigma,
            friction_enabled=self.config.enabled,
            total_candidate_pool=len(rows),
        )

    def _update_peer_directory(
        self,
        owner_id: str,
        peer_id: str,
        completed_signal: float,
        tick: int,
    ) -> float:
        sig = max(0.0, min(1.0, float(completed_signal)))
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT alpha_obs, beta_obs, interaction_count
                FROM tms_peer_directory
                WHERE owner_id = ? AND peer_id = ?;
                """,
                (owner_id, peer_id),
            ).fetchone()
            if row is None:
                new_a = 2.0 + sig
                new_b = 2.0 + (1.0 - sig)
                new_n = 1
            else:
                new_a = round(float(row["alpha_obs"]) + sig, 4)
                new_b = round(float(row["beta_obs"]) + (1.0 - sig), 4)
                new_n = int(row["interaction_count"]) + 1

            conn.execute(
                """
                INSERT OR REPLACE INTO tms_peer_directory
                (owner_id, peer_id, alpha_obs, beta_obs, interaction_count, last_tick)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (owner_id, peer_id, new_a, new_b, new_n, tick),
            )
            conn.commit()
        finally:
            self._close_if_not_shared(conn)
        return round(new_a / max(1e-6, new_a + new_b), 4)

    def _get_peer_directory_summary(
        self, owner_id: str, max_peers: int = 3
    ) -> List[Tuple[str, float, int]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT peer_id, alpha_obs, beta_obs, interaction_count
                FROM tms_peer_directory
                WHERE owner_id = ?
                ORDER BY interaction_count DESC, last_tick DESC
                LIMIT ?;
                """,
                (owner_id, max_peers),
            ).fetchall()
        finally:
            self._close_if_not_shared(conn)

        out: List[Tuple[str, float, int]] = []
        for r in rows:
            a = float(r["alpha_obs"])
            b = float(r["beta_obs"])
            rate = round(a / max(1e-6, a + b), 3)
            out.append((str(r["peer_id"]), rate, int(r["interaction_count"])))
        return out