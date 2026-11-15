"""
src/memory/scoped_memory.py
MODULE 5 (M5): Multi-Tier Scoped Memory Architecture with:
  1. Embedded SQLite WAL relational persistence across all agent & macro tables.
  2. Strict Epistemic Isolation (`ScopedAgentMemoryView(owner_id)`) raising
     `EpistemicIsolationViolationError` (`PermissionError`) on cross-agent access.
  3. Two-Pass Scoped Relational Retrieval:
     - Pass 1: Self state + Direct Counterparty B.
     - Pass 2: Fast Aho-Corasick / Multi-Entity Mention Scanner over incoming text
       to dynamically inject R(A -> C) ONLY when third party C is mentioned.
  4. Ebbinghaus-McGaugh emotional retention decay & STM -> LTM consolidation.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import uuid
from typing import Dict, List, Optional, Set, Tuple

from src.core.contracts import (
    BigFiveTraits,
    EpisodicMemoryRecord,
    ExpertiseDomain,
    ExternalAgentState,
    FehrSchmidtParams,
    InternalAgentState,
    LegalStatus,
    MacroEconomyState,
    MaslowNeeds,
    MemoryCognitionSpec,
    MoodPAD,
    PeerPerceivedModel,
    ProspectTheoryParams,
    RelationalEdge,
    RelationshipType,
    TMSPeerCompetenceRecord,
)


class EpistemicIsolationViolationError(PermissionError):
    """Raised whenever an agent attempts to read or mutate another agent's private memory."""


class AhoCorasickEntityScanner:
    """
    Fast single-pass entity mention automaton scanning incoming verbal messages or
    gossip for third-party agent IDs or names in O(len(text)) time.
    """

    def __init__(self, entity_directory: Dict[str, str]) -> None:
        """
        Args:
            entity_directory: mapping of surface strings (e.g., "Agent_3", "Elena")
                              to canonical `agent_id` (e.g., "Agent_3").
        """
        self.surface_to_id: Dict[str, str] = {
            k.lower(): v for k, v in entity_directory.items() if k.strip()
        }
        if self.surface_to_id:
            escaped = sorted((re.escape(k) for k in self.surface_to_id.keys()), key=len, reverse=True)
            self._regex = re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)
        else:
            self._regex = None

    def scan_mentioned_agent_ids(
        self,
        text: str,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[str]:
        if not text or self._regex is None:
            return []
        exclude = exclude_ids or set()
        found_ids: List[str] = []
        for match in self._regex.finditer(text):
            token = match.group(0).lower()
            canonical_id = self.surface_to_id.get(token)
            if canonical_id and canonical_id not in exclude and canonical_id not in found_ids:
                found_ids.append(canonical_id)
        return found_ids


class ScopedMemoryEngine:
    """
    Manages the SQLite WAL database and issues `ScopedAgentMemoryView` objects
    bound strictly to a single `owner_id`.
    """

    def __init__(self, db_path: str, spec: MemoryCognitionSpec) -> None:
        self.db_path = db_path
        self.spec = spec
        db_dir = os.path.dirname(os.path.abspath(db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_wal_and_schema()
        self._entity_directory: Dict[str, str] = {}
        self._scanner = AhoCorasickEntityScanner({})

    def _init_wal_and_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")

        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS agents_internal (
                agent_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agents_external_ledger (
                agent_id TEXT PRIMARY KEY,
                liquid_wealth_cash REAL NOT NULL,
                bank_deposit REAL NOT NULL,
                outstanding_debt REAL NOT NULL,
                credit_score REAL NOT NULL,
                inventory_json TEXT NOT NULL,
                base_wage REAL NOT NULL,
                social_status REAL NOT NULL,
                legal_status TEXT NOT NULL,
                spatial_node_id TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS relational_edges (
                owner_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_json TEXT NOT NULL,
                PRIMARY KEY (owner_id, target_id)
            );

            CREATE TABLE IF NOT EXISTS peer_perceived_models (
                owner_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                model_json TEXT NOT NULL,
                PRIMARY KEY (owner_id, target_id)
            );

            CREATE TABLE IF NOT EXISTS tms_directory (
                owner_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                alpha_success REAL NOT NULL,
                beta_failure REAL NOT NULL,
                last_queried_tick INTEGER NOT NULL,
                PRIMARY KEY (owner_id, target_id, domain)
            );

            CREATE TABLE IF NOT EXISTS episodic_memories (
                mem_id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                tick_created INTEGER NOT NULL,
                encoding_salience REAL NOT NULL,
                rpe_magnitude REAL NOT NULL,
                rehearsal_count INTEGER NOT NULL,
                valence_at_encoding REAL NOT NULL,
                arousal_at_encoding REAL NOT NULL,
                involved_peers_json TEXT NOT NULL,
                narrative_text TEXT NOT NULL,
                is_consolidated_ltm INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_episodic_owner
                ON episodic_memories(owner_id, tick_created DESC);

            CREATE TABLE IF NOT EXISTS macro_institutions (
                singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                state_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS acid_transaction_ledger (
                tx_id TEXT PRIMARY KEY,
                tick INTEGER NOT NULL,
                status TEXT NOT NULL,
                initiator_id TEXT NOT NULL,
                responder_id TEXT,
                cash_delta_initiator REAL NOT NULL,
                cash_delta_responder REAL NOT NULL,
                tax_collected REAL NOT NULL,
                item_id TEXT NOT NULL,
                quantity_transferred INTEGER NOT NULL,
                diagnostic_feedback TEXT NOT NULL,
                conservation_error REAL NOT NULL
            );
            """
        )
        self.conn.commit()

    def register_population(
        self,
        internal_states: List[InternalAgentState],
        external_states: List[ExternalAgentState],
        macro_state: MacroEconomyState,
    ) -> None:
        """Initializes all agents, pairwise relationships, TMS priors, and macro accounts."""
        cur = self.conn.cursor()
        self._entity_directory.clear()

        for istate in internal_states:
            cur.execute(
                "INSERT OR REPLACE INTO agents_internal (agent_id, state_json) VALUES (?, ?)",
                (istate.agent_id, istate.model_dump_json()),
            )
            self._entity_directory[istate.agent_id] = istate.agent_id
            self._entity_directory[istate.name] = istate.agent_id

        for estate in external_states:
            cur.execute(
                """
                INSERT OR REPLACE INTO agents_external_ledger (
                    agent_id, liquid_wealth_cash, bank_deposit, outstanding_debt,
                    credit_score, inventory_json, base_wage, social_status,
                    legal_status, spatial_node_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    estate.agent_id,
                    estate.liquid_wealth_cash,
                    estate.bank_deposit,
                    estate.outstanding_debt,
                    estate.credit_score,
                    json.dumps(estate.inventory),
                    estate.base_wage,
                    estate.social_status,
                    estate.legal_status.value,
                    estate.spatial_node_id,
                ),
            )

        # Initialize directed pairwise edges, perceived ToM models, and TMS directory
        for a in internal_states:
            for b in internal_states:
                if a.agent_id == b.agent_id:
                    continue
                edge = RelationalEdge(
                    owner_id=a.agent_id,
                    target_id=b.agent_id,
                    rel_type=RelationshipType.ACQUAINTANCE,
                    trust_tau=0.15 if a.specialty_domain == b.specialty_domain else 0.05,
                    intimacy_alpha=0.10,
                    credibility_kappa=0.55,
                    tie_strength_w=0.25,
                )
                cur.execute(
                    "INSERT OR REPLACE INTO relational_edges (owner_id, target_id, edge_json) VALUES (?, ?, ?)",
                    (a.agent_id, b.agent_id, edge.model_dump_json()),
                )

                pmodel = PeerPerceivedModel(
                    owner_id=a.agent_id,
                    target_id=b.agent_id,
                    est_honesty=0.55,
                    est_greed=0.45,
                    est_fehr_alpha=b.fehr_schmidt.disadvantageous_inequity_alpha,
                    narrative_summary=f"{b.name} ({b.agent_id}) works as {b.occupation} specializing in {b.specialty_domain.value}.",
                )
                cur.execute(
                    "INSERT OR REPLACE INTO peer_perceived_models (owner_id, target_id, model_json) VALUES (?, ?, ?)",
                    (a.agent_id, b.agent_id, pmodel.model_dump_json()),
                )

                for domain in ExpertiseDomain:
                    # Slightly higher prior for a peer's actual specialty domain
                    is_spec = b.specialty_domain == domain
                    alpha_init = 3.5 if is_spec else 2.0
                    beta_init = 1.5 if is_spec else 2.0
                    cur.execute(
                        """
                        INSERT OR REPLACE INTO tms_directory
                        (owner_id, target_id, domain, alpha_success, beta_failure, last_queried_tick)
                        VALUES (?, ?, ?, ?, ?, 0)
                        """,
                        (a.agent_id, b.agent_id, domain.value, alpha_init, beta_init),
                    )

        cur.execute(
            "INSERT OR REPLACE INTO macro_institutions (singleton_id, state_json) VALUES (1, ?)",
            (macro_state.model_dump_json(),),
        )
        self.conn.commit()
        self._scanner = AhoCorasickEntityScanner(self._entity_directory)

    def get_scoped_view(self, owner_id: str) -> "ScopedAgentMemoryView":
        """Returns an epistemically isolated memory view bound strictly to `owner_id`."""
        return ScopedAgentMemoryView(
            conn=self.conn,
            owner_id=owner_id,
            spec=self.spec,
            scanner=self._scanner,
        )

    def get_external_state(self, agent_id: str) -> ExternalAgentState:
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT * FROM agents_external_ledger WHERE agent_id = ?", (agent_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown agent_id in external ledger: {agent_id}")
        return ExternalAgentState(
            agent_id=row["agent_id"],
            liquid_wealth_cash=float(row["liquid_wealth_cash"]),
            bank_deposit=float(row["bank_deposit"]),
            outstanding_debt=float(row["outstanding_debt"]),
            credit_score=float(row["credit_score"]),
            inventory=json.loads(row["inventory_json"]),
            base_wage=float(row["base_wage"]),
            social_status=float(row["social_status"]),
            legal_status=LegalStatus(row["legal_status"]),
            spatial_node_id=row["spatial_node_id"],
        )

    def get_all_external_states(self) -> Dict[str, ExternalAgentState]:
        cur = self.conn.cursor()
        rows = cur.execute("SELECT agent_id FROM agents_external_ledger").fetchall()
        return {r["agent_id"]: self.get_external_state(r["agent_id"]) for r in rows}

    def get_macro_state(self) -> MacroEconomyState:
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT state_json FROM macro_institutions WHERE singleton_id = 1"
        ).fetchone()
        return MacroEconomyState.model_validate_json(row["state_json"])

    def update_macro_state(self, macro_state: MacroEconomyState) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE macro_institutions SET state_json = ? WHERE singleton_id = 1",
            (macro_state.model_dump_json(),),
        )
        self.conn.commit()


class ScopedAgentMemoryView:
    """
    Enforces strict partition-key isolation (`owner_id == self.owner_id`).
    Any attempt to read or write another agent's private state or relational edges
    raises `EpistemicIsolationViolationError` (a subclass of `PermissionError`).
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        owner_id: str,
        spec: MemoryCognitionSpec,
        scanner: AhoCorasickEntityScanner,
    ) -> None:
        self._conn = conn
        self.owner_id = owner_id
        self.spec = spec
        self._scanner = scanner

    def _assert_ownership(self, requested_owner_id: str) -> None:
        if requested_owner_id != self.owner_id:
            raise EpistemicIsolationViolationError(
                f"EPISTEMIC ISOLATION VIOLATION: ScopedView(owner='{self.owner_id}') "
                f"attempted illegal access to private partition of '{requested_owner_id}'."
            )

    def get_internal_state(self, requested_owner_id: Optional[str] = None) -> InternalAgentState:
        target = requested_owner_id or self.owner_id
        self._assert_ownership(target)
        cur = self._conn.cursor()
        row = cur.execute(
            "SELECT state_json FROM agents_internal WHERE agent_id = ?",
            (self.owner_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Internal state not found for {self.owner_id}")
        return InternalAgentState.model_validate_json(row["state_json"])

    def save_internal_state(self, state: InternalAgentState) -> None:
        self._assert_ownership(state.agent_id)
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE agents_internal SET state_json = ? WHERE agent_id = ?",
            (state.model_dump_json(), self.owner_id),
        )
        self._conn.commit()

    def get_relational_edge(
        self, target_id: str, requested_owner_id: Optional[str] = None
    ) -> RelationalEdge:
        owner = requested_owner_id or self.owner_id
        self._assert_ownership(owner)
        cur = self._conn.cursor()
        row = cur.execute(
            "SELECT edge_json FROM relational_edges WHERE owner_id = ? AND target_id = ?",
            (self.owner_id, target_id),
        ).fetchone()
        if row is None:
            return RelationalEdge(owner_id=self.owner_id, target_id=target_id)
        return RelationalEdge.model_validate_json(row["edge_json"])

    def save_relational_edge(self, edge: RelationalEdge) -> None:
        self._assert_ownership(edge.owner_id)
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO relational_edges (owner_id, target_id, edge_json) VALUES (?, ?, ?)",
            (self.owner_id, edge.target_id, edge.model_dump_json()),
        )
        self._conn.commit()

    def get_all_outgoing_edges(self) -> Dict[str, RelationalEdge]:
        cur = self._conn.cursor()
        rows = cur.execute(
            "SELECT target_id, edge_json FROM relational_edges WHERE owner_id = ?",
            (self.owner_id,),
        ).fetchall()
        return {
            r["target_id"]: RelationalEdge.model_validate_json(r["edge_json"])
            for r in rows
        }

    def get_peer_perceived_model(
        self, target_id: str, requested_owner_id: Optional[str] = None
    ) -> PeerPerceivedModel:
        owner = requested_owner_id or self.owner_id
        self._assert_ownership(owner)
        cur = self._conn.cursor()
        row = cur.execute(
            "SELECT model_json FROM peer_perceived_models WHERE owner_id = ? AND target_id = ?",
            (self.owner_id, target_id),
        ).fetchone()
        if row is None:
            return PeerPerceivedModel(owner_id=self.owner_id, target_id=target_id)
        return PeerPerceivedModel.model_validate_json(row["model_json"])

    def save_peer_perceived_model(self, model: PeerPerceivedModel) -> None:
        self._assert_ownership(model.owner_id)
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO peer_perceived_models (owner_id, target_id, model_json) VALUES (?, ?, ?)",
            (self.owner_id, model.target_id, model.model_dump_json()),
        )
        self._conn.commit()

    def dual_pass_scoped_retrieval(
        self,
        direct_partner_id: Optional[str],
        incoming_message_text: str,
        current_tick: int,
    ) -> Tuple[
        Dict[str, RelationalEdge],
        Dict[str, PeerPerceivedModel],
        List[EpisodicMemoryRecord],
    ]:
        """
        Executes the Two-Pass Scoped Relational Retrieval Architecture:
          - Pass 1: Loads R(owner -> direct_partner_id) and ToM model M(owner -> direct_partner_id).
          - Pass 2: Runs the Aho-Corasick Entity Scanner over `incoming_message_text`.
            If third-party Agent C is mentioned, dynamically loads R(owner -> C) and M(owner -> C).
            All uninvolved peers consume ZERO prompt tokens.
        """
        relevant_peers: List[str] = []
        if direct_partner_id and direct_partner_id != self.owner_id:
            relevant_peers.append(direct_partner_id)

        # Pass 2: Scan incoming text for third-party entity mentions
        mentioned_third_parties = self._scanner.scan_mentioned_agent_ids(
            text=incoming_message_text,
            exclude_ids={self.owner_id} | set(relevant_peers),
        )
        relevant_peers.extend(mentioned_third_parties)

        scoped_edges: Dict[str, RelationalEdge] = {}
        scoped_models: Dict[str, PeerPerceivedModel] = {}
        for peer_id in relevant_peers:
            scoped_edges[peer_id] = self.get_relational_edge(peer_id)
            scoped_models[peer_id] = self.get_peer_perceived_model(peer_id)

        # Retrieve top episodic memories involving relevant peers (or high salience)
        memories = self.retrieve_active_memories(
            current_tick=current_tick,
            relevant_peers=set(relevant_peers),
            top_k=5,
        )
        return scoped_edges, scoped_models, memories

    def record_episodic_memory(
        self,
        tick: int,
        narrative_text: str,
        involved_peers: List[str],
        arousal: float,
        valence: float,
        rpe_magnitude: float,
        goal_relevance: float = 0.5,
    ) -> EpisodicMemoryRecord:
        """
        Computes McGaugh amygdalar encoding salience:
            S_enc(m) = 1 + w_a * A(t) + w_rpe * |delta_RPE| + w_goal * GoalRel
        and stores the episodic trace in `episodic_memories`.
        """
        encoding_salience = (
            1.0
            + 0.85 * float(arousal)
            + 1.10 * min(2.0, float(rpe_magnitude))
            + 0.50 * float(goal_relevance)
        )
        is_ltm = encoding_salience >= self.spec.consolidation_salience_threshold
        rec = EpisodicMemoryRecord(
            mem_id=f"mem_{uuid.uuid4().hex[:10]}",
            owner_id=self.owner_id,
            tick_created=tick,
            encoding_salience=round(encoding_salience, 4),
            rpe_magnitude=round(rpe_magnitude, 4),
            rehearsal_count=0,
            valence_at_encoding=round(valence, 4),
            arousal_at_encoding=round(arousal, 4),
            involved_peers=involved_peers,
            narrative_text=narrative_text,
            is_consolidated_ltm=is_ltm,
        )
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO episodic_memories (
                mem_id, owner_id, tick_created, encoding_salience, rpe_magnitude,
                rehearsal_count, valence_at_encoding, arousal_at_encoding,
                involved_peers_json, narrative_text, is_consolidated_ltm
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.mem_id,
                self.owner_id,
                rec.tick_created,
                rec.encoding_salience,
                rec.rpe_magnitude,
                rec.rehearsal_count,
                rec.valence_at_encoding,
                rec.arousal_at_encoding,
                json.dumps(rec.involved_peers),
                rec.narrative_text,
                1 if rec.is_consolidated_ltm else 0,
            ),
        )
        self._conn.commit()
        return rec

    def retrieve_active_memories(
        self,
        current_tick: int,
        relevant_peers: Optional[Set[str]] = None,
        top_k: int = 5,
    ) -> List[EpisodicMemoryRecord]:
        """
        Applies Ebbinghaus-McGaugh retention curve:
            R_m(Delta_t) = exp( - Delta_t / (tau_0 * S_enc(m) * (1 + ln(1 + n_reh))) )
        Prunes decayed STM records (R_m < 0.15) and returns top-k memories.
        """
        cur = self._conn.cursor()
        rows = cur.execute(
            "SELECT * FROM episodic_memories WHERE owner_id = ? ORDER BY tick_created DESC",
            (self.owner_id,),
        ).fetchall()

        scored: List[Tuple[float, EpisodicMemoryRecord]] = []
        expired_ids: List[str] = []
        tau_0 = max(0.5, self.spec.ebbinghaus_base_half_life_ticks)

        for r in rows:
            peers = json.loads(r["involved_peers_json"])
            rec = EpisodicMemoryRecord(
                mem_id=r["mem_id"],
                owner_id=r["owner_id"],
                tick_created=int(r["tick_created"]),
                encoding_salience=float(r["encoding_salience"]),
                rpe_magnitude=float(r["rpe_magnitude"]),
                rehearsal_count=int(r["rehearsal_count"]),
                valence_at_encoding=float(r["valence_at_encoding"]),
                arousal_at_encoding=float(r["arousal_at_encoding"]),
                involved_peers=peers,
                narrative_text=r["narrative_text"],
                is_consolidated_ltm=bool(r["is_consolidated_ltm"]),
            )
            delta_t = max(0, current_tick - rec.tick_created)
            effective_stability = tau_0 * rec.encoding_salience * (
                1.0 + math.log(1.0 + rec.rehearsal_count)
            )
            retention = math.exp(-float(delta_t) / max(1e-4, effective_stability))

            if not rec.is_consolidated_ltm and retention < 0.15:
                expired_ids.append(rec.mem_id)
                continue

            peer_bonus = 1.5 if (relevant_peers and any(p in relevant_peers for p in peers)) else 1.0
            retrieval_score = retention * rec.encoding_salience * peer_bonus
            scored.append((retrieval_score, rec))

        if expired_ids:
            cur.executemany(
                "DELETE FROM episodic_memories WHERE mem_id = ?",
                [(mid,) for mid in expired_ids],
            )
            self._conn.commit()

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [item[1] for item in scored[:top_k]]

        # Increment rehearsal count and consolidate if threshold reached
        for rec in selected:
            new_reh = rec.rehearsal_count + 1
            new_sal = rec.encoding_salience + 0.15 * math.log(1.0 + new_reh)
            consolidate = 1 if (rec.is_consolidated_ltm or new_sal >= self.spec.consolidation_salience_threshold) else 0
            cur.execute(
                """
                UPDATE episodic_memories
                SET rehearsal_count = ?, encoding_salience = ?, is_consolidated_ltm = ?
                WHERE mem_id = ?
                """,
                (new_reh, round(new_sal, 4), consolidate, rec.mem_id),
            )
        if selected:
            self._conn.commit()

        return selected

    def get_tms_competence_directory(
        self, domain: ExpertiseDomain
    ) -> List[TMSPeerCompetenceRecord]:
        cur = self._conn.cursor()
        rows = cur.execute(
            """
            SELECT owner_id, target_id, domain, alpha_success, beta_failure, last_queried_tick
            FROM tms_directory
            WHERE owner_id = ? AND domain = ?
            """,
            (self.owner_id, domain.value),
        ).fetchall()
        return [
            TMSPeerCompetenceRecord(
                owner_id=r["owner_id"],
                target_id=r["target_id"],
                domain=ExpertiseDomain(r["domain"]),
                alpha_success=float(r["alpha_success"]),
                beta_failure=float(r["beta_failure"]),
                last_queried_tick=int(r["last_queried_tick"]),
            )
            for r in rows
        ]

    def update_tms_peer_belief(
        self,
        target_id: str,
        domain: ExpertiseDomain,
        outcome_quality_y: float,
        tick: int,
    ) -> TMSPeerCompetenceRecord:
        """
        Bayesian Beta-Bernoulli conjugate update:
            alpha^{(t+1)} = alpha^{(t)} + y
            beta^{(t+1)}  = beta^{(t)} + (1 - y)
        """
        y = max(0.0, min(1.0, float(outcome_quality_y)))
        cur = self._conn.cursor()
        row = cur.execute(
            """
            SELECT alpha_success, beta_failure FROM tms_directory
            WHERE owner_id = ? AND target_id = ? AND domain = ?
            """,
            (self.owner_id, target_id, domain.value),
        ).fetchone()
        alpha_new = (float(row["alpha_success"]) if row else 2.0) + y
        beta_new = (float(row["beta_failure"]) if row else 2.0) + (1.0 - y)
        cur.execute(
            """
            INSERT OR REPLACE INTO tms_directory
            (owner_id, target_id, domain, alpha_success, beta_failure, last_queried_tick)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (self.owner_id, target_id, domain.value, alpha_new, beta_new, tick),
        )
        self._conn.commit()
        return TMSPeerCompetenceRecord(
            owner_id=self.owner_id,
            target_id=target_id,
            domain=domain,
            alpha_success=alpha_new,
            beta_failure=beta_new,
            last_queried_tick=tick,
        )