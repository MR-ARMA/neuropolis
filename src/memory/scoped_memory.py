from __future__ import annotations

import json
import math
import sqlite3
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.contracts import SimulationConfig


class EpistemicIsolationViolationError(PermissionError):
    """Raised when an agent attempts to read or mutate another agent's private memory."""


class AhoCorasickEntityScanner:
    """
    Deterministic O(len(text)) Aho-Corasick trie automaton for scanning incoming
    messages and proposals to identify mentioned third-party agents.
    """

    def __init__(self, agent_ids: List[str]) -> None:
        self.trie: List[Dict[str, int]] = [{}]
        self.output: List[Set[str]] = [set()]
        self.fail: List[int] = [0]
        self._build_automaton(agent_ids)

    def _add_keyword(self, keyword: str, canonical_id: str) -> None:
        node = 0
        for ch in keyword.lower():
            if ch not in self.trie[node]:
                self.trie[node][ch] = len(self.trie)
                self.trie.append({})
                self.output.append(set())
                self.fail.append(0)
            node = self.trie[node][ch]
        self.output[node].add(canonical_id)

    def _build_automaton(self, agent_ids: List[str]) -> None:
        for aid in agent_ids:
            self._add_keyword(aid, aid)
            parts = aid.split("_")
            if len(parts) >= 2:
                self._add_keyword(parts[-1], aid)

        queue: List[int] = []
        for ch, next_node in self.trie[0].items():
            self.fail[next_node] = 0
            queue.append(next_node)

        while queue:
            curr = queue.pop(0)
            for ch, nxt in self.trie[curr].items():
                queue.append(nxt)
                f = self.fail[curr]
                while f > 0 and ch not in self.trie[f]:
                    f = self.fail[f]
                self.fail[nxt] = self.trie[f].get(ch, 0)
                self.output[nxt].update(self.output[self.fail[nxt]])

    def scan(self, text: str, exclude_ids: Optional[Set[str]] = None) -> List[str]:
        if not text:
            return []
        exclude = exclude_ids or set()
        node = 0
        found: Set[str] = set()
        for ch in text.lower():
            while node > 0 and ch not in self.trie[node]:
                node = self.fail[node]
            node = self.trie[node].get(ch, 0)
            if self.output[node]:
                for cid in self.output[node]:
                    if cid not in exclude:
                        found.add(cid)
        return sorted(found)


class ScopedAgentMemoryView:
    """Logically bound memory handle enforcing WHERE owner_id = self.owner_id."""

    def __init__(self, memory_engine: "MultiTierScopedMemory", owner_id: str) -> None:
        self._engine = memory_engine
        self.owner_id = owner_id

    def assert_access(self, requested_owner_id: str) -> None:
        if requested_owner_id != self.owner_id:
            raise EpistemicIsolationViolationError(
                f"Epistemic Isolation Breach: Agent '{self.owner_id}' cannot access private state of '{requested_owner_id}'."
            )

    def retrieve_context(
        self,
        requested_owner_id: str,
        partner_id: Optional[str],
        incoming_texts: List[str],
        current_tick: int,
    ) -> Dict[str, Any]:
        self.assert_access(requested_owner_id)
        return self._engine._retrieve_scoped_context_internal(
            owner_id=self.owner_id,
            partner_id=partner_id,
            incoming_texts=incoming_texts,
            current_tick=current_tick,
        )


class MultiTierScopedMemory:
    """
    M5: Remediated Multi-Tier Psychological, Relational & Episodic Memory Engine.
    - Enforces strict epistemic isolation (`ScopedAgentMemoryView`).
    - Provides full episodic trade and message narratives to the prompt (Fixing Section 2.6).
    - Records explicit `llm_valid` and `attention_measured` flags in `inference_telemetry`.
    """

    def __init__(self, db_path: str, config: SimulationConfig) -> None:
        self.db_path = db_path
        self.config = config
        self.omega = config.orthogonal_clamping
        self.trust_spec = config.trust_and_market_spec
        self.contagion_spec = config.contagion_and_opinion_spec
        self.agent_ids = [a.agent_id for a in config.agents]
        self.scanner = AhoCorasickEntityScanner(self.agent_ids)

        self.generalized_social_trust: float = (
            self.trust_spec.initial_generalized_trust
            if self.omega.omega_d2_trust_and_parochialism
            else 0.50
        )
        self.institutional_trust: float = (
            self.trust_spec.initial_institutional_trust
            if self.omega.omega_d2_trust_and_parochialism
            else 0.50
        )
        self._init_memory_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def get_scoped_view(self, owner_id: str) -> ScopedAgentMemoryView:
        return ScopedAgentMemoryView(self, owner_id)

    def _init_memory_tables(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS agents_internal (
                    agent_id TEXT PRIMARY KEY,
                    big_five_json TEXT NOT NULL,
                    fehr_schmidt_json TEXT NOT NULL,
                    cpt_params_json TEXT NOT NULL,
                    pad_mood_json TEXT NOT NULL,
                    maslow_json TEXT NOT NULL,
                    fatigue_f REAL NOT NULL DEFAULT 0.10,
                    identity_rigidity REAL NOT NULL DEFAULT 0.50,
                    innate_anchor_s REAL NOT NULL DEFAULT 0.0,
                    expressed_opinion_z REAL NOT NULL DEFAULT 0.0,
                    contagion_adopted_y INTEGER NOT NULL DEFAULT 0,
                    contagion_exposures_count INTEGER NOT NULL DEFAULT 0,
                    hebbian_vec_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS relational_edges (
                    owner_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    rel_type TEXT NOT NULL,
                    trust_tau_01 REAL NOT NULL DEFAULT 0.55,
                    trust_tau REAL NOT NULL DEFAULT 0.10,
                    intimacy_alpha REAL NOT NULL DEFAULT 0.10,
                    credibility_kappa REAL NOT NULL DEFAULT 0.50,
                    interaction_count_n INTEGER NOT NULL DEFAULT 0,
                    hk_confidence_eps REAL NOT NULL DEFAULT 0.50,
                    coop_stats_json TEXT NOT NULL,
                    PRIMARY KEY (owner_id, target_id)
                );

                CREATE TABLE IF NOT EXISTS peer_perceived_models (
                    owner_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    perceived_traits_json TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    PRIMARY KEY (owner_id, target_id)
                );

                CREATE TABLE IF NOT EXISTS tms_directory (
                    owner_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    domain_id TEXT NOT NULL,
                    alpha_prior REAL NOT NULL,
                    beta_prior REAL NOT NULL,
                    last_updated_tick INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (owner_id, target_id, domain_id)
                );

                CREATE TABLE IF NOT EXISTS stm_episodic (
                    mem_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    tick INTEGER NOT NULL,
                    partner_id TEXT,
                    consent_decision TEXT NOT NULL,
                    action_category TEXT NOT NULL,
                    internal_thought TEXT NOT NULL,
                    verbal_message TEXT NOT NULL,
                    encoding_salience REAL NOT NULL,
                    rpe_magnitude REAL NOT NULL DEFAULT 0.0,
                    rehearsal_count INTEGER NOT NULL DEFAULT 0,
                    affect_pad_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ltm_episodic (
                    mem_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    tick INTEGER NOT NULL,
                    partner_id TEXT,
                    narrative_summary TEXT NOT NULL,
                    encoding_salience REAL NOT NULL,
                    rehearsal_count INTEGER NOT NULL DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS inference_telemetry (
                    tick INTEGER NOT NULL,
                    agent_id TEXT NOT NULL,
                    partner_id TEXT,
                    wave INTEGER NOT NULL DEFAULT 1,
                    llm_valid INTEGER NOT NULL DEFAULT 1,
                    attention_measured INTEGER NOT NULL DEFAULT 0,
                    financial_stress REAL NOT NULL,
                    easterbrook_beta REAL NOT NULL,
                    mean_shannon_entropy REAL NOT NULL,
                    tms_triggered INTEGER NOT NULL DEFAULT 0,
                    attn_mass_drives_finance REAL NOT NULL,
                    attn_mass_peer_memory REAL NOT NULL,
                    attn_mass_incoming_msg REAL NOT NULL,
                    attn_mass_sink REAL NOT NULL
                );
                """
            )

            cur = conn.execute("SELECT COUNT(*) AS cnt FROM agents_internal;")
            if cur.fetchone()["cnt"] == 0:
                for ag in self.config.agents:
                    alpha_fs = (
                        ag.alpha_disadvantageous
                        if self.omega.omega_d2_trust_and_parochialism
                        else 0.0
                    )
                    beta_fs = (
                        ag.beta_advantageous
                        if self.omega.omega_d2_trust_and_parochialism
                        else 0.0
                    )
                    fs_json = json.dumps(
                        {
                            "tier": ag.fehr_schmidt_tier,
                            "alpha": alpha_fs,
                            "beta": beta_fs,
                        }
                    )
                    cpt_json = json.dumps(
                        {
                            "lambda": ag.loss_aversion_lambda,
                            "gamma": ag.risk_tolerance_gamma,
                            "delta": self.config.hank_and_crisis_spec.long_run_discount_delta,
                        }
                    )
                    pad_json = json.dumps(
                        {"valence": 0.10, "arousal": 0.45, "dominance": 0.50}
                    )
                    maslow_json = json.dumps(
                        {
                            "survival_satiety": 0.85,
                            "financial_safety": 0.70,
                        }
                    )
                    conn.execute(
                        """
                        INSERT INTO agents_internal VALUES (
                            ?, ?, ?, ?, ?, ?, 0.10, ?, ?, ?, 0, 0, ?
                        );
                        """,
                        (
                            ag.agent_id,
                            ag.big_five.model_dump_json(),
                            fs_json,
                            cpt_json,
                            pad_json,
                            maslow_json,
                            float(ag.identity_rigidity),
                            float(ag.innate_opinion_anchor_s),
                            float(ag.innate_opinion_anchor_s),
                            json.dumps([0.0] * 7),
                        ),
                    )

                for ag_i in self.config.agents:
                    for ag_j in self.config.agents:
                        if ag_i.agent_id == ag_j.agent_id:
                            continue
                        same_group = ag_i.tms_specialty == ag_j.tms_specialty
                        init_tau_01 = (
                            0.64
                            if same_group
                            else 0.52
                            if self.omega.omega_d2_trust_and_parochialism
                            else 0.50
                        )
                        init_tau_signed = round(2.0 * init_tau_01 - 1.0, 4)
                        init_aff = 0.25 if same_group else 0.05
                        init_eps = (
                            self.contagion_spec.hk_initial_confidence_eps
                            if self.omega.omega_d3_contagion_and_polarization
                            else 2.0
                        )
                        coop_stats = json.dumps(
                            {
                                "total_trades": 0,
                                "successful_coops": 0,
                                "defections_suffered": 0,
                            }
                        )
                        conn.execute(
                            """
                            INSERT INTO relational_edges VALUES (
                                ?, ?, ?, ?, ?, ?, 0.60, 1, ?, ?
                            );
                            """,
                            (
                                ag_i.agent_id,
                                ag_j.agent_id,
                                "COLLEAGUE" if same_group else "ACQUAINTANCE",
                                init_tau_01,
                                init_tau_signed,
                                init_aff,
                                init_eps,
                                coop_stats,
                            ),
                        )
                        conn.execute(
                            """
                            INSERT INTO peer_perceived_models VALUES (?, ?, ?, ?);
                            """,
                            (
                                ag_i.agent_id,
                                ag_j.agent_id,
                                json.dumps(
                                    {
                                        "perceived_reliability": init_tau_01,
                                    }
                                ),
                                f"Counterparty {ag_j.agent_id} ({ag_j.occupation_role}).",
                            ),
                        )
                        for dom in ("CONSUMABLE_NECESSITY", "PRODUCTIVE_CAPITAL_ASSET"):
                            is_spec = ag_j.tms_specialty == dom
                            alpha_p = 6.0 if is_spec else 2.0
                            beta_p = 1.5 if is_spec else 4.0
                            conn.execute(
                                """
                                INSERT INTO tms_directory VALUES (?, ?, ?, ?, ?, 0);
                                """,
                                (ag_i.agent_id, ag_j.agent_id, dom, alpha_p, beta_p),
                            )
            conn.commit()

    def compute_effective_trust(self, owner_id: str, target_id: str) -> float:
        if not self.omega.omega_d2_trust_and_parochialism:
            return 0.50
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT trust_tau_01, interaction_count_n
                FROM relational_edges
                WHERE owner_id = ? AND target_id = ?;
                """,
                (owner_id, target_id),
            ).fetchone()
            if not row:
                return self.generalized_social_trust
            tau_01 = float(row["trust_tau_01"])
            n_ijt = int(row["interaction_count_n"])
            omega_ijt = 1.0 - math.exp(
                -self.trust_spec.familiarity_decay_lambda_n * n_ijt
            )
            t_eff = (
                omega_ijt * tau_01
                + (1.0 - omega_ijt) * self.generalized_social_trust
            )
            return round(max(0.0, min(1.0, t_eff)), 4)

    def get_effective_trust_matrix(self) -> Dict[Tuple[str, str], float]:
        matrix: Dict[Tuple[str, str], float] = {}
        for u in self.agent_ids:
            for v in self.agent_ids:
                if u != v:
                    matrix[(u, v)] = self.compute_effective_trust(u, v)
        return matrix

    def _retrieve_scoped_context_internal(
        self,
        owner_id: str,
        partner_id: Optional[str],
        incoming_texts: List[str],
        current_tick: int,
    ) -> Dict[str, Any]:
        exclude = {owner_id}
        if partner_id:
            exclude.add(partner_id)
        combined_text = " ".join(incoming_texts)
        mentioned_ids = self.scanner.scan(combined_text, exclude_ids=exclude)

        scoped_targets: List[str] = []
        if partner_id and partner_id != owner_id:
            scoped_targets.append(partner_id)
        for mid in mentioned_ids:
            if mid not in scoped_targets:
                scoped_targets.append(mid)

        with self._connect() as conn:
            internal_row = dict(
                conn.execute(
                    "SELECT * FROM agents_internal WHERE agent_id = ?;", (owner_id,)
                ).fetchone()
            )
            internal_row["big_five"] = json.loads(internal_row["big_five_json"])
            internal_row["fehr_schmidt"] = json.loads(
                internal_row["fehr_schmidt_json"]
            )
            internal_row["cpt_params"] = json.loads(internal_row["cpt_params_json"])
            internal_row["pad_mood"] = json.loads(internal_row["pad_mood_json"])
            internal_row["maslow"] = json.loads(internal_row["maslow_json"])
            internal_row["hebbian_vec"] = json.loads(internal_row["hebbian_vec_json"])

            scoped_relations: Dict[str, Dict[str, Any]] = {}
            scoped_peer_models: Dict[str, Dict[str, Any]] = {}

            if scoped_targets:
                placeholders = ",".join("?" for _ in scoped_targets)
                rel_rows = conn.execute(
                    f"""
                    SELECT * FROM relational_edges
                    WHERE owner_id = ? AND target_id IN ({placeholders});
                    """,
                    [owner_id] + scoped_targets,
                ).fetchall()
                for r in rel_rows:
                    d = dict(r)
                    d["coop_stats"] = json.loads(d["coop_stats_json"])
                    d["effective_trust_01"] = self.compute_effective_trust(
                        owner_id, d["target_id"]
                    )
                    scoped_relations[d["target_id"]] = d

                pm_rows = conn.execute(
                    f"""
                    SELECT * FROM peer_perceived_models
                    WHERE owner_id = ? AND target_id IN ({placeholders});
                    """,
                    [owner_id] + scoped_targets,
                ).fetchall()
                for pm in pm_rows:
                    pmd = dict(pm)
                    pmd["perceived_traits"] = json.loads(pmd["perceived_traits_json"])
                    scoped_peer_models[pmd["target_id"]] = pmd

            stm_rows = conn.execute(
                "SELECT * FROM stm_episodic WHERE owner_id = ?;", (owner_id,)
            ).fetchall()
            tau_stm = 4.5
            scored_memories: List[Tuple[float, Dict[str, Any]]] = []
            for m in stm_rows:
                md = dict(m)
                delta_t = max(0, current_tick - int(md["tick"]))
                s_enc = max(0.5, float(md["encoding_salience"]))
                retention_r = math.exp(-delta_t / (tau_stm * s_enc))
                partner_bonus = (
                    1.35 if (partner_id and md["partner_id"] == partner_id) else 1.0
                )
                effective_salience = retention_r * s_enc * partner_bonus
                md["retention_r"] = round(retention_r, 4)
                md["effective_read_salience"] = round(effective_salience, 4)
                scored_memories.append((effective_salience, md))

            scored_memories.sort(key=lambda x: x[0], reverse=True)
            top_episodes = [item[1] for item in scored_memories[:4]]

            for ep in top_episodes:
                conn.execute(
                    """
                    UPDATE stm_episodic
                    SET rehearsal_count = rehearsal_count + 1,
                        encoding_salience = ROUND(encoding_salience + 0.12, 4)
                    WHERE mem_id = ?;
                    """,
                    (ep["mem_id"],),
                )
            conn.commit()

        return {
            "owner_id": owner_id,
            "internal_state": internal_row,
            "scoped_relations": scoped_relations,
            "scoped_peer_models": scoped_peer_models,
            "mentioned_third_parties": mentioned_ids,
            "retrieved_episodes": top_episodes,
            "generalized_social_trust": self.generalized_social_trust,
            "institutional_trust": self.institutional_trust,
        }

    def update_dyadic_trust_and_memory(
        self,
        tick: int,
        owner_id: str,
        partner_id: str,
        trade_approved: bool,
        own_surplus_x: float,
        partner_surplus_x: float,
        kkt_gouging_index: float,
        consent_decision: str,
        action_category: str,
        internal_thought: str,
        verbal_message: str,
        rpe_magnitude: float,
        updated_hebbian: List[float],
    ) -> float:
        with self._connect() as conn:
            int_row = conn.execute(
                "SELECT fehr_schmidt_json, pad_mood_json FROM agents_internal WHERE agent_id = ?;",
                (owner_id,),
            ).fetchone()
            fs = json.loads(int_row["fehr_schmidt_json"])
            pad = json.loads(int_row["pad_mood_json"])
            alpha_i = (
                float(fs.get("alpha", 0.5))
                if self.omega.omega_d2_trust_and_parochialism
                else 0.0
            )

            rel_row = conn.execute(
                """
                SELECT trust_tau_01, intimacy_alpha, interaction_count_n,
                       hk_confidence_eps, coop_stats_json
                FROM relational_edges
                WHERE owner_id = ? AND target_id = ?;
                """,
                (owner_id, partner_id),
            ).fetchone()

            tau_01 = float(rel_row["trust_tau_01"])
            intimacy = float(rel_row["intimacy_alpha"])
            n_count = int(rel_row["interaction_count_n"]) + 1
            hk_eps = float(rel_row["hk_confidence_eps"])
            stats = json.loads(rel_row["coop_stats_json"])
            stats["total_trades"] = int(stats.get("total_trades", 0)) + 1

            if not self.omega.omega_d2_trust_and_parochialism:
                new_tau_01 = 0.50
                s_ijt = 0.0
            else:
                if not trade_approved:
                    s_ijt = -0.65 if consent_decision == "REJECT" else -0.35
                    stats["defections_suffered"] = (
                        int(stats.get("defections_suffered", 0)) + 1
                    )
                else:
                    stats["successful_coops"] = (
                        int(stats.get("successful_coops", 0)) + 1
                    )
                    ineq_ratio = max(0.0, partner_surplus_x - own_surplus_x) / (
                        abs(own_surplus_x) + abs(partner_surplus_x) + 1.0
                    )
                    raw_fairness = (
                        self.trust_spec.trade_fairness_baseline_gamma
                        - alpha_i * ineq_ratio
                        - self.trust_spec.kkt_weight_lambda * kkt_gouging_index
                    )
                    s_ijt = math.tanh(raw_fairness)

                if s_ijt >= 0.0:
                    new_tau_01 = tau_01 + self.trust_spec.slovic_eta_pos * s_ijt * (
                        1.0 - tau_01
                    )
                    intimacy = min(1.0, intimacy + 0.06 * s_ijt)
                    if self.omega.omega_d3_contagion_and_polarization:
                        hk_eps = min(
                            1.0,
                            hk_eps + self.contagion_spec.hk_trust_inc_gamma * s_ijt,
                        )
                else:
                    new_tau_01 = (
                        tau_01 + self.trust_spec.slovic_eta_neg * s_ijt * tau_01
                    )
                    intimacy = max(-1.0, intimacy + 0.10 * s_ijt)
                    if self.omega.omega_d3_contagion_and_polarization:
                        hk_eps = max(
                            0.12,
                            hk_eps
                            - self.contagion_spec.hk_distrust_dec_gamma * abs(s_ijt),
                        )

                new_tau_01 = round(max(0.02, min(0.98, new_tau_01)), 4)

            new_tau_signed = round(2.0 * new_tau_01 - 1.0, 4)
            conn.execute(
                """
                UPDATE relational_edges
                SET trust_tau_01 = ?,
                    trust_tau = ?,
                    intimacy_alpha = ?,
                    interaction_count_n = ?,
                    hk_confidence_eps = ?,
                    coop_stats_json = ?
                WHERE owner_id = ? AND target_id = ?;
                """,
                (
                    new_tau_01,
                    new_tau_signed,
                    round(intimacy, 4),
                    n_count,
                    round(hk_eps, 4),
                    json.dumps(stats),
                    owner_id,
                    partner_id,
                ),
            )

            conn.execute(
                "UPDATE agents_internal SET hebbian_vec_json = ? WHERE agent_id = ?;",
                (
                    json.dumps([round(float(x), 4) for x in updated_hebbian[:7]]),
                    owner_id,
                ),
            )

            arousal_a = float(pad.get("arousal", 0.45))
            s_enc = round(
                1.0 + 1.2 * arousal_a + 0.8 * abs(rpe_magnitude) + 0.6 * abs(s_ijt), 4
            )
            mem_id = f"mem_{tick:03d}_{uuid.uuid4().hex[:8]}"
            conn.execute(
                """
                INSERT INTO stm_episodic VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?);
                """,
                (
                    mem_id,
                    owner_id,
                    tick,
                    partner_id,
                    consent_decision,
                    action_category,
                    internal_thought,
                    verbal_message,
                    s_enc,
                    round(abs(rpe_magnitude), 4),
                    json.dumps(pad),
                ),
            )

            if s_enc >= 2.10:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO ltm_episodic VALUES (?, ?, ?, ?, ?, ?, 1);
                    """,
                    (
                        f"ltm_{mem_id}",
                        owner_id,
                        tick,
                        partner_id,
                        f"Tick {tick} w/ {partner_id} ({consent_decision}): {verbal_message[:100]}",
                        s_enc,
                    ),
                )
            conn.commit()
            return new_tau_01

    def step_macro_trust_tiers(
        self,
        delta_gini: float,
        delta_theil_b: float,
        pg_compliance_rate: float,
        delta_mean_scarcity_f: float,
        universal_transfer_share_phi: float,
        fair_audit_rate: float = 1.0,
    ) -> Tuple[float, float]:
        if not self.omega.omega_d2_trust_and_parochialism:
            self.generalized_social_trust = 0.50
            self.institutional_trust = 0.50
            return (0.50, 0.50)

        prev_inst = self.institutional_trust
        univ_signal = 0.04 * (universal_transfer_share_phi - 0.45) + 0.02 * (
            fair_audit_rate - 0.50
        )
        new_inst = max(
            0.05,
            min(
                0.95,
                prev_inst + univ_signal - 0.15 * max(0.0, delta_mean_scarcity_f),
            ),
        )
        delta_inst = new_inst - prev_inst
        self.institutional_trust = round(new_inst, 4)

        cr_ref = 0.77
        mean_rev = 0.15 * (
            self.trust_spec.initial_generalized_trust - self.generalized_social_trust
        )
        delta_tsoc = (
            mean_rev
            - self.trust_spec.gini_trust_sensitivity_beta * max(0.0, delta_gini)
            - self.trust_spec.theil_trust_sensitivity_beta * max(0.0, delta_theil_b)
            + self.trust_spec.pg_compliance_trust_beta * (pg_compliance_rate - cr_ref)
            - self.trust_spec.scarcity_trust_beta * max(0.0, delta_mean_scarcity_f)
            + self.trust_spec.inst_trust_spillover_beta * delta_inst
        )
        self.generalized_social_trust = round(
            max(0.05, min(0.95, self.generalized_social_trust + delta_tsoc)), 4
        )
        return (self.generalized_social_trust, self.institutional_trust)

    def record_inference_telemetry(
        self,
        tick: int,
        agent_id: str,
        partner_id: Optional[str],
        wave: int,
        llm_valid: bool,
        attention_measured: bool,
        financial_stress: float,
        easterbrook_beta: float,
        mean_shannon_entropy: float,
        tms_triggered: bool,
        attn_telemetry: Dict[str, float],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO inference_telemetry VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                );
                """,
                (
                    tick,
                    agent_id,
                    partner_id,
                    wave,
                    int(llm_valid),
                    int(attention_measured),
                    round(financial_stress, 4),
                    round(easterbrook_beta, 4),
                    round(mean_shannon_entropy, 4),
                    int(tms_triggered),
                    round(float(attn_telemetry.get("SPAN_DRIVES_AND_FINANCE", 0.0)), 4),
                    round(float(attn_telemetry.get("SPAN_PEER_MEMORY", 0.0)), 4),
                    round(float(attn_telemetry.get("SPAN_INCOMING_MSG", 0.0)), 4),
                    round(float(attn_telemetry.get("SPAN_SINK", 0.0)), 4),
                ),
            )
            conn.commit()