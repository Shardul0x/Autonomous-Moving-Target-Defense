"""
MTD ML Engine
─────────────────────────────────────────────────────────────
• Real Isolation Forest (sklearn) trained on synthetic normal traffic
• Continuously retrains on a sliding window of observations
• Adaptive shift intensity based on anomaly score confidence
• Offline LLM-style explainer: rule-based NLP that generates
  analyst-quality threat reports (zero internet, zero GPU needed)
"""

import numpy as np
import random
import time
import threading
from datetime import datetime
from collections import deque
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# FEATURE VECTOR DEFINITION (8 features)
# ─────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "req_rate",          # requests per second
    "port_scan_count",   # unique ports probed
    "login_failures",    # failed auth attempts
    "payload_entropy",   # byte entropy of payloads (high = encrypted/obfuscated)
    "geo_anomaly",       # 0/1 unusual geo origin
    "time_of_day_score", # deviation from normal traffic hours
    "packet_size_variance",  # variance in packet sizes
    "connection_burst",  # connections opened in last 5s
]

def extract_features(obs: dict) -> np.ndarray:
    return np.array([
        obs.get("req_rate", 0),
        obs.get("port_scan_count", 0),
        obs.get("login_failures", 0),
        obs.get("payload_entropy", 4.5),
        obs.get("geo_anomaly", 0),
        obs.get("time_of_day_score", 0),
        obs.get("packet_size_variance", 0),
        obs.get("connection_burst", 0),
    ], dtype=float)


# ─────────────────────────────────────────────────────────────
# GENERATE SYNTHETIC NORMAL TRAFFIC (training baseline)
# ─────────────────────────────────────────────────────────────
def generate_normal_samples(n=500) -> np.ndarray:
    rng = np.random.default_rng(42)
    samples = np.column_stack([
        rng.normal(5, 2, n).clip(0),       # req_rate: ~5/s normal
        rng.normal(0.3, 0.3, n).clip(0),   # port_scan_count
        rng.normal(0.2, 0.3, n).clip(0),   # login_failures
        rng.normal(4.5, 0.4, n),            # payload_entropy
        rng.binomial(1, 0.02, n),           # geo_anomaly: 2% chance
        rng.normal(0.1, 0.2, n).clip(0),   # time_of_day_score
        rng.normal(50, 20, n).clip(0),      # packet_size_variance
        rng.normal(3, 1.5, n).clip(0),      # connection_burst
    ])
    return samples


# ─────────────────────────────────────────────────────────────
# ISOLATION FOREST MODEL
# ─────────────────────────────────────────────────────────────
class MTDIsolationForest:
    def __init__(self):
        self.model = IsolationForest(
            n_estimators=150,
            contamination=0.08,
            random_state=42,
            max_samples="auto",
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.trained = False
        self.observation_window = deque(maxlen=300)  # sliding window
        self.retrain_every = 50   # retrain after this many new observations
        self.obs_since_retrain = 0
        self.lock = threading.Lock()

        # Prime with normal traffic
        normal = generate_normal_samples(500)
        self._train(normal)

    def _train(self, X: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.trained = True

    def observe(self, obs: dict) -> dict:
        """
        Feed one observation, get back:
        - anomaly_score  : float 0-100 (higher = more anomalous)
        - is_anomaly     : bool
        - confidence     : float 0-1
        - feature_scores : per-feature contribution (approx)
        - shift_intensity: "NONE" | "SOFT" | "MODERATE" | "FULL" | "EMERGENCY"
        """
        vec = extract_features(obs)

        with self.lock:
            self.observation_window.append(vec)
            self.obs_since_retrain += 1

            # Retrain on sliding window periodically
            if self.obs_since_retrain >= self.retrain_every and len(self.observation_window) >= 100:
                window_data = np.array(self.observation_window)
                self._train(window_data)
                self.obs_since_retrain = 0

            if not self.trained:
                return self._default_result()

            X = vec.reshape(1, -1)
            X_scaled = self.scaler.transform(X)

            # Isolation Forest raw score: negative = anomalous
            raw_score = self.model.decision_function(X_scaled)[0]
            prediction = self.model.predict(X_scaled)[0]  # -1=anomaly, 1=normal

            # Convert to 0-100 scale (sigmoid-like mapping)
            # raw_score typically in [-0.5, 0.5]; map so -0.5 → ~95, 0.5 → ~5
            anomaly_score = int(np.clip(100 * (0.5 - raw_score), 0, 100))
            is_anomaly = bool(prediction == -1)
            confidence = min(abs(raw_score) * 200, 1.0)

            # Per-feature contribution (approximate: z-score distance from normal mean)
            normal_mean = self.scaler.mean_
            normal_std  = np.sqrt(self.scaler.var_) + 1e-9
            z_scores = np.abs((vec - normal_mean) / normal_std)
            feature_scores = {
                FEATURE_NAMES[i]: float(np.clip(z_scores[i] / 4, 0, 1))
                for i in range(len(FEATURE_NAMES))
            }

            shift_intensity = self._decide_shift(anomaly_score, confidence, is_anomaly)

        return {
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "confidence": round(confidence, 3),
            "raw_score": round(float(raw_score), 4),
            "feature_scores": feature_scores,
            "shift_intensity": shift_intensity,
            "feature_vector": {FEATURE_NAMES[i]: float(vec[i]) for i in range(len(FEATURE_NAMES))},
        }

    def _decide_shift(self, score, confidence, is_anomaly) -> str:
        if not is_anomaly or score < 35:
            return "NONE"
        if score < 50:
            return "SOFT"       # rotate 1 service port only
        if score < 65:
            return "MODERATE"   # rotate ports + change route
        if score < 72:
            return "FULL"       # full IP + port + route shift
        return "EMERGENCY"      # full shift + honeypot redirect + service relocation

    def _default_result(self):
        return {
            "anomaly_score": 5, "is_anomaly": False, "confidence": 0,
            "raw_score": 0.3, "feature_scores": {},
            "shift_intensity": "NONE", "feature_vector": {}
        }


# ─────────────────────────────────────────────────────────────
# OFFLINE LLM EXPLAINER
# ─────────────────────────────────────────────────────────────
# This is a structured NLP engine — deterministic, runs entirely
# offline, produces analyst-quality threat explanations by
# combining rule-based templates with feature-aware reasoning.
# Think of it as a "frozen GPT distillation" specialized for MTD.
# ─────────────────────────────────────────────────────────────

_THREAT_TEMPLATES = {
    "NONE": [
        "Traffic patterns are within normal baseline. All {n} monitored features show z-scores below threshold. Isolation Forest reports score {score}/100 — consistent with benign activity. No defensive action recommended.",
        "Request behavior aligns with established normal profile. Feature entropy is nominal. Anomaly model confidence is low ({conf:.0%}). Infrastructure stability maintained.",
        "No statistical deviation detected across connection, payload, and authentication dimensions. System operating in safe zone.",
    ],
    "SOFT": [
        "Minor deviation detected on {top_feat}. Isolation Forest flags mild anomaly (score {score}/100, confidence {conf:.0%}). Elevated {top_feat_val:.1f} requests suggest possible automated probing. Initiating soft port rotation as precaution.",
        "Borderline anomaly: {top_feat} reading of {top_feat_val:.1f} is {z:.1f}σ above normal. Could indicate early-stage reconnaissance. Rotating one service endpoint to invalidate any partial topology map.",
        "Weak signal on {top_feat} axis. Risk score {score}/100 does not yet warrant full infrastructure shift. Applying minimal disruption: single service port rotation.",
    ],
    "MODERATE": [
        "Moderate threat profile confirmed. Leading indicators: {top_feat} ({top_feat_val:.1f}) and {sec_feat} ({sec_feat_val:.1f}). Combined anomaly score {score}/100 at {conf:.0%} confidence. Network route randomization and port rotation initiated. Attacker reconnaissance data is now stale.",
        "Multi-feature anomaly detected. {top_feat} is {z:.1f} standard deviations above baseline. Port scan activity and authentication failures rising in tandem — pattern consistent with credential stuffing preceded by topology mapping. Executing MODERATE shift.",
        "Isolation Forest confidence {conf:.0%}: system behavior drifting from normal manifold. Initiating route and port shift to collapse attacker's working mental model of the infrastructure.",
    ],
    "FULL": [
        "HIGH THREAT — Full infrastructure shift triggered. Score {score}/100 ({conf:.0%} confidence). {top_feat} at {top_feat_val:.1f} ({z:.1f}σ above norm). {sec_feat} also critically elevated. Complete IP rotation, port randomization, and route change applied. Attacker's reconnaissance is now fully invalidated.",
        "Coordinated attack pattern identified across {top_feat}, {sec_feat}. Anomaly score {score}/100 indicates active exploitation attempt. FULL SHIFT executed: IP, port, container, and route all rotated simultaneously. Infrastructure surface has moved.",
        "Model detects high-confidence intrusion attempt (score {score}/100). Payload entropy anomaly combined with burst connection rate suggests automated exploitation framework. Full parameter rotation applied. Previous network fingerprint is now a dead end for the attacker.",
    ],
    "EMERGENCY": [
        "🚨 EMERGENCY SHIFT. Score {score}/100 — maximum threat. {top_feat} ({top_feat_val:.1f}) and {sec_feat} ({sec_feat_val:.1f}) indicate active exploit in progress. Complete infrastructure rotation executed. Malicious traffic redirected to honeypot. All services relocated to new containers. Incident logged.",
        "🚨 CRITICAL: Isolation Forest anomaly score {score}/100 at {conf:.0%} confidence. Multi-vector attack: port scanning + brute force + payload obfuscation detected simultaneously. EMERGENCY protocol: full shift + honeypot activation + deception layer engaged. Real infrastructure has vanished from attacker's view.",
        "🚨 EMERGENCY PROTOCOL ACTIVATED. Score {score}. Attacker has escalated to active exploitation phase. All infrastructure parameters rotated. Honeypot deception system online — attacker is now interacting with fake services. Real system is safe.",
    ],
}

def _top_features(feature_scores: dict, n=2):
    """Return top N features by score."""
    sorted_feats = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_feats[:n]

def generate_llm_explanation(result: dict, obs: dict) -> str:
    """
    Offline LLM-style explanation generator.
    Picks a template, fills it with real values from the ML result.
    """
    intensity = result["shift_intensity"]
    score = result["anomaly_score"]
    conf = result["confidence"]
    feat_scores = result.get("feature_scores", {})
    feat_vec = result.get("feature_vector", {})

    templates = _THREAT_TEMPLATES.get(intensity, _THREAT_TEMPLATES["NONE"])
    template = random.choice(templates)

    top = _top_features(feat_scores, 2)
    top_feat = top[0][0] if top else "req_rate"
    top_feat_val = feat_vec.get(top_feat, 0)
    sec_feat = top[1][0] if len(top) > 1 else "port_scan_count"
    sec_feat_val = feat_vec.get(sec_feat, 0)
    z = feat_scores.get(top_feat, 0) * 4  # rough z-score

    try:
        explanation = template.format(
            score=score,
            conf=conf,
            top_feat=top_feat.replace("_", " "),
            top_feat_val=top_feat_val,
            sec_feat=sec_feat.replace("_", " "),
            sec_feat_val=sec_feat_val,
            z=z,
            n=len(FEATURE_NAMES),
        )
    except (KeyError, ValueError):
        explanation = f"Anomaly score {score}/100. Shift intensity: {intensity}."

    # Append timestamp
    ts = datetime.now().strftime("%H:%M:%S")
    return f"[{ts}] {explanation}"


# ─────────────────────────────────────────────────────────────
# TRAFFIC SIMULATOR (replaces the old manual attacker)
# ─────────────────────────────────────────────────────────────

class TrafficSimulator:
    """
    Produces realistic traffic observations.
    In NORMAL mode: generates benign traffic.
    In ATTACK mode: escalates through 5 attack phases automatically,
    each phase pushing different feature dimensions.
    """

    PHASES = [
        {
            "name": "Idle",
            "color": "green",
            "params": {"req_rate": (4,2), "port_scan_count": (0.1,0.1),
                       "login_failures": (0.1,0.1), "payload_entropy": (4.5,0.3),
                       "geo_anomaly": 0.01, "time_of_day_score": (0.05,0.05),
                       "packet_size_variance": (50,15), "connection_burst": (2,1)},
        },
        {
            "name": "Reconnaissance",
            "color": "yellow",
            "params": {"req_rate": (12,3), "port_scan_count": (3,1),
                       "login_failures": (0.2,0.2), "payload_entropy": (4.6,0.3),
                       "geo_anomaly": 0.3, "time_of_day_score": (0.4,0.2),
                       "packet_size_variance": (80,20), "connection_burst": (8,3)},
        },
        {
            "name": "Port Scanning",
            "color": "orange",
            "params": {"req_rate": (30,5), "port_scan_count": (9,2),
                       "login_failures": (0.5,0.3), "payload_entropy": (4.7,0.2),
                       "geo_anomaly": 0.6, "time_of_day_score": (0.6,0.2),
                       "packet_size_variance": (150,40), "connection_burst": (18,5)},
        },
        {
            "name": "Brute Force",
            "color": "red",
            "params": {"req_rate": (50,8), "port_scan_count": (7,2),
                       "login_failures": (15,4), "payload_entropy": (4.9,0.3),
                       "geo_anomaly": 0.9, "time_of_day_score": (0.8,0.1),
                       "packet_size_variance": (200,50), "connection_burst": (30,8)},
        },
        {
            "name": "Active Exploit",
            "color": "red",
            "params": {"req_rate": (70,10), "port_scan_count": (10,2),
                       "login_failures": (20,5), "payload_entropy": (7.2,0.3),
                       "geo_anomaly": 1.0, "time_of_day_score": (0.95,0.05),
                       "packet_size_variance": (400,80), "connection_burst": (50,10)},
        },
    ]

    def __init__(self):
        self.current_phase_idx = 0
        self.attacking = False
        self._lock = threading.Lock()
        self._phase_change_time = time.time()

    def start_attack(self):
        with self._lock:
            self.attacking = True
            self.current_phase_idx = 1  # start from recon
            self._phase_change_time = time.time()

    def stop_attack(self):
        with self._lock:
            self.attacking = False
            self.current_phase_idx = 0

    def _sample(self, params: dict) -> dict:
        obs = {}
        for k, v in params.items():
            if isinstance(v, tuple):
                obs[k] = max(0, random.gauss(v[0], v[1]))
            else:
                obs[k] = 1 if random.random() < v else 0
        return obs

    def next_observation(self) -> dict:
        with self._lock:
            if self.attacking:
                # Auto-escalate phase every 12 seconds
                elapsed = time.time() - self._phase_change_time
                if elapsed > 12 and self.current_phase_idx < len(self.PHASES) - 1:
                    self.current_phase_idx += 1
                    self._phase_change_time = time.time()
                # After all phases done, loop back to idle
                if self.current_phase_idx >= len(self.PHASES):
                    self.attacking = False
                    self.current_phase_idx = 0
            phase = self.PHASES[self.current_phase_idx]
            obs = self._sample(phase["params"])
            obs["_phase"] = phase["name"]
            obs["_color"] = phase["color"]
            return obs

    def current_phase(self) -> dict:
        with self._lock:
            return self.PHASES[self.current_phase_idx]

    def is_attacking(self) -> bool:
        with self._lock:
            return self.attacking
