"""
Autonomous MTD — Flask Backend
Meraki Legion | VIIT Pune | India Innovates 2026
PROTOTYPE / UNDER DEVELOPMENT
"""
import random, time, threading, subprocess
from datetime import datetime
from flask import Flask, render_template, jsonify
from collections import deque
from ml_engine import MTDIsolationForest, TrafficSimulator, generate_llm_explanation, FEATURE_NAMES

app = Flask(__name__)

# ── DOCKER ────────────────────────────────────────────────────
def check_docker():
    try:
        r = subprocess.run(["docker","ps","--format","{{.Names}}"],
                           capture_output=True, timeout=4, text=True)
        names = r.stdout.strip().split("\n")
        return all(n in names for n in ["mtd_gateway","mtd_webapp","mtd_honeypot"])
    except:
        return False

def nginx_switch(target: str) -> bool:
    """
    Write nginx config by passing each line via printf to avoid ALL shell quoting issues.
    Uses container name (not IP) so Docker DNS resolves correctly.
    """
    upstream = "mtd_honeypot" if target == "honeypot" else "mtd_webapp"
    # Build config lines separately — no braces or quotes escaping nightmare
    cfg_lines = [
        "events {}",
        "http {",
        "  server {",
        "    listen 80;",
        "    location / {",
        f"      proxy_pass http://{upstream}:80;",
        "      proxy_connect_timeout 5s;",
        "    }",
        "  }",
        "}",
    ]
    # Join with newlines and write using printf (safe, no heredoc needed)
    cfg_escaped = "\\n".join(cfg_lines)
    try:
        write = subprocess.run(
            ["docker", "exec", "mtd_gateway", "sh", "-c",
             f"printf '{cfg_escaped}\\n' > /etc/nginx/nginx.conf"],
            capture_output=True, timeout=8, text=True)
        if write.returncode != 0:
            print(f"[Docker] write failed: {write.stderr.strip()}")
            return False

        # Validate config before reloading
        test = subprocess.run(
            ["docker", "exec", "mtd_gateway", "nginx", "-t"],
            capture_output=True, timeout=8, text=True)
        if test.returncode != 0:
            print(f"[Docker] nginx config invalid: {test.stderr.strip()}")
            return False

        rel = subprocess.run(
            ["docker", "exec", "mtd_gateway", "nginx", "-s", "reload"],
            capture_output=True, timeout=8, text=True)
        ok = rel.returncode == 0
        print(f"[Docker] nginx->{upstream}: {'OK' if ok else 'FAIL: '+rel.stderr.strip()}")
        return ok
    except Exception as e:
        print(f"[Docker] error: {e}")
        return False

DOCKER_MODE = check_docker()
print(f"[MTD] Docker: {'LIVE' if DOCKER_MODE else 'NOT DETECTED'}")
if DOCKER_MODE:
    nginx_switch("webapp")

# ── STATE ─────────────────────────────────────────────────────
IP_POOL    = [f"10.{a}.{b}.{c}" for a in range(0,10) for b in range(1,9)
              for c in [10,20,30,50,70,80,100,120,140,160]]
ROUTES     = ["A->B->C","A->D->C","B->E->C","A->F->D->C","D->G->C","B->H->C","A->J->K->C"]
CONTAINERS = [f"node-{i:02d}" for i in range(1,16)]
random.shuffle(IP_POOL)

INFRA = {
    "ip":"10.0.1.1","port":8080,"route":"A->B->C",
    "last_shift":"-","shift_count":0,"last_intensity":"NONE",
    "honeypot_active":False,"docker_mode":DOCKER_MODE,"gateway_target":"webapp",
}
SERVICES = {
    "web_server":  {"port":8080,"container":"mtd_webapp","real":True},
    "api_service": {"port":3000,"container":"node-02","real":False},
    "db_proxy":    {"port":5432,"container":"node-03","real":False},
    "auth_service":{"port":9000,"container":"node-04","real":False},
}
EVENT_LOG    = deque(maxlen=80)
LLM_LOG      = deque(maxlen=25)
RISK_HISTORY = deque(maxlen=100)

CURRENT_RESULT = {
    "anomaly_score":5,"is_anomaly":False,"confidence":0.0,
    "shift_intensity":"NONE",
    "feature_scores":{k:0.0 for k in FEATURE_NAMES},
    "feature_vector":{k:0.0 for k in FEATURE_NAMES},
}
CURRENT_OBS   = {k:0.0 for k in FEATURE_NAMES}
CURRENT_PHASE = {"name":"Idle","color":"green"}

_ip_idx = 0
def next_ip():
    global _ip_idx
    _ip_idx = (_ip_idx+1) % len(IP_POOL)
    return IP_POOL[_ip_idx]

# Honeypot stays engaged until explicitly stopped
_honeypot_engaged = False

def apply_shift(intensity: str, result: dict):
    global _honeypot_engaged
    changed = []
    docker_action = None

    if intensity == "SOFT":
        svc = random.choice(["api_service","db_proxy","auth_service"])
        old = SERVICES[svc]["port"]
        SERVICES[svc]["port"] = random.randint(10000,60000)
        changed.append(f"{svc} port {old}->{SERVICES[svc]['port']}")

    elif intensity == "MODERATE":
        for svc in ["api_service","db_proxy","auth_service"]:
            SERVICES[svc]["port"] = random.randint(10000,60000)
        old_r = INFRA["route"]
        INFRA["route"] = random.choice(ROUTES)
        changed.append("All service ports rotated")
        changed.append(f"Route {old_r}->{INFRA['route']}")

    elif intensity in ("FULL","EMERGENCY"):
        old_ip = INFRA["ip"]
        old_r  = INFRA["route"]
        INFRA["ip"]    = next_ip()
        INFRA["route"] = random.choice(ROUTES)
        for svc in ["api_service","db_proxy","auth_service"]:
            SERVICES[svc]["port"]      = random.randint(10000,60000)
            SERVICES[svc]["container"] = random.choice(CONTAINERS)
        changed.append(f"IP {old_ip}->{INFRA['ip']}")
        changed.append(f"Route {old_r}->{INFRA['route']}")
        changed.append("All services relocated")

        if intensity == "EMERGENCY" and not _honeypot_engaged:
            _honeypot_engaged        = True
            INFRA["honeypot_active"] = True
            INFRA["gateway_target"]  = "honeypot"
            docker_action            = "honeypot"
            changed.append("HONEYPOT ACTIVATED")

    if DOCKER_MODE and docker_action:
        ok = nginx_switch(docker_action)
        changed.append(f"[Docker->{docker_action}:{'OK' if ok else 'FAIL'}]")

    if intensity != "NONE" and changed:
        INFRA["last_shift"]    = datetime.now().strftime("%H:%M:%S")
        INFRA["shift_count"]  += 1
        INFRA["last_intensity"] = intensity
        ts = datetime.now().strftime("%H:%M:%S")
        EVENT_LOG.appendleft({
            "time":ts,"intensity":intensity,
            "score":result["anomaly_score"],"conf":round(result["confidence"]*100),
            "detail":" | ".join(changed),
            "color":{"SOFT":"cyan","MODERATE":"orange","FULL":"red","EMERGENCY":"red"}.get(intensity,"gray"),
        })

model     = MTDIsolationForest()
simulator = TrafficSimulator()

def ai_loop():
    tick = 0
    while True:
        obs    = simulator.next_observation()
        result = model.observe(obs)
        CURRENT_RESULT.update(result)
        CURRENT_OBS.update(obs)
        CURRENT_PHASE.update(simulator.current_phase())
        ts = datetime.now().strftime("%H:%M:%S")
        RISK_HISTORY.append({"t":ts,"score":result["anomaly_score"]})
        if result["is_anomaly"] or result["anomaly_score"]>40 or tick%15==0:
            LLM_LOG.appendleft({
                "time":ts,"text":generate_llm_explanation(result,obs),
                "intensity":result["shift_intensity"],
                "score":result["anomaly_score"],
                "color":obs.get("_color","green"),
            })
        apply_shift(result["shift_intensity"],result)
        tick += 1
        time.sleep(2)

@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/state")
def api_state():
    return jsonify({
        "infra":dict(INFRA),"services":dict(SERVICES),
        "result":dict(CURRENT_RESULT),"phase":dict(CURRENT_PHASE),
        "attacking":bool(simulator.is_attacking()),
        "risk_history":list(RISK_HISTORY)[-60:],
        "event_log":list(EVENT_LOG)[:25],
        "llm_log":list(LLM_LOG)[:8],
        "feature_names":FEATURE_NAMES,
        "model_info":{"type":"Isolation Forest","n_estimators":150,
                      "window_size":len(model.observation_window)},
    })

@app.route("/api/attacker/start", methods=["POST"])
def start_attack():
    global _honeypot_engaged
    _honeypot_engaged = False
    simulator.start_attack()
    EVENT_LOG.appendleft({"time":datetime.now().strftime("%H:%M:%S"),
        "intensity":"ALERT","score":0,"conf":0,
        "detail":"Attacker started — escalating through 5 phases","color":"red"})
    return jsonify({"ok":True})

@app.route("/api/attacker/stop", methods=["POST"])
def stop_attack():
    global _honeypot_engaged
    simulator.stop_attack()
    _honeypot_engaged        = False
    INFRA["honeypot_active"] = False
    INFRA["gateway_target"]  = "webapp"
    if DOCKER_MODE:
        nginx_switch("webapp")
    EVENT_LOG.appendleft({"time":datetime.now().strftime("%H:%M:%S"),
        "intensity":"INFO","score":0,"conf":0,
        "detail":"Attack stopped — traffic restored to real webapp","color":"green"})
    return jsonify({"ok":True})

if __name__ == "__main__":
    for i in range(20):
        RISK_HISTORY.append({"t":"boot","score":random.randint(3,20)})
    threading.Thread(target=ai_loop, daemon=True).start()
    print("\n"+"="*55)
    print("  Autonomous MTD  |  Meraki Legion  |  VIIT Pune")
    print(f"  Dashboard -> http://localhost:5000")
    print(f"  Gateway   -> http://localhost:8080  (WATCH THIS)")
    print(f"  Real app  -> http://localhost:8081")
    print(f"  Honeypot  -> http://localhost:8082")
    print(f"  Docker    : {'LIVE' if DOCKER_MODE else 'NOT RUNNING'}")
    print("="*55+"\n")
    app.run(debug=False, port=5000, threaded=True)