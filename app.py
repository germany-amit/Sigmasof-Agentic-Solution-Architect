# End-to-End Architecture & Integration MVP (Free-tier, single file)
# Covers each skill line: architecture, data/process flows, tech components, guidance/best-practices,
# complex issue resolution, impact assessment, EA/Security compliance, AWS app + SAP integration (simulated),
# Generative AI on AWS (simulated providers), API dev/integration with gateway/auth/monitoring.
#
# Run:  pip install -r requirements.txt && streamlit run app.py

import os, json, time, uuid, hmac, hashlib, base64, random
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import streamlit as st
from typing import List, Dict, Any
from pydantic import BaseModel, Field, validator

st.set_page_config(page_title="E2E Architecture & Integration — Skills MVP", layout="wide")

# ----------------------------- Utilities -----------------------------
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

def sign_payload(secret: str, payload: str) -> str:
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).decode().rstrip("=")

def make_jwt(payload: dict, secret: str, ttl_minutes: int=15) -> str:
    header = {"alg":"HS256","typ":"JWT"}
    now = int(time.time())
    body = {**payload, "iat": now, "exp": now + ttl_minutes*60}
    def b64(d):
        return base64.urlsafe_b64encode(json.dumps(d, separators=(',',':')).encode()).decode().rstrip('=')
    token = f"{b64(header)}.{b64(body)}"
    sig = sign_payload(secret, token)
    return f"{token}.{sig}"

def verify_jwt(token: str, secret: str) -> bool:
    try:
        head, body, sig = token.split('.')
        valid = sign_payload(secret, f"{head}.{body}") == sig
        if not valid: return False
        data = json.loads(base64.urlsafe_b64decode(body + '==').decode())
        return int(time.time()) < data.get("exp", 0)
    except Exception:
        return False

# ----------------------------- Data Models (Data Entities) -----------------------------
class Customer(BaseModel):
    customer_id: str
    email: str
    segment: str
    country: str

class Order(BaseModel):
    order_id: str
    customer_id: str
    amount: float
    created_at: str

class ChurnSignal(BaseModel):
    customer_id: str
    probability: float = Field(ge=0, le=1)
    reason: str

# ----------------------------- Sidebar: Scenario & Controls -----------------------------
scenario = st.sidebar.radio("Demo Scenario", [
    "AWS↔SAP Integration", "Generative AI on AWS", "API Gateway & Monitoring"
], index=0)

st.sidebar.caption("All features are free-tier friendly; AWS/SAP calls are simulated.")

# ----------------------------- Tabs -----------------------------
A,B,C,D,E,F = st.tabs([
    "Architecture Designer", "Data & Process Flows", "AWS x SAP Integration",
    "GenAI on AWS (Sim)", "API Dev & Gateway", "EA & Security Compliance"
])

# ----------------------------- Tab A: Architecture Designer -----------------------------
with A:
    st.subheader("Design end-to-end solution from business requirements")
    reqs = st.text_area("Business Requirements", value=(
        "Reduce churn by 10% in 2 quarters; integrate AWS apps with SAP; expose APIs; comply with security/EA; "
        "use cost-efficient, scalable architecture; add GenAI summaries for CSMs."), height=100)

    col1, col2 = st.columns(2)
    with col1:
        components = st.multiselect("Technology Components", [
            "AWS Lambda","AWS API Gateway","Amazon S3","Amazon RDS","AWS SageMaker","AWS Bedrock",
            "EventBridge","Step Functions","Redis (ElastiCache)","SAP (RFC/BAPI)","SAP PI/PO","Kafka (MSK)"
        ], default=["AWS Lambda","AWS API Gateway","Amazon S3","SAP (RFC/BAPI)"])
        protocols = st.multiselect("Protocols", ["HTTPS/REST","OAuth2","JWT","mTLS","SQS","Kafka","SFTP"],
                                   default=["HTTPS/REST","JWT"])
    with col2:
        nonfunc = st.multiselect("Non-Functional Goals", ["Scalable","Resilient","Cost-Efficient","Secure","Compliant"],
                                 default=["Scalable","Secure","Cost-Efficient"])
        risks = st.text_area("Known Risks/Assumptions", value="PII handling, SAP throughput limits, cost spikes, model drift", height=70)

    if st.button("Generate Architecture", key="arch"):
        arch = {
            "requirements": reqs,
            "components": components,
            "protocols": protocols,
            "nfr": nonfunc,
            "risks": risks,
            "flows": [
                "Client→API Gateway→Lambda→(S3/RDS)→SageMaker/Bedrock→Response",
                "SAP→(PI/PO|RFC)→EventBridge→Lambda→S3/RDS",
                "Monitoring→Metrics+Logs→Alerts",
            ]
        }
        st.json(arch)
        # Tiny diagram with matplotlib
        fig,ax=plt.subplots(figsize=(6,2)); ax.axis('off')
        ax.text(0.1,0.6,'Client',bbox=dict(boxstyle='round',fc='w'))
        ax.text(0.3,0.6,'API GW',bbox=dict(boxstyle='round',fc='w'))
        ax.text(0.5,0.6,'Lambda',bbox=dict(boxstyle='round',fc='w'))
        ax.text(0.7,0.6,'S3/RDS',bbox=dict(boxstyle='round',fc='w'))
        ax.text(0.9,0.6,'AI',bbox=dict(boxstyle='round',fc='w'))
        ax.annotate('',(0.28,0.62),(0.12,0.62),arrowprops=dict(arrowstyle='->'))
        ax.annotate('',(0.48,0.62),(0.32,0.62),arrowprops=dict(arrowstyle='->'))
        ax.annotate('',(0.68,0.62),(0.52,0.62),arrowprops=dict(arrowstyle='->'))
        ax.annotate('',(0.88,0.62),(0.72,0.62),arrowprops=dict(arrowstyle='->'))
        st.pyplot(fig)
        st.success("Architecture generated. Use other tabs to validate implementation details.")

# ----------------------------- Tab B: Data & Process Flows -----------------------------
with B:
    st.subheader("Define data entities & business process mappings")
    # Tiny catalog
    entities = [Customer(customer_id="C001", email="a@x.com", segment="Pro", country="IN"),
                Order(order_id="O100", customer_id="C001", amount=199.0, created_at=str(datetime.utcnow())),
                ChurnSignal(customer_id="C001", probability=0.73, reason="Low engagement")]
    st.json([e.dict() for e in entities])

    st.markdown("**Process Flow (happy path)**")
    st.code("""
    1) SAP posts order event → Integration Layer
    2) Event normalized → stored to S3/RDS, triggers churn scoring Lambda
    3) Model returns probability → retention recommendation persisted
    4) API exposes insights to CRM & CSM dashboard
    5) Monitoring tracks latency, drift, and failures; alerts on thresholds
    """, language="text")

    st.markdown("**Protocol choices:** REST over HTTPS + JWT; SFTP as fallback for batch; mTLS for partner links.")

# ----------------------------- Tab C: AWS x SAP Integration (Simulated) -----------------------------
with C:
    st.subheader("Design & implement AWS-based apps that integrate with ERP (SAP) — Simulated")
    secret = st.text_input("Shared Secret (JWT & signatures)", value="demo-secret")
    mode = st.selectbox("Exchange Mechanism", ["REST Push","SFTP Batch","EventBridge"], index=0)

    col1,col2 = st.columns(2)
    with col1:
        st.caption("Simulate SAP→AWS secure message")
        sample_payload = {"order_id":"O100","amount":199.0,"ts":datetime.utcnow().isoformat()+"Z"}
        if st.button("Sign & Send", key="sap_send"):
            token = make_jwt({"iss":"SAP","scope":"orders:write"}, secret)
            signature = sign_payload(secret, json.dumps(sample_payload, separators=(',',':')))
            st.code(json.dumps({"headers":{"Authorization":f"Bearer {token}","X-Signature":signature},"body":sample_payload}, indent=2))
            st.info(f"Transport: {mode}")
    with col2:
        st.caption("Validate at AWS Integration Layer")
        inbound_token = st.text_input("Paste JWT", value="")
        inbound_sig = st.text_input("Paste Signature", value="")
        inbound_body = st.text_area("Paste Body (JSON)", value="")
        if st.button("Validate", key="validate"):
            ok_jwt = verify_jwt(inbound_token, secret)
            try: ok_sig = inbound_sig==sign_payload(secret, json.dumps(json.loads(inbound_body), separators=(',',':')))
            except Exception: ok_sig=False
            st.metric("JWT valid", ok_jwt); st.metric("Signature valid", ok_sig)
            if ok_jwt and ok_sig:
                st.success("Message accepted → persisted (S3/RDS) → triggers Lambda")
            else:
                st.error("Rejected. Check keys, payload canonicalization, or clock skew.")

    st.markdown("**3rd‑party integration**: add adapter functions to normalize payloads, implement retries/backoff, and idempotency keys.")

# ----------------------------- Tab D: Generative AI on AWS (Simulated Providers) -----------------------------
with D:
    st.subheader("Design & deploy GenAI using SageMaker/Bedrock/Lambda — Simulated")
    provider = st.selectbox("Provider", ["AWS Bedrock","AWS SageMaker","AWS Lambda (serverless inference)"], index=0)
    task = st.selectbox("Task", ["Summarize RFP","Generate retention email","Explain churn reason"], index=0)
    text = st.text_area("Input", value="Customer has reduced usage; contract renewal in 30 days.", height=80)

    def local_gen(p: str, t: str) -> str:
        # Free-tier deterministic generator (no heavy models)
        if "Summarize" in t:
            return "Summary: Reduce churn risk with targeted offers, proactive outreach, and SLA-backed success plan."
        if "retention" in t:
            return "Email: We value your partnership. Based on your goals, we can improve ROI via tailored plan and credits."
        return "Top factors: Low engagement, high ticket volume, expiring contract. Actions: CSM call, enablement, discount."

    if st.button("Run GenAI", key="genai"):
        # If an external key exists, still keep it local to stay free-tier safe
        out = local_gen(provider, task)
        st.markdown(out)

    st.markdown("**Optimization knobs**: batch size, max tokens, temperature; **Cost model**: requests × unit price; **Guardrails**: PII scrubbing, rate limits.")

# ----------------------------- Tab E: API Development & Gateway -----------------------------
with E:
    st.subheader("Architect and develop robust, scalable, secure APIs")
    basepath = st.text_input("Base Path", value="/v1")
    resources = st.multiselect("Resources", ["customers","orders","churn"], default=["customers","orders","churn"])
    auth = st.selectbox("Auth", ["API Key","JWT (HS256)","OAuth2"] , index=1)
    rate = st.slider("Rate Limit (req/min)", 30, 600, 120)

    if st.button("Generate OpenAPI", key="oas"):
        paths = {}
        for r in resources:
            paths[f"{basepath}/{r}"] = {
                "get": {"summary": f"List {r}", "security": [{auth: []}], "responses": {"200": {"description": "OK"}}},
                "post": {"summary": f"Create {r[:-1]}", "security": [{auth: []}], "responses": {"201": {"description": "Created"}}}
            }
        oas = {"openapi":"3.0.0","info":{"title":"Demo API","version":"1.0.0"},"x-gateway":{"rate_limit_rpm":rate},"paths":paths}
        st.json(oas)
        st.download_button("Download openapi.json", data=json.dumps(oas, indent=2), file_name="openapi.json")

    st.markdown("**Gateway design**: WAF rules, JWT validation, IP allowlists, CORS, request/response transforms, canary routing.")

    st.markdown("**Monitoring**: below chart simulates p95 latency under load")
    if st.button("Simulate Load", key="load"):
        x=np.arange(0,60)
        p95 = 120 + 30*np.sin(x/6) + np.random.RandomState(0).randn(60)*10
        fig,ax=plt.subplots(); ax.plot(x,p95); ax.set_xlabel('Minute'); ax.set_ylabel('p95 latency (ms)'); st.pyplot(fig)

# ----------------------------- Tab F: EA & Security Compliance -----------------------------
with F:
    st.subheader("Ensure solutions comply with Enterprise Architecture & Security principles")
    checks = {
        "Least Privilege": st.checkbox("IAM roles are least-privilege", value=True),
        "Encryption": st.checkbox("Data encrypted in transit (TLS) & at rest", value=True),
        "PII Controls": st.checkbox("PII masked/hashed; access logged", value=True),
        "Network": st.checkbox("Private subnets / SGs / no public DB", value=True),
        "Secrets": st.checkbox("Secrets in vault (no hardcoding)", value=True),
        "Audit": st.checkbox("Audit trails & tamper evidence", value=True),
    }
    st.json(checks)
    score = sum(1 for v in checks.values() if v) / len(checks)
    st.metric("Compliance Score", f"{score*100:.0f}%")
    st.markdown("**Impact assessment**: changes to data contracts affect downstream scoring; recompute features, update versioned schemas, run backfills.")

st.caption("Single-file MVP • free/open-source • simulates AWS/SAP, GenAI, APIs, and compliance to cover every line of the skills list.")
