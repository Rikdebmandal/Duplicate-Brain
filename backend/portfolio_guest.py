import os
import secrets
import tempfile
from pathlib import Path

runtime = Path(tempfile.mkdtemp(prefix="portfolio-twin-"))
os.environ.update(DATABASE_URL="sqlite+pysqlite:///" + str(runtime / "history.db"),
                  MODEL_DIR=str(runtime / "models"), SECRET_KEY=secrets.token_hex(32),
                  LLM_ENABLED="false", DEBUG="false", OPENBLAS_NUM_THREADS="1")
from cryptography.fernet import Fernet
os.environ["FIELD_ENCRYPTION_KEY"] = Fernet.generate_key().decode()

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from seeds.seed_data import DEFAULT_EMAIL, load_seed, seed
from app.database.session import SessionLocal
from app.models import User
from app.services.prediction.engine import PredictionEngine, PredictionRequest

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
class Scenario(BaseModel):
    scenario: str = Field(min_length=5, max_length=2500)
    options: list[str] = Field(min_length=2, max_length=5)

@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE

@app.get("/api/history")
def history():
    return [{"scenario": d["situation"], "options": d["options"], "chosen": d["decision"]}
            for d in load_seed()["decisions"]]

@app.post("/api/predict")
def predict(item: Scenario):
    labels = [s.strip() for s in item.options]
    if any(not s or len(s)>180 for s in labels) or len(set(labels)) != len(labels):
        raise HTTPException(422, "Enter distinct options under 181 characters.")
    with SessionLocal() as db:
        user = db.execute(select(User).where(User.email==DEFAULT_EMAIL)).scalar_one()
        result = PredictionEngine(db, user).predict(PredictionRequest(
            scenario=item.scenario.strip(), options=labels, use_llm=False, persist=False))
        return {k:result[k] for k in ("options","predicted_option","confidence_label","layers","similar_decisions")}

PAGE = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Duplicate Brain | Rikdeb Mandal</title><style>:root{font:16px/1.55 system-ui,sans-serif;color:#172e34;background:#f5f5ee}*{box-sizing:border-box}body{margin:0}main{max-width:1100px;margin:auto;padding:28px clamp(18px,5vw,64px) 70px}nav{border-bottom:1px solid #bdc9c1;padding-bottom:18px}a{color:#215a50}h1{font:normal clamp(42px,7vw,82px)/1.02 Georgia,serif;letter-spacing:-.06em;margin:32px 0 12px}h2{font-size:22px}.grid{display:grid;grid-template-columns:1.1fr .9fr;gap:20px}section{background:#fffefa;border:1px solid #c3cfc4;border-radius:12px;padding:22px;margin:18px 0}label{display:block;font-weight:600;margin:14px 0}textarea,select,input{font:inherit;border:1px solid #8caaa2;border-radius:6px;padding:9px;width:100%;background:white;color:#142824}button{font:inherit;border:0;border-radius:6px;padding:10px 18px;background:#245650;color:white;cursor:pointer}button:disabled{opacity:.55}article{padding:10px 0;border-top:1px solid #d0d9d2}progress{display:block;width:100%;accent-color:#2e7465}small{color:#49615a}details{margin:14px 0}summary{cursor:pointer;font-weight:600}.hint{max-width:76ch}@media(max-width:650px){.grid{grid-template-columns:1fr;gap:0}section{margin:10px 0}}</style>
<main><nav><a href="https://rikdeb-mandal.higgsfield.app/projects">← Portfolio projects</a></nav><h1>Duplicate Brain</h1><p class="hint">Test a decision against the repository’s synthetic decision history. The original Python engine combines the persona’s traits, its fitted statistical model and similar past choices.</p>
<div class="grid"><section><h2>Decision scenario</h2><label>Start with an example<select id="example"></select></label><form id="form"><label>Scenario<textarea id="scenario" required minlength="5" maxlength="2500" rows="5"></textarea></label><label>Options · one per line<textarea id="options" required rows="4"></textarea></label><button id="go" disabled>Run prediction</button></form></section>
<section><h2>Model result</h2><p id="status" aria-live="polite">Loading decision history…</p><div id="result"></div></section></div><section><h2>Historical evidence</h2><div id="evidence">Run a prediction to see the closest cases.</div></section>
</main><script>
const el=id=>document.getElementById(id);
function node(tag,text){const n=document.createElement(tag);n.textContent=text;return n}
async function start(){const r=await fetch('/api/history');if(!r.ok)throw Error();const data=await r.json();for(const [i,d] of data.entries())el('example').add(new Option(d.scenario.slice(0,75)+'…',i));
function choose(){const d=data[+el('example').value];el('scenario').value=d.scenario;el('options').value=d.options.join('\n')}choose();el('example').onchange=choose;el('status').textContent=data.length+' original records loaded.';el('go').disabled=false;
el('form').onsubmit=async e=>{e.preventDefault();const options=el('options').value.split('\n').map(x=>x.trim()).filter(Boolean);if(options.length<2||options.length>5){el('status').textContent='Enter two to five options.';return}
el('go').disabled=true;el('status').textContent='Scoring options…';el('result').replaceChildren();el('evidence').replaceChildren();
try{const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({scenario:el('scenario').value,options})});if(!r.ok)throw Error();const d=await r.json();el('status').textContent='Predicted choice · '+d.predicted_option;
for(const o of d.options){const a=node('article',o.label+' · '+(o.probability*100).toFixed(1)+'%'),b=document.createElement('progress');b.max=1;b.value=o.probability;a.append(b);el('result').append(a)}
el('result').append(node('small','Active layers · '+d.layers.filter(l=>l.available).map(l=>l.name).join(', ')));
for(const s of d.similar_decisions){const a=node('article',s.scenario_text);a.append(node('p','Historical choice · '+s.chosen_option));el('evidence').append(a)}
}catch{el('status').textContent='Prediction unavailable. Check the options and retry.'}finally{el('go').disabled=false}}}
start().catch(()=>el('status').textContent='The decision history could not load. Reload to retry.');</script></html>'''

if __name__ == "__main__":
    import uvicorn
    seed(DEFAULT_EMAIL, secrets.token_urlsafe(32), False)
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
