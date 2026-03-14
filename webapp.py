import time
import torch
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import hashlib
import os
import re

from bitnet.model import BitNetDecoder
from data.dataset import UniProtDataset  # not DNADataset

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "./checkpoints/checkpoint_step382500.pth"

# ---------------- FUNCTION NAMES ----------------
FUNCTION_NAMES = [
    "Enzyme",
    "Binding",
    "Transport",
    "Structural",
    "Regulatory",
    "Signal",
    "Unknown",
    "Other"
]

DOMAIN_NAMES = [
    "ATP-binding domain",
    "Kinase domain",
    "Transmembrane region",
    "Zinc finger",
    "SH3 domain",
    "WD repeat"
]

LOCALIZATION_NAMES = [
    "Cytoplasm",
    "Nucleus",
    "Membrane",
    "Mitochondria",
    "Secreted",
    "Extracellular"
]

GO_NAMES = [
    "GO:0004672 (protein kinase activity)",
    "GO:0005524 (ATP binding)",
    "GO:0000166 (nucleotide binding)",
    "GO:0005634 (nucleus)",
    "GO:0005737 (cytoplasm)"
]


# ---------------- LOAD DATASET & MODEL ----------------
dataset = UniProtDataset(tsv_path="data/uniprot_annotations.tsv", max_len=128, max_samples=1000)
tokenizer = dataset.tokenizer


model = BitNetDecoder(
    vocab_size=tokenizer.vocab_size,
    d_model=96,
    n_layers=4,
    n_heads=4,
    max_seq_len=256
)

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
# ---------------- EXTRA MODEL STATS ----------------
# Model size in KB
model_size_kb = os.path.getsize(CHECKPOINT_PATH) // 1024

# Extract step from filename (assumes 'checkpoint_stepXXXX.pth')
match = re.search(r'step(\d+)', CHECKPOINT_PATH)
step_number = match.group(1) if match else "—"

# Compute checkpoint hash (SHA1, short 8 chars)
sha1 = hashlib.sha1()
with open(CHECKPOINT_PATH, "rb") as f:
    while True:
        data = f.read(65536)  # read 64kb chunks
        if not data:
            break
        sha1.update(data)
checkpoint_hash = sha1.hexdigest()[:8]

NUM_PARAMS = sum(p.numel() for p in model.parameters())
BITS = 1

# ---------------- FASTAPI ----------------
app = FastAPI(title="BitNet-1 Edge Demo")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------- HTML TEMPLATE ----------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>BitNet-1 | Edge AI Demo</title>
<link rel="icon" type="image/png" href="/static/favicon.png">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

:root {
    --bg:#f8fafc;
    --card:#ffffff;
    --border:#e5e7eb;
    --text:#0f172a;
    --muted:#475569;
    --accent:linear-gradient(135deg,#6366f1,#22c55e);
}

*{box-sizing:border-box;font-family:Inter;}

body{
    margin:0;
    background:linear-gradient(180deg,#f1f5f9,#ffffff);
    display:flex;
    justify-content:center;
    padding:30px;
}

.container{
    width:100%;
    max-width:900px;
    background:var(--card);
    border-radius:24px;
    padding:40px;
    border:1px solid var(--border);
    box-shadow:0 25px 60px rgba(0,0,0,.08);
}

.header{
    display:flex;
    align-items:center;
    gap:14px;
}
.title{
    font-size:42px;
    font-weight:800;
    letter-spacing:.08em;
    display:flex;
    gap:2px;
}

.title span{
    background:linear-gradient(135deg,#6366f1,#22c55e);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    animation:glitch 3s infinite;
}

.title .one{
    color:#22c55e;
    text-shadow:0 0 12px rgba(34,197,94,.7);
}

.title .dash{
    opacity:.5;
}

@keyframes glitch{
    0%,100%{ transform:translate(0); }
    20%{ transform:translate(1px,-1px); }
    40%{ transform:translate(-1px,1px); }
    60%{ transform:translate(1px,0); }
    80%{ transform:translate(-1px,0); }
}

.logo{
    width:56px;
    height:56px;
    border-radius:18px;
    background:linear-gradient(135deg,#6366f1,#22c55e);
    display:flex;
    align-items:center;
    justify-content:center;
    color:white;
    font-weight:800;
    letter-spacing:1px;
    box-shadow:0 0 0 rgba(99,102,241,.6);
    animation:pulse 2.5s infinite;
}

.logo span{
    font-size:18px;
}

@keyframes pulse{
    0%{ box-shadow:0 0 0 0 rgba(99,102,241,.6); }
    70%{ box-shadow:0 0 0 18px rgba(99,102,241,0); }
    100%{ box-shadow:0 0 0 0 rgba(99,102,241,0); }
}

.subtitle{
    margin:12px 0 18px;
    color:var(--muted);
}

.badges{
    display:flex;
    flex-wrap:wrap;
    gap:8px;
    margin-bottom:22px;
}

.badge{
    padding:7px 14px;
    border-radius:999px;
    font-size:12px;
    font-weight:600;
    border:1px solid transparent;
    background:#f1f5f9;
    color:#0f172a;
}

.badge.bit{ background:#eef2ff; border-color:#a5b4fc; color:#3730a3; }
.badge.edge{ background:#ecfeff; border-color:#67e8f9; color:#0369a1; }
.badge.power{ background:#f0fdf4; border-color:#86efac; color:#166534; }
.badge.research{ background:#fff7ed; border-color:#fdba74; color:#9a3412; }
.badge.status{ background:#ecfdf5; border-color:#6ee7b7; color:#065f46; }

textarea{
    width:100%;
    min-height:120px;
    padding:16px;
    border-radius:14px;
    border:1px solid var(--border);
    background:#f8fafc;
    font-size:15px;
}

.actions{
    display:flex;
    gap:14px;
    margin-top:18px;
}

button{
    padding:14px;
    border-radius:14px;
    border:none;
    font-weight:600;
    cursor:pointer;
}

.generate{
    background:var(--accent);
    color:white;
    flex:1;
}

.copy{
    background:#f1f5f9;
    border:1px solid var(--border);
}

.output-wrapper {
    margin-top: 15px;
}

.output-container {
    border-radius: 14px;
    background: #f8fafc;
    border: 1px solid var(--border);
    padding: 12px 16px 16px 16px;
    font-family: monospace;
    font-size: 14px;
    line-height: 1.3;
    font-weight: 500;
    white-space: pre-wrap;
    min-height: 60px;
    overflow-x: auto;
}

.output-header {
    display: flex;
    justify-content: space-between; /* badge left, copy button right */
    align-items: center;
    margin-bottom: 6px;
}

.output-badge {
    background: linear-gradient(135deg,#6366f1,#22c55e); /* green-blue gradient */
    color: white;
    font-size: 12px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 999px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.2);
}


.output-text {
    min-height: 60px;
}


/* Compact function bars */
#functionOutput .bar {
    display: grid;
    grid-template-columns: 80px 1fr 35px;
    gap: 2px;
    align-items: center;
    margin-bottom: 2px;   /* less vertical space */
    font-size: 12px;
}
#functionOutput .bar-fill,
#sequenceOutput + .probs .bar-fill {
    height: 4px;
    border-radius: 4px;
}



.probs {
    margin-top: 30px;   /* increased spacing above bars */
}


/* Bars */
.bar{
    display:grid;
    grid-template-columns:60px 1fr 50px;
    gap:8px;
    align-items:center;
    margin-bottom:6px;
}

.bar-fill{
    height:10px;
    border-radius:6px;
    background:linear-gradient(90deg,#6366f1,#22c55e);
}




/* Stats */
.stats{
    margin-top:20px;
    display:grid;
    grid-template-columns:repeat(2,1fr);
    gap:10px;
    font-size:14px;
    color:var(--muted);
}
.stat-note{
    grid-column:1 / -1;
    font-size:12px;
    color:#64748b;
    margin-top:6px;
}

/* Footer */
.footer{
    margin-top:30px;
    text-align:center;
    font-size:12px;
    color:#64748b;
}
/* ================= AI THINKING LOADER ================= */
/* ===== HEADER ALIGNMENT ===== */

.header{
    display:flex;
    align-items:center;
    justify-content:space-between;
}

/* ===== AI BADGE ===== */

.ai-badge{
    width:46px;
    height:46px;
    border-radius:14px;
    background:linear-gradient(135deg,#6366f1,#22c55e);
    display:flex;
    align-items:center;
    justify-content:center;
    position:relative;
    opacity:0;
    transform:scale(.8);
    transition:.3s ease;
}

/* inner neural core */
/* 🧬 HELIX */

/* 🧬 DOUBLE HELIX */

/* ===== 3D HELIX CORE ===== */

.ai-badge{
    perspective:800px;
}

.helix{
    position:relative;
    width:30px;
    height:30px;
    transform-style:preserve-3d;
    animation:helixTilt var(--helixSpeed,6s) linear infinite;
}

/* strands */
.strand span{
    position:absolute;
    left:50%;
    width:6px;
    height:6px;
    margin-left:-3px;
    border-radius:50%;
    background:white;
    box-shadow:0 0 8px rgba(255,255,255,.9);
}

/* vertical layout */
.strand span:nth-child(1){ top:2px; }
.strand span:nth-child(2){ top:10px; }
.strand span:nth-child(3){ top:18px; }
.strand span:nth-child(4){ top:26px; }

/* animated motion */
.strand-a span{
    animation:waveA var(--helixSpeed,1.6s) ease-in-out infinite;
}
.strand-b span{
    animation:waveB var(--helixSpeed,1.6s) ease-in-out infinite;
}

/* phase shift */
.strand span:nth-child(1){ animation-delay:0s; }
.strand span:nth-child(2){ animation-delay:.2s; }
.strand span:nth-child(3){ animation-delay:.4s; }
.strand span:nth-child(4){ animation-delay:.6s; }

/* wave motion */
@keyframes waveA{
    0%   { transform:translateX(-7px) translateZ(-6px) scale(.6); opacity:.4; }
    50%  { transform:translateX(7px)  translateZ(8px)  scale(1.2); opacity:1; }
    100% { transform:translateX(-7px) translateZ(-6px) scale(.6); opacity:.4; }
}

@keyframes waveB{
    0%   { transform:translateX(7px)  translateZ(-6px) scale(.6); opacity:.4; }
    50%  { transform:translateX(-7px) translateZ(8px)  scale(1.2); opacity:1; }
    100% { transform:translateX(7px)  translateZ(-6px) scale(.6); opacity:.4; }
}

/* cinematic tilt */
@keyframes helixTilt{
    0%   { transform:rotateY(55deg) rotateX(18deg) rotate(0deg); }
    100% { transform:rotateY(55deg) rotateX(18deg) rotate(360deg); }
}
.strand::before{
    content:"";
    position:absolute;
    left:50%;
    top:0;
    bottom:0;
    width:2px;
    background:linear-gradient(to bottom,transparent,#22c55e,transparent);
    transform:translateX(-50%);
    opacity:.35;
    filter:blur(.6px);
}



/* animated state */
.ai-badge.active{
    opacity:1;
    transform:scale(1);
    animation:badgePulse 1.4s infinite ease-in-out;
}
.ai-badge.done{
    opacity:1;
    transform:scale(1);
    background:linear-gradient(135deg,#22c55e,#16a34a);
}

.ai-badge.done::before{
    content:"";
    position:absolute;
    inset:-6px;
    border-radius:18px;
    border:2px solid rgba(34,197,94,.5);
}

/* glowing ring */
.ai-badge.active::before{
    content:"";
    position:absolute;
    inset:-6px;
    border-radius:18px;
    border:2px solid rgba(99,102,241,.35);
    animation:ringSpin 2.5s linear infinite;
}

@keyframes badgePulse{
    0%,100%{ transform:scale(1); }
    50%{ transform:scale(1.08); }
}

@keyframes ringSpin{
    from{ transform:rotate(0deg); }
    to{ transform:rotate(360deg); }
}

.ai-loader{
    display:flex;
    justify-content:center;
    gap:10px;
    margin-top:18px;
}

.ai-loader span{
    width:10px;
    height:10px;
    border-radius:50%;
    background:linear-gradient(135deg,#6366f1,#22c55e);
    animation:aiBounce 1.2s infinite ease-in-out;
}

.ai-loader span:nth-child(2){ animation-delay:.15s; }
.ai-loader span:nth-child(3){ animation-delay:.30s; }
.ai-loader span:nth-child(4){ animation-delay:.45s; }
.ai-loader span:nth-child(5){ animation-delay:.60s; }

@keyframes aiBounce{
    0%, 80%, 100%{
        transform:scale(.6);
        opacity:.5;
    }
    40%{
        transform:scale(1.4);
        opacity:1;
    }
}

</style>
</head>

<body>
<div class="container">

<div class="header">
    <div class="logo"><span>BN</span></div>
    <h1 class="title">
  <span>B</span><span>i</span><span>t</span><span>N</span><span>e</span><span>t</span>
  <span class="dash">-</span>
  <span class="one">1</span>
</h1>


    <div id="loaderBadge" class="ai-badge">
    <div class="helix">
    <div class="strand strand-a">
        <span></span><span></span><span></span><span></span>
    </div>
    <div class="strand strand-b">
        <span></span><span></span><span></span><span></span>
    </div>
</div>

</div>

</div>


<div class="subtitle">
1-Bit Transformer optimized for ultra-low power Edge Devices
</div>

<div class="badges">
    <span class="badge bit">⚡ 1-Bit Model</span>
    <span class="badge edge">📱 Edge-Optimized</span>
    <span class="badge power">🔋 Ultra-Low Power</span>
    <span class="badge research">🧪 Research Grade</span>
    <span class="badge status">🟢 Live Demo</span>
</div>

<form id="dnaForm">
<textarea name="dna_input" placeholder="Enter DNA / Protein sequence (30–120 amino acids recommended)">%%INPUT%%</textarea>
<div class="actions">
<button class="generate" id="genBtn">Generate</button>

</div>
</form>




<!-- Outputs -->
<div class="output-wrapper">
  <div class="output-header">
    <span class="output-badge">Sequence Output</span>
    <button class="copy" onclick="copySequence()">Copy</button>
  </div>
  <div id="sequenceOutput" class="output-container">%%OUTPUT%%</div>
</div>

<!-- Function Predictions -->
<div class="output-wrapper">
  <div class="output-header">
    <span class="output-badge">Function Prediction</span>
    <button class="copy" onclick="copyFunctions()">Copy</button>
  </div>
  <div id="functionOutput" class="output-container">%%FUNC_OUTPUT%%</div>
</div>

<!-- Domain Predictions -->
<div class="output-wrapper">
  <div class="output-header">
    <span class="output-badge">Domain Detection</span>
    <button class="copy" onclick="copyDomains()">Copy</button>
  </div>
  <div id="domainOutput" class="output-container">%%DOMAIN_OUTPUT%%</div>
</div>

<!-- GO Term Predictions -->
<div class="output-wrapper">
  <div class="output-header">
    <span class="output-badge">GO Terms</span>
    <button class="copy" onclick="copyGO()">Copy</button>
  </div>
  <div id="goOutput" class="output-container">%%GO_OUTPUT%%</div>
</div>

<!-- Localization Predictions -->
<div class="output-wrapper">
  <div class="output-header">
    <span class="output-badge">Localization</span>
    <button class="copy" onclick="copyLoc()">Copy</button>
  </div>
  <div id="locOutput" class="output-container">%%LOC_OUTPUT%%</div>
</div>



<!-- Top-k probabilities -->
<div class="probs">
<h3>Next-Token Probability</h3>
%%BARS%%
</div>

<!-- Stats -->
<div class="stats">
  <div><b>Parameters:</b> %%PARAMS%%</div>
  <div><b>Precision:</b> %%BITS%%-bit</div>
  <div><b>Device:</b> %%DEVICE%%</div>
  <div><b>Latency:</b> %%LATENCY%% ms</div>
  <div><b>Inference Mode:</b> Greedy</div>
  <div><b>Max Tokens:</b> 50</div>
  <div><b>Input Tokens:</b> %%IN_TOKENS%%</div>
  <div><b>Output Tokens:</b> %%OUT_TOKENS%%</div>

  <div class="stat-note">
    Model Checkpoint: <b>checkpoint_step%%STEP%%.pth</b> 
    (hash: <code>%%HASH%%</code>, size: %%SIZE%% KB)
  </div>
</div>

<div class="footer">
© 2026 • Academic Edge-AI Demonstration
</div>

</div>

<script>
function copySequence(){
    navigator.clipboard.writeText(
        document.getElementById("sequenceOutput").innerText
    );
}

function copyFunctions(){
    navigator.clipboard.writeText(
        document.getElementById("functionOutput").innerText
    );
}
function copyFunctions(){
    navigator.clipboard.writeText(document.getElementById("functionOutput").innerText);
}
function copyDomains(){
    navigator.clipboard.writeText(document.getElementById("domainOutput").innerText);
}
function copyGO(){
    navigator.clipboard.writeText(document.getElementById("goOutput").innerText);
}
function copyLoc(){
    navigator.clipboard.writeText(document.getElementById("locOutput").innerText);
}

const form = document.getElementById("dnaForm");
const loader = document.getElementById("loaderBadge");
const btn = document.getElementById("genBtn");


form.addEventListener("submit", async (e)=>{
    e.preventDefault();
    loader.classList.remove("done");
    loader.classList.add("active");   // SHOW LOADER
    btn.disabled = true;
btn.innerText = "Generating...";

    const fd = new FormData(form);
    const res = await fetch("/", { method:"POST", body:fd });
    const html = await res.text();

    const doc = new DOMParser().parseFromString(html,"text/html");

    // Typewriter effect for sequence
    const newText = doc.getElementById("sequenceOutput").innerText;
    const output = document.getElementById("sequenceOutput");
    output.innerText = "";

    let i = 0;
    function type(){
        if(i < newText.length){
            output.innerText += newText.charAt(i++);
            setTimeout(type, 18);
        }else{
            loader.classList.remove("active");
loader.classList.add("done");
btn.disabled = false;
btn.innerText = "Generate";

setTimeout(()=>{
    loader.classList.remove("done");
}, 1200);   // HIDE LOADER WHEN DONE
        }
    }
    type();

    // --- Update outputs safely ---


const funcDiv = doc.getElementById("functionOutput");
if(funcDiv){
    document.getElementById("functionOutput").innerHTML = funcDiv.innerHTML;
}

const domainDiv = doc.getElementById("domainOutput");
if(domainDiv){
    document.getElementById("domainOutput").innerHTML = domainDiv.innerHTML;
}

const goDiv = doc.getElementById("goOutput");
if(goDiv){
    document.getElementById("goOutput").innerHTML = goDiv.innerHTML;
}

const locDiv = doc.getElementById("locOutput");
if(locDiv){
    document.getElementById("locOutput").innerHTML = locDiv.innerHTML;
}

// Update stats + top-k probabilities
const statsDiv = doc.querySelector(".stats");
if(statsDiv){
    document.querySelector(".stats").innerHTML = statsDiv.innerHTML;
}

const probsDiv = doc.querySelector(".probs");
if(probsDiv){
    document.querySelector(".probs").innerHTML = probsDiv.innerHTML;
}

});
</script>
</body>
</html>
"""


# ---------------- ROUTES ----------------
@app.get("/", response_class=HTMLResponse)
async def home():
    return (HTML_PAGE
        .replace("%%OUTPUT%%","")
        .replace("%%INPUT%%","")
        .replace("%%PARAMS%%",f"{NUM_PARAMS:,}")
        .replace("%%BITS%%",str(BITS))
        .replace("%%DEVICE%%",DEVICE.upper())
        .replace("%%LATENCY%%","—")
        .replace("%%BARS%%","")
        .replace("%%IN_TOKENS%%","—")
        .replace("%%OUT_TOKENS%%","—")
        .replace("%%STEP%%", step_number)
        .replace("%%HASH%%", checkpoint_hash)
        .replace("%%SIZE%%", str(model_size_kb))
        .replace("%%FUNC_OUTPUT%%", "")
        .replace("%%DOMAIN_OUTPUT%%", "")
        .replace("%%GO_OUTPUT%%", "")   
        .replace("%%LOC_OUTPUT%%", "")
    )

@app.post("/", response_class=HTMLResponse)
async def generate(dna_input: str = Form(...)):
    dna_input = dna_input.strip()
    if not dna_input:
        return await home()

    input_ids = torch.tensor(
        [tokenizer.encode(dna_input)], device=DEVICE
    )
    input_token_count = input_ids.shape[1]

    start = time.time()
    with torch.no_grad():
        logits, hidden_states, _ = model(
            input_ids,
            attention_mask=torch.ones_like(input_ids)
        )

        generated = model.generate(
            input_ids, max_new_tokens=50
        )

        func_logits = model.predict_function(
            hidden_states,
            attention_mask=torch.ones_like(input_ids)
        )
        domain_logits = model.predict_domain(
        hidden_states,
        attention_mask=torch.ones_like(input_ids)
        )

        loc_logits = model.predict_localization(
            hidden_states,
            attention_mask=torch.ones_like(input_ids)
        )

        go_logits = model.predict_go(
            hidden_states,
            attention_mask=torch.ones_like(input_ids)
        )

    func_probs = torch.sigmoid(func_logits)[0]
    domain_probs = torch.sigmoid(domain_logits)[0]
    loc_probs = torch.sigmoid(loc_logits)[0]
    go_probs = torch.sigmoid(go_logits)[0]
        # ---------------- DEBUG: Print probabilities ----------------
    print("Function probabilities:", func_probs.tolist())
    print("Domain probabilities:", domain_probs.tolist())
    print("Localization probabilities:", loc_probs.tolist())
    print("GO term probabilities:", go_probs.tolist())

    latency = round((time.time() - start) * 1000, 2)
    output_token_count = generated.shape[1] - input_ids.shape[1]

    decoded = tokenizer.decode(generated[0].tolist())
    output_text = " ".join(decoded) if isinstance(decoded, list) else decoded

    # ---------------- Function predictions ----------------
    report_text = ""

    # ---------------- FUNCTIONS ----------------
    func_report = "=== Predicted Protein Function ===\n"

    for name, prob in zip(FUNCTION_NAMES, func_probs):
        if prob.item() > 0.3:
            func_report += f"• {name:<18} Confidence: {prob.item():.2f}\n"

    report_text += "\n"

    # ---------------- DOMAINS ----------------
    domain_report ="=== Predicted Domains ===\n"

    for name, prob in zip(DOMAIN_NAMES, domain_probs):
        if prob.item() > 0.03:
            domain_report += f"• {name}\n"


    # ---------------- GO TERMS ----------------
    go_report = "=== Predicted GO Terms ===\n"

    for name, prob in zip(GO_NAMES, go_probs):
        if prob.item() > 0.05:
            go_report += f"• {name}\n"


    # ---------------- LOCALIZATION ----------------
    loc_report = "=== Predicted Localization ===\n"

    for name, prob in zip(LOCALIZATION_NAMES, loc_probs):
        if prob.item() > 0.4:
            loc_report += f"• {name}\n"

    if loc_report.strip() == "":
        loc_report = "No confident localization predictions."

    # fallback if no confident function
    if func_report.strip() == "":
        func_report = "• No confident function detected\n"
    domain_report_html = domain_report.replace("\n", "<br>")
    go_report_html = go_report.replace("\n", "<br>")
    loc_report_html = loc_report.replace("\n", "<br>")
    func_report_html = func_report.replace("\n", "<br>")

    # ---------------- Next-token probabilities ----------------

    probs = torch.softmax(logits[:, -1, :], dim=-1)
    topk = torch.topk(probs, 5)

    bars = ""
    for idx, val in zip(topk.indices[0], topk.values[0]):
        tok = tokenizer.decode([idx.item()])
        bars += f"""
        <div class="bar">
            <span>{tok}</span>
            <div class="bar-fill" style="width:{val.item()*100:.1f}%"></div>
            <small>{val.item():.2f}</small>
        </div>
        """

    return (HTML_PAGE
    .replace("%%OUTPUT%%", output_text)
    .replace("%%FUNC_OUTPUT%%", func_report_html)
    .replace("%%DOMAIN_OUTPUT%%", domain_report_html)
    .replace("%%GO_OUTPUT%%", go_report_html)
    .replace("%%LOC_OUTPUT%%", loc_report_html)
    .replace("%%INPUT%%", dna_input)
    .replace("%%PARAMS%%", f"{NUM_PARAMS:,}")
    .replace("%%BITS%%", str(BITS))
    .replace("%%DEVICE%%", DEVICE.upper())
    .replace("%%LATENCY%%", str(latency))
    .replace("%%BARS%%", bars)
    .replace("%%IN_TOKENS%%", str(input_token_count))
    .replace("%%OUT_TOKENS%%", str(output_token_count))
    .replace("%%STEP%%", step_number)
    .replace("%%HASH%%", checkpoint_hash)
    .replace("%%SIZE%%", str(model_size_kb))
)

