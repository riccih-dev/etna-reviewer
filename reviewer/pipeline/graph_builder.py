"""Stage 5 — assemble claims + evidence + verification into an explicit graph.

Node kinds: CLAIM, EVIDENCE, DATASET, ASSUMPTION.
Edge relations: supports, partially_supports, refutes, insufficient, evaluated_on, assumes.

Rendered as a dark, canvas-based force-directed graph — no charting library,
just a small physics loop (repulsion + spring edges) and 2D canvas drawing,
matching the standalone graph.html viewer this project's design is based on.
"""

from __future__ import annotations

import json

from ..models import Claim, Evidence, Sentence, Verification
from .checkers import datasets_mentioned

RELATION_FOR_LABEL = {
    "SUPPORTS": "supports",
    "PARTIALLY_SUPPORTS": "partially_supports",
    "CONTRADICTS": "refutes",
    "INSUFFICIENT_INFORMATION": "insufficient",
}

ASSUMPTION_TEXT = {
    "generalization": "the evaluated setting(s) represent the claimed scope",
    "causal": "the observed association is identified as causal",
}


def build_graph_data(
    claims: list[Claim],
    sentences: list[Sentence],
    graph_edges: list[tuple[Claim, Evidence, Verification]],
) -> dict:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    for c in claims:
        nodes[c.id] = {
            "id": c.id, "kind": "CLAIM", "label": c.text,
            "type": c.type, "section": c.section, "location": f"{c.section}, p.{c.page}",
        }

    for claim, evidence, verification in graph_edges:
        if evidence.id not in nodes:
            nodes[evidence.id] = {
                "id": evidence.id, "kind": "EVIDENCE", "label": evidence.text,
                "section": evidence.section, "location": f"{evidence.section}, p.{evidence.page}",
            }
        edges.append({
            "source": evidence.id, "target": claim.id,
            "relation": RELATION_FOR_LABEL[verification.label],
            "score": verification.entailment_score, "rationale": verification.rationale,
        })

    for name in datasets_mentioned(sentences):
        did = "D_" + name.replace(" ", "_")
        nodes[did] = {"id": did, "kind": "DATASET", "label": name}
        for c in claims:
            if name.lower() in c.text.lower():
                edges.append({"source": c.id, "target": did, "relation": "evaluated_on"})

    for c in claims:
        if c.type in ASSUMPTION_TEXT:
            aid = f"A_{c.id}"
            nodes[aid] = {"id": aid, "kind": "ASSUMPTION", "label": ASSUMPTION_TEXT[c.type]}
            edges.append({"source": c.id, "target": aid, "relation": "assumes"})

    return {"nodes": list(nodes.values()), "edges": edges}


SEVERITY_ICON = {"major": "\U0001F6A9", "minor": "⚠️", "info": "✅"}

_TEMPLATE = r"""
<div id="ceg-root" style="background:#14161a;border:1px solid #2a2f3a;border-radius:10px;
     overflow:hidden;display:flex;height:__HEIGHT__px;font:13px/1.5 ui-sans-serif,system-ui,sans-serif;color:#e6e9ef">
<div id="ceg-cards" style="width:420px;flex:none;overflow:auto;padding:12px;border-right:1px solid #2a2f3a"></div>
<div style="flex:1;position:relative;display:flex;flex-direction:column">
 <canvas id="ceg-c" style="width:100%;flex:1;display:block;cursor:grab"></canvas>
 <div style="padding:6px 12px;font:11.5px ui-sans-serif,system-ui,sans-serif;color:#8b94a7;
      border-top:1px solid #2a2f3a;display:flex;gap:14px;flex-wrap:wrap;flex:none">
  <span><i style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#e8562a;margin-right:5px"></i>claim</span>
  <span><i style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#9ece6a;margin-right:5px"></i>evidence</span>
  <span><i style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#e0af68;margin-right:5px"></i>dataset</span>
  <span><i style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#bb9af7;margin-right:5px"></i>assumption</span>
  <span style="margin-left:auto">drag to pan · scroll to zoom · double-click to reset view · click a finding or a node to highlight the other · click empty space to clear</span>
 </div>
</div>
<div id="ceg-tip" style="position:fixed;max-width:380px;background:#0b0d12;border:1px solid #2a2f3a;
     border-radius:8px;padding:8px 10px;font:12.5px/1.5 ui-sans-serif,system-ui,sans-serif;color:#e6e9ef;
     pointer-events:none;display:none;z-index:9999"></div>
</div>
<style>
 #ceg-cards .card{border:1px solid #2a2f3a;border-radius:8px;padding:10px;margin-bottom:9px;cursor:pointer}
 #ceg-cards .card:hover{border-color:#e8562a}
 #ceg-cards .card.sel{border-color:#e8562a;background:#1d2230}
 #ceg-cards .card .head{font-weight:600}
 #ceg-cards .card .conf{float:right;font-size:11px;color:#8b94a7}
 #ceg-cards .card .loc{color:#8b94a7;font-size:11.5px;margin:2px 0 6px}
 #ceg-cards .card .row{margin-top:4px;font-size:12.5px}
 #ceg-cards .card .row b{color:#c8cede}
</style>
<script>
(function(){
const CARDS = __CARDS__;
const DATA = __GRAPH__;
const COL={CLAIM:'#e8562a',EVIDENCE:'#9ece6a',DATASET:'#e0af68',ASSUMPTION:'#bb9af7'};
const EDGE={supports:'#9ece6a',partially_supports:'#e0af68',insufficient:'#565f73',
            refutes:'#f7768e',evaluated_on:'#3d4657',assumes:'#bb9af7'};

// ---- findings panel ----
const cardsEl=document.getElementById('ceg-cards');
function esc(s){return String(s==null?'':s).replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]));}
cardsEl.innerHTML=CARDS.map((c,i)=>{
 let h='<div class="card" data-i="'+i+'"><div class="head">'+esc(c.icon)+' '+esc(c.badge)+
  (c.confidence!=null?'<span class="conf">confidence '+c.confidence.toFixed(2)+'</span>':'')+'</div>';
 h+='<div class="loc">'+esc(c.locations.join(' · '))+'</div>';
 if(c.claim_text) h+='<div class="row"><b>Claim:</b> '+esc(c.claim_text)+'</div>';
 if(c.evidence_text) h+='<div class="row"><b>Evidence found:</b> '+esc(c.evidence_text)+'</div>';
 if(c.missing) h+='<div class="row"><b>Missing / weak part:</b> '+esc(c.missing)+'</div>';
 if(c.question) h+='<div class="row"><b>Targeted reviewer question:</b> '+esc(c.question)+'</div>';
 if(c.action) h+='<div class="row"><b>Suggested author action:</b> '+esc(c.action)+'</div>';
 if(c.rationale) h+='<div class="row" style="color:#8b94a7">Rationale: '+esc(c.rationale)+'</div>';
 return h+'</div>';
}).join('');

// ---- graph ----
const c=document.getElementById('ceg-c'), ctx=c.getContext('2d');
let W,H,selNodes=null,selCardIdx=null;
const nodes=DATA.nodes.map(n=>({...n,x:Math.random()*600+100,y:Math.random()*400+60,vx:0,vy:0}));
const idx=new Map(nodes.map(n=>[n.id,n]));
const edges=DATA.edges.filter(e=>idx.has(e.source)&&idx.has(e.target));
const nodeToCards=new Map();
CARDS.forEach((card,i)=>(card.node_ids||[]).forEach(nid=>{
 if(!nodeToCards.has(nid)) nodeToCards.set(nid,[]);
 nodeToCards.get(nid).push(i);
}));

function size(){
  const rect=c.getBoundingClientRect();
  W=c.width=rect.width*devicePixelRatio; H=c.height=rect.height*devicePixelRatio;
}
size();
new ResizeObserver(size).observe(c);

function step(){
 for(const n of nodes){n.vx*=.86;n.vy*=.86;}
 for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){
  const a=nodes[i],b=nodes[j];let dx=b.x-a.x,dy=b.y-a.y,d2=dx*dx+dy*dy+.01;
  if(d2<90000){const f=1400/d2,d=Math.sqrt(d2);a.vx-=f*dx/d;a.vy-=f*dy/d;b.vx+=f*dx/d;b.vy+=f*dy/d;}}
 for(const e of edges){const a=idx.get(e.source),b=idx.get(e.target);
  let dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy)||1,f=(d-130)*.012;
  a.vx+=f*dx/d;a.vy+=f*dy/d;b.vx-=f*dx/d;b.vy-=f*dy/d;}
 const cx=c.clientWidth/2,cy=c.clientHeight/2;
 for(const n of nodes){n.vx+=(cx-n.x)*.002;n.vy+=(cy-n.y)*.002;n.x+=n.vx;n.y+=n.vy;}
}
function hot(id){return !selNodes || selNodes.has(id);}

// ---- pan / zoom camera ----
let camX=0, camY=0, camScale=1;

function toWorld(clientX, clientY){
 const r=c.getBoundingClientRect();
 return {x:(clientX-r.left-camX)/camScale, y:(clientY-r.top-camY)/camScale};
}

function draw(){
 ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
 ctx.clearRect(0,0,c.clientWidth,c.clientHeight);
 ctx.translate(camX,camY);
 ctx.scale(camScale,camScale);
 for(const e of edges){const a=idx.get(e.source),b=idx.get(e.target);
  const on=hot(e.source)&&hot(e.target);
  ctx.strokeStyle=on?(EDGE[e.relation]||'#444b5c'):'#1a1d24';ctx.lineWidth=(on?1.4:.6)/camScale;
  ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();}
 for(const n of nodes){const on=hot(n.id);
  const r=n.kind==='CLAIM'?9:6;
  ctx.globalAlpha=on?1:.2;ctx.fillStyle=COL[n.kind]||'#888';
  ctx.beginPath();ctx.arc(n.x,n.y,r,0,7);ctx.fill();
  if(on&&(n.kind==='CLAIM'||selNodes)){ctx.fillStyle='#c8cede';ctx.font=(11/camScale)+'px ui-sans-serif,system-ui,sans-serif';
   ctx.fillText(n.id,n.x+r+3,n.y+3);}
  ctx.globalAlpha=1;}
}
(function loop(){step();draw();requestAnimationFrame(loop);})();

function highlightCards(indices){
 selCardIdx=indices;
 cardsEl.querySelectorAll('.card').forEach(el=>{
  const on=indices && indices.has(Number(el.dataset.i));
  el.classList.toggle('sel', !!on);
 });
 if(indices && indices.size){
  const first=cardsEl.querySelector('.card.sel');
  if(first) first.scrollIntoView({block:'nearest', behavior:'smooth'});
 }
}

function selectCard(i){
 const card=CARDS[i];
 selNodes = card.node_ids && card.node_ids.length ? new Set(card.node_ids) : null;
 highlightCards(new Set([i]));
}

function selectNode(n){
 const cardIdxs=nodeToCards.get(n.id);
 if(cardIdxs && cardIdxs.length){
  const nodeIds=new Set();
  cardIdxs.forEach(i=>(CARDS[i].node_ids||[]).forEach(id=>nodeIds.add(id)));
  selNodes=nodeIds;
  highlightCards(new Set(cardIdxs));
 } else {
  const neighbors=new Set([n.id]);
  for(const e of edges){ if(e.source===n.id) neighbors.add(e.target); if(e.target===n.id) neighbors.add(e.source); }
  selNodes=neighbors;
  highlightCards(null);
 }
}

function clearSelection(){ selNodes=null; highlightCards(null); }

cardsEl.addEventListener('click',ev=>{
 const el=ev.target.closest('.card');
 if(!el) return;
 const i=Number(el.dataset.i);
 if(selCardIdx && selCardIdx.size===1 && selCardIdx.has(i)){ clearSelection(); return; }
 selectCard(i);
});

const tip=document.getElementById('ceg-tip');
let dragging=false, dragMoved=false, dragStart=null, camStart=null;

c.addEventListener('mousedown',ev=>{
 dragging=true; dragMoved=false;
 dragStart={x:ev.clientX,y:ev.clientY};
 camStart={x:camX,y:camY};
 c.style.cursor='grabbing';
 tip.style.display='none';
});

window.addEventListener('mousemove',ev=>{
 if(dragging){
  const dx=ev.clientX-dragStart.x, dy=ev.clientY-dragStart.y;
  if(Math.hypot(dx,dy)>3) dragMoved=true;
  if(dragMoved){ camX=camStart.x+dx; camY=camStart.y+dy; tip.style.display='none'; return; }
 }
 const r=c.getBoundingClientRect();
 if(ev.clientX<r.left||ev.clientX>r.right||ev.clientY<r.top||ev.clientY>r.bottom){ tip.style.display='none'; return; }
 const w=toWorld(ev.clientX,ev.clientY);
 const n=nodes.find(n=>Math.hypot(n.x-w.x,n.y-w.y)<11);
 if(n){tip.style.display='block';tip.style.left=(ev.clientX+14)+'px';tip.style.top=(ev.clientY+8)+'px';
  tip.innerHTML='<b>'+n.id+'</b> &middot; '+n.kind+(n.type?' ('+n.type+')':'')+'<br>'+
   (n.location?'<i style="color:#8b94a7">'+n.location+'</i><br>':'')+
   String(n.label||'').slice(0,160);}
 else tip.style.display='none';
});

window.addEventListener('mouseup',ev=>{
 if(!dragging) return;
 dragging=false;
 c.style.cursor='grab';
 if(!dragMoved){
  const w=toWorld(ev.clientX,ev.clientY);
  const n=nodes.find(n=>Math.hypot(n.x-w.x,n.y-w.y)<11);
  if(!n){ clearSelection(); } else { selectNode(n); }
 }
});

c.addEventListener('wheel',ev=>{
 ev.preventDefault();
 const r=c.getBoundingClientRect();
 const mx=ev.clientX-r.left, my=ev.clientY-r.top;
 const wx=(mx-camX)/camScale, wy=(my-camY)/camScale;
 const factor=Math.exp(-ev.deltaY*0.001);
 const newScale=Math.min(4,Math.max(0.25,camScale*factor));
 camX=mx-wx*newScale; camY=my-wy*newScale; camScale=newScale;
},{passive:false});

c.addEventListener('dblclick',()=>{ camX=0; camY=0; camScale=1; });
})();
</script>
"""


def render_combined_html(cards: list[dict], graph_data: dict, height: int = 620) -> str:
    """Findings list + canvas graph in one document so clicking either side highlights the other."""
    payload = []
    for c in cards:
        payload.append({
            "icon": SEVERITY_ICON.get(c["severity"], "•"),
            "badge": c["badge"], "kind": c["kind"], "locations": c["locations"],
            "claim_text": c.get("claim_text"), "evidence_text": c.get("evidence_text"),
            "missing": c.get("missing"), "question": c.get("question"), "action": c.get("action"),
            "rationale": c.get("rationale"), "confidence": c.get("confidence"),
            "node_ids": c.get("node_ids", []),
        })
    return (_TEMPLATE
            .replace("__CARDS__", json.dumps(payload))
            .replace("__GRAPH__", json.dumps(graph_data))
            .replace("__HEIGHT__", str(height)))
