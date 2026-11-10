const $ = (s, r = document) => r.querySelector(s);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const level = s => (s >= .6 ? 'high' : s >= .3 ? 'mid' : 'low');
const LABEL = { high: 'High confidence', mid: 'Neutral', low: 'Low confidence' };
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
let ENGINE = 'intelligent', HEALTH = null;

/* ---------- typewriter ---------- */
async function typeInto(el, text, speed = 28) {
  el.textContent = '';
  for (const ch of text) { el.textContent += ch; await sleep(speed + Math.random() * 25); }
}
(async () => {
  await typeInto($('#typed'), 'Which section does this story belong to?', 38);
  $('.caret').classList.add('done');
})();

/* ---------- engines / status ---------- */
function paintStatus() {
  const s = $('#status'), h = HEALTH;
  s.className = 'status';
  if (!h) { $('span', s).textContent = 'offline'; return; }
  const e = h.engines.find(x => x.id === ENGINE) || h.engines[0];
  if (h.demo) { s.classList.add('demo'); $('span', s).textContent = 'demo mode · no model files'; return; }
  s.classList.add(e.id === 'intelligent' && e.semantic !== 'ready' ? 'demo' : 'ok');
  $('span', s).textContent = e.id === 'intelligent'
    ? `intelligent · ${e.labels.length} classes · ${e.semantic === 'ready' ? 'SVM + semantic' : e.semantic === 'loading' ? 'semantic loading…' : 'SVM only'}`
    : `standard · ${e.labels.length} classes`;
  s.title = (e.notes || []).join('\n');
  document.querySelectorAll('.engine button').forEach(b => {
    const eng = h.engines.find(x => x.id === b.dataset.engine);
    b.disabled = !eng || !eng.available; b.classList.toggle('on', b.dataset.engine === ENGINE);
  });
}
async function loadHealth() {
  try {
    HEALTH = await (await fetch('/api/health')).json();
    if (!HEALTH.engines.find(e => e.id === ENGINE && e.available)) ENGINE = HEALTH.default;
  } catch { HEALTH = null; }
  paintStatus();
  if (HEALTH && HEALTH.ocr === 'none') $('#drop small').textContent = `PDF with selectable text · up to ${HEALTH.max_upload_mb} MB · image OCR not available on this server`;
  if (HEALTH && HEALTH.engines.some(e => e.semantic === 'loading')) setTimeout(loadHealth, 4000);
}
document.querySelectorAll('.engine button').forEach(b => b.addEventListener('click', () => { ENGINE = b.dataset.engine; paintStatus(); }));
loadHealth();

/* ---------- tabs ---------- */
document.querySelectorAll('.tab').forEach((t, i) => t.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x === t));
  document.querySelectorAll('.pane').forEach(p => p.classList.toggle('active', p.id === 'pane-' + t.dataset.tab));
  $('.tabs').dataset.i = i;
}));

/* ---------- shared renderers ---------- */
function countUp(el, to, ms = 1100) {
  const t0 = performance.now();
  const tick = now => { const k = Math.min(1, (now - t0) / ms), e = 1 - Math.pow(1 - k, 3); el.textContent = (to * e).toFixed(1); if (k < 1) requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
}
const loader = msg => `<div class="loadtxt">${msg}</div><div class="loader"></div>`;
const barsHTML = preds => `<div class="bars">${preds.map(p => {
  const l = level(p.score);
  return `<div class="bar"><span class="n">${esc(p.label)}</span><div class="t"><div class="f ${l}" data-w="${p.score * 100}"></div></div><span class="v">${(p.score * 100).toFixed(1)}%</span></div>`;
}).join('')}</div>`;
const growBars = root => requestAnimationFrame(() => root.querySelectorAll('.f[data-w]').forEach(f => f.style.width = f.dataset.w + '%'));
function agreeBadge(m) {
  if (m.agree === true) return '<span class="pill agree">✓ engines agree</span>';
  if (m.agree === false) return `<span class="pill mixed">mixed signal · SVM ${esc(m.svm)} / semantic ${esc(m.semantic)}</span>`;
  return '';
}

async function renderResult(host, text, preds, meta) {
  const top = preds[0], lv = level(top.score);
  host.innerHTML = `
    <div class="result ${lv}">
      <p class="q">“${esc(text)}”</p>
      <div class="top-row">
        <div class="ring" style="--c:var(--${lv})"><span><b class="num">0.0</b><small>%</small></span></div>
        <div><div class="cat" id="cat"></div><span class="tag">${LABEL[lv]}</span> ${meta ? agreeBadge(meta) : ''}</div>
      </div>
      ${barsHTML(preds.slice(0, 6))}
    </div>`;
  const ring = $('.ring', host);
  growBars(host); countUp($('.num', host), top.score * 100);
  const t0 = performance.now();
  (function a(now) { const k = Math.min(1, (now - t0) / 1100); ring.style.setProperty('--p', top.score * 100 * (1 - Math.pow(1 - k, 3))); if (k < 1) requestAnimationFrame(a); })(t0);
  typeInto($('#cat', host), top.label, 45);
}

/* ---------- single headline ---------- */
const EXAMPLES = ['Central bank holds rates as inflation cools', 'Underdogs stun champions in extra-time final', 'New chip promises faster on-device AI', 'Police arrest suspects in city robbery'];
EXAMPLES.forEach(t => { const b = document.createElement('button'); b.className = 'chip'; b.textContent = t.split(' ').slice(0, 3).join(' ') + '…'; b.onclick = () => { $('#headline').value = t; classifyText(); }; $('#examples').append(b); });

async function classifyText() {
  const text = $('#headline').value.trim(), host = $('#text-result'), btn = $('#go-text');
  if (text.length < 2) return $('#headline').focus();
  btn.disabled = true; host.innerHTML = loader('reading headline');
  try {
    const [r] = await Promise.all([fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, engine: ENGINE }) }), sleep(650)]);
    if (!r.ok) throw new Error((await r.json()).detail || 'Request failed');
    const d = await r.json();
    await renderResult(host, text, d.predictions, d.meta);
  } catch (e) { host.innerHTML = `<div class="err">${esc(e.message)}</div>`; }
  btn.disabled = false;
}
$('#go-text').onclick = classifyText;
$('#headline').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); classifyText(); } });

/* ---------- newspaper upload ---------- */
const drop = $('#drop'), input = $('#file');
['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => e.dataTransfer.files[0] && analyze(e.dataTransfer.files[0]));
input.addEventListener('change', () => input.files[0] && analyze(input.files[0]));

async function analyze(file) {
  const host = $('#file-result');
  $('#fname').textContent = file.name;
  drop.classList.add('busy');
  const steps = ['reading pages', 'detecting headlines', 'reading each story in full', 'classifying every story'];
  let i = 0; host.innerHTML = loader(steps[0]);
  const timer = setInterval(() => { i = Math.min(i + 1, steps.length - 1); const t = $('.loadtxt', host); if (t) t.textContent = steps[i]; }, 2000);
  try {
    const fd = new FormData(); fd.append('file', file); fd.append('engine', ENGINE);
    const r = await fetch('/api/analyze', { method: 'POST', body: fd });
    if (r.status === 413) throw new Error(`File too large for this server (limit ${HEALTH ? HEALTH.max_upload_mb : 4} MB). Try a smaller PDF or fewer pages.`);
    let d; try { d = await r.json(); } catch { throw new Error('The server could not process this file (' + r.status + ').'); }
    if (!r.ok) throw new Error(d.detail || 'Upload failed');
    renderStories(host, d);
  } catch (e) { host.innerHTML = `<div class="err">${esc(e.message)}</div>`; }
  clearInterval(timer); drop.classList.remove('busy'); input.value = '';
}

function renderStories(host, d) {
  if (!d.total) { host.innerHTML = '<div class="empty">No headlines detected. Try a sharper, higher-resolution scan.</div>'; return; }
  const state = { view: 'category', filter: null };
  host.innerHTML = `
    <div class="summary"><b>${d.total} stor${d.total > 1 ? 'ies' : 'y'} · ${d.categories.length} categor${d.categories.length > 1 ? 'ies' : 'y'}</b><span>${d.pages.length} page${d.pages.length > 1 ? 's' : ''} · ${d.engine} engine</span></div>
    <div class="toolbar"><div class="seg"><button data-v="category" class="on">By category</button><button data-v="page">By page</button></div></div>
    <div class="cats" id="cats"></div><div id="list"></div>`;
  const cats = $('#cats', host), list = $('#list', host), byId = Object.fromEntries(d.stories.map(s => [s.id, s]));

  const chips = () => {
    cats.innerHTML = `<button class="cat-chip ${state.filter ? '' : 'on'}" data-c="">All <b>${d.total}</b></button>` +
      d.categories.map(c => `<button class="cat-chip ${state.filter === c.label ? 'on' : ''}" data-c="${esc(c.label)}">${esc(c.label)} <b>${c.count}</b></button>`).join('');
    cats.querySelectorAll('.cat-chip').forEach(b => b.onclick = () => { state.filter = b.dataset.c || null; chips(); draw(); });
  };
  const card = (s, n) => {
    const top = s.predictions[0], lv = level(top.score), m = s.meta;
    const el = document.createElement('div'); el.className = 'hl'; el.style.animationDelay = (n * 60) + 'ms';
    el.innerHTML = `<div><div class="txt"></div><div class="src">page ${s.page}</div></div>
      <div class="meta"><b>${esc(top.label)}</b><em>${(top.score * 100).toFixed(1)}%</em><br><span class="tag ${lv}">${LABEL[lv]}</span></div>
      <div class="mini"><i class="f ${lv}" style="width:0"></i></div>
      <div class="badges"><span class="pill">${m.mode === 'story' ? `read full story · ${m.body_words} words` : 'headline only'}</span>${agreeBadge(m)}<span class="chev">details ▾</span></div>
      <div class="detail">${s.snippet ? `<p>“${esc(s.snippet)}…”</p>` : ''}${barsHTML(s.predictions.slice(0, 5))}</div>`;
    el.onclick = () => { el.classList.toggle('open'); growBars(el); };
    typeInto($('.txt', el), s.text, 14);
    setTimeout(() => $('.mini i', el).style.width = top.score * 100 + '%', 150);
    return el;
  };
  const draw = () => {
    list.innerHTML = ''; let n = 0;
    if (state.view === 'category') {
      d.categories.filter(c => !state.filter || c.label === state.filter).forEach(c => {
        const g = document.createElement('div'); g.className = 'group';
        g.innerHTML = `<h3>${esc(c.label)} <em>${c.count} stor${c.count > 1 ? 'ies' : 'y'} · avg ${(c.avg_score * 100).toFixed(0)}%</em></h3>`;
        c.headlines.forEach(h => g.append(card(byId[h.id], n++)));
        list.append(g);
      });
    } else {
      d.pages.forEach(p => {
        const ids = p.story_ids.filter(id => !state.filter || byId[id].predictions[0].label === state.filter);
        const g = document.createElement('div'); g.className = 'group';
        g.innerHTML = `<h3>PAGE ${String(p.page).padStart(2, '0')}</h3>` + (ids.length ? '' : '<div class="empty">No matching stories on this page</div>');
        ids.forEach(id => g.append(card(byId[id], n++)));
        list.append(g);
      });
    }
  };
  host.querySelectorAll('.seg button').forEach(b => b.onclick = () => {
    state.view = b.dataset.v; host.querySelectorAll('.seg button').forEach(x => x.classList.toggle('on', x === b)); draw();
  });
  chips(); draw();
}
