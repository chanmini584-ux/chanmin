"use strict";
// SafeWatch 관제 대시보드 — vanilla JS (no build step)

const LV = ["NORMAL", "ATTENTION", "WARNING", "HIGH_RISK", "EMERGENCY"];
const LV_KO = { NORMAL: "정상", ATTENTION: "관심", WARNING: "주의", HIGH_RISK: "고위험", EMERGENCY: "긴급" };
const REVIEW_KO = { NEW: "미확인", VIEWED: "열람", ACKED: "확인", REPORT_READY: "신고준비", REPORTED: "신고완료", DISMISSED: "종결" };
const ST = { site: null, inc: new Map(), sel: null, detail: null, vtab: null, dtab: "overview", popupMin: "WARNING", alerted: new Set(), lastPointer: 0 };
document.addEventListener("pointerdown", (e) => { if (e.target.closest("#detail")) ST.lastPointer = Date.now(); });

const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const lvIdx = (l) => LV.indexOf(l);
const lvVar = (l) => `var(--${l || "NORMAL"})`;
// show times in the site's local time as recorded in the ISO string (not the browser's timezone)
const hms = (iso) => { const m = /T(\d{2}:\d{2}:\d{2})/.exec(iso || ""); return m ? m[1] : (iso || ""); };
const TYPE_KO = { ASSAULT: "폭행/싸움", WEAPON_THREAT: "흉기·위험물 위협", CHASE: "추격", FALL: "쓰러짐", INTRUSION: "침입", LOITERING: "비정상 배회" };
const operator = () => localStorage.getItem("sw_operator") || "";

async function api(path, opt = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opt,
    body: opt.body && typeof opt.body !== "string" ? JSON.stringify(opt.body) : opt.body });
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch {} throw new Error(m); }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r.text();
}

// ------------------------------------------------------------------ boot
async function boot() {
  setInterval(() => ($("#clock").textContent = new Date().toLocaleString("ko-KR", { hour12: false })), 1000);
  $("#viewTabs").addEventListener("click", (e) => { const v = e.target.dataset.view; if (v) showView(v); });
  $("#btnRun").onclick = openRunDialog;
  $("#btnSource").onclick = openSourceDialog;
  $("#soundToggle").checked = localStorage.getItem("sw_sound") === "1";
  $("#soundToggle").onchange = (e) => localStorage.setItem("sw_sound", e.target.checked ? "1" : "0");
  $("#fltAttention").onchange = renderQueue; $("#fltClosed").onchange = renderQueue;
  ST.site = await api("/api/site");
  ST.popupMin = ST.site.policy.popup_min_level;
  $("#siteName").textContent = ST.site.name;
  renderCams();
  (await api("/api/incidents")).forEach((c) => ST.inc.set(c.id, c));
  renderQueue();
  connectWS();
  pollStatus(); setInterval(pollStatus, 2000);
}

function showView(v) {
  document.querySelectorAll("#viewTabs button").forEach((b) => b.classList.toggle("active", b.dataset.view === v));
  for (const id of ["ops", "kpi", "eval"]) $(`#view-${id}`).classList.toggle("hidden", id !== v);
  if (v === "kpi") renderKPI();
  if (v === "eval") renderEval();
}

function connectWS() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => { $("#wsState").textContent = "실시간 연결"; $("#wsState").className = "ws on"; };
  ws.onclose = () => { $("#wsState").textContent = "연결 끊김"; $("#wsState").className = "ws off"; setTimeout(connectWS, 2000); };
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.incident) onIncident(msg.kind, msg.incident);
    if (msg.kind === "job") pollStatus();
  };
}

let detailTimer = null;
function onIncident(kind, card) {
  const prev = ST.inc.get(card.id);
  ST.inc.set(card.id, { ...(prev || {}), ...card });
  const popup = lvIdx(card.peak_level) >= lvIdx(ST.popupMin);
  if (popup && !ST.alerted.has(card.id) && card.review === "NEW") { ST.alerted.add(card.id); toast(card); }
  renderQueue(kind === "incident_alert" ? card.id : null);
  highlightCams();
  if (ST.sel === card.id && !detailTimer) detailTimer = setTimeout(() => { detailTimer = null; loadDetail(card.id, true); }, 700);
}

// ------------------------------------------------------------------ queue
function renderQueue(flashId) {
  const showAtt = $("#fltAttention").checked, showClosed = $("#fltClosed").checked;
  const items = [...ST.inc.values()].filter((c) =>
    (showAtt || lvIdx(c.peak_level) >= lvIdx("WARNING")) && (showClosed || c.status === "OPEN"));
  items.sort((a, b) => (b.status === "OPEN") - (a.status === "OPEN") || (a.review === "NEW" ? 0 : 1) - (b.review === "NEW" ? 0 : 1)
    || lvIdx(b.peak_level) - lvIdx(a.peak_level) || String(b.start_time).localeCompare(a.start_time));
  $("#queueCount").textContent = `${items.length}건 · 미확인 ${items.filter((c) => c.review === "NEW").length}`;
  $("#queue").innerHTML = items.map((c) => `
    <div class="card ${c.id === ST.sel ? "sel" : ""} ${c.id === flashId ? "flash" : ""}" data-id="${c.id}" style="--lvl:${lvVar(c.peak_level)}">
      <div class="row1"><span class="type">${esc(c.type_ko)}</span>
        <span><span class="badge" style="--lvl:${lvVar(c.peak_level)}">${LV_KO[c.peak_level]}</span>${c.status === "OPEN" && c.level !== c.peak_level ? `<span class="chip">현재 ${LV_KO[c.level]}</span>` : ""}</span></div>
      <div class="meta">${hms(c.start_time)} · ${esc(c.location?.name)} (${c.cameras.join("→")})</div>
      <div class="meta">${c.status === "OPEN" ? "<b style='color:#ffa198'>진행중</b>" : "종료"}
        <span class="chip ${c.review}">${REVIEW_KO[c.review] || c.review}</span>${(c.objects || []).map((o) => `<span class="chip">${esc(o)}</span>`).join("")}</div>
    </div>`).join("") || `<div class="empty muted" style="padding:20px">표시할 사건이 없습니다.<br>상단의 '시나리오 재생'으로 시작해 보세요.</div>`;
  $("#queue").querySelectorAll(".card").forEach((el) => (el.onclick = () => select(el.dataset.id)));
}

function toast(c) {
  const el = document.createElement("div");
  el.className = "toast"; el.style.setProperty("--lvl", lvVar(c.peak_level));
  el.innerHTML = `<b>${LV_KO[c.peak_level]} · ${esc(c.type_ko)}</b><div class="small muted">${esc(c.location?.name)} · ${hms(c.start_time)}</div>`;
  el.onclick = () => { select(c.id); el.remove(); };
  $("#toasts").prepend(el);
  setTimeout(() => el.remove(), 12000);
  if ($("#soundToggle").checked) beep(lvIdx(c.peak_level) >= 3 ? 3 : 1);
}

function beep(n) {
  try {
    const ctx = new AudioContext();
    for (let i = 0; i < n; i++) {
      const o = ctx.createOscillator(), g = ctx.createGain();
      o.frequency.value = 880; o.connect(g); g.connect(ctx.destination);
      g.gain.setValueAtTime(0.15, ctx.currentTime + i * 0.25); g.gain.setValueAtTime(0, ctx.currentTime + i * 0.25 + 0.15);
      o.start(ctx.currentTime + i * 0.25); o.stop(ctx.currentTime + i * 0.25 + 0.16);
    }
  } catch {}
}

// ------------------------------------------------------------------ cameras & jobs
function renderCams() {
  $("#cams").innerHTML = ST.site.cameras.map((c) => `
    <div class="cam" id="cam-${c.id}" data-id="${c.id}">
      <img data-src="/api/cameras/${c.id}/live.mjpg" alt="${esc(c.name)}" class="hidden">
      <div class="off">신호 없음 (분석 작업 없음)</div>
      <div class="label"><b>${c.id}</b> ${esc(c.name)}</div>
    </div>`).join("");
}

function setCamLive(id, live) {
  const el = $(`#cam-${id}`); if (!el) return;
  const img = $("img", el), off = $(".off", el);
  if (live && img.classList.contains("hidden")) { img.src = img.dataset.src + "?t=" + Date.now(); img.classList.remove("hidden"); off.classList.add("hidden"); }
  if (!live && !img.classList.contains("hidden")) { img.removeAttribute("src"); img.classList.add("hidden"); off.classList.remove("hidden"); }
}

function highlightCams() {
  const open = [...ST.inc.values()].filter((c) => c.status === "OPEN");
  for (const c of ST.site.cameras) {
    const el = $(`#cam-${c.id}`); if (!el) continue;
    const lv = Math.max(-1, ...open.filter((i) => i.cameras.includes(c.id)).map((i) => lvIdx(i.level)));
    el.classList.toggle("hl", lv >= 3); el.classList.toggle("hl2", lv === 2);
  }
}

async function pollStatus() {
  let s; try { s = await api("/api/status"); } catch { return; }
  const running = s.jobs.filter((j) => ["running", "starting"].includes(j.status));
  // running cameras stream live; finished ones keep showing their last analysed frame
  const liveCams = new Set([...running.flatMap((j) => j.cameras), ...Object.keys(s.pipelines)]);
  ST.site.cameras.forEach((c) => setCamLive(c.id, liveCams.has(c.id)));
  $("#jobs").innerHTML = s.jobs.slice(-8).reverse().map((j) => `
    <div class="job">${esc(j.title)} <span class="st">${j.status}${j.error ? " · " + esc(j.error) : ""}${j.stream ? " · " + esc(j.stream) : ""}</span>
      ${["running", "starting"].includes(j.status) ? `<button class="btn" style="padding:0 6px;margin-left:6px" data-stop="${j.id}">중지</button>` : ""}</div>`).join("")
    + Object.entries(s.pipelines).map(([k, p]) => `<div class="job muted">${k}: ${p.frames}f · 처리 ${p.proc_ms_avg}ms/f</div>`).join("")
    || `<span class="muted small">실행 중인 작업이 없습니다.</span>`;
  $("#jobs").querySelectorAll("[data-stop]").forEach((b) => (b.onclick = () => api(`/api/jobs/${b.dataset.stop}/stop`, { method: "POST" }).then(pollStatus)));
}

// ------------------------------------------------------------------ detail
async function select(id) {
  ST.sel = id; ST.vtab = null; ST.dtab = "overview";
  renderQueue();
  await api(`/api/incidents/${id}/view`, { method: "POST", body: { operator: operator() } }).catch(() => {});
  loadDetail(id);
}

async function loadDetail(id, soft = false) {
  const d = await api(`/api/incidents/${id}`);
  if (ST.sel !== id) return;
  ST.detail = d;
  ST.inc.set(id, { ...(ST.inc.get(id) || {}), review: d.review, status: d.status, level: d.level, peak_level: d.peak_level });
  if (soft && ST.dtab === "report") return; // do not disturb an operator editing the report
  if (soft && Date.now() - ST.lastPointer < 1200) { setTimeout(() => loadDetail(id, true), 800); return; }
  renderDetail(soft);
}

function renderDetail(soft = false) {
  const d = ST.detail; if (!d) return;
  // keep the running <video>/<img> element across soft refreshes so playback does not restart
  const oldPlayer = soft ? $("#detail .player") : null;
  const cams = Object.keys(d.clips || {});
  if (!ST.vtab) ST.vtab = cams.find((c) => d.clips[c].status === "ready") || d.primary_camera;
  const sits = Object.values(d.situations).sort((a, b) => b.peak_score - a.peak_score);
  $("#detail").innerHTML = `
    <div class="d-head">
      <h2><span class="badge" style="--lvl:${lvVar(d.peak_level)}">${LV_KO[d.peak_level]}</span>${esc(d.type_ko)}
        <span class="chip ${d.review}">${REVIEW_KO[d.review] || d.review}</span></h2>
      <div class="muted small">${esc(d.start_time.replace("T", " "))} · ${esc(d.location.name)} · ${esc(d.location.address)}<br>
        ${d.status === "OPEN" ? `<b style="color:#ffa198">진행중</b> · 현재 ${LV_KO[d.level]}` : "종료"} · 사건 ID ${esc(d.id)}</div>
      <div class="d-actions">
        <button class="btn primary" id="aAck" ${["ACKED", "REPORT_READY", "REPORTED"].includes(d.review) ? "disabled" : ""}>상황 확인(ACK)</button>
        <button class="btn" id="aReport">신고 지원</button>
        <button class="btn good" data-fb="TP">정탐</button>
        <button class="btn danger" data-fb="FP">오탐</button>
        <button class="btn" id="aDismiss">종결</button>
        ${d.feedback ? `<span class="muted small">피드백: ${d.feedback.label}</span>` : ""}
      </div>
      <div class="vtabs" style="margin:8px 0 0">${["overview", "evidence", "report"].map((t) =>
        `<button data-dtab="${t}" class="${ST.dtab === t ? "active" : ""}">${{ overview: "개요", evidence: "판단 근거", report: "신고 지원" }[t]}</button>`).join("")}</div>
    </div>
    <div id="dbody"></div>`;
  $("#aAck").onclick = async () => { await api(`/api/incidents/${d.id}/ack`, { method: "POST", body: { operator: operator() } }); loadDetail(d.id); };
  $("#aReport").onclick = () => { ST.dtab = "report"; renderDetail(); };
  $("#aDismiss").onclick = async () => { const r = prompt("종결 사유 (예: 상황 해소, 오탐)"); if (r !== null) { await api(`/api/incidents/${d.id}/dismiss`, { method: "POST", body: { reason: r, operator: operator() } }); loadDetail(d.id); } };
  document.querySelectorAll("[data-fb]").forEach((b) => (b.onclick = async () => { await api(`/api/incidents/${d.id}/feedback`, { method: "POST", body: { label: b.dataset.fb, operator: operator() } }); loadDetail(d.id); }));
  document.querySelectorAll("[data-dtab]").forEach((b) => (b.onclick = () => { ST.dtab = b.dataset.dtab; renderDetail(); }));
  const body = $("#dbody");
  if (ST.dtab === "overview") {
    body.innerHTML = overviewHTML(d, cams);
    const np = $(".player", body);
    if (oldPlayer && np && oldPlayer.dataset.sig === np.dataset.sig) np.replaceWith(oldPlayer);
  }
  if (ST.dtab === "evidence") body.innerHTML = evidenceHTML(sits);
  if (ST.dtab === "report") return renderReport(d);
  body.querySelectorAll("[data-vtab]").forEach((b) => (b.onclick = () => { ST.vtab = b.dataset.vtab; renderDetail(); }));
}

function playerHTML(d, cams) {
  const all = [...new Set([...cams, ...d.cameras, ...d.related_cameras.filter((r) => r.status !== "nearby").map((r) => r.camera)])];
  const tab = ST.vtab;
  const clip = d.clips?.[tab];
  let media;
  if (clip && clip.status === "ready") media = `<video controls autoplay muted src="/media/${esc(clip.path)}"></video><div class="muted small">사건 영상 (${clip.start_ts}s ~ ${clip.end_ts}s, 사건 전 버퍼 포함)</div>`;
  else media = `<img src="/api/cameras/${tab}/live.mjpg?t=${Date.now()}" onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'muted small',textContent:'영상 없음 (클립 생성 전이거나 카메라 미연결)'}))"><div class="muted small">${clip ? "사건 영상 인코딩/녹화 중 — 실시간 영상 표시" : "실시간 영상"}</div>`;
  const sig = `${tab}|${clip ? clip.status : "live"}`;
  return `<div class="vtabs">${all.map((c) => `<button data-vtab="${c}" class="${c === tab ? "active" : ""}">${c}${d.clips?.[c]?.status === "ready" ? " 🎞" : ""}</button>`).join("")}</div><div class="player" data-sig="${esc(sig)}">${media}</div>`;
}

function overviewHTML(d, cams) {
  const persons = Object.values(d.actors);
  return `
    <div class="sec"><h3>AI 사건 요약</h3><div class="summary">${esc(d.summary)}</div></div>
    <div class="sec"><h3>관련 영상</h3>${playerHTML(d, cams)}</div>
    <div class="sec"><h3>관련 인원 ${persons.length}</h3><div class="persons">${persons.map((a) => `
      <div><b>#${a.track_id}</b> <span class="chip">${a.camera}</span> ${esc({ actor: "행위자(추정)", target: "상대방", subject: "당사자" }[a.role] || "")}
        ${a.color ? `· 상의 ${esc(a.color)}` : ""} ${a.weapon ? `· <b style="color:#ff7b72">${esc(a.weapon)}</b>` : ""} ${a.direction ? `· ${esc(a.direction)}` : ""}
        ${a.linked_from ? `<span class="muted small">· ${esc(a.linked_from)} 동일인 후보 (${a.link_score})</span>` : ""}</div>`).join("")}</div></div>
    <div class="sec"><h3>주변 CCTV / 이동 경로</h3>${mapSVG(d)}<div class="rel">${d.related_cameras.map((r) => `
      <div><span class="st-${r.status}">●</span><b>${r.camera}</b> ${esc(r.name)} <span class="muted small">${esc(r.reason)}</span></div>`).join("")}</div></div>
    <div class="sec"><h3>타임라인</h3><ul class="timeline">${d.timeline.map((e) => `
      <li><span class="t">${hms(e.time)}</span><span class="c">${e.camera}</span><span>${e.level ? `<span class="badge" style="--lvl:${lvVar(e.level)}">${LV_KO[e.level] || e.level}</span> ` : ""}${esc(e.text)}</span></li>`).join("")}</ul></div>
    <div class="sec"><h3>처리 이력</h3><div class="small muted">${(d.audit || []).map((a) => `${hms(a.at)} ${esc(a.action)} ${esc(a.actor)}`).join("<br>") || "없음"}</div></div>`;
}

function mapSVG(d) {
  const cams = ST.site.cameras.filter((c) => c.lat != null && c.lon != null);
  if (cams.length < 2) return "";
  const W = 440, H = 150, pad = 40;
  const lats = cams.map((c) => c.lat), lons = cams.map((c) => c.lon);
  const [la0, la1, lo0, lo1] = [Math.min(...lats), Math.max(...lats), Math.min(...lons), Math.max(...lons)];
  const P = (c) => [pad + ((c.lon - lo0) / ((lo1 - lo0) || 1)) * (W - 2 * pad), H - pad - ((c.lat - la0) / ((la1 - la0) || 1)) * (H - 2 * pad) * 0.4 - 20];
  const pos = Object.fromEntries(cams.map((c) => [c.id, P(c)]));
  const rel = Object.fromEntries(d.related_cameras.map((r) => [r.camera, r.status]));
  let s = `<svg class="map" viewBox="0 0 ${W} ${H}">`;
  for (const c of cams) for (const n of c.neighbors) if (pos[n] && c.id < n) s += `<line x1="${pos[c.id][0]}" y1="${pos[c.id][1]}" x2="${pos[n][0]}" y2="${pos[n][1]}" stroke="#2a3442" stroke-width="6" stroke-linecap="round"/>`;
  const route = d.cameras;
  for (let i = 0; i + 1 < route.length; i++) { const a = pos[route[i]], b = pos[route[i + 1]]; if (a && b) s += `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="#f0883e" stroke-width="3" marker-end="url(#arr)"/>`; }
  const last = route[route.length - 1];
  for (const [cid, st] of Object.entries(rel)) if (st === "predicted" && pos[last] && pos[cid]) s += `<line x1="${pos[last][0]}" y1="${pos[last][1]}" x2="${pos[cid][0]}" y2="${pos[cid][1]}" stroke="#e3b341" stroke-dasharray="5 4" stroke-width="2" marker-end="url(#arr2)"/>`;
  s += `<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#f0883e"/></marker><marker id="arr2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#e3b341"/></marker></defs>`;
  for (const c of cams) {
    const [x, y] = pos[c.id];
    const col = c.id === d.primary_camera ? "#f85149" : d.cameras.includes(c.id) ? "#f0883e" : rel[c.id] === "predicted" ? "#e3b341" : "#8b98a8";
    s += `<circle cx="${x}" cy="${y}" r="8" fill="${col}"/><text x="${x}" y="${y + 24}" fill="#c9d1d9" font-size="11" text-anchor="middle">${c.id}</text>`;
  }
  return s + `<text x="8" y="14" fill="#8b98a8" font-size="10">● 발생 ● 이동 확인(후보) ● 이동 예상 — 카메라 위치(위경도) 기준 개략도</text></svg>`;
}

function evidenceHTML(sits) {
  return `<div class="sec"><h3>상황별 판단 근거 (최고점 시점)</h3>
    <p class="muted small">관문 = 이 상황이 성립하기 위한 필수 증거 · 가중 = 보강 증거(가중치) · 반대 = 점수를 낮추는 정상 정황.
      점수 = 관문 × (기본점 + (1−기본점) × noisy-OR(가중×강도)) × (1−반대) → 장소/시간 보정 → 정책 임계값으로 단계 산정.</p>
    ${sits.map((s) => `<div class="sit"><div class="h"><span>${esc(s.type_ko)} <span class="chip">${s.camera}</span> <span class="muted small">${s.actors.join(" → ")}</span></span>
      <span class="badge" style="--lvl:${lvVar(s.peak_level)}">${LV_KO[s.peak_level]} ${s.peak_score.toFixed(2)}</span></div>
      ${s.evidence.map((e) => { const neg = e.weight !== null && e.weight < 0; const kind = e.weight === null ? "관문" : neg ? "반대" : `가중 ${e.weight}`;
        return `<div class="ev"><span class="k ${e.weight === null ? "gate" : neg ? "neg" : ""}">${kind}</span><span>${esc(e.text)}</span><span class="bar ${neg ? "neg" : ""}"><i style="width:${Math.round(e.strength * 100)}%"></i></span></div>`; }).join("")}
    </div>`).join("")}</div>`;
}

// ------------------------------------------------------------------ report support
async function renderReport(d) {
  const body = $("#dbody");
  body.innerHTML = `<div class="sec muted">신고 정보 생성 중…</div>`;
  let r;
  try { r = await api(`/api/incidents/${d.id}/report`); } catch (e) { body.innerHTML = `<div class="sec notice">${esc(e.message)}</div>`; return; }
  const f = r.fields, op = r.operator, locked = r.status === "REPORTED";
  const list = (a) => (a || []).join(", ");
  body.innerHTML = `
  <div class="sec">
    <div class="notice">${esc(r.disclaimer)}</div>
    <p>상태: <span class="rstatus">${{ DRAFT: "초안(AI 자동 정리)", CONFIRMED: "확정 — 신고 준비 완료", REPORTED: "신고 완료 기록됨" }[r.status]}</span></p>
    <div class="form" id="rform">
      <label>발생시간<input class="ro" value="${esc(f.occurred_at)}" readonly></label>
      <label>위치<input id="fLoc" value="${esc(f.location.name)} / ${esc(f.location.address)} (${f.location.lat}, ${f.location.lon})" ${locked ? "readonly" : ""}></label>
      <label>사건 유형 / 위험도<input id="fType" value="${esc(f.incident_type)} / ${esc(f.risk_level)}" ${locked ? "readonly" : ""}></label>
      <label>위험물<input id="fObj" value="${esc(list(f.dangerous_objects))}" ${locked ? "readonly" : ""}></label>
      <label>관련 인원 (${f.persons.count}명)<textarea id="fPersons" ${locked ? "readonly" : ""}>${esc((f.persons.descriptions || []).join("\n"))}</textarea></label>
      <label>행동<input id="fBeh" value="${esc(list(f.behaviors))}" ${locked ? "readonly" : ""}></label>
      <label>이동방향<input id="fDir" value="${esc(list(f.movement_direction))}" ${locked ? "readonly" : ""}></label>
      <label>관련 CCTV<input class="ro" value="${esc(f.related_cctv.map((c) => c.camera + "(" + c.name + ")").join(", "))}" readonly></label>
      <label>사건 영상<span>${f.video.map((v) => `<a href="${esc(v.url)}" target="_blank">${v.camera} (${v.status})</a>`).join(" · ") || "녹화 중"}</span></label>
      <label>AI 분석 결과<textarea class="ro" readonly style="min-height:120px">${esc(f.ai_analysis.summary)}</textarea></label>
      <label>권장 신고/연계 기관<input id="fAgency" value="${esc(list(f.recommended_agency))}" ${locked ? "readonly" : ""}></label>
      <label>신고 문안 (전화 신고 시 참고)<textarea id="fScript" ${locked ? "readonly" : ""}>${esc(r.call_script)}</textarea></label>
      <hr style="border-color:#2a3442;width:100%">
      <label>확인자(관제요원)<input id="oName" value="${esc(op.name || operator())}" ${locked ? "readonly" : ""}></label>
      <label>메모<textarea id="oMemo" ${locked ? "readonly" : ""}>${esc(op.memo || "")}</textarea></label>
      <label style="display:flex;gap:6px;align-items:center;color:var(--text)"><input type="checkbox" id="oVerified" ${op.verified ? "checked" : ""} ${locked ? "disabled" : ""}> 영상으로 상황을 직접 확인했습니다 (확정 필수)</label>
      <div class="d-actions">
        <button class="btn" id="rSave" ${locked ? "disabled" : ""}>저장</button>
        <button class="btn primary" id="rConfirm" ${r.status !== "DRAFT" ? "disabled" : ""}>확정 (신고 준비 완료)</button>
        <a class="btn" href="/api/incidents/${d.id}/report.txt" target="_blank">텍스트 보기/인쇄</a>
        <button class="btn" id="rCopy">신고 문안 복사</button>
      </div>
      <div class="form" style="border-top:1px solid #2a3442;padding-top:8px">
        <span class="muted small">관제자가 직접 신고(전화/시스템)한 뒤 기록합니다. 본 시스템은 자동 신고하지 않습니다.</span>
        <label>신고 기관<select id="rAgency" ${r.status !== "CONFIRMED" ? "disabled" : ""}>${["112", "119", "112+119", "시설관리자", "기타"].map((a) => `<option ${op.agency === a ? "selected" : ""}>${a}</option>`).join("")}</select></label>
        <label>접수번호(선택)<input id="rReceipt" value="${esc(op.receipt_no || "")}" ${r.status !== "CONFIRMED" ? "disabled" : ""}></label>
        <button class="btn good" id="rReported" ${r.status !== "CONFIRMED" ? "disabled" : ""}>신고 완료 기록</button>
        ${op.reported_at ? `<span class="small">신고 기록: ${hms(op.reported_at)} · ${esc(op.agency)} ${esc(op.receipt_no)}</span>` : ""}
      </div>
    </div>
  </div>`;
  const payload = () => ({
    fields: { dangerous_objects: $("#fObj").value.split(",").map((s) => s.trim()).filter(Boolean),
      persons: { count: f.persons.count, descriptions: $("#fPersons").value.split("\n").filter(Boolean) },
      behaviors: $("#fBeh").value.split(",").map((s) => s.trim()).filter(Boolean),
      movement_direction: $("#fDir").value.split(",").map((s) => s.trim()).filter(Boolean),
      recommended_agency: $("#fAgency").value.split(",").map((s) => s.trim()).filter(Boolean),
      location_note: $("#fLoc").value, type_note: $("#fType").value },
    call_script: $("#fScript").value,
    operator: { name: $("#oName").value, memo: $("#oMemo").value, verified: $("#oVerified").checked },
  });
  const save = async () => { localStorage.setItem("sw_operator", $("#oName").value); return api(`/api/incidents/${d.id}/report`, { method: "PUT", body: payload() }); };
  if (!locked) $("#rSave").onclick = async () => { await save(); toastText("저장했습니다"); };
  $("#rConfirm").onclick = async () => {
    try { await save(); await api(`/api/incidents/${d.id}/report/confirm`, { method: "POST", body: { name: $("#oName").value, verified: $("#oVerified").checked } });
      toastText("신고 정보가 확정되었습니다. 관제자가 직접 신고하세요."); loadDetail(d.id); } catch (e) { alert(e.message); }
  };
  $("#rCopy").onclick = () => navigator.clipboard?.writeText($("#fScript").value).then(() => toastText("복사했습니다"));
  $("#rReported").onclick = async () => {
    try { await api(`/api/incidents/${d.id}/report/reported`, { method: "POST", body: { agency: $("#rAgency").value, receipt_no: $("#rReceipt").value } });
      toastText("신고 완료가 기록되었습니다"); loadDetail(d.id); } catch (e) { alert(e.message); }
  };
}

function toastText(t) {
  const el = document.createElement("div"); el.className = "toast"; el.style.setProperty("--lvl", "var(--accent)"); el.textContent = t;
  $("#toasts").prepend(el); setTimeout(() => el.remove(), 3000);
}

// ------------------------------------------------------------------ dialogs
async function openRunDialog() {
  const list = await api("/api/scenarios");
  $("#scnList").innerHTML = list.map((s) => `
    <div class="scn"><div><b>${esc(s.title)}</b><div class="small muted">${s.name} · ${s.duration.toFixed(0)}초 · ${s.cameras.join(", ")}
      ${s.hard_negative ? '<span class="neg">· 정상-유사(오탐 시험)</span>' : `· 정답: ${s.events.join(", ")}`}</div></div>
      <button class="btn primary" data-scn="${s.name}">재생</button></div>`).join("");
  $("#scnList").querySelectorAll("[data-scn]").forEach((b) => (b.onclick = async (e) => {
    e.preventDefault();
    await api("/api/jobs", { method: "POST", body: { kind: "scenario", name: b.dataset.scn, speed: +$("#scnSpeed").value } });
    $("#dlgRun").close(); pollStatus();
  }));
  $("#dlgRun").showModal();
}

function openSourceDialog() {
  $("#srcCam").innerHTML = ST.site.cameras.map((c) => `<option value="${c.id}">${c.id} ${esc(c.name)}</option>`).join("");
  $("#srcStart").onclick = async (e) => {
    e.preventDefault();
    try { await api("/api/jobs", { method: "POST", body: { kind: "source", camera_id: $("#srcCam").value, source: $("#srcPath").value, analysis_fps: +$("#srcFps").value } });
      $("#dlgSource").close(); pollStatus(); } catch (err) { alert(err.message); }
  };
  $("#dlgSource").showModal();
}

// ------------------------------------------------------------------ KPI & eval
async function renderKPI() {
  const k = await api("/api/kpi");
  const L = { detect_latency_s: "탐지 지연", alert_latency_s: "알림 지연", recognition_s: "사건 인지 시간", understanding_s: "상황 파악 시간", report_prep_s: "신고 준비 시간", total_response_s: "전체 대응 시간" };
  $("#kpiSummary").innerHTML = Object.entries(L).map(([key, l]) => `<div class="kcard"><div class="v">${k.summary[key].mean ?? "-"}<span class="small muted"> s</span></div><div class="l">${l} (n=${k.summary[key].n})</div></div>`).join("")
    + `<div class="kcard"><div class="v">${k.feedback.operator_precision ?? "-"}</div><div class="l">운영자 정밀도 (정탐 ${k.feedback.TP} / 오탐 ${k.feedback.FP})</div></div>`;
  const cols = ["id", "type", "peak_level", "review", "feedback", ...Object.keys(L), "report_status"];
  $("#kpiTable").innerHTML = `<tr>${cols.map((c) => `<th>${L[c] || c}</th>`).join("")}</tr>` + k.rows.map((r) => `<tr>${cols.map((c) => `<td>${esc(r[c] ?? "-")}</td>`).join("")}</tr>`).join("");
}

async function renderEval() {
  const r = await api("/api/eval/latest");
  const el = $("#evalPanel");
  if (!r.available) { el.innerHTML = `<div class="panel-title">AI 성능 평가</div><p style="padding:12px">${esc(r.hint)}<br><code>safewatch eval --seeds 1 2 3</code></p>`; return; }
  const ops = Object.entries(r.operating_points);
  el.innerHTML = `<div class="panel-title">AI 성능 평가 — 사건 단위 (${esc(r.generated_at)})</div>
  <div class="eval-grid"><div>
    <p class="small">데이터: ${r.data.kind} · 시나리오 ${r.data.scenarios.length}종 × seed ${r.data.seeds.length} = ${r.data.runs}회 · GT ${r.data.gt_events}건 · 노이즈 ${r.data.noise} · 정책 ${esc(r.policy.provenance?.source)}</p>
    <table class="tbl"><tr><th>단계</th><th>θ</th><th>Precision</th><th>Recall</th><th>F1</th><th>FNR</th><th>FP</th><th>FP/cam-h</th><th>평균 지연</th></tr>
    ${ops.map(([lv, o]) => `<tr><td><span class="badge" style="--lvl:${lvVar(lv)}">${LV_KO[lv]}</span></td><td>${o.threshold}</td><td>${o.strict.precision}</td><td>${o.strict.recall}</td><td>${o.strict.f1}</td><td>${o.strict.fnr}</td><td>${o.strict.fp}</td><td>${o.strict.fp_per_cam_hour}</td><td>${o.strict.latency_mean_s ?? "-"}s</td></tr>`).join("")}</table>
    <h4>유형별 (WARNING 기준)</h4>
    <table class="tbl"><tr><th>유형</th><th>GT</th><th>P</th><th>R</th><th>F1</th><th>FP</th><th>평균 지연</th></tr>
    ${Object.entries(r.per_type_at_warning).map(([t, s]) => `<tr><td>${TYPE_KO[t] || t}</td><td>${s.n_gt}</td><td>${s.precision}</td><td>${s.recall}</td><td>${s.f1}</td><td>${s.fp}</td><td>${s.latency_mean_s ?? "-"}s</td></tr>`).join("")}</table>
    <h4>정상-유사 상황 최고 점수</h4><div class="small">${Object.entries(r.hard_negative_max_score).map(([k, v]) => `${k}: ${v.join(", ")}`).join("<br>")}</div>
    ${r.caveats.map((c) => `<p class="caveat">⚠ ${esc(c)}</p>`).join("")}
  </div><div><img src="data:image/svg+xml;base64,${btoa(unescape(encodeURIComponent(prSVG(r))))}"><p class="small muted">PR 곡선 (θ 0.05~0.99), 점: 현재 정책의 단계 경계. AP strict ${r.ap.strict} / lenient ${r.ap.lenient}</p></div></div>`;
}

function prSVG(r) {
  const W = 420, H = 320, p = 40, X = (v) => p + v * (W - p - 10), Y = (v) => H - p - v * (H - p - 10);
  const pts = [...r.pr_curve].sort((a, b) => a.theta - b.theta).map((c) => `${X(c.recall)},${Y(c.precision)}`).join(" ");
  let s = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" font-family="sans-serif"><rect width="100%" height="100%" fill="white"/>`;
  for (const v of [0, .2, .4, .6, .8, 1]) s += `<line x1="${X(v)}" y1="${Y(0)}" x2="${X(v)}" y2="${Y(1)}" stroke="#ddd"/><line x1="${X(0)}" y1="${Y(v)}" x2="${X(1)}" y2="${Y(v)}" stroke="#ddd"/><text x="${X(v) - 6}" y="${H - p + 14}" font-size="9">${v}</text><text x="8" y="${Y(v) + 3}" font-size="9">${v}</text>`;
  s += `<polyline points="${pts}" fill="none" stroke="#2563eb" stroke-width="2"/>`;
  Object.entries(r.operating_points).forEach(([lv, o], i) => { s += `<circle cx="${X(o.strict.recall)}" cy="${Y(o.strict.precision)}" r="4" fill="#d33"/><text x="${Math.min(X(o.strict.recall) - 70, W - 90)}" y="${Y(o.strict.precision) + 16 + i * 12}" font-size="10" fill="#b91c1c">${lv} θ=${o.threshold}</text>`; });
  return s + `<text x="${W / 2 - 20}" y="${H - 6}" font-size="11">Recall</text><text x="2" y="14" font-size="11">Precision</text></svg>`;
}

boot();
