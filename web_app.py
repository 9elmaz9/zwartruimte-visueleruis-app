# web_app.py — Frontend zonder autoscan, met spinner en % voortgang (pseudo), PRG, verwijderen en CSV
# Start:: python web_app.py → http://127.0.0.1:5009

import os
from datetime import datetime
from flask import Flask, request, redirect, url_for, render_template_string, flash, session
from werkzeug.utils import secure_filename

# --- veilige import analyzer_core ---
import sys, importlib
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try: 
    core = importlib.import_module("analyzer_core")
except Exception as e:
    raise RuntimeError("Het is niet gelukt om analyzer_core.py te importeren naast web_app.py") from e

REQUIRED = [
    "get_video_duration_seconds","detect_black_segments","detect_glitches",
    "detect_freezes","detect_1khz_tone","detect_ruis_gray_stripes",
    "to_hms","hms_to_seconds","merge_intervals",
]
missing = [n for n in REQUIRED if not hasattr(core, n)]
if missing:
    raise RuntimeError(f"В analyzer_core.py отсутствует: {missing}")
# --- Einde van de import ---

ALLOWED_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR)
app.secret_key = "elmaz"  # mijn

PAGE = r"""
<!doctype html>
<html lang="nl">
<head>
  <meta charset="utf-8">
  <title>Zwartruimte – Upload & Analyse</title>
  <link rel="icon" type="image/x-icon" href="https://meemoo.be/favicon.ico">
  <link rel="shortcut icon" href="https://meemoo.be/favicon.ico">
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
           padding: 24px; background:#0b0c10; color:#eaecef; }
    .wrap { max-width: 1100px; margin: 0 auto; }
    .card { background:#111317; border:1px solid #1f2328; border-radius:14px; padding:16px; margin-bottom:16px; }
    .muted { color:#9aa4b2; }
    .row { display:flex; gap:16px; flex-wrap: wrap; align-items:center; }
    .kv { background:#0f1115; border:1px solid #23262d; border-radius:10px; padding:8px 10px; }
    .badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; border:1px solid #2d333b; background:#0f1115; }
    table { width:100%; border-collapse: collapse; margin-top:10px; font-size: 14px;}
    th, td { border-bottom:1px solid #23262d; padding:6px 6px; text-align:left; }
    th { color:#b7c1ce; position: sticky; top:0; background:#111317; }
    details summary { cursor:pointer; margin-top:8px; }
    code { background:#0f1115; padding:2px 6px; border-radius:6px; }

    /* Knoppen  */
    .btn {
      display:inline-block; padding:10px 20px; border-radius:30px;
      background:#00c9a7; color:white; text-decoration:none; border:none;
      cursor:pointer; font-weight:600;
    }
    .btn:hover { background:#00b394; }
    .btn:disabled { opacity:.6; cursor:not-allowed; }
    .btn.secondary { background:#2d333b; }
    .btn.secondary:hover { background:#3a414a; }
    .btn.danger { background:#b94b55; }
    .btn.danger:hover { background:#a23f48; }

    /* Drag & drop */
    input[type=file] { display:none; }
    .drop { border:2px dashed #2d333b; border-radius:14px; padding:18px; text-align:center; background:#0f1115; }
    .drop.drag { border-color:#00c9a7; background:#0f1220; }

    /* Lijst van geselecteerde bestanden */
    .picked-item { display:flex; align-items:center; justify-content:space-between;
      gap:10px; padding:6px 10px; border:1px solid #23262d; background:#0f1115;
      border-radius:10px; margin-top:6px; font-size:14px; }
    .picked-remove { background:#b94b55; border:none; color:#fff; border-radius:8px;
      padding:4px 10px; cursor:pointer; font-weight:600; }
    .picked-remove:hover { background:#a23f48; }

    /* Bovenste banner */
    .hero {
  width: 70%;
  height: auto;
  display: block;
  margin-left: 0;       /* to the left side  */
  margin-right: auto;   /* without auticentreren */
  margin-top: 10px;
  margin-bottom: 20px;
  border: none;
  background: transparent;
}


    .topbar { display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; gap:12px; }
    .top-actions { display:flex; gap:8px; flex-wrap:wrap; }

/* Volledig scherm overlay met spinner en % */
    #overlay {
      position: fixed; inset: 0; background: rgba(0,0,0,.55);
      display: none; align-items: center; justify-content: center; z-index: 9999;
      backdrop-filter: blur(1px);
    }
    .spinner {
      width: 74px; height: 74px; border-radius: 50%;
      border: 6px solid rgba(255,255,255,0.25);
      border-top-color: #00c9a7;
      animation: spin 1s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .ovl-box {
      display:flex; flex-direction:column; gap:12px; align-items:center; padding:22px 26px;
      background:#111317; border:1px solid #1f2328; border-radius:16px; color:#eaecef;
      box-shadow: 0 10px 40px rgba(0,0,0,.35);
      min-width: 300px;
    }
    .ovl-text { color:#c7d0db; font-size:14px; }

/* +++ procenten + voortgangsbalk +++ */
    .ovl-pct { font-weight:700; font-size:16px; text-align:center; }
    .pbar { width:260px; height:10px; background:#0f1115; border:1px solid #23262d; border-radius:999px; overflow:hidden; }
    .pbar > div { height:100%; width:0%; background:#00c9a7; transition: width .25s ease; }
  </style>
</head>
<body>
<div id="overlay">
  <div class="ovl-box">
    <div class="spinner"></div>
    <div class="ovl-text">Analyseren… even geduld aub</div>
    <div class="pbar"><div id="ovlbar"></div></div>
    <div class="ovl-pct" id="ovlpct">0%</div>
  </div>
</div>

<div class="wrap">

<!-- Banner ( de afbeelding in static/brand_header.png) -->
<img class="hero" src="{{ url_for('static', filename='brand_header.svg') }}" alt="banner">

  <div class="topbar">
    <div class="muted"> Laad de bestanden hieronder op. Op de hoofdpagina wordt er niets automatisch geanalyseerd.</div>
    {% if results %}
    <div class="top-actions">
      <button class="btn secondary" type="button" onclick="downloadSummaryPage()">Download Summary CSV</button>
      <button class="btn secondary" type="button" onclick="downloadEventsPage()">Download Events CSV</button>
    </div>
    {% endif %}
  </div>
  <br>

  <div class="card">
    <h2 style="margin:0 0 10px 0;">Zwartruimte – upload & analyse</h2>
    <br>
    <form id="form" method="post" enctype="multipart/form-data" action="{{ url_for('analyze') }}">
      <div id="drop" class="drop">
        <div class="muted">Sleep & zet hier je video’s neer<br>of</div>
        <label class="btn secondary" for="file">Kies bestanden</label>
        <input id="file" type="file" name="videos" multiple accept=".mp4,.mov,.mkv,.avi,.m4v">
        <div id="picked" class="muted" style="margin-top:8px;">Geen bestanden geselecteerd</div>
        <ul id="pickedList" style="list-style:none; padding:0; margin:10px 0 0 0;"></ul>
      </div>
      <div class="row" style="margin-top:12px;">
        <button id="go" class="btn" type="submit">Analyze</button>
        <span id="busy" class="muted" style="display:none;">Analyseren…</span>
      </div>
    </form>
  </div>

  {% with messages = get_flashed_messages() %}
    {% if messages %}
      <div class="card">
        {% for m in messages %}<div>{{ m }}</div>{% endfor %}
      </div>
    {% endif %}
  {% endwith %}

  {% if results %}
    {% for item in results %}
    <div class="card" data-file="{{ item.filename }}">
      <div class="row" style="justify-content:space-between;">
        <div>
          <div style="font-weight:600; font-size:16px;">{{ item.filename }}</div>
          <div class="muted" style="font-size:13px;">Uploaded: {{ item.uploaded_at }}</div>
        </div>
        <div class="row">
          <span class="kv"><span class="muted">Video</span> <code class="video_hms">{{ item.video_hms }}</code></span>
          <span class="kv"><span class="muted">Events</span> <code class="events_cnt">{{ item.errors_count }}</code></span>
          <span class="kv"><span class="muted">Beschadiging</span> <code class="damage_pct">{{ "%.2f"|format(item.damage_percent) }}%</code></span>
        </div>
      </div>

      <div class="row" style="margin-top:10px;">
        <!-- короткая строка про длительности -->
        <span class="badge">Video: <b>{{ item.video_hms }}</b> — Beschadigd: <b>{{ item.covered_hms }}</b></span>

        <span class="row" style="gap:8px; margin-left:auto;">
          <button class="btn secondary" onclick="copySummary(this)" type="button">Copy summary</button>
          <button class="btn secondary" onclick="downloadCSV(this)" type="button">Download CSV</button>
          <form method="post" action="{{ url_for('delete') }}" style="display:inline;">
            <input type="hidden" name="filename" value="{{ item.filename }}">
            <button class="btn danger" type="submit" onclick="return confirm('Verwijderen?')">Verwijderen</button>
          </form>
        </span>
      </div>

      <details style="margin-top:12px;">
        <summary>Toon gebeurtenissen ({{ item.errors_count }})</summary>

         <div class="filters">
          <label><input type="checkbox" class="flt" data-type="BLACK" checked>BLACK</label>
          <label><input type="checkbox" class="flt" data-type="FREEZE" checked>FREEZE</label>
          <label><input type="checkbox" class="flt" data-type="GLITCH" checked>GLITCH</label>
          <label><input type="checkbox" class="flt" data-type="RUIS/STRIPES" checked>RUIS/STRIPES</label>
          <label><input type="checkbox" class="flt" data-type="1KHZ_TONE" checked>1KHZ_TONE</label>
        </div>

        {% if item.events %}
        <table class="events">
          <thead>
            <tr>
              <th>Type</th>
              <th>Start</th>
              <th>Einde</th>
              <th>Duur (sec)</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            {% for ev in item.events %}
            <tr data-type="{{ ev.type }}">
              <td>{{ ev.type }}</td>
              <td>{{ ev.start }}</td>
              <td>{{ ev.end }}</td>
              <td>{{ "%.2f"|format(ev.duration) }}</td>
              <td>{{ ev.details }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
        {% else %}
          <p class="muted">Geen gebeurtenissen gevonden.</p>
        {% endif %}
      </details>
    </div>
    {% endfor %}
  {% endif %}
</div>

<script>
  // Drag & drop + счётчик + удаление перед отправкой
  const drop = document.getElementById('drop');
  const file = document.getElementById('file');
  const picked = document.getElementById('picked');
  const pickedList = document.getElementById('pickedList');
  const form = document.getElementById('form');
  const goBtn = document.getElementById('go');
  const busy = document.getElementById('busy');
  const overlay = document.getElementById('overlay');

  // overlay progress (псевдо)
  const ovlbar = document.getElementById('ovlbar');
  const ovlpct = document.getElementById('ovlpct');
  let progTimer = null;
  let progValue = 0;

  // Хранилище выбранных файлов
  let dt = new DataTransfer();
  const allowed = new Set(['.mp4','.mov','.mkv','.avi','.m4v']);

  function renderPicked() {
    if (dt.files.length === 0) {
      picked.textContent = 'Geen bestanden geselecteerd';
    } else {
      const names = Array.from(dt.files).map(f => f.name);
      picked.textContent = `${names.length} geselecteerd: ` + names.slice(0,3).join(', ') + (names.length>3 ? '…' : '');
    }
    pickedList.innerHTML = '';
    Array.from(dt.files).forEach((f, idx) => {
      const li = document.createElement('li');
      li.className = 'picked-item';
      li.innerHTML = `
        <span title="${f.name}">${f.name}</span>
        <button type="button" class="picked-remove" data-idx="${idx}">Verwijderen</button>
      `;
      pickedList.appendChild(li);
    });
  }
  function syncInputFromDT() { file.files = dt.files; renderPicked(); }
  function addFiles(fileList) {
    for (const f of Array.from(fileList)) {
      const ext = '.' + (f.name.split('.').pop() || '').toLowerCase();
      if (!allowed.has(ext)) continue;
      const dup = Array.from(dt.files).some(x => x.name === f.name && x.size === f.size && x.lastModified === f.lastModified);
      if (!dup) dt.items.add(f);
    }
    syncInputFromDT();
  }
  pickedList.addEventListener('click', (e)=>{
    const btn = e.target.closest('.picked-remove'); if (!btn) return;
    const idx = parseInt(btn.dataset.idx, 10);
    const nextDT = new DataTransfer();
    Array.from(dt.files).forEach((f, i)=>{ if (i!==idx) nextDT.items.add(f); });
    dt = nextDT; syncInputFromDT();
  });
  file.addEventListener('change', ()=>{ dt = new DataTransfer(); addFiles(file.files); });
  drop.addEventListener('dragover', (e)=>{ e.preventDefault(); drop.classList.add('drag'); });
  drop.addEventListener('dragleave', ()=> drop.classList.remove('drag'));
  drop.addEventListener('drop', (e)=>{
    e.preventDefault(); drop.classList.remove('drag');
    if (e.dataTransfer.files && e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
  });
  renderPicked();

  // Показать/скрыть спиннер + псевдо-прогресс вокруг запроса
  form.addEventListener('submit', ()=>{
    file.files = dt.files;            // синхронизируем перед отправкой
    goBtn.disabled = true;
    if (busy) busy.style.display = 'inline';
    if (overlay) overlay.style.display = 'flex';

    // старт псевдо-прогресса: растём до 95%
    progValue = 0;
    if (ovlbar) ovlbar.style.width = '0%';
    if (ovlpct) ovlpct.textContent = '0%';
    if (progTimer) clearInterval(progTimer);
    progTimer = setInterval(()=>{
      const delta = Math.max(0.2, (95 - progValue) * 0.03); // ускорение в начале, замедление к концу
      progValue = Math.min(95, progValue + delta);
      if (ovlbar) ovlbar.style.width = progValue.toFixed(1) + '%';
      if (ovlpct) ovlpct.textContent = Math.floor(progValue) + '%';
    }, 250);
  });

  window.addEventListener('pageshow', ()=>{ // после возврата со страницы результатов
    if (progTimer) { clearInterval(progTimer); progTimer = null; }
    // добиваем до 100% и скрываем
    if (ovlbar) ovlbar.style.width = '100%';
    if (ovlpct) ovlpct.textContent = '100%';
    setTimeout(()=>{ if (overlay) overlay.style.display = 'none'; }, 150);
    goBtn.disabled = false;
    if (busy) busy.style.display = 'none';
  });

// Filteren van rijen op type gebeurtenis
  document.querySelectorAll('.card .filters').forEach(fltBlock=>{
    fltBlock.addEventListener('change', (e)=>{
      const card = e.currentTarget.closest('.card');
      const checks = card.querySelectorAll('.flt');
      const active = new Set(Array.from(checks).filter(c=>c.checked).map(c=>c.dataset.type));
      const rows = card.querySelectorAll('table.events tbody tr');
      rows.forEach(tr=>{
        const t = tr.getAttribute('data-type');
        tr.style.display = active.has(t) ? '' : 'none';
      });
    });
  });

  

  // CSV from tabel(per video)
  function tableToCSV(table) {
    const rows = Array.from(table.querySelectorAll('tr'));
    const cellsToText = cells => cells.map(td => {
      const v = td.textContent.replace(/\r?\n/g, ' ').trim();
      return /[",;\t]/.test(v) ? `"${v.replace(/"/g,'""')}"` : v;
    }).join(',');
    return rows.map(tr => cellsToText(Array.from(tr.children))).join('\n');
  }
  window.downloadCSV = function(btn) {
    const card = btn.closest('.card');
    const table = card.querySelector('table.events');
    if (!table) { alert('Geen data tabel'); return; }
    const csv = tableToCSV(table);
    const fn = (card.getAttribute('data-file') || 'events') + '_events.csv';
    const blob = new Blob([csv], {type: 'text/csv;charset=utf-8;'});
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = fn;
    document.body.appendChild(a); a.click(); a.remove();
  }

  // Copy summary
  window.copySummary = function(btn) {
    const card = btn.closest('.card');
    const fileName = card.getAttribute('data-file') || 'video';
    const video = card.querySelector('.video_hms').textContent.trim();
    const eventsNum = card.querySelector('.events_cnt').textContent.trim();
    const perc = card.querySelector('.damage_pct').textContent.trim();
    // берём текст справа "Video: ... — Beschadigd: ..."
    const badge = card.querySelector('.badge').textContent;
    const covered = (badge.split('Beschadigd:').pop() || '').trim();
    const line = `📊 ${fileName} — Events ${eventsNum}; Video ${video}; Beschadigd ${covered}; Beschadiging: ${perc}`;
    navigator.clipboard.writeText(line).then(()=> {
      btn.textContent = 'Copied!'; setTimeout(()=> btn.textContent = 'Copy summary', 1200);
    });
  }

  // Summary/Events CSV пvoor alle kaarten op de pagina
  window.downloadSummaryPage = function() {
    const cards = Array.from(document.querySelectorAll('.card[data-file]'));
    const rows = [["video_file","video_duration_hms","errors_count","covered_defects_hms","damage_percent"]];
    cards.forEach(card=>{
      const file = card.getAttribute('data-file');
      const video = card.querySelector('.video_hms').textContent.trim();
      const eventsNum = card.querySelector('.events_cnt').textContent.trim();
      const badge = card.querySelector('.badge').textContent;
      const covered = (badge.split('Beschadigd:').pop() || '').trim();
      const perc = card.querySelector('.damage_pct').textContent.trim();
      rows.push([file, video, eventsNum, covered, perc]);
    });
    const csv = rows.map(r => r.map(v => /[",;\t]/.test(v) ? `"${v.replace(/"/g,'""')}"` : v).join(',')).join('\n');
    const blob = new Blob([csv], {type: 'text/csv;charset=utf-8;'});
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'summary_page.csv';
    document.body.appendChild(a); a.click(); a.remove();
  }
  window.downloadEventsPage = function() {
    const rows = [["video_file","type","start_time","end_time","duration_sec","details"]];
    document.querySelectorAll('.card[data-file]').forEach(card=>{
      const file = card.getAttribute('data-file');
      card.querySelectorAll('table.events tbody tr').forEach(tr=>{
        const tds = tr.querySelectorAll('td');
        rows.push([file,
          tds[0].textContent.trim(),
          tds[1].textContent.trim(),
          tds[2].textContent.trim(),
          tds[3].textContent.trim(),
          tds[4].textContent.trim()]);
      });
    });
    const csv = rows.map(r => r.map(v => /[",;\t]/.test(v) ? `"${v.replace(/"/g,'""')}"` : v).join(',')).join('\n');
    const blob = new Blob([csv], {type: 'text/csv;charset=utf-8;'});
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'events_page.csv';
    document.body.appendChild(a); a.click(); a.remove();
  }
</script>
</body>
</html>
"""

def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT

def analyze_one(filepath: str):
    """Start detect_* en bereidt data voor de frontend (alleen voor de aangeleverde bestanden)."""
    video_duration = core.get_video_duration_seconds(filepath)

    all_results = []
    all_results += core.detect_black_segments(filepath)
    all_results += core.detect_glitches(filepath)
    all_results += core.detect_freezes(filepath)
    all_results += core.detect_1khz_tone(filepath)
    all_results += core.detect_ruis_gray_stripes(filepath)

    total_defect_sec = float(sum(float(r["duration"]) for r in all_results))
    total_hms = core.to_hms(total_defect_sec)

    intervals = [(core.hms_to_seconds(r["start"]), core.hms_to_seconds(r["end"])) for r in all_results]
    merged = core.merge_intervals(intervals)
    covered_sec = float(sum(e - s for s, e in merged))
    covered_hms = core.to_hms(covered_sec)

    damage_percent = (covered_sec / video_duration * 100.0) if video_duration > 0 else 0.0
    video_hms = core.to_hms(video_duration) if video_duration > 0 else "00:00:00"

    events = [{
        "type": r["type"],
        "start": r["start"],
        "end": r["end"],
        "duration": float(r["duration"]),
        "details": r.get("details", "")
    } for r in all_results]

    return {
        "video_duration": float(video_duration),
        "video_hms": video_hms,
        "total_sec": int(round(total_defect_sec)),
        "total_hms": total_hms,
        "covered_sec": int(round(covered_sec)),
        "covered_hms": covered_hms,
        "damage_percent": float(damage_percent),
        "events": events,
        "errors_count": len(events),
    }

@app.get("/")
def index():
    # without nothig, do nothing
    return render_template_string(PAGE, results=None)

@app.get("/favicon.ico")
def favicon():
    return redirect("https://meemoo.be/favicon.ico", code=302)


@app.get("/result")
def result():
# Resultaten uit de sessie weergeven (PRG)
    results = session.get('last_results')
    return render_template_string(PAGE, results=results)

@app.post("/analyze")
def analyze():
    if "videos" not in request.files:
        flash("Geen bestanden ontvangen.")
        return redirect(url_for("index"))

    files = request.files.getlist("videos")
    results = []
    any_saved = False

    for f in files:
        if not f or f.filename == "":
            continue
        if not allowed_file(f.filename):
            flash(f"Overgeslagen: {f.filename} (niet-untersteunde extensie)")
            continue

        fname = secure_filename(f.filename)
        save_path = os.path.join(UPLOAD_DIR, fname)
        f.save(save_path)
        any_saved = True

# Alleen het zojuist geüploade bestand analyseren
        res = analyze_one(save_path)
        res.update({
            "filename": fname,
            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        results.append(res)

    if not any_saved:
        flash("Niets geüpload.")
        return redirect(url_for("index"))

# Resultaten in de sessie opslaan en PRG uitvoeren → /result
    session['last_results'] = results
    return redirect(url_for("result"))

@app.post("/delete")
def delete():
# Bestand verwijderen uit uploads/ en uit de huidige resultaten in de sessie
    fname = request.form.get("filename", "")
    if not fname:
        flash("Geen bestandsnaam.")
        return redirect(url_for("result"))
    path = os.path.join(UPLOAD_DIR, os.path.basename(fname))
    if os.path.isfile(path) and allowed_file(path):
        os.remove(path)
        flash(f"Verwijderd: {fname}")
    else:
        flash("Bestand niet gevonden.")

# Resultaten in de sessie bijwerken (kaart verwijderen)
    if 'last_results' in session and isinstance(session['last_results'], list):
        session['last_results'] = [r for r in session['last_results'] if r.get('filename') != fname]

    return redirect(url_for("result"))

if __name__ == "__main__":
    app.run(debug=True, port=5009, threaded=True)
