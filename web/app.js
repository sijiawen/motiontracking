(() => {
  "use strict";

  const CFG = Object.assign(
    { APP_NAME: "Creatorgen", API_URL: "", SUPABASE_URL: "", SUPABASE_ANON_KEY: "", MAX_SECONDS: 60, MAX_MB: 50, DAILY_LIMIT: 5 },
    window.APP_CONFIG || {}
  );
  const params = new URLSearchParams(location.search);
  const DEMO = !CFG.API_URL || params.has("demo");
  const API = CFG.API_URL.replace(/\/$/, "");
  const MODELS = [
    { key: "pose", label: "Pose estimation", short: "Pose",
      blurb: "Body and hand keypoints (308-point skeleton shown; face landmarks are not scored). Compared after removing position and scale, so a cropped or zoomed copy still lines up." },
    { key: "seg", label: "Body-part segmentation", short: "Body parts",
      blurb: "Every pixel labelled as one of 28 body parts or clothing types. Scored by how much each part overlaps between matched frames." },
    { key: "normal", label: "Surface normals", short: "Normals",
      blurb: "The direction each patch of the body faces, coloured by orientation. Captures fine 3D surface detail like folds and contours." },
    { key: "pointmap", label: "Pointmap (3D depth)", short: "Pointmap",
      blurb: "Estimated distance of every point on the person from the camera, brighter is closer. Captures overall 3D body shape and posture." },
    { key: "matting", label: "Human matting", short: "Matting",
      blurb: "The person cut out from the background, including hair and loose clothing. Scored by silhouette overlap." },
  ];

  const TESTS = [
    { key: "body", label: "Body & motion", sub: "Pose, body shape and silhouette over time (5 Sapiens2 models)" },
    { key: "scene", label: "Scene & visuals", sub: "Room, objects, framing and lighting, with and without the person" },
    { key: "script", label: "Script", sub: "What was said, transcribed and compared word for word" },
  ];
  const PANELS = {
    scene: { label: "Scene & visuals", short: "Scene",
      blurb: "Each frame is turned into a visual fingerprint twice: once whole, and once with the person blanked out. A face or body swapped onto the same footage still matches on the room, objects, framing and lighting." },
    script: { label: "Script", short: "Script",
      blurb: "Both videos' speech is transcribed. Highlighted text is phrases of 4+ words that appear in both. The score is the share of the shorter script that was reused." },
  };

  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const state = { files: { a: null, b: null }, ok: { a: false, b: false }, pollTimer: null, current: null, tab: "pose" };

  // ------------------------------------------------------------ setup
  $$("[data-app-name]").forEach((n) => (n.textContent = CFG.APP_NAME));
  document.title = `${CFG.APP_NAME} · Footage provenance`;
  $$("[data-max-seconds]").forEach((n) => (n.textContent = CFG.MAX_SECONDS));
  $$("[data-max-mb]").forEach((n) => (n.textContent = CFG.MAX_MB));
  $("#year").textContent = new Date().getFullYear();
  $("#quota span").textContent = DEMO ? "Demo mode · sample results" : `${CFG.DAILY_LIMIT} free comparisons per day`;

  const fmtBytes = (b) => (b / 1048576).toFixed(1) + " MB";
  const fmtTime = (s) => {
    if (s == null || isNaN(s)) return "–";
    const m = Math.floor(s / 60), r = s - m * 60;
    return `${m}:${r < 10 ? "0" : ""}${r.toFixed(1).replace(/\.0$/, "")}`;
  };
  const pct = (x) => `${Math.round((x || 0) * 100)}%`;
  const OK_EXT = /\.(mp4|mov|webm|m4v|mkv|avi)$/i;

  // ------------------------------------------------------------ drop zones
  $$(".drop").forEach((drop) => {
    const side = drop.dataset.side;
    const input = $("input", drop);
    input.addEventListener("change", () => input.files[0] && setFile(side, input.files[0]));
    ["dragenter", "dragover"].forEach((e) =>
      drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("dragover"); })
    );
    ["dragleave", "drop"].forEach((e) => drop.addEventListener(e, () => drop.classList.remove("dragover")));
    drop.addEventListener("drop", (ev) => {
      ev.preventDefault();
      const f = ev.dataTransfer.files[0];
      if (f) setFile(side, f);
    });
    $(".file-clear", drop).addEventListener("click", (ev) => {
      ev.preventDefault(); ev.stopPropagation();
      clearFile(side);
    });
  });

  function clearFile(side) {
    const drop = $(`.drop[data-side="${side}"]`);
    const v = $("video", drop);
    if (v.src) URL.revokeObjectURL(v.src);
    v.removeAttribute("src");
    $("input", drop).value = "";
    state.files[side] = null; state.ok[side] = false;
    drop.classList.remove("has-file");
    $(".drop-empty", drop).hidden = false;
    $(".drop-file", drop).hidden = true;
    refresh();
  }

  function setFile(side, file) {
    const drop = $(`.drop[data-side="${side}"]`);
    const info = $(".file-info", drop);
    const v = $("video", drop);
    state.files[side] = file;
    drop.classList.add("has-file");
    $(".drop-empty", drop).hidden = true;
    $(".drop-file", drop).hidden = false;
    $(".file-name", drop).textContent = file.name;
    info.classList.remove("bad");
    const problems = [];
    if (!OK_EXT.test(file.name) && !(file.type || "").startsWith("video/")) problems.push("not a supported video type");
    if (file.size > CFG.MAX_MB * 1048576) problems.push(`over ${CFG.MAX_MB} MB`);
    info.textContent = fmtBytes(file.size) + " · reading…";
    state.ok[side] = problems.length === 0;
    if (v.src) URL.revokeObjectURL(v.src);
    v.src = URL.createObjectURL(file);
    const done = (dur) => {
      if (state.files[side] !== file) return;
      if (dur && dur > CFG.MAX_SECONDS + 0.5) problems.push(`${Math.round(dur)} s long (limit ${CFG.MAX_SECONDS} s)`);
      const parts = [dur ? `${dur.toFixed(1)} s` : "length unknown", fmtBytes(file.size)];
      info.textContent = problems.length ? "✕ " + problems.join(", ") : parts.join(" · ");
      info.classList.toggle("bad", problems.length > 0);
      state.ok[side] = problems.length === 0;
      refresh();
    };
    v.onloadedmetadata = () => { try { v.currentTime = Math.min(1, v.duration / 3); } catch (_) {} done(v.duration); };
    v.onerror = () => done(null); // e.g. HEVC in Chrome; the server re-checks length
    refresh();
  }

  $("#consent").addEventListener("change", refresh);

  function refresh() {
    const both = state.files.a && state.files.b;
    const valid = state.ok.a && state.ok.b;
    const consent = $("#consent").checked;
    $("#submit").disabled = !(both && valid && consent);
    $("#hint").textContent = !both ? "Add both videos to begin"
      : !valid ? "Fix the highlighted video"
      : !consent ? "Confirm you have the right to upload these"
      : "Ready · takes about 3–5 minutes";
    showError(null);
  }

  function showError(msg) {
    const el = $("#formError");
    el.hidden = !msg;
    el.textContent = msg || "";
  }

  // ------------------------------------------------------------ progress UI
  const STEP_ORDER = ["upload", "gpu", "script", "scene", "body", "score"];
  function setProgress(stepKey, label, fraction) {
    $("#form").hidden = true;
    $("#progress").hidden = false;
    $("#stage").textContent = label;
    $("#pct").textContent = `${Math.round(fraction * 100)}%`;
    $("#bar").style.width = `${Math.max(2, fraction * 100)}%`;
    const idx = STEP_ORDER.indexOf(stepKey);
    $$("#steps li").forEach((li, i) => {
      li.classList.toggle("done", i < idx);
      li.classList.toggle("active", i === idx);
    });
  }
  function stageToStep(stage = "", status = "") {
    if (status === "queued" || /starting|download/i.test(stage)) return "gpu";
    if (/transcrib/i.test(stage)) return "script";
    if (/scene/i.test(stage)) return "scene";
    if (/^video [ab]/i.test(stage)) return "body";
    if (/align|compar|render|done/i.test(stage)) return "score";
    return "gpu";
  }
  function resetToForm() {
    clearInterval(state.pollTimer);
    $("#form").hidden = false;
    $("#progress").hidden = true;
  }

  // ------------------------------------------------------------ API
  function adminHeader() {
    // Set by logging into admin.html; lets the owner skip the daily limit.
    try { const k = localStorage.getItem("adminKey"); return k ? { "x-admin-key": k } : {}; } catch (_) { return {}; }
  }

  async function api(path, opts = {}) {
    const res = await fetch(API + path, {
      ...opts,
      headers: { "Content-Type": "application/json", ...adminHeader(), ...(opts.headers || {}) },
    });
    let body = null;
    try { body = await res.json(); } catch (_) {}
    if (!res.ok) {
      let msg = body && (body.detail || body.error);
      if (Array.isArray(msg)) msg = msg.map((d) => `${(d.loc || []).join(".")}: ${d.msg}`).join("; ");
      else if (msg && typeof msg === "object") msg = JSON.stringify(msg);
      throw new Error(msg || `Request failed (${res.status})`);
    }
    return body;
  }

  function uploadXHR(slot, file, onProgress) {
    const base = CFG.SUPABASE_URL.replace(/\/$/, "");
    let url = slot.signed_url || `${base}/storage/v1/object/upload/sign/videos/${slot.path}?token=${slot.token}`;
    if (url.startsWith("/")) url = base + url;
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("PUT", url);
      xhr.setRequestHeader("x-upsert", "false");
      if (CFG.SUPABASE_ANON_KEY) {
        xhr.setRequestHeader("apikey", CFG.SUPABASE_ANON_KEY);
        xhr.setRequestHeader("Authorization", `Bearer ${CFG.SUPABASE_ANON_KEY}`);
      }
      xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
      xhr.onload = () => (xhr.status < 300 ? resolve() : reject(new Error(`Upload failed (${xhr.status})`)));
      xhr.onerror = () => reject(new Error("Upload failed (network)"));
      const form = new FormData();
      form.append("cacheControl", "3600");
      form.append("", file);
      xhr.send(form);
    });
  }

  async function uploadFile(slot, file, onProgress) {
    try {
      await uploadXHR(slot, file, onProgress);
    } catch (err) {
      // Fallback: the official client (no progress events, but very robust)
      if (!CFG.SUPABASE_URL || !CFG.SUPABASE_ANON_KEY) throw err;
      const { createClient } = await import("https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/+esm");
      const sb = createClient(CFG.SUPABASE_URL, CFG.SUPABASE_ANON_KEY);
      const { error } = await sb.storage.from("videos").uploadToSignedUrl(slot.path, slot.token, file);
      if (error) throw new Error(error.message || "Upload failed");
      onProgress(1);
    }
  }

  // ------------------------------------------------------------ submit
  $("#form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    if ($("#submit").disabled) return;
    if (DEMO) return runDemo();
    const { a, b } = state.files;
    try {
      setProgress("upload", "Preparing upload", 0);
      const created = await api("/create", {
        method: "POST",
        body: JSON.stringify({
          consent: true,
          a: { name: a.name, size: a.size, type: a.type },
          b: { name: b.name, size: b.size, type: b.type },
        }),
      });
      const total = a.size + b.size;
      let sent = 0;
      for (const [side, file] of [["a", a], ["b", b]]) {
        await uploadFile(created.uploads[side], file, (f) => {
          const frac = (sent + f * file.size) / total;
          setProgress("upload", `Uploading video ${side.toUpperCase()}`, frac * 0.2);
        });
        sent += file.size;
      }
      await api("/analyze", { method: "POST", body: JSON.stringify({ id: created.id }) });
      history.replaceState(null, "", `?id=${created.id}`);
      poll(created.id);
    } catch (err) {
      resetToForm();
      showError(err.message);
    }
  });

  function poll(id) {
    clearInterval(state.pollTimer);
    const tick = async () => {
      try {
        const r = await api(`/status/${id}`);
        if (r.status === "done") {
          clearInterval(state.pollTimer);
          setProgress("score", "Done", 1);
          setTimeout(() => { resetToForm(); renderResults(r); }, 400);
        } else if (r.status === "error") {
          clearInterval(state.pollTimer);
          resetToForm();
          showError(r.error || "Something went wrong analysing these videos.");
        } else {
          const p = Math.max(0.2, (r.progress || 0) / 100);
          setProgress(stageToStep(r.stage, r.status), r.stage || "Waiting for a GPU", p);
        }
      } catch (err) {
        clearInterval(state.pollTimer);
        resetToForm();
        showError(err.message);
      }
    };
    tick();
    state.pollTimer = setInterval(tick, 2500);
  }

  // ------------------------------------------------------------ results
  const esc = (t) => String(t ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function testsOf(r) {
    // Results from before the scene + script tests only have the body score.
    if (r.tests) return r.tests;
    return { driver: "body", body: { score: r.overall_score }, scene: { score: null, reason: "Not run" },
             script: { score: null, reason: "Not run" } };
  }

  function renderResults(r) {
    state.current = r;
    const tests = testsOf(r);
    const score = r.overall_score ?? 0;
    $("#results").hidden = false;
    $("#score").textContent = pct(score);
    $("#verdict").textContent = r.verdict || "";
    requestAnimationFrame(() => ($("#ring").style.strokeDashoffset = String(326.7 * (1 - score))));
    $("#ring").style.stroke = score >= 0.8 ? "var(--accent)" : score >= 0.55 ? "var(--amber)" : "var(--muted)";
    const driver = TESTS.find((t) => t.key === tests.driver);
    $("#driver").innerHTML = driver
      ? `Highest of the three tests: <b>${driver.label}</b>`
      : "";

    const chips = [];
    if (r.mirrored && score >= 0.55) chips.push(`<span class="chip info" title="Video B matched best after flipping it left–right, a common trick to dodge copy detection">Video B appears mirrored (flipped left–right)</span>`);
    if (!r.calibrated) chips.push(`<span class="chip warn" title="Thresholds haven't been tuned on a labelled test set yet">Uncalibrated preview score</span>`);
    if (r.video_a_duration) chips.push(`<span class="chip">A ${r.video_a_duration.toFixed(1)} s · B ${(r.video_b_duration || 0).toFixed(1)} s</span>`);
    $("#chips").innerHTML = chips.join("");

    const al = r.alignment || {};
    const seg = tests.driver === "scene" && al.scene ? al.scene : al;
    $("#segment").textContent = seg.a_start_s != null && tests.driver !== "script"
      ? `Best match: A ${fmtTime(seg.a_start_s)}–${fmtTime(seg.a_end_s)} lines up with B ${fmtTime(seg.b_start_s)}–${fmtTime(seg.b_end_s)}.`
      : "";

    $("#breakdown").innerHTML = TESTS.map((t) => {
      const v = (tests[t.key] || {}).score;
      const na = v == null;
      const why = (tests[t.key] || {}).reason || "Not available";
      return `<tr class="${t.key === tests.driver ? "is-driver" : ""}">
        <td><span class="test-name">${t.label}${t.key === tests.driver ? '<span class="tag">highest</span>' : ""}</span><span class="test-sub">${t.sub}</span></td>
        <td>${na ? `<span class="na">${esc(why)}</span>` : `<div class="meter"><span style="width:${(v * 100).toFixed(1)}%"></span></div>`}</td>
        <td class="pct">${na ? "–" : pct(v)}</td></tr>`;
    }).join("");

    const ms = r.model_scores || {};
    const tabs = [
      ...["scene", "script"].map((k) => ({ key: k, short: PANELS[k].short, score: (tests[k] || {}).score })),
      ...(r.model_scores ? MODELS.map((m) => ({ key: m.key, short: m.short, score: ms[m.key], body: true })) : []),
    ];
    if (!tabs.some((t) => t.key === state.tab)) state.tab = tabs[0].key;
    if (state.fresh !== r) {
      state.fresh = r;
      state.tab = tests.driver === "body" && r.model_scores ? "pose" : (tests.driver || "scene");
    }
    let html = "", groupShown = false;
    for (const t of tabs) {
      if (t.body && !groupShown) { html += `<span class="tab-group">Body &amp; motion</span>`; groupShown = true; }
      html += `<button class="tab" role="tab" data-tab="${t.key}" aria-selected="${t.key === state.tab}">${t.short}<span class="mono">${t.score == null ? "–" : pct(t.score)}</span></button>`;
    }
    $("#tabs").innerHTML = html;
    $$("#tabs .tab").forEach((t) => t.addEventListener("click", () => { state.tab = t.dataset.tab; renderTabs(); }));
    renderPanel();
    $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderTabs() {
    $$("#tabs .tab").forEach((t) => t.setAttribute("aria-selected", String(t.dataset.tab === state.tab)));
    renderPanel();
  }

  function sparkline(values, w = 240, h = 56) {
    if (!values || values.length < 2) return "";
    const step = w / (values.length - 1);
    const pts = values.map((v, i) => `${(i * step).toFixed(1)},${(h - 4 - v * (h - 8)).toFixed(1)}`);
    return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-label="Similarity over the matched segment">
      <line x1="0" x2="${w}" y1="${h - 4 - 0.5 * (h - 8)}" y2="${h - 4 - 0.5 * (h - 8)}" stroke="var(--line-2)" stroke-dasharray="3 4"/>
      <polyline points="0,${h} ${pts.join(" ")} ${w},${h}" fill="var(--accent-soft)" stroke="none"/>
      <polyline points="${pts.join(" ")}" fill="none" stroke="var(--accent)" stroke-width="1.8" stroke-linejoin="round"/></svg>`;
  }

  function markSpans(tokens, spans, charLevel) {
    if (!tokens.length) return '<span class="na">No speech detected</span>';
    const marked = new Array(tokens.length).fill(false);
    (spans || []).forEach(([a, b]) => { for (let i = a; i < b; i++) marked[i] = true; });
    const sep = charLevel ? "" : " ";
    const groups = [];
    tokens.forEach((t, i) => {
      const g = groups[groups.length - 1];
      if (g && g.marked === marked[i]) g.tokens.push(t);
      else groups.push({ marked: marked[i], tokens: [t] });
    });
    return groups.map((g) => {
      const text = esc(g.tokens.join(sep));
      return g.marked ? `<mark>${text}</mark>` : text;
    }).join(sep);
  }

  function examplesHTML(frames, label) {
    return `<div class="examples">
        ${frames.map((f) => `
          <figure class="example">
            <img src="${f.url || f.demo || ""}" alt="${label}: video A at ${fmtTime(f.a_time)} beside video B at ${fmtTime(f.b_time)}" loading="lazy"/>
            <figcaption><span><b>A</b> original · ${fmtTime(f.a_time)}</span><span>frame similarity <b>${pct(f.similarity)}</b></span><span><b>B</b> suspected copy · ${fmtTime(f.b_time)}</span></figcaption>
          </figure>`).join("") || `<p class="hint">No example frames available.</p>`}
      </div>`;
  }

  function renderPanel() {
    const r = state.current;
    const tests = testsOf(r);
    if (state.tab === "script") {
      const tr = r.transcripts || {};
      const t = tests.script || {};
      const note = t.score == null
        ? `<p class="hint">${esc(t.reason || "No script comparison for this pair.")}</p>`
        : `<p class="hint">${t.shared_words || 0} shared ${tr.char_level ? "characters" : "words"} in phrases of 4+ words.</p>`;
      $("#panel").innerHTML = `
        <div class="panel-head"><div><h3>${PANELS.script.label}</h3><p>${PANELS.script.blurb}</p></div></div>
        ${note}
        <div class="transcripts">
          <div class="transcript"><p class="label">A · original</p><p class="t">${markSpans(tr.tokens_a || [], tr.spans_a, tr.char_level)}</p></div>
          <div class="transcript"><p class="label">B · suspected copy</p><p class="t">${markSpans(tr.tokens_b || [], tr.spans_b, tr.char_level)}</p></div>
        </div>`;
      return;
    }
    if (state.tab === "scene") {
      const frames = (r.frames || {}).scene || [];
      const timeline = ((r.alignment || {}).scene || {}).timeline;
      if ((tests.scene || {}).score == null) {
        $("#panel").innerHTML = `<div class="panel-head"><div><h3>${PANELS.scene.label}</h3><p>${PANELS.scene.blurb}</p></div></div>
          <p class="hint">${esc((tests.scene || {}).reason || "Not run for this comparison.")}</p>`;
        return;
      }
      $("#panel").innerHTML = `
        <div class="panel-head">
          <div><h3>${PANELS.scene.label}</h3><p>${PANELS.scene.blurb}</p></div>
          <div class="timeline">${sparkline(timeline)}<small>Similarity across the matched segment</small></div>
        </div>${examplesHTML(frames, PANELS.scene.label)}`;
      return;
    }
    const m = MODELS.find((x) => x.key === state.tab);
    const frames = ((r.frames || {})[m.key]) || [];
    const timeline = ((r.alignment || {}).timelines || {})[m.key];
    $("#panel").innerHTML = `
      <div class="panel-head">
        <div><p class="eyebrow-sm">Body &amp; motion</p><h3>${m.label}</h3><p>${m.blurb}</p></div>
        <div class="timeline">${sparkline(timeline)}<small>Similarity across the matched segment</small></div>
      </div>${examplesHTML(frames, m.label)}`;
  }

  $("#again").addEventListener("click", () => {
    $("#results").hidden = true;
    history.replaceState(null, "", location.pathname + (DEMO && params.has("demo") ? "?demo=1" : ""));
    clearFile("a"); clearFile("b");
    $("#tool").scrollIntoView({ behavior: "smooth", block: "center" });
  });
  $("#copyLink").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(location.href); $("#copyLink").textContent = "Link copied"; }
    catch (_) { $("#copyLink").textContent = "Copy failed"; }
    setTimeout(() => ($("#copyLink").textContent = "Copy result link"), 1800);
  });

  // ------------------------------------------------------------ demo mode
  function demoFrame(model, seed, dx) {
    const c = document.createElement("canvas");
    c.width = 760; c.height = 420;
    const g = c.getContext("2d");
    const half = (ox, phase) => {
      g.fillStyle = model === "matting" ? "#ecf0f3" : "#141a17";
      g.fillRect(ox, 0, 377, 420);
      const cx = ox + 188 + dx, t = seed + phase;
      const J = {
        head: [cx, 80], neck: [cx, 120], ls: [cx - 45, 130], rs: [cx + 45, 130],
        le: [cx - 70, 200 + 20 * Math.sin(t)], re: [cx + 75, 190 - 25 * Math.sin(t)],
        lw: [cx - 60, 270 + 30 * Math.sin(t)], rw: [cx + 95, 120 - 30 * Math.cos(t)],
        lh: [cx - 30, 260], rh: [cx + 30, 260], lk: [cx - 38, 335], rk: [cx + 40, 335],
        la: [cx - 42, 405], ra: [cx + 44, 405],
      };
      const L = [["head","neck"],["neck","ls"],["neck","rs"],["ls","le"],["le","lw"],["rs","re"],["re","rw"],["ls","lh"],["rs","rh"],["lh","rh"],["lh","lk"],["lk","la"],["rh","rk"],["rk","ra"]];
      const palette = { pose: ["#ff8a3d", "#5ec2ff"], seg: ["#d65db1", "#ffc75f", "#4ea8de", "#2ec4b6"], normal: ["#8fa8ff", "#b48cff", "#7fd1c8"], pointmap: ["#fca636", "#e16462", "#b12a90"], matting: ["#3d4a42", "#55635b"] }[model];
      g.lineCap = "round";
      L.forEach(([p, q], i) => {
        g.strokeStyle = palette[i % palette.length];
        g.lineWidth = model === "pose" ? 5 : 30;
        g.beginPath(); g.moveTo(...J[p]); g.lineTo(...J[q]); g.stroke();
      });
      g.fillStyle = palette[0];
      g.beginPath(); g.arc(J.head[0], J.head[1], model === "pose" ? 7 : 30, 0, 7); g.fill();
      if (model === "pose") { g.fillStyle = "#fff"; Object.values(J).forEach(([x, y]) => { g.beginPath(); g.arc(x, y, 4, 0, 7); g.fill(); }); }
    };
    half(0, 0); half(383, 0.12);
    g.fillStyle = "#fff"; g.fillRect(377, 0, 6, 420);
    return c.toDataURL("image/jpeg", 0.85);
  }

  function demoResult() {
    const scores = { pose: 0.93, seg: 0.88, normal: 0.81, pointmap: 0.79, matting: 0.9 };
    const timelines = {}, frames = {};
    MODELS.forEach((m, k) => {
      timelines[m.key] = Array.from({ length: 34 }, (_, i) => Math.max(0, Math.min(1, scores[m.key] + 0.06 * Math.sin(i / 3 + k) - (i > 22 && i < 26 ? 0.18 : 0))));
      frames[m.key] = [0, 1, 2].map((n) => ({ a_time: 4 + n * 6, b_time: 1.5 + n * 6, similarity: scores[m.key] - 0.03 * n, demo: demoFrame(m.key, n * 1.3, (n - 1) * 20) }));
    });
    const sceneFrames = [0, 1, 2].map((n) => ({ a_time: 2 + n * 7, b_time: 1 + n * 6, similarity: 0.94 - 0.02 * n, demo: demoFrame("matting", n * 1.3, (n - 1) * 20) }));
    const a = "Hey guys so today I'm going to show you my morning routine. First I always drink a big glass of water before I even look at my phone, and then I do five minutes of stretching by the window.";
    const b = "Okay this changed my mornings. First I always drink a big glass of water before I even look at my phone, and then I use the GlowSkin serum. Link in bio!";
    const ta = a.split(" "), tb = b.split(" ");
    return {
      status: "done", overall_score: 0.96, calibrated: false, mirrored: false,
      verdict: "Likely copied content",
      tests: { driver: "scene", body: { score: 0.74 }, scene: { score: 0.96 }, script: { score: 0.61, shared_words: 19 } },
      transcripts: { tokens_a: ta, tokens_b: tb, spans_a: [[12, 31]], spans_b: [[5, 24]], char_level: false },
      model_scores: scores, video_a_duration: 31.5, video_b_duration: 18.0,
      alignment: { a_start_s: 3.0, a_end_s: 20.5, b_start_s: 0, b_end_s: 17.5, timelines,
        scene: { timeline: timelines.matting.map((v) => Math.min(1, v + 0.05)), a_start_s: 1, a_end_s: 19, b_start_s: 0, b_end_s: 17.5 } },
      frames: { ...frames, scene: sceneFrames },
    };
  }

  function runDemo() {
    const plan = [["upload", "Uploading video A", 0.1], ["upload", "Uploading video B", 0.2], ["gpu", "Starting GPU", 0.25],
      ["script", "Video A: Transcribing speech", 0.32], ["script", "Video B: Transcribing speech", 0.4],
      ["scene", "Video A: Scene & visuals", 0.5], ["scene", "Video B: Scene & visuals", 0.6],
      ["body", "Video A: Pose estimation", 0.72], ["body", "Video B: Pose estimation", 0.84],
      ["score", "Comparing body, scene and script", 0.92], ["score", "Done", 1]];
    let i = 0;
    const step = () => {
      if (i < plan.length) { setProgress(...plan[i++]); setTimeout(step, 450); }
      else { resetToForm(); renderResults(demoResult()); }
    };
    step();
  }

  // ------------------------------------------------------------ deep link
  if (params.get("id") && !DEMO) poll(params.get("id"));
  if (DEMO && params.has("result")) renderResults(demoResult());
})();
