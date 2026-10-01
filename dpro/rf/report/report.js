(function () {
  "use strict";
  const D = JSON.parse(document.getElementById("report-data").textContent);
  const S = D.summary.methods, PR = D.summary.per_run;
  const NS = "http://www.w3.org/2000/svg";
  const REDRAW = [];
  const el = (p, tag, a, text) => { const e = document.createElementNS(NS, tag); for (const k in a || {}) e.setAttribute(k, a[k]); if (text !== undefined) e.textContent = text; p.appendChild(e); return e; };
  const f2 = (v) => (v === undefined || v === null || isNaN(v)) ? "–" : (+v).toFixed(2);
  const runLab = (r) => r.replace("golden_run_", "run ");
  const RUNS = Object.keys(D.runs).sort((a, b) => +a.split("_").pop() - +b.split("_").pop());

  // ---------------------------------------------------------------- method names and families
  const FAM = { rf: "var(--c1)", ref: "var(--c2)", new: "var(--c3)", ceil: "var(--ceil)" };
  const M = [
    // [key in results, short name, family, group]
    ["radio flow + abs, 3-scan average + raw revisits (phase 3 start)", "flow + absolute + revisits (best hand)", "rf", "wifi"],
    ["abs, all APs (v1 solver, no patches)", "absolute RSSI only", "rf", "wifi"],
    ["radio flow + abs, 3-scan average", "flow + absolute, 3-scan avg", "rf", "wifi"],
    ["radio flow, single scans", "radio flow, single scans", "rf", "wifi"],
    ["radio flow, 3-scan average + raw revisits", "radio flow, 3-scan avg + revisits", "rf", "wifi"],
    ["radio flow, 3-scan average", "radio flow, 3-scan avg", "rf", "wifi"],
    ["radio flow, 3-scan average + learned revisits", "flow, 3-scan avg + learned revisits", "new", "wifi"],
    ["learned update (final refinement)", "learned update, after the run", "new", "wifi"],
    ["learned update (also while streaming)", "learned update, also online", "new", "wifi"],
    ["v1 network off, tuned", "DPRO v1, network off, tuned", "ref", "wifi"],
    ["v1 DPRO", "DPRO v1 with its network", "ref", "wifi"],
    ["WkNN + radio map", "WkNN fingerprinting (needs a survey)", "ref", "wifi"],
    ["v1 network off", "DPRO v1, network off", "ref", "wifi"],
    ["motion rules only", "motion rules only (no WiFi)", "ref", "wifi"],
    ["ceiling: abs with true APs", "ceiling: absolute, true AP map", "ceil", "wifi"],
    ["ceiling: radio flow with true APs", "ceiling: radio flow, true AP map", "ceil", "wifi"],
    ["wheel odometry", "wheel odometry", "ref", "odo"],
    ["odometry in the solver, no WiFi", "odometry in the solver, no WiFi", "rf", "odo"],
    ["odometry + radio flow", "odometry + radio flow", "rf", "odo"],
    ["odometry + radio flow + raw revisits", "odometry + radio flow + revisits", "rf", "odo"],
    ["odometry + WiFi gyro-bias check", "odometry + WiFi gyro-bias check", "new", "odo"],
    ["ceiling: odometry with the true gyro bias removed", "ceiling: true gyro bias removed", "ceil", "odo"],
  ].filter((m) => S[m[0]]);
  const NAME = Object.fromEntries(M.map((m) => [m[0], m[1]]));
  const COL = Object.fromEntries(M.map((m) => [m[0], FAM[m[2]]]));
  const GROUPS = [
    ["Radio Flow SLAM, hand-weighted", (m) => m[3] === "wifi" && m[2] === "rf"],
    ["New learned parts (phases 2–3)", (m) => m[3] === "wifi" && m[2] === "new"],
    ["Earlier methods", (m) => m[3] === "wifi" && m[2] === "ref"],
    ["Ceilings (use the truth)", (m) => m[3] === "wifi" && m[2] === "ceil"],
  ];
  const ODO_GROUP = ["With wheel odometry (phase 4)", (m) => m[3] === "odo"];

  // ---------------------------------------------------------------- tooltip
  const tip = document.getElementById("tip");
  const show = (t, x, y) => { tip.textContent = t; tip.style.display = "block"; const w = tip.offsetWidth; tip.style.left = Math.max(8, Math.min(innerWidth - w - 8, x + 12)) + "px"; tip.style.top = (y + 14) + "px"; };
  const hide = () => (tip.style.display = "none");
  const tipOf = (e) => e.target && e.target.closest && e.target.closest("[data-tip]");
  document.addEventListener("mouseover", (e) => { const t = tipOf(e); if (t) show(t.getAttribute("data-tip"), e.clientX, e.clientY); });
  document.addEventListener("mousemove", (e) => { const t = tipOf(e); if (t) show(t.getAttribute("data-tip"), e.clientX, e.clientY); });
  document.addEventListener("mouseout", (e) => { if (tipOf(e)) hide(); });
  document.addEventListener("focusin", (e) => { const t = tipOf(e); if (t) { const r = t.getBoundingClientRect(); show(t.getAttribute("data-tip"), r.left, r.bottom); } });
  document.addEventListener("focusout", hide);

  // numbers in the text: data-num="method|metric"
  document.querySelectorAll("[data-num]").forEach((n) => { const [m, k] = n.dataset.num.split("|"); n.textContent = S[m] ? f2(S[m][k]) + " m" : "–"; });

  // shared axis helpers
  const gridX = (svg, X, y0, y1, ticks, fmt) => ticks.forEach((t) => { el(svg, "line", { x1: X(t), x2: X(t), y1: y0, y2: y1, stroke: "var(--grid)" }); el(svg, "text", { x: X(t), y: y1 + 14, "text-anchor": "middle", "font-size": 10, fill: "var(--muted)", "font-family": "var(--mono)" }, fmt ? fmt(t) : t); });
  const gridY = (svg, Y, x0, x1, ticks, fmt) => ticks.forEach((t) => { el(svg, "line", { x1: x0, x2: x1, y1: Y(t), y2: Y(t), stroke: "var(--grid)" }); el(svg, "text", { x: x0 - 6, y: Y(t) + 3, "text-anchor": "end", "font-size": 10, fill: "var(--muted)", "font-family": "var(--mono)" }, fmt ? fmt(t) : t); });
  const xTitle = (svg, x, y, t) => el(svg, "text", { x, y, "text-anchor": "middle", "font-size": 11, fill: "var(--ink-2)" }, t);
  const yTitle = (svg, x, y, t) => el(svg, "text", { x, y, "text-anchor": "middle", "font-size": 11, fill: "var(--ink-2)", transform: `rotate(-90 ${x} ${y})` }, t);
  const path = (pts) => pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ");
  const tipText = (m) => { const e = S[m]; let t = `${NAME[m]}: aligned ${f2(e.aligned)} m · causal ${f2(e.causal)} m · online ${f2(e.online)} m · drift ${f2(e.drift5)} m per 5 m`; if (e.trainings > 1) t += ` · ±${f2(e.aligned_sd_train)} over trainings`; else if (e.seeds > 1) t += ` · ±${f2(e.aligned_sd_seed)} over seeds`; return t; };

  // ---------------------------------------------------------------- 1. main results (one svg: labels + 4 metric panels)
  (function results() {
    const svg = document.getElementById("res-chart"); if (!svg) return;
    const rows = [];
    GROUPS.forEach(([g, f]) => { const ms = M.filter(f).sort((a, b) => S[a[0]].aligned - S[b[0]].aligned); if (ms.length) { rows.push({ g }); ms.forEach((m) => rows.push({ m: m[0] })); } });
    const LW = 286, PW = 158, GAP = 20, rh = 22, top = 34, H = top + rows.length * rh + 30, W = LW + 4 * PW + 3 * GAP;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const best = S["v1 network off, tuned"];
    [["aligned", "Aligned (m)", 4], ["causal", "Causal (m)", 4], ["online", "Online (m)", 12], ["drift5", "Drift per 5 m (m)", 4]].forEach(([k, title, vmax], c) => {
      const x0 = LW + c * (PW + GAP), X = (v) => x0 + Math.min(v, vmax) / vmax * (PW - 34);
      el(svg, "text", { x: x0, y: 14, "font-size": 12, "font-weight": 600, fill: "var(--ink)", "font-family": "var(--display)" }, title);
      gridX(svg, X, top - 8, H - 26, vmax > 6 ? [0, 4, 8, 12] : [0, 1, 2, 3, 4]);
      if (best) el(svg, "line", { x1: X(best[k]), x2: X(best[k]), y1: top - 8, y2: H - 26, stroke: "var(--c2)", "stroke-dasharray": "3 3", "stroke-width": 1.2 });
      rows.forEach((r, i) => {
        if (!r.m) return;
        const y = top + i * rh, v = S[r.m][k];
        el(svg, "rect", { x: X(0), y: y + 5, width: Math.max(1.5, X(v) - X(0)), height: rh - 10, rx: 3, fill: COL[r.m], tabindex: 0, "data-tip": tipText(r.m) });
        el(svg, "text", { x: X(v) + 4, y: y + rh / 2 + 4, "font-size": 10.5, fill: "var(--ink)", "font-family": "var(--mono)" }, v > vmax ? f2(v) + " ›" : f2(v));
      });
    });
    rows.forEach((r, i) => {
      const y = top + i * rh;
      if (r.g) el(svg, "text", { x: 0, y: y + rh / 2 + 5, "font-size": 10, "letter-spacing": "0.08em", fill: "var(--muted)", "font-family": "var(--display)", "font-weight": 600 }, r.g.toUpperCase());
      else el(svg, "text", { x: 10, y: y + rh / 2 + 4, "font-size": 11.5, fill: "var(--ink)", "font-weight": r.m === "radio flow + abs, 3-scan average + raw revisits (phase 3 start)" || r.m === "v1 network off, tuned" ? 600 : 400 }, NAME[r.m]);
    });
  })();

  // ---------------------------------------------------------------- 2. full table
  (function table() {
    const tb = document.getElementById("all-tbody"); if (!tb) return;
    GROUPS.concat([ODO_GROUP]).forEach(([g, f]) => {
      const ms = M.filter(f).sort((a, b) => S[a[0]].aligned - S[b[0]].aligned); if (!ms.length) return;
      const tr = document.createElement("tr"); tr.className = "grp"; tr.innerHTML = `<td colspan="7">${g}</td>`; tb.appendChild(tr);
      ms.forEach(([m]) => {
        const e = S[m], sd = e.trainings > 1 ? `±${f2(e.aligned_sd_train)} (trainings)` : e.seeds > 1 ? `±${f2(e.aligned_sd_seed)} (seeds)` : "";
        const r = document.createElement("tr");
        r.innerHTML = `<td><span class="sw" style="background:${COL[m]}"></span>${NAME[m]}</td><td class="n">${f2(e.aligned)}</td><td class="n muted">${sd}</td><td class="n">${f2(e.causal)}</td><td class="n">${f2(e.online)}</td><td class="n">${f2(e.drift5)}</td><td class="n muted">${e.trainings > 1 ? e.trainings + " × " : ""}${e.seeds}</td>`;
        tb.appendChild(r);
      });
    });
  })();

  // ---------------------------------------------------------------- 3. tuning on synthetic sites
  (function tuning() {
    const svg = document.getElementById("tune-chart"); if (!svg || !D.tune) return;
    const R = D.tune.results, W = 560, H = 290, m = { l: 46, r: 16, t: 14, b: 42 };
    const lx = Math.log10, X = (v) => m.l + (lx(v) - lx(0.01)) / (lx(10) - lx(0.01)) * (W - m.l - m.r), Y = (v) => H - m.b - (Math.min(v, 4.2) - 1.8) / (4.2 - 1.8) * (H - m.t - m.b);
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    gridX(svg, X, m.t, H - m.b, [0.01, 0.03, 0.1, 0.3, 1, 3, 10]);
    gridY(svg, Y, m.l, W - m.r, [2, 2.5, 3, 3.5, 4]);
    xTitle(svg, (m.l + W - m.r) / 2, H - 6, "weight multiplier (log scale)"); yTitle(svg, 13, (m.t + H - m.b) / 2, "error, 24 synthetic sites (m)");
    const ser = [["abs_W1", "absolute RSSI, single scans", "var(--c2)"], ["flow_W1", "radio flow, single scans", "var(--c1)"], ["flow_W3", "radio flow, 3-scan average", "var(--c3)"], ["flow_W3_rev", "+ revisits (revisit weight)", "var(--c4)"]];
    ser.forEach(([key, name, col]) => {
      const pts = Object.entries(R[key] || {}).map(([c, v]) => [+c, v]).sort((a, b) => a[0] - b[0]); if (!pts.length) return;
      el(svg, "path", { d: path(pts.map((p) => [X(p[0]), Y(p[1])])), fill: "none", stroke: col, "stroke-width": 2 });
      pts.forEach((p) => el(svg, "circle", { cx: X(p[0]), cy: Y(p[1]), r: 4.5, fill: col, stroke: "var(--surface)", "stroke-width": 2, tabindex: 0, "data-tip": `${name}, weight ${p[0]}: ${p[1].toFixed(2)} m` }));
    });
    const tkey = document.getElementById("tune-key");
    if (tkey) tkey.innerHTML = ser.filter((s) => R[s[0]]).map(([, n, c]) => `<span><span class="sw" style="background:${c}"></span>${n}</span>`).join("");
    const ks = document.getElementById("k-chart"); if (!ks) return;
    const K = R.K, Wk = 280, Hk = 210, mk = { l: 40, r: 10, t: 16, b: 42 }, keys = Object.keys(K), bw = (Wk - mk.l - mk.r) / keys.length, Yk = (v) => Hk - mk.b - v / 4 * (Hk - mk.t - mk.b);
    ks.setAttribute("viewBox", `0 0 ${Wk} ${Hk}`);
    gridY(ks, Yk, mk.l, Wk - mk.r, [0, 1, 2, 3, 4]);
    keys.forEach((k, i) => {
      const x = mk.l + i * bw + bw * 0.2, v = K[k];
      el(ks, "rect", { x, y: Yk(v), width: bw * 0.6, height: Yk(0) - Yk(v), rx: 3, fill: k === "8" ? "var(--c1)" : "var(--c1-soft)", tabindex: 0, "data-tip": `links up to ${k} scans back: ${v.toFixed(2)} m` });
      el(ks, "text", { x: x + bw * 0.3, y: Yk(v) - 5, "text-anchor": "middle", "font-size": 10.5, fill: "var(--ink)", "font-family": "var(--mono)" }, v.toFixed(2));
      el(ks, "text", { x: x + bw * 0.3, y: Hk - mk.b + 14, "text-anchor": "middle", "font-size": 10, fill: "var(--muted)", "font-family": "var(--mono)" }, k);
    });
    xTitle(ks, (mk.l + Wk - mk.r) / 2, Hk - 8, "links to the K previous scans");
  })();

  // ---------------------------------------------------------------- 4. per-run heat table
  (function perrun() {
    const t = document.getElementById("perrun"); if (!t) return;
    const ms = ["radio flow + abs, 3-scan average + raw revisits (phase 3 start)", "learned update (final refinement)", "v1 network off, tuned", "ceiling: abs with true APs", "wheel odometry", "odometry + WiFi gyro-bias check"].filter((m) => PR[m]);
    const hex = (c) => { c = c.replace("#", ""); return [0, 2, 4].map((i) => parseInt(c.slice(i, i + 2), 16)); };
    const paint = () => {
      const cs = getComputedStyle(document.documentElement), lo = hex(cs.getPropertyValue("--seq-lo").trim()), hi = hex(cs.getPropertyValue("--seq-hi").trim());
      const dark = cs.getPropertyValue("--is-dark").trim() === "1";
      t.innerHTML = "";
      const head = document.createElement("tr"); head.innerHTML = `<th>run</th>` + ms.map((m) => `<th>${NAME[m]}</th>`).join(""); t.appendChild(head);
      RUNS.forEach((r) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td class="n">${runLab(r)} <span class="muted">${D.runs[r].scans} scans · ${D.runs[r].path.toFixed(0)} m</span></td>` + ms.map((m) => {
          const v = PR[m].aligned[r], f = Math.min(1, v / 3), rgb = lo.map((a, i) => Math.round(a + (hi[i] - a) * f));
          const ink = (f > 0.5) !== dark ? "#ffffff" : "#11161c";
          return `<td class="n hc" style="background:rgb(${rgb.join(",")});color:${ink}">${v.toFixed(2)}</td>`;
        }).join("");
        t.appendChild(tr);
      });
    };
    paint(); REDRAW.push(paint);
  })();

  // ---------------------------------------------------------------- 5. maps
  function maps(boxId, runs, methods) {
    const box = document.getElementById(boxId); if (!box || !D.maps) return;
    runs.forEach((r) => {
      const MM = D.maps[r]; if (!MM) return;
      const have = methods.filter((m) => MM.methods[m[0]]);
      const fig = document.createElement("figure"); fig.className = "map"; box.appendChild(fig);
      const svg = document.createElementNS(NS, "svg"); svg.setAttribute("viewBox", "0 0 300 300"); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", `Estimated paths and ground truth, ${runLab(r)}`); fig.appendChild(svg);
      const all = [MM.G].concat(have.map((m) => MM.methods[m[0]].X)).flat();
      const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
      const x0 = Math.min(...xs) - 0.5, x1 = Math.max(...xs) + 0.5, y0 = Math.min(...ys) - 0.5, y1 = Math.max(...ys) + 0.5;
      const k = Math.min(280 / (x1 - x0), 272 / (y1 - y0)), ox = 10 + (280 - (x1 - x0) * k) / 2, oy = 6 + (272 - (y1 - y0) * k) / 2;
      const P = (p) => [ox + (p[0] - x0) * k, oy + (y1 - p[1]) * k];
      el(svg, "line", { x1: 10, x2: 10 + 2 * k, y1: 293, y2: 293, stroke: "var(--muted)", "stroke-width": 2 });
      el(svg, "text", { x: 14 + 2 * k, y: 296, "font-size": 9.5, fill: "var(--muted)", "font-family": "var(--mono)" }, "2 m");
      el(svg, "path", { d: path(MM.G.map(P)), fill: "none", stroke: "var(--ink)", "stroke-width": 3, "stroke-linejoin": "round", opacity: 0.8, "data-tip": "ground truth" });
      el(svg, "circle", { cx: P(MM.G[0])[0], cy: P(MM.G[0])[1], r: 4, fill: "var(--ink)" });
      have.forEach(([m, col, dash]) => el(svg, "path", { d: path(MM.methods[m].X.map(P)), fill: "none", stroke: col, "stroke-width": 2, "stroke-dasharray": dash || "", "stroke-linejoin": "round", "data-tip": `${NAME[m]}: ${f2(MM.methods[m].aligned)} m (seed 0)` }));
      const cap = document.createElement("figcaption");
      cap.innerHTML = `<b>${runLab(r)}</b> · ${D.runs[r].scans} scans, ${D.runs[r].path.toFixed(0)} m<br><span class="sw" style="background:var(--ink)"></span>ground truth<br>` + have.map(([m, col]) => `<span class="sw" style="background:${col}"></span>${NAME[m]} <span class="num">${f2(MM.methods[m].aligned)} m</span>`).join("<br>");
      fig.appendChild(cap);
    });
  }
  maps("maps-wifi", ["golden_run_7", "golden_run_10", "golden_run_5"], [
    ["v1 network off, tuned", "var(--c2)"], ["radio flow + abs, 3-scan average + raw revisits (phase 3 start)", "var(--c1)"],
    ["learned update (final refinement)", "var(--c3)"], ["ceiling: abs with true APs", "var(--ceil)", "4 3"]]);
  maps("maps-odo", ["golden_run_4", "golden_run_9"], [
    ["wheel odometry", "var(--c2)"], ["odometry + WiFi gyro-bias check", "var(--c3)"], ["ceiling: odometry with the true gyro bias removed", "var(--ceil)", "4 3"]]);

  // ---------------------------------------------------------------- 6. phase 2: revisit proposals
  (function revisits() {
    const svg = document.getElementById("rev-chart"); if (!svg || !D.revisit_extra) return;
    const P = D.revisit_extra.props, ceil = D.revisit_extra.ceiling_lt3;
    const items = [["raw fingerprint distance", P["raw"]], ["learned (synthetic + other zone)", P["learned"]], ["same, comparing 3-scan sequences", P["learned, 3-scan sequence"]]].filter((x) => x[1]);
    const W = 600, m = { l: 236, r: 30, t: 30, b: 34 }, rowH = 44, H = m.t + items.length * rowH + m.b, X = (v) => m.l + v * (W - m.l - m.r);
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    gridX(svg, X, m.t - 10, H - m.b, [0, 0.25, 0.5, 0.75, 1], (v) => `${v * 100}%`);
    items.forEach(([name, v], i) => {
      const y = m.t + i * rowH;
      el(svg, "text", { x: m.l - 8, y: y + 19, "text-anchor": "end", "font-size": 11.5, fill: "var(--ink)" }, name);
      el(svg, "rect", { x: X(0), y: y + 4, width: X(v.lt3) - X(0), height: 22, rx: 3, fill: "var(--c1-soft)", tabindex: 0, "data-tip": `${name}: ${(v.lt3 * 100).toFixed(0)} % of ${v.n} proposals within 3 m, ${(v.lt2 * 100).toFixed(0)} % within 2 m` });
      el(svg, "rect", { x: X(0), y: y + 4, width: Math.max(1, X(v.lt2) - X(0)), height: 22, rx: 3, fill: "var(--c1)", "pointer-events": "none" });
      el(svg, "text", { x: X(v.lt3) + 5, y: y + 19, "font-size": 10.5, fill: "var(--ink)", "font-family": "var(--mono)" }, `${(v.lt3 * 100).toFixed(0)} % (${(v.lt2 * 100).toFixed(0)} % < 2 m)`);
    });
    el(svg, "line", { x1: X(ceil), x2: X(ceil), y1: m.t - 10, y2: H - m.b, stroke: "var(--ink)", "stroke-dasharray": "4 3", "stroke-width": 1.4 });
    el(svg, "text", { x: X(ceil) + 4, y: m.t - 14, "font-size": 10.5, fill: "var(--ink)" }, `best possible ${(ceil * 100).toFixed(0)} %`);
    el(svg, "line", { x1: X(0.6), x2: X(0.6), y1: m.t - 10, y2: H - m.b, stroke: "var(--bad)", "stroke-dasharray": "2 3", "stroke-width": 1.2 });
    el(svg, "text", { x: X(0.6) - 4, y: m.t - 14, "text-anchor": "end", "font-size": 10.5, fill: "var(--bad)" }, "gate 60 %");
    const av = document.getElementById("avail-chart"); if (!av) return;
    const A = D.revisit_extra.available, Wa = 560, Ha = 180, ma = { l: 38, r: 10, t: 14, b: 36 };
    const vmax = Math.max(...Object.values(A)), bw = (Wa - ma.l - ma.r) / RUNS.length, Ya = (v) => Ha - ma.b - v / vmax * (Ha - ma.t - ma.b);
    av.setAttribute("viewBox", `0 0 ${Wa} ${Ha}`);
    gridY(av, Ya, ma.l, Wa - ma.r, [0, 20, 40].filter((v) => v <= vmax));
    RUNS.forEach((r, i) => {
      const x = ma.l + i * bw + bw * 0.18, v = A[r];
      el(av, "rect", { x, y: Ya(v), width: bw * 0.64, height: Math.max(1.5, Ya(0) - Ya(v)), rx: 3, fill: "var(--c1)", tabindex: 0, "data-tip": `${runLab(r)}: ${v} scan pairs at least 7 scans apart and less than 3 m apart` });
      el(av, "text", { x: x + bw * 0.32, y: Ya(v) - 4, "text-anchor": "middle", "font-size": 10, fill: "var(--ink)", "font-family": "var(--mono)" }, v);
      el(av, "text", { x: x + bw * 0.32, y: Ha - ma.b + 14, "text-anchor": "middle", "font-size": 10, fill: "var(--muted)", "font-family": "var(--mono)" }, r.split("_").pop());
    });
    xTitle(av, (ma.l + Wa - ma.r) / 2, Ha - 4, "run");
  })();

  // ---------------------------------------------------------------- 7. phase 3: training, synthetic test, behaviour
  (function phase3() {
    const svg = document.getElementById("train-chart");
    if (svg && D.train && Object.keys(D.train).length) {
      const smooth = (a, k) => a.map((_, i) => { const lo = Math.max(0, i - k), hi = Math.min(a.length, i + k + 1); let s = 0; for (let j = lo; j < hi; j++) s += a[j]; return s / (hi - lo); });
      const seeds = Object.keys(D.train), diffs = seeds.map((s) => smooth(D.train[s].pose.map((v, j) => v - D.train[s].start[j]), 7));
      const n = Math.max(...diffs.map((d) => d.length)), W = 560, H = 250, m = { l: 50, r: 70, t: 14, b: 42 };
      const lo = Math.min(-0.2, Math.floor(Math.min(...diffs.flat()) * 20) / 20), hi = Math.max(0.1, Math.ceil(Math.max(...diffs.flat()) * 20) / 20);
      const X = (i) => m.l + i / (n - 1) * (W - m.l - m.r), Y = (v) => H - m.b - (v - lo) / (hi - lo) * (H - m.t - m.b);
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      const ticks = []; for (let v = Math.ceil(lo * 10) / 10; v <= hi + 1e-9; v += 0.1) ticks.push(+v.toFixed(1));
      gridY(svg, Y, m.l, W - m.r, ticks, (v) => (v > 0 ? "+" : "") + v.toFixed(1));
      el(svg, "line", { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), stroke: "var(--ink-2)", "stroke-width": 1.2 });
      el(svg, "text", { x: W - m.r, y: Y(0) - 6, "text-anchor": "end", "font-size": 10.5, fill: "var(--ink-2)" }, "same as the hand-weighted solver");
      const dashes = ["", "6 3", "2 3"];
      diffs.forEach((d, i) => el(svg, "path", { d: path(d.map((v, j) => [X(j), Y(v)])), fill: "none", stroke: "var(--c3)", "stroke-width": 2, "stroke-dasharray": dashes[i % 3] }));
      let prevY = -1e9;
      diffs.map((d, i) => ({ i, y: Y(d[d.length - 1]) + 4 })).sort((a, b) => a.y - b.y).forEach(({ i, y }) => {
        const yy = Math.max(y, prevY + 12); prevY = yy;
        el(svg, "text", { x: W - m.r + 6, y: yy, "font-size": 10.5, fill: "var(--ink)" }, `seed ${seeds[i]}`);
      });
      const steps = D.train_steps || 1500, last = D.train_last || steps;
      gridX(svg, (v) => X(v / last * (n - 1)), H - m.b, H - m.b, [0, 500, 1000, 1500].filter((v) => v <= last));
      xTitle(svg, (m.l + W - m.r) / 2, H - 6, "training step (rolling mean over 50 synthetic clips)");
      yTitle(svg, 13, (m.t + H - m.b) / 2, "learned − hand (m)");
    }
    const sy = document.getElementById("synth-chart");
    if (sy && D.synth) {
      const ps = D.synth.per_site, keys = Object.keys(ps).filter((k) => k !== "hand"), Ws = 320, Hs = 320, ms = { l: 46, r: 14, t: 14, b: 44 };
      const vmax = Math.ceil(Math.max(...Object.values(ps).flat()));
      const Xs = (v) => ms.l + v / vmax * (Ws - ms.l - ms.r), Ys = (v) => Hs - ms.b - v / vmax * (Hs - ms.t - ms.b);
      sy.setAttribute("viewBox", `0 0 ${Ws} ${Hs}`);
      const tk = []; for (let v = 0; v <= vmax; v++) tk.push(v);
      gridX(sy, Xs, ms.t, Hs - ms.b, tk); gridY(sy, Ys, ms.l, Ws - ms.r, tk);
      el(sy, "line", { x1: Xs(0), y1: Ys(0), x2: Xs(vmax), y2: Ys(vmax), stroke: "var(--ink-2)", "stroke-dasharray": "4 3" });
      el(sy, "text", { x: Xs(vmax * 0.97), y: Ys(vmax * 0.97) + 14, "text-anchor": "end", "font-size": 10, fill: "var(--ink-2)" }, "no change");
      el(sy, "text", { x: Xs(vmax * 0.95), y: Ys(vmax * 0.2), "text-anchor": "end", "font-size": 10.5, fill: "var(--ink-2)" }, "below the line: learned is better");
      keys.forEach((k, i) => ps[k].forEach((v, j) => el(sy, "circle", { cx: Xs(ps.hand[j]), cy: Ys(v), r: 4, fill: "var(--c3)", "fill-opacity": 0.55, stroke: "var(--surface)", "stroke-width": 1, tabindex: 0, "data-tip": `synthetic site ${j + 1}: hand-weighted ${ps.hand[j].toFixed(2)} m → learned (training ${i}) ${v.toFixed(2)} m` })));
      xTitle(sy, (ms.l + Ws - ms.r) / 2, Hs - 8, "hand-weighted solver (m)"); yTitle(sy, 13, (ms.t + Hs - ms.b) / 2, "learned update (m)");
    }
    const bh = document.getElementById("behav-chart");
    if (bh && D.net) {
      const B = D.net.by_k, rv = D.net.revisit || {};
      const cats = Object.keys(B).map((k) => [k, B[k], "var(--c3)", `${k}`]).concat(
        [["true (< 3 m)", rv["true (< 3 m)"], "var(--good)", "rev ✓"], ["wrong (>= 3 m)", rv["wrong (>= 3 m)"], "var(--bad)", "rev ✗"]].filter((c) => c[1] && c[1].mult_median !== null));
      const Wb = 340, Hb = 240, mb = { l: 46, r: 10, t: 16, b: 46 }, bw = (Wb - mb.l - mb.r) / cats.length;
      const vmaxb = Math.max(1.5, ...cats.map((c) => c[1].mult_median)) * 1.12, Yb = (v) => Hb - mb.b - v / vmaxb * (Hb - mb.t - mb.b);
      bh.setAttribute("viewBox", `0 0 ${Wb} ${Hb}`);
      gridY(bh, Yb, mb.l, Wb - mb.r, [0, 0.5, 1, 1.5, 2, 3, 4].filter((v) => v <= vmaxb), (v) => `×${v}`);
      el(bh, "line", { x1: mb.l, x2: Wb - mb.r, y1: Yb(1), y2: Yb(1), stroke: "var(--ink-2)", "stroke-dasharray": "4 3" });
      cats.forEach(([lab, v, col, tick], i) => {
        const x = mb.l + i * bw + bw * 0.16;
        el(bh, "rect", { x, y: Yb(v.mult_median), width: bw * 0.68, height: Math.max(1.5, Yb(0) - Yb(v.mult_median)), rx: 3, fill: col, tabindex: 0,
          "data-tip": `${isNaN(+lab) ? "revisit links, " + lab : "links " + lab + " scans apart"}: median weight ×${v.mult_median.toFixed(2)}, median target shift ${v.shift_abs_median !== undefined ? v.shift_abs_median.toFixed(2) + " dB" : "–"} (${v.n} factors)` });
        el(bh, "text", { x: x + bw * 0.34, y: Hb - mb.b + 14, "text-anchor": "middle", "font-size": 9.5, fill: "var(--muted)", "font-family": "var(--mono)" }, tick);
      });
      xTitle(bh, (mb.l + Wb - mb.r) / 2, Hb - 8, "scans apart, and revisit links");
    }
  })();

  // ---------------------------------------------------------------- 8. phase 4: odometry
  (function phase4() {
    const svg = document.getElementById("odo-chart");
    const ser = [["wheel odometry", "var(--c2)"], ["odometry + radio flow", "var(--c1)"], ["odometry + WiFi gyro-bias check", "var(--c3)"]].filter((s) => PR[s[0]]);
    if (svg && ser.length) {
      const W = 680, H = 250, m = { l: 40, r: 10, t: 22, b: 40 }, vmax = 3, gw = (W - m.l - m.r) / RUNS.length, Y = (v) => H - m.b - Math.min(v, vmax) / vmax * (H - m.t - m.b);
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      gridY(svg, Y, m.l, W - m.r, [0, 1, 2, 3]);
      const bw = gw * 0.78 / ser.length;
      RUNS.forEach((r, i) => {
        const x = m.l + i * gw + gw * 0.11;
        ser.forEach(([name, col], k) => {
          const v = PR[name].aligned[r];
          el(svg, "rect", { x: x + k * bw + 1, y: Y(v), width: bw - 2, height: Math.max(1.5, Y(0) - Y(v)), rx: 2, fill: col, tabindex: 0, "data-tip": `${runLab(r)} · ${NAME[name]}: ${v.toFixed(2)} m aligned, ${PR[name].online[r].toFixed(2)} m online` });
        });
        el(svg, "text", { x: x + gw * 0.39, y: H - m.b + 14, "text-anchor": "middle", "font-size": 10, fill: "var(--muted)", "font-family": "var(--mono)" }, r.split("_").pop());
      });
      ["golden_run_4", "golden_run_9"].forEach((r) => { const i = RUNS.indexOf(r); if (i >= 0) el(svg, "text", { x: m.l + i * gw + gw / 2, y: m.t - 8, "text-anchor": "middle", "font-size": 10, fill: "var(--ink-2)" }, "gyro bias"); });
      xTitle(svg, (m.l + W - m.r) / 2, H - 4, "run (aligned error, m)");
      const key = document.getElementById("odo-key"); if (key) key.innerHTML = ser.map(([n, c]) => `<span><span class="sw" style="background:${c}"></span>${NAME[n]}</span>`).join("");
    }
    const dg = document.getElementById("odo-diag-chart");
    if (dg && D.odo_diag && PR["wheel odometry"]) {
      const keys = Object.keys(D.odo_diag).filter((k) => k.startsWith("radio flow, ")), ws = keys.map((k) => +k.split("c_flow ").pop());
      const bad = ["golden_run_4", "golden_run_9"], good = RUNS.filter((r) => !bad.includes(r));
      const mean = (k, rs) => rs.reduce((a, r) => a + D.odo_diag[k][r].aligned, 0) / rs.length;
      const base = (rs) => rs.reduce((a, r) => a + PR["wheel odometry"].aligned[r], 0) / rs.length;
      const lines = [["runs 4 and 9 (gyro bias)", bad, "var(--c2)"], ["the other 10 runs", good, "var(--c1)"]];
      const W = 340, H = 240, m = { l: 40, r: 14, t: 16, b: 44 }, lx = Math.log10;
      const X = (v) => m.l + (lx(v) - lx(0.1)) / (lx(3) - lx(0.1)) * (W - m.l - m.r), Y = (v) => H - m.b - v / 2.5 * (H - m.t - m.b);
      dg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      gridX(dg, X, m.t, H - m.b, [0.1, 0.3, 1, 3]); gridY(dg, Y, m.l, W - m.r, [0, 0.5, 1, 1.5, 2, 2.5]);
      lines.forEach(([name, rs, col]) => {
        const pts = ws.map((w, i) => [w, mean(keys[i], rs)]);
        el(dg, "line", { x1: m.l, x2: W - m.r, y1: Y(base(rs)), y2: Y(base(rs)), stroke: col, "stroke-dasharray": "2 3", "stroke-width": 1.2 });
        el(dg, "path", { d: path(pts.map((p) => [X(p[0]), Y(p[1])])), fill: "none", stroke: col, "stroke-width": 2 });
        pts.forEach((p) => el(dg, "circle", { cx: X(p[0]), cy: Y(p[1]), r: 4.5, fill: col, stroke: "var(--surface)", "stroke-width": 2, tabindex: 0, "data-tip": `${name}, radio-flow weight ${p[0]}: ${p[1].toFixed(2)} m (odometry alone ${base(rs).toFixed(2)} m)` }));
        el(dg, "text", { x: X(pts[1][0]), y: Math.min(Y(pts[1][1]), Y(base(rs))) - 9, "text-anchor": "middle", "font-size": 10, fill: "var(--ink)" }, name);
      });
      xTitle(dg, (m.l + W - m.r) / 2, H - 8, "weight on radio flow (×, log scale)");
      yTitle(dg, 12, (m.t + H - m.b) / 2, "mean aligned error (m)");
    }
    const gc = document.getElementById("gyro-curve-chart");
    if (gc && D.gyro) {
      const G = D.gyro.runs, shown = [["golden_run_4", "var(--c2)"], ["golden_run_9", "var(--c3)"], ["golden_run_7", "var(--c1)"], ["golden_run_1", "var(--c4)"]].filter((s) => G[s[0]]);
      const W = 560, H = 280, m = { l: 46, r: 16, t: 16, b: 46 }, ytop = 1.6;
      const rel = (r) => G[r].coarse_cost.map((c) => c / G[r].cost0);
      const lo = Math.min(...shown.map(([r]) => Math.min(...rel(r)))), ylo = Math.floor(lo * 10) / 10;
      const X = (v) => m.l + (v + 30) / 60 * (W - m.l - m.r), Y = (v) => H - m.b - (v - ylo) / (ytop - ylo) * (H - m.t - m.b);
      gc.setAttribute("viewBox", `0 0 ${W} ${H}`);
      const cp = el(el(gc, "defs"), "clipPath", { id: "gyro-clip" }); el(cp, "rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b });
      gridX(gc, X, m.t, H - m.b, [-30, -20, -10, 0, 10, 20, 30]);
      const tk = []; for (let v = ylo; v <= ytop + 1e-9; v += 0.2) tk.push(+v.toFixed(1)); gridY(gc, Y, m.l, W - m.r, tk);
      el(gc, "line", { x1: m.l, x2: W - m.r, y1: Y(1), y2: Y(1), stroke: "var(--ink-2)", "stroke-dasharray": "4 3" });
      el(gc, "text", { x: W - m.r - 4, y: Y(1) - 5, "text-anchor": "end", "font-size": 10, fill: "var(--ink-2)" }, "same as no bias");
      shown.forEach(([r, col]) => {
        const g = G[r];
        el(gc, "path", { d: path(g.coarse.map((b, i) => [X(b), Y(rel(r)[i])])), fill: "none", stroke: col, "stroke-width": 2, "clip-path": "url(#gyro-clip)" });
        el(gc, "line", { x1: X(g.truth_deg_s), x2: X(g.truth_deg_s), y1: H - m.b, y2: H - m.b - 9, stroke: col, "stroke-width": 3 });
      });
      shown.forEach(([r, col]) => {
        const g = G[r];
        el(gc, "circle", { cx: X(g.b_deg_s), cy: Y(g.cost / g.cost0), r: 5, fill: col, stroke: "var(--surface)", "stroke-width": 2, tabindex: 0, "data-tip": `${runLab(r)}: best bias ${g.b_deg_s.toFixed(1)} °/s (true ${g.truth_deg_s.toFixed(1)}), fits the RSSI ${g.gain.toFixed(2)}× better than no bias → ${g.accepted ? "corrected" : "kept as is"}` });
      });
      if (G.golden_run_4) { const g = G.golden_run_4; el(gc, "text", { x: X(g.b_deg_s) + 9, y: Y(g.cost / g.cost0) + 4, "font-size": 10.5, fill: "var(--ink)" }, "run 4"); }
      xTitle(gc, (m.l + W - m.r) / 2, H - 8, "candidate gyro bias (°/s) · thick ticks on the axis = true bias");
      yTitle(gc, 13, (m.t + H - m.b) / 2, "RSSI misfit, relative to no bias");
      const key = document.getElementById("gyro-key");
      if (key) key.innerHTML = shown.map(([r, c]) => `<span><span class="sw" style="background:${c}"></span>${runLab(r)}${G[r].accepted ? " (corrected)" : ""}${["golden_run_7", "golden_run_1"].includes(r) ? " (healthy)" : ""}</span>`).join("");
    }
    const cc = document.getElementById("gyro-cal-chart");
    if (cc && D.gyro) {
      const C = D.gyro.calibration, tau = C.tau, rows = C.rows, real = D.gyro.runs;
      const W = 560, H = 196, m = { l: 150, r: 20, t: 26, b: 40 }, lmax = Math.log(Math.max(2, ...rows.map((r) => r.gain), ...Object.values(real).map((r) => r.gain)) * 1.05);
      const X = (g) => m.l + Math.log(Math.max(g, 1)) / lmax * (W - m.l - m.r);
      cc.setAttribute("viewBox", `0 0 ${W} ${H}`);
      gridX(cc, X, m.t, H - m.b, [1, 1.1, 1.3, 1.5, 2, 3, 5, 10].filter((v) => Math.log(v) <= lmax), (v) => v + "×");
      const lanes = [["synthetic, no bias", rows.filter((r) => r.b_true === 0), "var(--ceil)"], ["synthetic, with a bias", rows.filter((r) => r.b_true !== 0), "var(--c3)"], ["our 12 real runs", Object.entries(real).map(([k, v]) => Object.assign({ name: k }, v)), "var(--c1)"]];
      lanes.forEach(([lab, pts, col], li) => {
        const y = m.t + 18 + li * 44;
        el(cc, "text", { x: m.l - 10, y: y + 4, "text-anchor": "end", "font-size": 11, fill: "var(--ink)" }, lab);
        pts.forEach((p, j) => {
          const jit = ((j * 37) % 11 - 5) * 1.6;
          el(cc, "circle", { cx: X(p.gain), cy: y + jit, r: 4, fill: col, "fill-opacity": 0.75, stroke: "var(--surface)", "stroke-width": 1, tabindex: 0,
            "data-tip": p.name ? `${runLab(p.name)}: fits ${p.gain.toFixed(2)}× better with a bias of ${p.b_deg_s.toFixed(1)} °/s (true ${p.truth_deg_s.toFixed(1)} °/s) → ${p.accepted ? "corrected" : "kept"}` : `synthetic run, ${p.n} scans: fits ${p.gain.toFixed(2)}× better; true bias ${p.b_true.toFixed(1)} °/s, estimated ${p.b_hat.toFixed(1)} °/s` });
          if (p.name === "golden_run_4" || p.name === "golden_run_9") el(cc, "text", { x: X(p.gain), y: y - 10, "text-anchor": "middle", "font-size": 10, fill: "var(--ink)" }, runLab(p.name));
        });
      });
      el(cc, "line", { x1: X(tau), x2: X(tau), y1: m.t - 6, y2: H - m.b, stroke: "var(--ink)", "stroke-dasharray": "4 3", "stroke-width": 1.4 });
      el(cc, "text", { x: X(tau) + 4, y: m.t - 10, "font-size": 10.5, fill: "var(--ink)" }, `margin ${tau}× (chosen on synthetic runs)`);
      xTitle(cc, (m.l + W - m.r) / 2, H - 6, "how much better the best bias fits the RSSI than no bias (log scale)");
    }
  })();

  function redrawAll() { REDRAW.forEach((f) => f()); }
  try { matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redrawAll); } catch (e) { /* older browsers */ }
  new MutationObserver(redrawAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})();
