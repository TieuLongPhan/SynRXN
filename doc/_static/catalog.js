(function () {
  "use strict";
  const initializeCatalog = () => {
  const root = document.getElementById("synrxn-catalog");
  if (!root) return;

  const $ = (id) => document.getElementById(id);
  const state = { datasets: [], filtered: [], compare: new Set(), release: {} };
  const task = $("catalog-task");
  const license = $("catalog-license");
  const search = $("catalog-search");
  const split = $("catalog-split");
  const target = $("catalog-target");
  const size = $("catalog-size");
  const results = $("catalog-results");
  const detail = $("catalog-detail");
  const compare = $("catalog-compare");
  const clear = $("catalog-clear");
  const example = $("catalog-example");

  const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  })[char]);
  const formatNumber = (value, digits = 0) => Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
  const formatBytes = (bytes) => bytes < 1e6 ? `${formatNumber(bytes / 1e3, 1)} KB` : `${formatNumber(bytes / 1e6, 1)} MB`;
  const scale = (rows) => rows < 10000 ? "small" : rows <= 100000 ? "medium" : "large";

  function download(name, content, type) {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([content], { type }));
    link.download = name; link.click(); URL.revokeObjectURL(link.href);
  }

  function copyText(value, button) {
    const copied = navigator.clipboard ? navigator.clipboard.writeText(value) : Promise.reject();
    copied.catch(() => { const field = document.createElement("textarea"); field.value = value; document.body.appendChild(field); field.select(); document.execCommand("copy"); field.remove(); }).then(() => { button.textContent = "Copied"; });
  }

  function optionValues(select, values) {
    [...new Set(values)].sort().forEach((value) => {
      const option = document.createElement("option");
      option.value = value; option.textContent = value; select.appendChild(option);
    });
  }

  function datasetText(item) {
    return [item.id, item.title, item.description, item.benchmark_role, item.license,
      ...(item.targets || []), ...(item.columns || []), ...(item.citations || [])].join(" ").toLowerCase();
  }

  function applyFilters() {
    const query = search.value.trim().toLowerCase();
    state.filtered = state.datasets.filter((item) =>
      (!query || datasetText(item).includes(query)) &&
      (!task.value || item.task === task.value) &&
      (!license.value || item.license === license.value) &&
      (!target.value || (item.targets || []).includes(target.value)) &&
      (!size.value || scale(item.rows) === size.value) &&
      (!split.checked || Object.keys(item.split_counts || {}).length > 0));
    renderResults();
    const params = new URLSearchParams();
    if (query) params.set("q", query);
    if (task.value) params.set("task", task.value);
    if (license.value) params.set("license", license.value);
    if (target.value) params.set("target", target.value);
    if (size.value) params.set("size", size.value);
    if (split.checked) params.set("split", "1");
    history.replaceState(null, "", `${location.pathname}${params.size ? `?${params}` : ""}${location.hash}`);
  }

  function resetFilters() {
    search.value = ""; task.value = ""; license.value = ""; target.value = "";
    size.value = ""; split.checked = false; applyFilters(); search.focus();
  }

  function renderResults() {
    $("catalog-count").textContent = `${state.filtered.length} dataset${state.filtered.length === 1 ? "" : "s"}`;
    results.innerHTML = state.filtered.map((item) => `
      <article class="catalog-card">
        <div class="catalog-card-top"><div><span class="catalog-task">${escapeHtml(item.task)}</span><span class="catalog-license">${escapeHtml(item.license)}</span></div><span class="catalog-ready"><i class="fa-solid fa-vial-circle-check" aria-hidden="true"></i> benchmark</span></div>
        <div class="catalog-card-main">
          <div><h2>${escapeHtml(item.title)}</h2><p>${escapeHtml(item.description)}</p></div>
          ${item.depiction ? `<img class="catalog-thumbnail" src="${escapeHtml(item.depiction)}" loading="lazy" width="220" height="150" alt="Reaction preview for ${escapeHtml(item.title)}">` : `<div class="catalog-thumbnail catalog-thumbnail-empty" aria-hidden="true"><i class="fa-solid fa-arrow-right-arrow-left"></i></div>`}
        </div>
        <dl><div><dt>Records</dt><dd>${formatNumber(item.rows)}</dd></div><div><dt>Targets</dt><dd>${escapeHtml((item.targets || []).join(", ") || "—")}</dd></div><div><dt>Splits</dt><dd>${escapeHtml(Object.keys(item.split_counts || {}).join(", ") || "—")}</dd></div></dl>
        <p class="catalog-role"><i class="fa-solid fa-bullseye" aria-hidden="true"></i> ${escapeHtml(item.benchmark_role)}</p>
        <div class="catalog-actions">
          <button type="button" data-detail="${escapeHtml(item.id)}">Details</button>
          <label><input type="checkbox" data-compare="${escapeHtml(item.id)}" ${state.compare.has(item.id) ? "checked" : ""}> Compare</label>
        </div>
      </article>`).join("") || `<p class="catalog-empty">No datasets match these filters.</p>`;
  }

  function summaryHtml(item) {
    return Object.entries(item.target_summaries || {}).map(([name, summary]) => {
      if (summary.kind === "numeric") return `<li><strong>${escapeHtml(name)}</strong>: ${formatNumber(summary.min, 3)}–${formatNumber(summary.max, 3)}, mean ${formatNumber(summary.mean, 3)}</li>`;
      return `<li><strong>${escapeHtml(name)}</strong>: ${formatNumber(summary.unique)} unique; top ${summary.top.slice(0, 3).map((x) => `${escapeHtml(x.value)} (${formatNumber(x.count)})`).join(", ")}</li>`;
    }).join("") || "<li>No target summary declared.</li>";
  }

  function splitSummary(item) {
    const entries = Object.entries(item.split_counts || {});
    return entries.length ? entries.map(([name, count]) => `${escapeHtml(name)} ${formatNumber(count)}`).join(" · ") : "No published split";
  }

  function renderExample(item) {
    if (!example || !item) return;
    example.innerHTML = `<div class="catalog-example-copy"><p class="catalog-eyebrow">Default example</p><h2>${escapeHtml(item.title)}</h2><p>${escapeHtml(item.benchmark_role)}</p><dl><div><dt>Task</dt><dd>${escapeHtml(item.task)}</dd></div><div><dt>Records</dt><dd>${formatNumber(item.rows)}</dd></div><div><dt>Split</dt><dd>${splitSummary(item)}</dd></div></dl><button type="button" class="catalog-example-action" data-example-detail="${escapeHtml(item.id)}">Inspect benchmark <span aria-hidden="true">→</span></button></div>${item.depiction ? `<img src="${escapeHtml(item.depiction)}" width="440" height="300" alt="Reaction example from ${escapeHtml(item.title)}">` : ""}`;
    example.hidden = false;
  }

  function showDetail(id) {
    const item = state.datasets.find((entry) => entry.id === id); if (!item) return;
    const sampleColumns = item.columns.slice(0, 8);
    const rows = item.sample.map((row) => `<tr>${sampleColumns.map((column) => `<td title="${escapeHtml(row[column])}">${escapeHtml(row[column])}</td>`).join("")}</tr>`).join("");
    const code = `from synrxn import DataLoader\n\n# Release ${state.release.version}; DOI ${state.release.doi}\n# SHA-256 ${item.sha256}\nloader = DataLoader(task="${item.task}", source="zenodo", version="${state.release.version}")\ndf = loader.load("${item.name}")`;
    const citation = `${item.title}. SynRXN release ${state.release.version}. ${state.release.doi}. Artifact SHA-256: ${item.sha256}. Sources: ${item.citations.join(", ")}.`;
    const schemaRows = item.columns.map((column) => { const meta = item.column_metadata[column] || {}; return `<tr><th>${escapeHtml(column)}</th><td>${escapeHtml(meta.logical_type || "string")}</td><td>${meta.nullable ? "yes" : "no"}</td><td>${escapeHtml(meta.unit || "—")}</td><td>${escapeHtml(meta.description || "No description supplied")}</td></tr>`; }).join("");
    detail.innerHTML = `<button class="catalog-close" type="button" aria-label="Close details">×</button>
      <span class="catalog-task">${escapeHtml(item.id)}</span><h2>${escapeHtml(item.title)}</h2><p>${escapeHtml(item.benchmark_role)}</p>
      <div class="catalog-detail-grid"><div><h3>Release facts</h3><ul><li>${formatNumber(item.rows)} records · ${formatBytes(item.size)}</li><li>License: ${escapeHtml(item.license)}</li><li>SHA-256: <code>${escapeHtml(item.sha256)}</code></li><li>Citations: ${escapeHtml(item.citations.join(", "))}</li></ul><h3>Target summary</h3><ul>${summaryHtml(item)}</ul></div>
      <div><h3>Reaction preview</h3>${item.depiction ? `<img src="${escapeHtml(item.depiction)}" loading="lazy" width="660" height="450" alt="Rendered reaction example from ${escapeHtml(item.title)}">` : ""}<code class="catalog-reaction">${escapeHtml(item.reaction_text || "No reaction preview available")}</code></div></div>
      <h3>Sample rows</h3><div class="catalog-table-wrap"><table><thead><tr>${sampleColumns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>
      <h3>Schema</h3><div class="catalog-table-wrap"><table><thead><tr><th>Column</th><th>Type</th><th>Nullable</th><th>Unit</th><th>Description</th></tr></thead><tbody>${schemaRows}</tbody></table></div>
      <h3>Pinned loading snippet</h3><button type="button" data-copy-code>Copy snippet</button><pre><code>${escapeHtml(code)}</code></pre><h3>Citation record</h3><button type="button" data-copy-citation>Copy citation</button><p>${escapeHtml(citation)}</p>`;
    detail.dataset.code = code; detail.dataset.citation = citation;
    detail.hidden = false; detail.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderCompare() {
    const selected = state.datasets.filter((item) => state.compare.has(item.id));
    compare.hidden = selected.length < 2;
    if (selected.length < 2) return;
    compare.innerHTML = `<h2>Dataset comparison</h2><p><button type="button" data-export="markdown">Export Markdown</button> <button type="button" data-export="json">Export JSON</button></p><div class="catalog-table-wrap"><table><thead><tr><th>Field</th>${selected.map((x) => `<th>${escapeHtml(x.name)}</th>`).join("")}</tr></thead><tbody>
      <tr><th>Task</th>${selected.map((x) => `<td>${escapeHtml(x.task)}</td>`).join("")}</tr><tr><th>Records</th>${selected.map((x) => `<td>${formatNumber(x.rows)}</td>`).join("")}</tr>
      <tr><th>Compressed size</th>${selected.map((x) => `<td>${formatBytes(x.size)}</td>`).join("")}</tr><tr><th>Typical pandas memory</th>${selected.map((x) => `<td>~${formatBytes(x.size * 4)}</td>`).join("")}</tr><tr><th>Columns</th>${selected.map((x) => `<td>${formatNumber(x.columns.length)}</td>`).join("")}</tr><tr><th>Targets</th>${selected.map((x) => `<td>${escapeHtml(x.targets.join(", ") || "—")}</td>`).join("")}</tr>
      <tr><th>Splits</th>${selected.map((x) => `<td>${escapeHtml(Object.keys(x.split_counts).join(", ") || "—")}</td>`).join("")}</tr><tr><th>License</th>${selected.map((x) => `<td>${escapeHtml(x.license)}</td>`).join("")}</tr></tbody></table></div>`;
  }

  results.addEventListener("click", (event) => {
    const button = event.target.closest("[data-detail]"); if (button) showDetail(button.dataset.detail);
  });
  results.addEventListener("change", (event) => {
    const input = event.target.closest("[data-compare]"); if (!input) return;
    if (input.checked && state.compare.size >= 4) { input.checked = false; return; }
    input.checked ? state.compare.add(input.dataset.compare) : state.compare.delete(input.dataset.compare); renderCompare();
  });
  detail.addEventListener("click", (event) => { if (event.target.closest(".catalog-close")) detail.hidden = true; const codeButton = event.target.closest("[data-copy-code]"); if (codeButton) copyText(detail.dataset.code, codeButton); const citationButton = event.target.closest("[data-copy-citation]"); if (citationButton) copyText(detail.dataset.citation, citationButton); });
  if (example) example.addEventListener("click", (event) => { const button = event.target.closest("[data-example-detail]"); if (button) showDetail(button.dataset.exampleDetail); });
  compare.addEventListener("click", (event) => { const button = event.target.closest("[data-export]"); if (!button) return; const selected = state.datasets.filter((item) => state.compare.has(item.id)); if (button.dataset.export === "json") { download("synrxn-comparison.json", JSON.stringify(selected, null, 2), "application/json"); return; } const fields = ["id", "rows", "size", "targets", "license"]; const markdown = `| Field | ${selected.map((x) => x.name).join(" | ")} |\n|---|${selected.map(() => "---").join("|")}|\n${fields.map((field) => `| ${field} | ${selected.map((x) => Array.isArray(x[field]) ? x[field].join(", ") : x[field]).join(" | ")} |`).join("\n")}`; download("synrxn-comparison.md", markdown, "text/markdown"); });
  [search, task, license, target, size, split].forEach((control) => control.addEventListener(control === search ? "input" : "change", applyFilters));
  if (clear) clear.addEventListener("click", resetFilters);

  const catalogRequest = window.SYNRXN_CATALOG
    ? Promise.resolve(window.SYNRXN_CATALOG)
    : fetch("_static/catalog-data.json").then((response) => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); });

  catalogRequest.then((catalog) => {
    state.datasets = catalog.datasets; state.release = catalog.release; optionValues(task, state.datasets.map((x) => x.task)); optionValues(license, state.datasets.map((x) => x.license)); optionValues(target, state.datasets.flatMap((x) => x.targets || []));
    const params = new URLSearchParams(location.search); search.value = params.get("q") || ""; task.value = params.get("task") || ""; license.value = params.get("license") || ""; target.value = params.get("target") || ""; size.value = params.get("size") || ""; split.checked = params.get("split") === "1";
    const withSplits = state.datasets.filter((item) => Object.keys(item.split_counts || {}).length > 0).length;
    const recordCount = state.datasets.reduce((total, item) => total + Number(item.rows || 0), 0);
    $("catalog-datasets").textContent = formatNumber(state.datasets.length);
    $("catalog-records").textContent = formatNumber(recordCount);
    $("catalog-splits").textContent = formatNumber(withSplits);
    renderExample(state.datasets.find((item) => item.id === "classification/schneider_b") || state.datasets.find((item) => Object.keys(item.split_counts || {}).length > 0) || state.datasets[0]);
    $("catalog-release").textContent = `Release ${catalog.release.version} · ${catalog.release.doi}`; root.setAttribute("aria-busy", "false"); applyFilters();
  }).catch((error) => { root.setAttribute("aria-busy", "false"); $("catalog-error").hidden = false; $("catalog-error").textContent = `Catalog data could not be loaded: ${error.message}`; });
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeCatalog, { once: true });
  } else {
    initializeCatalog();
  }
})();
