(function () {
  "use strict";

  const page = location.pathname.split("/").pop();
  if (page === "catalog.html" || page === "catalog_sharealike.html") {
    const destination = new URL("data_records.html", location.href);
    destination.search = location.search;
    destination.hash = location.hash;
    if (page === "catalog_sharealike.html" && !destination.searchParams.has("license")) {
      destination.searchParams.set("license", "CC-BY-SA-4.0");
    }
    location.replace(destination);
    return;
  }

  // Keep catalogue filters separate from Sphinx's full-text search parameter.
  if (["data_records.html", "families.html"].includes(page)) {
    const url = new URL(location.href);
    if (url.searchParams.has("q")) {
      if (!url.searchParams.has("filter")) url.searchParams.set("filter", url.searchParams.get("q"));
      url.searchParams.delete("q");
      try { history.replaceState(null, "", url); } catch (_) { /* file:// may restrict history changes. */ }
    }
  }

  function initializeCatalog() {
    const root = document.getElementById("synrxn-catalog");
    if (!root) return;
    const sections = [...root.parentElement.querySelectorAll(":scope > section")];
    if (sections.length) {
      const reference = document.createElement("details");
      reference.className = "catalog-written-reference";
      reference.id = "written-records";
      const summary = document.createElement("summary");
      summary.textContent = "Written reference";
      sections[0].before(reference);
      reference.append(summary, ...sections);
      const target = document.getElementById(location.hash.slice(1));
      if (target === reference || reference.contains(target)) reference.open = true;
    }
    const $ = (id) => document.getElementById(id);
    const controls = {
      search: $("catalog-search"), task: $("catalog-task"),
      license: $("catalog-license"), target: $("catalog-target"),
      size: $("catalog-size"), split: $("catalog-split")
    };
    const results = $("catalog-results");
    const detail = $("catalog-detail");
    const comparison = $("catalog-compare");
    const state = { datasets: [], filtered: [], selected: new Set(), release: {} };
    let detailTrigger = null;
    let reactionViewer = null;
    const notice = document.createElement("p");
    notice.className = "synrxn-sr-only";
    notice.setAttribute("role", "status");
    root.appendChild(notice);
    $("catalog-count").setAttribute("aria-live", "polite");
    detail.setAttribute("aria-labelledby", "catalog-detail-title");

    const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
    })[char]);
    const number = (value, digits = 0) => Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
    const bytes = (value) => value < 1e6 ? `${number(value / 1e3, 1)} kB` : `${number(value / 1e6, 1)} MB`;
    const scale = (rows) => rows < 10000 ? "small" : rows <= 100000 ? "medium" : "large";
    const recordedSplit = (item) => item.split_status !== "proposed" && Object.keys(item.split_counts || {}).length > 0;
    const selectedItems = () => state.datasets.filter((item) => state.selected.has(item.id));
    const setText = (id, text) => { const node = $(id); if (node) node.textContent = text; };

    function download(name, content, type) {
      const link = document.createElement("a");
      const url = URL.createObjectURL(new Blob([content], { type }));
      link.href = url;
      link.download = name;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    async function copyText(value, button) {
      let copied = false;
      if (navigator.clipboard) {
        try { await navigator.clipboard.writeText(value); copied = true; } catch (_) { /* Use the selectable text fallback. */ }
      }
      if (!copied) {
        const field = document.createElement("textarea");
        field.value = value;
        field.className = "synrxn-sr-only";
        document.body.appendChild(field);
        field.select();
        try { copied = document.execCommand("copy"); } catch (_) { copied = false; }
        field.remove();
        button.focus();
      }
      button.textContent = copied ? "Copied" : "Copy unavailable";
      notice.textContent = copied ? "Text copied to clipboard." : "Select and copy the displayed text.";
    }

    function addOptions(select, values) {
      [...new Set(values.filter(Boolean))].sort().forEach((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = select === controls.license && value === "CC-BY-SA-4.0"
          ? "CC BY-SA 4.0 (ShareAlike)" : value;
        select.appendChild(option);
      });
    }

    function searchableText(item) {
      return [item.id, item.title, item.description, item.benchmark_role, item.license,
        ...(item.targets || []), ...(item.columns || []), ...(item.citations || [])].join(" ").toLowerCase();
    }

    function splitSummary(item) {
      const entries = Object.entries(item.split_counts || {});
      if (!entries.length) return "Not recorded";
      const counts = entries.map(([name, count]) => `${escapeHtml(name)} ${number(count)}`).join("; ");
      return `${item.split_status === "proposed" ? "Proposed" : "Recorded"}: ${counts}`;
    }

    function applyFilters() {
      const query = controls.search.value.trim().toLowerCase();
      state.filtered = state.datasets.filter((item) =>
        (!query || searchableText(item).includes(query)) &&
        (!controls.task.value || item.task === controls.task.value) &&
        (!controls.license.value || item.license === controls.license.value) &&
        (!controls.target.value || (item.targets || []).includes(controls.target.value)) &&
        (!controls.size.value || scale(item.rows) === controls.size.value) &&
        (!controls.split.checked || recordedSplit(item)));
      renderResults();
      const params = new URLSearchParams(location.search);
      if (query) params.set("filter", query);
      else params.delete("filter");
      ["task", "license", "target", "size"].forEach((key) => {
        if (controls[key].value) params.set(key, controls[key].value);
        else params.delete(key);
      });
      if (controls.split.checked) params.set("split", "1");
      else params.delete("split");
      try {
        history.replaceState(null, "", `${location.pathname}${params.size ? `?${params}` : ""}${location.hash}`);
      } catch (_) { /* Some browsers restrict history changes on local files. Filtering still works. */ }
    }

    function setRecordUrl(id) {
      const url = new URL(location.href);
      if (id) url.searchParams.set("record", id);
      else url.searchParams.delete("record");
      if (url.href !== location.href) history.pushState(null, "", url);
    }

    function renderResults() {
      setText("catalog-count", `${state.filtered.length} of ${state.datasets.length} source tables`);
      if (!state.filtered.length) {
        results.innerHTML = '<p class="catalog-empty">No source tables match these filters.</p>';
        return;
      }
      results.innerHTML = `<div class="catalog-table-wrap" tabindex="0" role="region" aria-label="Filtered source tables">
        <table class="catalog-record-table"><caption>Select two to four tables for comparison. Row counts retain source-specific units and may overlap.</caption>
        <thead><tr><th scope="col">Source table</th><th scope="col">Task group</th><th scope="col" class="catalog-number">Rows</th><th scope="col">Targets</th><th scope="col">Split assignments</th><th scope="col">License expression</th><th scope="col">Compare</th></tr></thead>
        <tbody>${state.filtered.map((item) => `<tr>
          <th scope="row"><button type="button" class="catalog-title-button" data-detail="${escapeHtml(item.id)}" aria-controls="catalog-detail">${escapeHtml(item.title)}</button><code class="catalog-id">${escapeHtml(item.id)}</code>${item.release_eligible === false ? '<span class="catalog-id">Metadata only; release withheld</span>' : ""}</th>
          <td>${escapeHtml(item.task)}</td><td class="catalog-number">${number(item.rows)}</td>
          <td>${escapeHtml((item.targets || []).join(", ") || "—")}</td><td>${splitSummary(item)}</td><td>${escapeHtml(item.license)}</td>
          <td><input type="checkbox" data-compare="${escapeHtml(item.id)}" aria-label="Compare ${escapeHtml(item.title)}" ${state.selected.has(item.id) ? "checked" : ""}></td>
        </tr>`).join("")}</tbody></table></div>`;
    }

    function targetSummary(item) {
      return Object.entries(item.target_summaries || {}).map(([name, summary]) => {
        if (summary.kind === "numeric") return `<li><strong>${escapeHtml(name)}</strong>: range ${number(summary.min, 3)}–${number(summary.max, 3)}; mean ${number(summary.mean, 3)}</li>`;
        const frequent = (summary.top || []).filter((entry) => entry.count > 1 && String(entry.value).length <= 60 && !String(entry.value).includes(">"));
        return `<li><strong>${escapeHtml(name)}</strong>: ${number(summary.unique)} distinct values.${frequent.length ? ` Most frequent: ${frequent.slice(0, 3).map((entry) => `${escapeHtml(entry.value)} (${number(entry.count)})`).join("; ")}` : ""}</li>`;
      }).join("") || "<li>No target summary recorded.</li>";
    }

    function showDetail(id, trigger) {
      const item = state.datasets.find((entry) => entry.id === id);
      if (!item) return;
      if (reactionViewer) reactionViewer.destroy();
      reactionViewer = null;
      detailTrigger = trigger;
      const metadataOnly = item.release_eligible === false || item.redistribution_partition === "excluded";
      const columns = item.columns || [];
      const sampleColumns = columns.slice(0, 8);
      const sampleRows = (item.sample || []).map((row) => `<tr>${sampleColumns.map((column) => `<td>${escapeHtml(row[column])}</td>`).join("")}</tr>`).join("");
      const candidate = state.release.release_status === "candidate";
      const source = candidate ? 'source="local", data_dir="Data"' : `source="zenodo", version="${state.release.version}"`;
      const code = `from synrxn import DataLoader\n\n# SynRXN ${state.release.version}\nloader = DataLoader(task="${item.task}", ${source})\ndf = loader.load("${item.name}")`;
      const sources = (item.citations || []).map((key) => (window.SYNRXN_CITATIONS || {})[key]).filter(Boolean);
      const citation = [`${item.title}. SynRXN, version ${state.release.version}.${state.release.doi ? ` https://doi.org/${state.release.doi}` : ""}`,
        ...sources.map((entry) => entry.text)].join("\n\n");
      const sourceList = sources.map((entry) => {
        const url = entry.doi ? `https://doi.org/${entry.doi}` : entry.url;
        const link = url && /^https?:\/\//i.test(url) ? ` <a href="${escapeHtml(url)}" target="_blank" rel="noopener">Publication</a>` : "";
        return `<li>${escapeHtml(entry.text)}${link}</li>`;
      }).join("");
      const schemaRows = columns.map((column) => {
        const meta = (item.column_metadata || {})[column] || {};
        const nulls = (item.null_counts || {})[column];
        return `<tr><th scope="row">${escapeHtml(column)}</th><td>${escapeHtml(meta.logical_type || "Not recorded")}</td><td>${nulls === undefined ? "—" : number(nulls)}</td><td>${escapeHtml(meta.unit || "—")}</td><td>${escapeHtml(meta.description || "Not recorded")}</td></tr>`;
      }).join("");
      detail.innerHTML = `<div class="catalog-detail-header"><h2 id="catalog-detail-title" tabindex="-1">${escapeHtml(item.title)}</h2><button class="catalog-close" type="button">Close details</button></div>
        <p>${escapeHtml(item.description || "")}</p><p>${escapeHtml(item.benchmark_role || "")}</p>
        ${!metadataOnly && item.reaction_text ? `<section class="catalog-reaction-section"><h3>Example reaction</h3><div data-reaction-viewer></div>
          <details class="catalog-smiles"><summary>Reaction SMILES</summary><code class="catalog-reaction">${escapeHtml(item.reaction_text)}</code></details></section>` : ""}
        <div class="catalog-detail-grid"><div><h3>Table overview</h3><dl class="catalog-facts">
          <dt>Stored rows</dt><dd>${number(item.rows)}</dd>
          <dt>Targets</dt><dd>${escapeHtml((item.targets || []).join(", ") || "Not recorded")}</dd>
          <dt>License</dt><dd>${escapeHtml(item.license)}</dd>
          <dt>Splits</dt><dd>${splitSummary(item)}</dd></dl>
          ${metadataOnly ? '<p class="catalog-detail-note">Redistribution rights remain unresolved. This entry provides metadata only.</p>' : ""}</div>
          ${metadataOnly ? "" : `<div><h3>Target summaries</h3><ul>${targetSummary(item)}</ul></div>`}</div>
        <section class="catalog-citation-section"><h3>Sources and citation</h3>
          ${sourceList ? `<ol class="catalog-source-list">${sourceList}</ol>` : '<p>See the <a href="reference.html">bibliography</a> for source publications.</p>'}
          <p class="catalog-resource-citation">${escapeHtml(citation.split("\n\n")[0])}</p><button type="button" data-copy-citation>Copy citations</button></section>
        ${metadataOnly ? "" : `<details class="catalog-disclosure"><summary>Load this table in Python</summary>
          ${candidate ? '<p class="catalog-detail-note">Requires the matching local data files. For the portable DuckDB collection, see <a href="query_and_service.html">Database access</a>.</p>' : ""}
          <button type="button" data-copy-code>Copy code</button><pre><code>${escapeHtml(code)}</code></pre></details>
          <details class="catalog-disclosure"><summary>Sample rows</summary><p class="catalog-detail-note">Preview values may be truncated. The first ${sampleColumns.length} columns are shown.</p>
          <div class="catalog-table-wrap" tabindex="0" role="region" aria-label="Sample rows"><table><thead><tr>${sampleColumns.map((column) => `<th scope="col">${escapeHtml(column)}</th>`).join("")}</tr></thead><tbody>${sampleRows}</tbody></table></div></details>`}
        <details class="catalog-disclosure"><summary>Column schema</summary><div class="catalog-table-wrap" tabindex="0" role="region" aria-label="Column schema"><table><thead><tr><th scope="col">Column</th><th scope="col">Type</th><th scope="col">Missing rows</th><th scope="col">Unit</th><th scope="col">Description</th></tr></thead><tbody>${schemaRows}</tbody></table></div></details>
        <details class="catalog-disclosure"><summary>Source and file details</summary><dl class="catalog-facts catalog-provenance">
          <dt>Table identifier</dt><dd><code>${escapeHtml(item.id)}</code></dd>
          <dt>Compressed file</dt><dd>${bytes(item.size)}</dd>
          <dt>SHA-256</dt><dd><code>${escapeHtml(item.sha256)}</code></dd>
          ${item.rights_status ? `<dt>Rights status</dt><dd><code>${escapeHtml(item.rights_status)}</code></dd>` : ""}
          <dt>Source keys</dt><dd>${escapeHtml((item.citations || []).join(", ") || "Not recorded")}</dd>
          ${item.license_obligations?.length ? `<dt>License conditions</dt><dd><ul>${item.license_obligations.map((text) => `<li>${escapeHtml(text)}</li>`).join("")}</ul></dd>` : ""}</dl>
          <p class="catalog-detail-note">See <a href="families.html">Families and tasks</a> for table lineage and source-specific reuse terms.</p></details>`;
      detail.dataset.code = metadataOnly ? "" : code;
      detail.dataset.citation = citation;
      detail.hidden = false;
      const viewer = detail.querySelector("[data-reaction-viewer]");
      if (viewer && item.depiction && window.SynRXNReactionViewer) {
        reactionViewer = window.SynRXNReactionViewer.mount(viewer, item);
      }
      $("catalog-detail-title").focus({ preventScroll: true });
      detail.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
    }

    function closeDetail(updateUrl = true) {
      if (reactionViewer) reactionViewer.destroy();
      reactionViewer = null;
      detail.hidden = true;
      if (updateUrl) {
        setRecordUrl();
        (detailTrigger && detailTrigger.isConnected ? detailTrigger : controls.search).focus();
      }
    }

    function renderComparison() {
      const selected = selectedItems();
      comparison.hidden = selected.length === 0;
      if (!selected.length) return;
      const fields = [
        ["Task group", (item) => escapeHtml(item.task)],
        ["Stored rows", (item) => number(item.rows)],
        ["Compressed artifact size", (item) => bytes(item.size)],
        ["Columns", (item) => number(item.columns.length)],
        ["Targets", (item) => escapeHtml((item.targets || []).join(", ") || "—")],
        ["Split assignments", splitSummary],
        ["License expression", (item) => escapeHtml(item.license)]
      ];
      comparison.innerHTML = `<h2>Selected source tables</h2><p>${selected.length} of 4 tables selected. ${selected.length === 1 ? "Select another table to export a comparison." : "Row counts retain source-specific units; shared source records may occur in multiple tables."}</p>
        <p class="catalog-compare-actions"><button type="button" data-export="markdown" ${selected.length < 2 ? "disabled" : ""}>Export Markdown</button><button type="button" data-export="json" ${selected.length < 2 ? "disabled" : ""}>Export JSON</button><button type="button" data-clear-comparison>Clear selection</button></p>
        <div class="catalog-table-wrap" tabindex="0" role="region" aria-label="Selected table comparison"><table><thead><tr><th scope="col">Field</th>${selected.map((item) => `<th scope="col">${escapeHtml(item.title)}<code class="catalog-id">${escapeHtml(item.id)}</code></th>`).join("")}</tr></thead><tbody>${fields.map(([label, value]) => `<tr><th scope="row">${label}</th>${selected.map((item) => `<td>${value(item)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
    }

    results.addEventListener("click", (event) => {
      const button = event.target.closest("[data-detail]");
      if (!button) return;
      setRecordUrl(button.dataset.detail);
      showDetail(button.dataset.detail, button);
    });
    results.addEventListener("change", (event) => {
      const input = event.target.closest("[data-compare]");
      if (!input) return;
      if (input.checked && state.selected.size >= 4) {
        input.checked = false;
        notice.textContent = "Four tables are already selected. Deselect one before adding another.";
        return;
      }
      input.checked ? state.selected.add(input.dataset.compare) : state.selected.delete(input.dataset.compare);
      notice.textContent = `${state.selected.size} tables selected for comparison.`;
      renderComparison();
    });
    detail.addEventListener("click", (event) => {
      if (event.target.closest(".catalog-close")) closeDetail();
      const codeButton = event.target.closest("[data-copy-code]");
      const citationButton = event.target.closest("[data-copy-citation]");
      if (codeButton) copyText(detail.dataset.code, codeButton);
      if (citationButton) copyText(detail.dataset.citation, citationButton);
    });
    detail.addEventListener("keydown", (event) => {
      if (event.target.closest("dialog[open]")) return;
      if (event.key === "Escape") { event.preventDefault(); closeDetail(); }
    });
    comparison.addEventListener("click", (event) => {
      if (event.target.closest("[data-clear-comparison]")) {
        state.selected.clear();
        renderResults();
        renderComparison();
        controls.search.focus();
        notice.textContent = "Comparison selection cleared.";
        return;
      }
      const button = event.target.closest("[data-export]");
      if (!button) return;
      const selected = selectedItems();
      if (button.dataset.export === "json") {
        download("synrxn-comparison.json", JSON.stringify(selected, null, 2), "application/json");
        return;
      }
      const cell = (value) => String(Array.isArray(value) ? value.join(", ") : value ?? "").replace(/\|/g, "\\|").replace(/\r?\n/g, " ");
      const fields = ["id", "rows", "size", "targets", "license", "sha256"];
      const markdown = `| Field | ${selected.map((item) => cell(item.title)).join(" | ")} |\n|---|${selected.map(() => "---").join("|")}|\n${fields.map((field) => `| ${field} | ${selected.map((item) => cell(item[field])).join(" | ")} |`).join("\n")}\n`;
      download("synrxn-comparison.md", markdown, "text/markdown");
    });
    Object.entries(controls).forEach(([name, control]) => control.addEventListener(name === "search" ? "input" : "change", applyFilters));
    const clear = $("catalog-clear");
    if (clear) clear.addEventListener("click", () => {
      ["search", "task", "license", "target", "size"].forEach((key) => { controls[key].value = ""; });
      controls.split.checked = false;
      applyFilters();
      controls.search.focus();
    });

    const restoreView = () => {
      const params = new URLSearchParams(location.search);
      controls.search.value = params.get("filter") || "";
      ["task", "license", "target", "size"].forEach((key) => { controls[key].value = params.get(key) || ""; });
      controls.split.checked = params.get("split") === "1";
      applyFilters();
      const id = params.get("record");
      if (state.datasets.some((item) => item.id === id)) showDetail(id);
      else if (!detail.hidden) closeDetail(false);
    };
    const catalogue = window.SYNRXN_CATALOG;
    root.setAttribute("aria-busy", "false");
    if (!catalogue) {
      const failure = $("catalog-error");
      failure.hidden = false;
      failure.textContent = "The record browser could not load. See the written reference below.";
      return;
    }
    state.datasets = catalogue.datasets;
    state.release = catalogue.release;
    addOptions(controls.task, state.datasets.map((item) => item.task));
    addOptions(controls.license, state.datasets.map((item) => item.license));
    addOptions(controls.target, state.datasets.flatMap((item) => item.targets || []));
    const releaseLabel = state.release.release_status === "candidate" ? "Unpublished candidate" : "Release";
    setText("catalog-release", `${releaseLabel} ${state.release.version}`);
    restoreView();
    window.addEventListener("popstate", restoreView);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initializeCatalog, { once: true });
  else initializeCatalog();
})();
