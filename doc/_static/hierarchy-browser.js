/* Local filtering changes visibility only; source counts and metadata are fixed. */
(() => {
  "use strict";

  function addSharedTables(introduction, families) {
    const tables = new Map();
    for (const family of families) {
      for (const task of family.tasks) {
        for (const row of task.rows) {
          if (row.admission !== "admitted") continue;
          if (!tables.has(row.key)) {
            tables.set(row.key, {
              title: row.element.querySelector(".dataset-title").textContent.trim(),
              uses: new Map(),
            });
          }
          const uses = tables.get(row.key).uses;
          if (!uses.has(family)) uses.set(family, new Set());
          uses.get(family).add(task.element);
        }
      }
    }
    const pairs = new Map();
    for (const table of tables.values()) {
      const members = [...table.uses.keys()];
      members.forEach((left, i) => {
        for (const right of members.slice(i + 1)) {
          const id = `${left.element.id}/${right.element.id}`;
          if (!pairs.has(id)) pairs.set(id, { left, right, tables: [] });
          pairs.get(id).tables.push(table);
        }
      });
    }
    if (!pairs.size) return;

    const disclosure = document.createElement("details");
    disclosure.className = "shared-tables";
    disclosure.id = "shared-tables";
    const label = document.createElement("summary");
    label.textContent = "Shared tables";
    const description = document.createElement("p");
    description.textContent = "Select a connection, then open a table to see the tasks that use it.";
    const layout = document.createElement("div");
    layout.className = "shared-table-layout";
    const diagram = document.createElement("div");
    diagram.className = "shared-table-diagram";
    const results = document.createElement("section");
    results.className = "shared-table-results";
    results.setAttribute("aria-live", "polite");
    layout.append(diagram, results);
    disclosure.append(label, description, layout);
    introduction.append(disclosure);
    disclosure.open = location.hash === "#shared-tables";

    const links = [...pairs.values()].sort((a, b) => b.tables.length - a.tables.length);
    const buttons = new Map();
    const select = (link) => {
      for (const [pair, button] of buttons) {
        button.setAttribute("aria-pressed", String(pair === link));
      }
      const heading = document.createElement("h3");
      heading.textContent = `${link.left.title} ↔ ${link.right.title}`;
      const list = document.createElement("ul");
      for (const table of [...link.tables].sort((a, b) => a.title.localeCompare(b.title))) {
        const entry = document.createElement("li");
        const details = document.createElement("details");
        const title = document.createElement("summary");
        title.textContent = table.title;
        const tasks = document.createElement("ul");
        for (const family of [link.left, link.right]) {
          for (const task of table.uses.get(family)) {
            const item = document.createElement("li");
            const anchor = document.createElement("a");
            const url = new URL(location.href);
            url.searchParams.delete("filter");
            url.searchParams.delete("availability");
            url.hash = task.id;
            anchor.href = url.href;
            anchor.textContent = task.querySelector(".task-title").textContent;
            item.append(anchor);
            tasks.append(item);
          }
        }
        details.append(title, tasks);
        entry.append(details);
        list.append(entry);
      }
      results.replaceChildren(heading, list);
    };

    const byFamily = new Map(families.map((family) => [
      family, links.filter((link) => link.left === family || link.right === family),
    ]));
    const pending = new Set(links);
    for (const [family, adjacent] of [...byFamily].sort((a, b) => b[1].length - a[1].length)) {
      const connections = adjacent.filter((link) => pending.has(link));
      if (!connections.length) continue;
      const group = document.createElement("div");
      group.className = "shared-table-branches";
      const root = document.createElement("span");
      root.className = "shared-table-node shared-table-root";
      root.textContent = family.title;
      root.style.gridRow = `span ${connections.length}`;
      group.append(root);
      for (const link of connections) {
        pending.delete(link);
        const neighbor = link.left === family ? link.right : link.left;
        const leaf = document.createElement("span");
        leaf.className = "shared-table-node";
        leaf.textContent = neighbor.title;
        const edge = document.createElement("button");
        edge.type = "button";
        edge.className = "shared-table-edge";
        const count = document.createElement("span");
        count.textContent = link.tables.length;
        const unit = link.tables.length === 1 ? "shared table" : "shared tables";
        edge.setAttribute("aria-label", `${link.tables.length} ${unit} between ${family.title} and ${neighbor.title}`);
        edge.append(count);
        edge.addEventListener("click", () => select(link));
        buttons.set(link, edge);
        group.append(edge, leaf);
      }
      diagram.append(group);
    }
    select(links[0]);
  }

  const taskCells = [...document.querySelectorAll(".synrxn-home-families tbody td:nth-child(2)")];
  if (taskCells.length) {
    const counts = taskCells.map((cell) => Number(cell.textContent.trim()));
    const maximum = Math.max(...counts);
    taskCells.forEach((cell, index) => {
      const meter = document.createElement("meter");
      meter.className = "family-task-meter";
      meter.max = maximum;
      meter.value = counts[index];
      meter.setAttribute("aria-hidden", "true");
      (cell.querySelector("p") || cell).prepend(meter);
    });
  }

  document.querySelectorAll("[data-family-browser]").forEach((browser) => {
    const introduction = browser.querySelector(".catalogue-introduction");
    const scope = document.createElement("details");
    scope.className = "catalogue-scope";
    const scopeLabel = document.createElement("summary");
    scopeLabel.textContent = "Scope and exports";
    scope.append(scopeLabel, ...introduction.children);
    introduction.append(scope);

    const controls = browser.querySelector(".catalogue-controls");
    const search = controls.querySelector('input[type="search"]');
    search.placeholder = "Task, table, or license";
    const availability = controls.querySelector("select");
    const result = controls.querySelector("output");
    const expand = controls.querySelector('[data-action="expand"]');
    controls.querySelector('[data-action="collapse"]').remove();
    const navigation = browser.querySelector(".catalogue-family-index");
    const familyList = navigation.querySelector("ol");
    const familyFilter = document.createElement("details");
    const familyLabel = document.createElement("summary");
    const narrowScreen = window.matchMedia("(max-width: 767px)");
    familyFilter.open = !narrowScreen.matches;
    familyLabel.textContent = "Families";
    familyFilter.append(familyLabel, familyList);
    navigation.append(familyFilter);
    const allFamilies = document.createElement("li");
    const allLink = document.createElement("a");
    allLink.href = "#family-sections";
    allLink.textContent = "All families";
    allFamilies.append(allLink);
    familyList.prepend(allFamilies);

    const sections = browser.querySelector(".catalogue-family-sections");
    if (location.pathname.endsWith("/families.html")) {
      for (const row of sections.querySelectorAll(".dataset-row")) {
        const key = row.querySelector(".dataset-key").textContent;
        if (!key.startsWith("canonical:")) continue;
        const title = row.querySelector(".dataset-title");
        const link = document.createElement("a");
        link.href = `data_records.html?record=${encodeURIComponent(key.slice("canonical:".length))}`;
        link.textContent = title.textContent;
        title.replaceChildren(link);
      }
    }
    sections.querySelectorAll(".task-content").forEach((panel) => {
      const definition = document.createElement("details");
      definition.className = "task-definition-details";
      const label = document.createElement("summary");
      label.textContent = "Task definition and metrics";
      definition.append(label, panel.querySelector(".task-definition"));
      const qualification = panel.querySelector(".task-scope");
      if (qualification) definition.append(qualification);
      panel.append(definition);
    });
    sections.querySelectorAll(".family-datasets thead tr").forEach((header) => {
      ["Table", "Records", "License"].forEach((label, index) => {
        header.cells[index].textContent = label;
      });
    });
    const noResults = browser.querySelector(".catalogue-no-results");
    const layout = document.createElement("div");
    layout.className = "catalogue-browse";
    const results = document.createElement("div");
    results.className = "catalogue-results";
    navigation.before(layout);
    layout.append(navigation, results);
    results.append(result, noResults, sections);

    const families = [...sections.querySelectorAll(".catalogue-family")];
    const normalize = (value) => value.normalize("NFKC").toLocaleLowerCase();
    const index = families.map((family) => {
      const heading = family.querySelector("h2").cloneNode(true);
      heading.querySelector(".family-number").remove();
      const title = heading.textContent.trim();
      const link = navigation.querySelector(`a[href="#${family.id}"]`);
      const count = link.querySelector(".family-size");
      link.replaceChildren(document.createTextNode(title), count);
      return {
        element: family, title, text: normalize(title), link, count,
        tasks: [...family.querySelectorAll(".catalogue-task")].map((task) => ({
          element: task,
          text: normalize(task.querySelector(".task-title").textContent + " " +
            task.querySelector(".task-definition").textContent),
          rows: [...task.querySelectorAll(".dataset-row")].map((row) => ({
            element: row,
            key: row.querySelector(".dataset-key").textContent,
            text: normalize(row.textContent),
            admission: row.dataset.admission,
          })),
        })),
      };
    });
    addSharedTables(introduction, index);
    let selectedFamily = "";
    const visibleTasks = () => [...sections.querySelectorAll(
      ".catalogue-family:not([hidden]) .catalogue-task:not([hidden])"
    )];
    const updateExpand = () => {
      const tasks = visibleTasks();
      expand.disabled = tasks.length === 0;
      expand.textContent = tasks.length && tasks.every((task) => task.open)
        ? "Collapse tasks" : "Expand tasks";
    };
    const filter = ({ openMatches = true } = {}) => {
      const query = normalize(search.value.trim());
      const terms = query.split(/\s+/).filter(Boolean);
      const state = availability.value;
      const active = terms.length > 0 || state !== "all";
      let taskCount = 0;
      const tableKeys = new Set();
      index.forEach((family) => {
        const selected = !selectedFamily || family.element.id === selectedFamily;
        const familyKeys = new Set();
        let visibleTasks = 0;
        family.tasks.forEach((task) => {
          let visibleRows = 0;
          task.rows.forEach((row) => {
            const text = family.text + " " + task.text + " " + row.text;
            const match = terms.every((term) => text.includes(term)) &&
              (state === "all" || row.admission === state);
            row.element.hidden = !match;
            if (match) {
              visibleRows += 1;
              familyKeys.add(row.key);
            }
          });
          task.element.hidden = visibleRows === 0;
          if (visibleRows > 0) {
            visibleTasks += 1;
            if (active && selected && openMatches) task.element.open = true;
          }
        });
        family.count.textContent = familyKeys.size;
        family.count.title = `${familyKeys.size} matching table views`;
        family.element.hidden = !selected || visibleTasks === 0;
        if (selected) {
          taskCount += visibleTasks;
          familyKeys.forEach((key) => tableKeys.add(key));
        }
      });
      noResults.hidden = taskCount > 0;
      result.textContent = `${tableKeys.size} ${tableKeys.size === 1 ? "table view" : "table views"} · ${taskCount} ${taskCount === 1 ? "task" : "tasks"}`;
      updateExpand();
    };
    const selectFromLocation = () => {
      const target = document.getElementById(location.hash.slice(1));
      const previousFamily = selectedFamily;
      selectedFamily = target?.closest(".catalogue-family")?.id || "";
      if (!selectedFamily) allLink.setAttribute("aria-current", "true");
      else allLink.removeAttribute("aria-current");
      index.forEach((family) => {
        if (family.element.id === selectedFamily) family.link.setAttribute("aria-current", "true");
        else family.link.removeAttribute("aria-current");
      });
      const current = index.find((family) => family.element.id === selectedFamily);
      familyLabel.textContent = narrowScreen.matches ? current?.title || "All families" : "Families";
      filter({ openMatches: !selectedFamily });
      if (target?.matches(".catalogue-task")) {
        target.open = true;
        if (!target.closest("[hidden]")) target.scrollIntoView({ block: "start" });
      }
      else if (selectedFamily && previousFamily !== selectedFamily) {
        visibleTasks().forEach((task) => { task.open = true; });
      }
      const matchingTasks = visibleTasks();
      if (matchingTasks.length && !matchingTasks.some((task) => task.open)) matchingTasks[0].open = true;
      updateExpand();
    };
    const applyFilters = () => {
      const url = new URL(location.href);
      for (const [name, value] of [
        ["filter", search.value.trim()],
        ["availability", availability.value === "all" ? "" : availability.value],
      ]) {
        if (value) url.searchParams.set(name, value);
        else url.searchParams.delete(name);
      }
      history.replaceState(null, "", url);
      filter();
    };
    controls.hidden = false;
    search.addEventListener("input", applyFilters);
    availability.addEventListener("change", applyFilters);
    expand.addEventListener("click", () => {
      const tasks = visibleTasks();
      const open = tasks.some((task) => !task.open);
      tasks.forEach((task) => {
        task.open = open;
      });
      updateExpand();
    });
    sections.addEventListener("toggle", updateExpand, true);
    navigation.addEventListener("click", (event) => {
      const link = event.target.closest('a[href^="#"]');
      if (!link || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      if (location.hash !== link.hash) history.pushState(null, "", link.hash);
      selectFromLocation();
      if (narrowScreen.matches) {
        familyFilter.open = false;
        familyLabel.focus();
      }
    });
    const restoreFilters = () => {
      const query = new URLSearchParams(location.search);
      search.value = query.get("filter") || "";
      const state = query.get("availability");
      availability.value = ["admitted", "held"].includes(state) ? state : "all";
      selectFromLocation();
    };
    window.addEventListener("popstate", restoreFilters);
    window.addEventListener("hashchange", restoreFilters);
    window.addEventListener("pageshow", () => window.setTimeout(restoreFilters, 0));
    narrowScreen.addEventListener("change", () => {
      familyFilter.open = !narrowScreen.matches;
      selectFromLocation();
    });
    restoreFilters();
  });
})();
