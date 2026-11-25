(function () {
  "use strict";

  function mount(host, item) {
    const shell = document.createElement("div");
    shell.className = "reaction-viewer";
    shell.innerHTML = `<div class="reaction-toolbar">
      <label>Drawing <select aria-label="Reaction drawing engine"><option value="cdk">CDK</option><option value="rdkit">RDKit</option></select></label>
      <div class="reaction-zoom"><button type="button" data-action="out" aria-label="Zoom out">−</button>
      <button type="button" data-action="fit">Fit</button>
      <button type="button" data-action="in" aria-label="Zoom in">+</button></div>
      <button type="button" data-action="expand">Expand</button>
      <a target="_blank" rel="noopener" class="reaction-svg-link">Open SVG</a>
    </div><div class="reaction-viewport" tabindex="0" role="region" aria-label="Reaction diagram; scroll to inspect when zoomed">
      <div class="reaction-stage"></div></div><p class="reaction-status" role="status"></p>`;
    host.appendChild(shell);
    const select = shell.querySelector("select");
    const viewport = shell.querySelector(".reaction-viewport");
    const stage = shell.querySelector(".reaction-stage");
    const status = shell.querySelector(".reaction-status");
    const openLink = shell.querySelector("a");
    const expand = shell.querySelector('[data-action="expand"]');
    const dialog = document.createElement("dialog");
    dialog.className = "reaction-dialog";
    const heading = document.createElement("h3");
    heading.id = "reaction-dialog-title";
    heading.textContent = item.title;
    dialog.setAttribute("aria-labelledby", heading.id);
    const close = document.createElement("button");
    close.type = "button";
    close.className = "reaction-dialog-close";
    close.textContent = "Close diagram";
    close.addEventListener("click", () => dialog.close());
    dialog.append(heading, close);
    host.appendChild(dialog);
    let image = null;
    let zoom = 1;
    let generation = 0;
    let destroyed = false;

    function layout(center = false) {
      if (!image || !image.naturalWidth || !viewport.clientWidth || destroyed) return;
      const fit = Math.min((viewport.clientWidth - 32) / image.naturalWidth,
        (viewport.clientHeight - 32) / image.naturalHeight, 1.6);
      image.style.width = `${Math.max(1, image.naturalWidth * fit * zoom)}px`;
      image.style.height = `${Math.max(1, image.naturalHeight * fit * zoom)}px`;
      if (center) {
        viewport.scrollLeft = Math.max(0, (viewport.scrollWidth - viewport.clientWidth) / 2);
        viewport.scrollTop = Math.max(0, (viewport.scrollHeight - viewport.clientHeight) / 2);
      }
      shell.querySelector('[data-action="out"]').disabled = zoom <= 0.5;
      shell.querySelector('[data-action="in"]').disabled = zoom >= 8;
    }

    function load(engine, fallback = false) {
      const current = ++generation;
      const next = new Image();
      next.alt = `Example reaction from ${item.title}`;
      const src = engine === "cdk" ? item.depiction.replace(/\.svg$/, ".cdk.svg") : item.depiction;
      status.textContent = "Loading reaction diagram…";
      openLink.hidden = true;
      next.onload = () => {
        if (current !== generation || destroyed) return;
        image = next;
        stage.replaceChildren(image);
        zoom = 1;
        layout(true);
        openLink.href = src;
        openLink.hidden = false;
        status.textContent = fallback ? "CDK diagram unavailable; showing the local RDKit drawing."
          : `Drawn with ${engine === "cdk" ? "CDK" : "RDKit"}.`;
      };
      next.onerror = () => {
        if (current !== generation || destroyed) return;
        if (engine === "cdk") {
          select.querySelector('[value="cdk"]').disabled = true;
          select.value = "rdkit";
          load("rdkit", true);
        } else {
          image = null;
          stage.replaceChildren();
          status.textContent = "No diagram is available. The reaction SMILES is provided below.";
        }
      };
      next.src = src;
    }

    select.addEventListener("change", () => load(select.value));
    shell.addEventListener("click", (event) => {
      const action = event.target.closest("[data-action]")?.dataset.action;
      if (action === "expand") {
        dialog.appendChild(shell);
        expand.hidden = true;
        dialog.showModal();
        zoom = 1;
        layout(true);
        return;
      }
      if (!["in", "out", "fit"].includes(action)) return;
      zoom = action === "fit" ? 1 : Math.max(0.5, Math.min(8, zoom * (action === "in" ? 1.5 : 1 / 1.5)));
      layout(true);
    });
    dialog.addEventListener("close", () => {
      if (destroyed) return;
      host.insertBefore(shell, dialog);
      expand.hidden = false;
      zoom = 1;
      layout(true);
      expand.focus();
    });
    const observer = new ResizeObserver(() => layout());
    observer.observe(viewport);
    load("cdk");
    return {
      destroy() {
        destroyed = true;
        generation += 1;
        observer.disconnect();
        if (dialog.open) dialog.close();
        dialog.remove();
        shell.remove();
      }
    };
  }

  window.SynRXNReactionViewer = { mount };
})();
