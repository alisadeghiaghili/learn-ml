/** Modal dialog with light markdown rendering. */

export function escapeHtml(s: string): string {
  return s
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderInline(raw: string): string {
  let t = escapeHtml(raw);
  t = t.replace(/`([^`]+)`/g, "<code>$1</code>");
  t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  t = t.replace(
    /\[([^\]]+)\]\(([^)\s]+)\)/g,
    (_m, label: string, href: string) =>
      `<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`,
  );
  return t;
}

export function renderMarkdown(md: string): string {
  const blocks = md.split(/```/);
  let html = "";
  blocks.forEach((block, i) => {
    if (i % 2 === 1) {
      html += `<pre>${escapeHtml(block.replace(/^\w*\n/, ""))}</pre>`;
      return;
    }
    const paragraphs = block
      .split(/\n{2,}/)
      .map((p) => p.trim())
      .filter(Boolean)
      .map((p) => `<p>${p.split("\n").map(renderInline).join("<br/>")}</p>`)
      .join("");
    html += paragraphs;
  });
  return html;
}

export interface ModalAction {
  label: string;
  className?: string;
  onClick: () => void;
}

export function showModal(opts: {
  title: string;
  bodyHtml: string;
  actions?: ModalAction[];
  onClose?: () => void;
  variant?: "default" | "celebrate" | "welcome";
}): { close: () => void; el: HTMLElement } {
  const overlay = document.createElement("div");
  overlay.className = `overlay${opts.variant === "celebrate" ? " overlay-celebrate" : ""}${opts.variant === "welcome" ? " overlay-welcome" : ""}`;
  const titleClass = opts.variant === "celebrate" ? ' class="visually-hidden"' : opts.variant === "welcome" ? ' class="visually-hidden"' : "";
  const modalClass =
    opts.variant === "celebrate"
      ? " modal-celebrate"
      : opts.variant === "welcome"
        ? " modal-welcome"
        : "";
  overlay.innerHTML = `
    <div class="modal${modalClass}" role="dialog" aria-modal="true" aria-label="${escapeHtml(opts.title)}">
      <h2${titleClass}>${escapeHtml(opts.title)}</h2>
      <div class="markdown">${opts.bodyHtml}</div>
      <div class="modal-actions"></div>
    </div>
  `;
  const actionsEl = overlay.querySelector(".modal-actions") as HTMLElement;
  const close = () => {
    overlay.remove();
    opts.onClose?.();
  };

  const actions = opts.actions?.length ? opts.actions : [{ label: "Close", onClick: () => close() }];

  for (const action of actions) {
    const btn = document.createElement("button");
    btn.className = action.className ?? "";
    btn.textContent = action.label;
    btn.addEventListener("click", () => {
      action.onClick();
      if (document.body.contains(overlay)) overlay.remove();
    });
    actionsEl.appendChild(btn);
  }

  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) close();
  });
  document.body.appendChild(overlay);
  const modalEl = overlay.querySelector<HTMLElement>(".modal");
  if (modalEl) {
    modalEl.tabIndex = -1;
    requestAnimationFrame(() => modalEl.focus());
  }
  return { close, el: overlay };
}
