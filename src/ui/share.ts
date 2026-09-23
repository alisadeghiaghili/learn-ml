/** Progress-aware social share (LinkedIn / X / Facebook + copy). */

import type { CurriculumSummary } from "./progress";

export const LIVE_URL = "https://alisadeghiaghili.github.io/learn-ml/";
export const SHARE_URL = "https://alisadeghiaghili.github.io/learn-ml/";
export const REPO_URL = "https://github.com/alisadeghiaghili/learn-ml";
export const COFFEE_URL = "https://www.buymeacoffee.com/alisadeghil";
export const PUBLISHER = "Ali Sadeghi Aghili";
export const COFFEE_BUTTON_HTML = `<a href="${COFFEE_URL}" target="_blank" rel="noopener noreferrer"><img src="https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=&slug=alisadeghil&button_colour=2a3a4a&font_colour=ffffff&font_family=Cookie&outline_colour=ffffff&coffee_colour=FFDD00" alt="Buy me a coffee" /></a>`;

export const TITLE = "LearnML — interactive scikit-learn sandbox";

export interface ShareContext {
  levelName: string;
  levelId: string;
  commands: number | null;
  curriculum: CurriculumSummary;
}

function bulletList(items: { name: string; worldTitle: string; learning?: string[] }[], limit?: number): string[] {
  const list = limit ? items.slice(0, limit) : items;
  const lines = list.map((l) => `• ${l.worldTitle}: ${l.name}`);
  if (limit && items.length > limit) lines.push(`• …and ${items.length - limit} more`);
  return lines;
}

export function shareMessageLinkedIn(ctx: ShareContext): string {
  const c = ctx.curriculum;
  const learned = c.learned.length ? bulletList(c.learned) : [];
  const topics = c.learned.flatMap((l) => l.learning ?? []).slice(0, 8);
  const topicLines = topics.map((t) => `• ${t}`);
  const parts = [
    "I am really happy — I just learned practical machine learning with scikit-learn on LearnML!",
    "",
    c.solvedCount > 0
      ? `Latest win: ${ctx.levelName} (${ctx.levelId})${
          ctx.commands !== null ? ` · ${ctx.commands} commands` : ""
        }`
      : "Just starting the curriculum.",
    "",
    learned.length ? "What I have learned so far:" : "",
    ...learned,
    topicLines.length ? "" : "",
    ...topicLines,
    "",
    `Progress: ${c.solvedCount}/${c.total} levels.`,
    "",
    "If you want to understand fit/predict, leakage, and honest metrics — not just type code — try it. Free, no login:",
    SHARE_URL,
  ];
  return parts.filter(Boolean).join("\n").replace(/\n{3,}/g, "\n\n");
}

export function shareMessageX(ctx: ShareContext): string {
  const c = ctx.curriculum;
  const head = `Learned real ML concepts on LearnML — ${c.solvedCount}/${c.total} levels.`;
  const first = c.learned[0] ? `• ${c.learned[0].name}` : "fit is not understanding";
  let text = `${head}\n${first}\n${SHARE_URL}`;
  if (text.length > 275) text = `${head}\n${SHARE_URL}`;
  return text;
}

export interface ShareTargets {
  linkedin: string;
  x: string;
  facebook: string;
  text: string;
  shortText: string;
  url: string;
  learnedLines: string[];
}

export function buildShareTargets(ctx: ShareContext): ShareTargets {
  const longText = shareMessageLinkedIn(ctx);
  const shortText = shareMessageX(ctx);
  const url = SHARE_URL;
  return {
    linkedin: `https://www.linkedin.com/shareArticle?mini=true&url=${encodeURIComponent(url)}&title=${encodeURIComponent(TITLE)}&summary=${encodeURIComponent(longText)}&source=LearnML`,
    x: `https://twitter.com/intent/tweet?text=${encodeURIComponent(shortText)}`,
    facebook: `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(url)}&quote=${encodeURIComponent(longText)}`,
    text: longText,
    shortText,
    url,
    learnedLines: bulletList(ctx.curriculum.learned),
  };
}

export function openShareWindow(url: string): void {
  window.open(url, "_blank", "noopener,noreferrer,width=720,height=640");
}

export async function copySharePayload(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      ta.remove();
      return ok;
    } catch {
      return false;
    }
  }
}

export async function shareWithClipboard(
  kind: "linkedin" | "facebook" | "x" | "copy",
  targets: ShareTargets,
): Promise<{ opened: boolean; copied: boolean }> {
  if (kind === "copy") {
    return { opened: false, copied: await copySharePayload(targets.text) };
  }
  const copied = await copySharePayload(kind === "x" ? targets.shortText : targets.text);
  const href =
    kind === "linkedin" ? targets.linkedin : kind === "facebook" ? targets.facebook : targets.x;
  openShareWindow(href);
  return { opened: true, copied };
}
