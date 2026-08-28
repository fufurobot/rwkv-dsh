export type RoundSection = {
  index: number;
  heading: string;
  kind: string;
  body: string;
};

const HEADING = /^##\s+(第\s*\d+\s*轮[^\n]*|综合[^\n]*)\s*$/gm;

export function splitRoundSections(content: string): RoundSection[] {
  const text = content.replace(/\r\n/g, "\n");
  const matches = [...text.matchAll(HEADING)];
  if (matches.length === 0) {
    const body = text.trim();
    return body ? [{ index: 0, heading: "", kind: "", body }] : [];
  }

  const sections: RoundSection[] = [];
  const lead = text.slice(0, matches[0].index).trim();
  if (lead) {
    sections.push({ index: 0, heading: "", kind: "", body: lead });
  }

  for (let i = 0; i < matches.length; i += 1) {
    const match = matches[i];
    const heading = (match[1] ?? "").trim();
    const start = (match.index ?? 0) + match[0].length;
    const end = i + 1 < matches.length ? (matches[i + 1].index ?? text.length) : text.length;
    sections.push({
      index: parseRoundIndex(heading),
      heading,
      kind: parseRoundKind(heading),
      body: text.slice(start, end).trim(),
    });
  }
  return sections;
}

function parseRoundIndex(heading: string): number {
  const found = heading.match(/第\s*(\d+)\s*轮/);
  return found ? Number(found[1]) : 0;
}

function parseRoundKind(heading: string): string {
  if (heading.includes("意图")) return "意图识别";
  if (heading.includes("并行")) return "并行阅读";
  if (heading.includes("总结")) return "总结";
  if (heading.includes("综合")) return "综合";
  const parts = heading.split("·");
  return (parts[1] ?? heading).trim();
}
