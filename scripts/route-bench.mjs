#!/usr/bin/env node
/**
 * G1I router bench — Primitive Bench native format, not ChatGPT prompts.
 *
 * Official G1x (README / primitive_bench.py):
 *   System: Tools:
 *   [{name, description, arguments}, ...]
 *   Return only a JSON function call.
 *   User: ...
 *   Assistant: <think></think>
 *   ```json
 *
 *   node scripts/route-bench.mjs --once
 *   node scripts/route-bench.mjs --loop
 *   node scripts/route-bench.mjs --variant v2 --loop
 *
 * Env: RWKV_BASE_URL RWKV_MODEL RWKV_PASSWORD
 */

const BASE_URL = (process.env.RWKV_BASE_URL || "http://192.168.0.115:8000/v1").replace(
  /\/$/,
  "",
);
const MODEL = process.env.RWKV_MODEL || "rwkv-g1i-13b-4922";
const PASSWORD = process.env.RWKV_PASSWORD || "";
const MAX_TOKENS = Number(process.env.RWKV_ROUTE_MAX_TOKENS || 512);
const LOOP_MS = Number(process.env.RWKV_BENCH_SLEEP_MS || 2000);

const CATALOGS = {
  v1: [
    {
      name: "respond",
      description: "Answer directly. Greetings, thanks, or questions about available tools.",
      arguments: {},
    },
    {
      name: "inspect",
      description: "One lookup: list files, read one named file, or one calculation.",
      arguments: {},
    },
    {
      name: "run_agents",
      description:
        "Split this user request into 2-10 file-read subtasks. Invent goals from the wording.",
      arguments: { tasks: { type: "array" } },
    },
  ],
  v2: [
    {
      name: "respond",
      description: "Hello, thanks, chit-chat, or what tools can you use. Not arithmetic.",
      arguments: {},
    },
    {
      name: "inspect",
      description: "Arithmetic, one file list, or one named file. No arguments.",
      arguments: {},
    },
    {
      name: "run_agents",
      description:
        "Analyze, introduce, walk through, or compare a project; or several questions. Emit 2-10 short goals.",
      arguments: {
        tasks: {
          type: "array",
          items: { type: "object", properties: { goal: { type: "string" } } },
        },
      },
    },
  ],
};

const CASES = [
  { id: "hi", user: "你好", route: "respond", tasks: false },
  { id: "thanks", user: "谢谢", route: "respond", tasks: false },
  { id: "tools-zh", user: "你有哪些工具？", route: "respond", tasks: false },
  { id: "tools-en", user: "What tools can you use?", route: "respond", tasks: false },
  { id: "list-files", user: "我这个项目有什么文件啊", route: "inspect", tasks: false },
  { id: "list-root", user: "列出根目录", route: "inspect", tasks: false },
  { id: "read-readme", user: "读一下 README.md 的标题", route: "inspect", tasks: false },
  { id: "read-en", user: "Read README.md and report its title.", route: "inspect", tasks: false },
  { id: "calc", user: "4827乘以391等于多少", route: "inspect", tasks: false },
  { id: "analyze", user: "请你帮我分析一下这个项目", route: "inspect", tasks: true },
  { id: "analyze-short", user: "分析一下这个项目", route: "inspect", tasks: true },
  { id: "overview", user: "介绍一下这个仓库的架构和技术栈", route: "inspect", tasks: true },
  { id: "how-run-cases", user: "这个评测怎么跑，有哪些 case", route: "inspect", tasks: true },
  { id: "compare", user: "对比 src/a.py 和 src/b.py 的错误处理", route: "inspect", tasks: true },
  { id: "three-asks", user: "这个项目怎么启动，依赖是什么，测试怎么跑", route: "inspect", tasks: true },
  { id: "walkthrough", user: "walk through this codebase", route: "inspect", tasks: true },
];

function parseArgs(argv) {
  const args = { loop: false, variant: "v2", once: false };
  for (let i = 2; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === "--loop") args.loop = true;
    else if (token === "--once") args.once = true;
    else if (token === "--variant") args.variant = argv[++i] || args.variant;
    else if (token.startsWith("--variant=")) args.variant = token.slice("--variant=".length);
  }
  if (!args.once && !args.loop) args.loop = true;
  return args;
}

function cleanG1Text(text) {
  return String(text)
    .replace(/\r\n/g, "\n")
    .trim()
    .replace(/\n{2,}/g, "\n");
}

function renderG1I(user, catalog) {
  const toolsPretty =
    "[\n" + catalog.map((entry) => JSON.stringify(entry)).join(",\n") + "\n]";
  return (
    "System: Tools:\n" +
    toolsPretty +
    "\n" +
    'Exact tool names only. Call shape: {"name":"inspect","arguments":{}} or {"name":"run_agents","arguments":{"tasks":[{"goal":"a"},{"goal":"b"}]}}.\n' +
    "Return only a JSON function call.\n\n" +
    `User: ${cleanG1Text(user)}\n\n` +
    "Assistant: <think></think>\n```json\n"
  );
}

function extractJsonObject(text) {
  const start = text.indexOf("{");
  if (start < 0) return null;
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (escape) escape = false;
      else if (ch === "\\") escape = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) return text.slice(start, i + 1);
    }
  }
  return repairTruncatedJson(text.slice(start));
}

function repairTruncatedJson(fragment) {
  let value = fragment.trim();
  if (!value.startsWith("{")) return null;
  const stack = [];
  let inString = false;
  let escape = false;
  for (const ch of value) {
    if (inString) {
      if (escape) escape = false;
      else if (ch === "\\") escape = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === "{" || ch === "[") stack.push(ch);
    else if (ch === "}" || ch === "]") stack.pop();
  }
  if (inString) value += '"';
  if (value.endsWith(",")) value = value.slice(0, -1);
  while (stack.length) {
    value += stack.pop() === "{" ? "}" : "]";
  }
  try {
    JSON.parse(value);
    return value;
  } catch {
    return null;
  }
}

function collectGoals(value) {
  if (typeof value === "string") {
    try {
      value = JSON.parse(value);
    } catch {
      return [];
    }
  }
  if (!Array.isArray(value)) return [];
  const goals = [];
  for (const item of value.slice(0, 10)) {
    const goal = String(
      typeof item === "string" ? item : item?.goal || item?.path || item?.file || "",
    ).trim();
    if (!goal) return [];
    goals.push([...goal].slice(0, 240).join(""));
  }
  return goals;
}

function filePathsInText(text) {
  const ext = new Set(["py", "rs", "go", "md", "ts", "tsx", "js", "jsx", "json", "toml", "yml", "yaml", "txt"]);
  const paths = [];
  const seen = new Set();
  for (const token of String(text).split(/[^A-Za-z0-9./_-]+/)) {
    const suffix = token.split(".").pop()?.toLowerCase();
    if (!token.includes(".") || !ext.has(suffix) || seen.has(token)) continue;
    seen.add(token);
    paths.push(token);
  }
  return paths;
}

function parseFunctionCall(raw, user) {
  let text = String(raw || "").trim();
  text = text.replace(/^```json\s*/i, "").replace(/```\s*$/g, "");
  const object = extractJsonObject(text);
  if (!object) {
    const legacy = text.match(/<route>\s*(inspect|respond)\s*<\/route>/i);
    if (legacy) {
      return { ok: true, name: legacy[1].toLowerCase(), tasks: [], raw: text, legacy: true };
    }
    return { ok: false, error: "no json object", raw: text };
  }
  let parsed;
  try {
    parsed = JSON.parse(object);
  } catch (err) {
    return { ok: false, error: `json: ${err.message}`, raw: text };
  }
  let args = parsed.arguments ?? {};
  if (typeof args === "string") {
    try {
      args = JSON.parse(args);
    } catch {
      args = {};
    }
  }
  const name = String(parsed.name || parsed.route || "").trim();
  let tasks = collectGoals(
    args.tasks ?? args.files ?? args.paths ?? parsed.tasks ?? parsed.files,
  );
  if (name === "respond") return { ok: true, name, tasks: [], raw: text };
  if (name === "inspect" || name === "run_agents") {
    if (tasks.length < 2) {
      const files = filePathsInText(user);
      if (files.length >= 2) tasks = files;
    }
    return {
      ok: true,
      name: tasks.length >= 2 ? "run_agents" : name === "run_agents" ? "run_agents" : "inspect",
      tasks,
      raw: text,
    };
  }
  return { ok: false, error: `unknown tool ${name}`, raw: text, json: parsed };
}

function cannedChecklist(tasks) {
  const blob = tasks.join(" ");
  return /README/i.test(blob) && /入口|源码/.test(blob) && /测试/.test(blob) && tasks.length === 3;
}

function scoreCase(expect, got) {
  if (!got.ok) return { pass: false, reason: got.error || "parse fail" };
  const route = got.name === "respond" ? "respond" : "inspect";
  const hasTasks = got.name === "run_agents" && got.tasks.length >= 2;
  if (route !== expect.route) {
    return { pass: false, reason: `route ${route} != ${expect.route}` };
  }
  if (expect.tasks) {
    if (!hasTasks) return { pass: false, reason: `want run_agents 2-10, got ${got.name} ${got.tasks.length}` };
    if (got.tasks.length > 10) return { pass: false, reason: `too many tasks ${got.tasks.length}` };
    const unique = new Set(got.tasks.map((t) => t.replace(/\s+/g, "")));
    if (unique.size < 2) return { pass: false, reason: "tasks not distinct" };
    if (cannedChecklist(got.tasks)) return { pass: false, reason: "canned README/入口/测试 checklist" };
  } else if (hasTasks) {
    return { pass: false, reason: `unexpected tasks ${JSON.stringify(got.tasks)}` };
  }
  return { pass: true, reason: "ok" };
}

function completionsUrl(base) {
  const root = base.endsWith("/v1") ? base : `${base}/v1`;
  return `${root}/chat/completions`;
}

function splitAtStop(text) {
  const stops = ["```", "\nUser:", "\nSystem:", "\nTool:"];
  let cut = text.length;
  for (const stop of stops) {
    const index = text.indexOf(stop);
    if (index >= 0 && index < cut) cut = index;
  }
  return text.slice(0, cut);
}

async function complete(prompt) {
  const url = completionsUrl(BASE_URL);
  const body = {
    model: MODEL,
    contents: [prompt],
    max_tokens: MAX_TOKENS,
    temperature: 0.001,
    top_k: 50,
    top_p: 1,
    alpha_presence: 0,
    alpha_frequency: 0,
    alpha_decay: 0.99,
    enable_think: false,
    stream: true,
    chunk_size: 1,
    password: PASSWORD,
  };
  const headers = { "Content-Type": "application/json" };
  if (PASSWORD) headers.Authorization = `Bearer ${PASSWORD}`;
  const response = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`HTTP ${response.status}: ${text.slice(0, 240)}`);
  }
  const raw = await response.text();
  let content = "";
  for (const line of raw.split(/\r?\n/)) {
    if (!line.startsWith("data:")) continue;
    const data = line.slice(5).trim();
    if (!data || data === "[DONE]") continue;
    let payload;
    try {
      payload = JSON.parse(data);
    } catch {
      continue;
    }
    const choice =
      (payload.choices || []).find((item) => (item.index || 0) === 0) || payload.choices?.[0];
    content += choice?.delta?.content ?? choice?.message?.content ?? "";
  }
  return splitAtStop(content);
}

async function runRound(variantName, catalog) {
  const started = Date.now();
  const rows = [];
  for (const test of CASES) {
    const prompt = renderG1I(test.user, catalog);
    let got;
    try {
      const raw = await complete(prompt);
      got = parseFunctionCall(raw, test.user);
    } catch (err) {
      got = { ok: false, error: err.message, raw: "" };
    }
    const scored = scoreCase(test, got);
    rows.push({
      id: test.id,
      user: test.user,
      expect: { route: test.route, tasks: test.tasks },
      got: {
        ok: got.ok,
        name: got.name,
        tasks: got.tasks,
        error: got.error,
        preview: String(got.raw || "").replace(/\s+/g, " ").slice(0, 200),
      },
      pass: scored.pass,
      reason: scored.reason,
    });
    process.stdout.write(scored.pass ? "." : "x");
  }
  process.stdout.write("\n");
  const passed = rows.filter((row) => row.pass).length;
  return {
    variant: variantName,
    passed,
    total: rows.length,
    accuracy: passed / rows.length,
    elapsedMs: Date.now() - started,
    fails: rows.filter((row) => !row.pass),
    rows,
  };
}

function printRound(round, n) {
  const pct = (round.accuracy * 100).toFixed(1);
  console.log(
    `\n[round ${n}] g1i/${round.variant}  ${round.passed}/${round.total}  ${pct}%  ${round.elapsedMs}ms`,
  );
  for (const fail of round.fails) {
    console.log(
      `  FAIL ${fail.id}  ${fail.reason}\n    user: ${fail.user}\n    got: ${fail.got.preview || fail.got.error}`,
    );
  }
}

async function main() {
  const args = parseArgs(process.argv);
  const catalog = CATALOGS[args.variant];
  if (!catalog) {
    console.error(`unknown variant ${args.variant}. have: ${Object.keys(CATALOGS).join(", ")}`);
    process.exit(1);
  }
  console.log(
    `route-bench g1i  ${completionsUrl(BASE_URL)}  model=${MODEL}  variant=${args.variant}  loop=${args.loop}`,
  );
  let n = 0;
  let best = 0;
  for (;;) {
    n += 1;
    const round = await runRound(args.variant, catalog);
    printRound(round, n);
    if (round.accuracy > best) best = round.accuracy;
    console.log(`best so far ${(best * 100).toFixed(1)}%`);
    if (!args.loop) process.exit(round.accuracy === 1 ? 0 : 2);
    await new Promise((resolve) => setTimeout(resolve, LOOP_MS));
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
