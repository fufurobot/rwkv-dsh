#!/usr/bin/env node
/**
 * Package RWKV Agent for macOS and Windows with one command.
 *
 *   pnpm package              Mac host build + Windows NSIS (cross-compile on Mac/Linux)
 *   pnpm package:mac          macOS only
 *   pnpm package:win          Windows only
 *   pnpm package:setup        install packaging toolchains only
 *
 * Flags: --mac --win --setup --universal --skip-setup
 *
 * Artifacts land in ./release/
 */

import { spawnSync } from "node:child_process";
import {
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const releaseDir = path.join(root, "release");
const xwinCacheDir = path.join(root, "src-tauri", ".xwin-cache");
const host = process.platform;
const isMac = host === "darwin";
const isWin = host === "win32";
const isLinux = host === "linux";

const flags = new Set(process.argv.slice(2));
const setupOnly = flags.has("--setup");
const skipSetup = flags.has("--skip-setup");
const universalMac = flags.has("--universal");
const explicitMac = flags.has("--mac");
const explicitWin = flags.has("--win");
const wantMac = setupOnly || explicitMac || (!explicitMac && !explicitWin);
const wantWin = setupOnly || explicitWin || (!explicitMac && !explicitWin);

if (isMac) {
  process.env.HOMEBREW_NO_AUTO_UPDATE = "1";
  process.env.HOMEBREW_NO_ENV_HINTS = "1";
}

class CommandError extends Error {
  constructor(command, status) {
    super(`${command} failed (exit ${status ?? "spawn"})`);
    this.name = "CommandError";
  }
}

function log(message) {
  console.log(`\n==> ${message}`);
}

function fail(message) {
  console.error(`\nerror: ${message}`);
  process.exit(1);
}

function commandExists(command) {
  const probe = isWin ? "where" : "command";
  const probeArgs = isWin ? [command] : ["-v", command];
  return spawnSync(probe, probeArgs, { stdio: "ignore", shell: isWin }).status === 0;
}

function run(command, commandArgs, options = {}) {
  const result = spawnSync(command, commandArgs, {
    cwd: root,
    stdio: "inherit",
    shell: isWin && !command.includes(path.sep),
    env: options.env ?? process.env,
  });
  if (result.status !== 0) {
    throw new CommandError(`${command} ${commandArgs.join(" ")}`, result.status);
  }
}

function runCapture(command, commandArgs) {
  const result = spawnSync(command, commandArgs, {
    cwd: root,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    shell: isWin && !command.includes(path.sep),
  });
  return (result.stdout || "").trim();
}

function tauriBin() {
  const name = isWin ? "tauri.cmd" : "tauri";
  const local = path.join(root, "node_modules", ".bin", name);
  if (existsSync(local)) return local;
  fail("Tauri CLI is missing. Run pnpm install in the project root first.");
}

function toolBinDir(formula, binary) {
  const candidates = [];
  if (commandExists("brew")) {
    const prefix = runCapture("brew", ["--prefix", formula]);
    if (prefix) candidates.push(path.join(prefix, "bin"));
  }
  candidates.push(`/opt/homebrew/opt/${formula}/bin`, `/usr/local/opt/${formula}/bin`);
  return candidates.find((dir) => existsSync(path.join(dir, binary))) ?? null;
}

function llvmBinDir() {
  return toolBinDir("llvm", "llvm-rc");
}

function lldBinDir() {
  return toolBinDir("lld", "lld-link");
}

function withBuildEnv() {
  const env = {
    ...process.env,
    CI: "true",
    CARGO_TERM_COLOR: "always",
    HOMEBREW_NO_AUTO_UPDATE: "1",
    XWIN_CACHE_DIR: xwinCacheDir,
    XWIN_ACCEPT_LICENSE: "1",
  };
  const extraBins = [lldBinDir(), llvmBinDir(), path.join(os.homedir(), ".cargo", "bin")].filter(
    (dir) => dir && existsSync(dir),
  );
  if (extraBins.length > 0) {
    env.PATH = `${extraBins.join(path.delimiter)}${path.delimiter}${env.PATH ?? ""}`;
  }
  return env;
}

function brewHas(pkg) {
  if (!commandExists("brew")) return false;
  const prefix = runCapture("brew", ["--prefix", pkg]);
  return Boolean(prefix) && existsSync(prefix);
}

function ensureNodeDeps() {
  if (existsSync(path.join(root, "node_modules", ".bin", isWin ? "tauri.cmd" : "tauri"))) {
    return;
  }
  log("Installing frontend dependencies");
  if (existsSync(path.join(root, "pnpm-lock.yaml")) && commandExists("pnpm")) {
    run("pnpm", ["install"]);
  } else {
    run("npm", ["install"]);
  }
}

function brewEnv(extra = {}) {
  return {
    ...process.env,
    HOMEBREW_NO_AUTO_UPDATE: "1",
    HOMEBREW_NO_ENV_HINTS: "1",
    ...extra,
  };
}

function ensureBrewPackages(packages) {
  if (!commandExists("brew")) {
    fail("Homebrew is required on macOS to install NSIS and LLVM. Install it from https://brew.sh");
  }
  const missing = packages.filter((pkg) => {
    if (pkg === "nsis" && commandExists("makensis")) return false;
    return !brewHas(pkg);
  });
  if (missing.length === 0) return;
  log(`Installing Homebrew packages: ${missing.join(", ")}`);
  const mirrors = [
    brewEnv(),
    brewEnv({ HOMEBREW_BOTTLE_DOMAIN: "https://mirrors.ustc.edu.cn/homebrew-bottles" }),
    brewEnv({ HOMEBREW_BOTTLE_DOMAIN: "https://mirrors.tuna.tsinghua.edu.cn/homebrew-bottles" }),
  ];
  let lastError = null;
  for (const env of mirrors) {
    try {
      run("brew", ["install", ...missing], { env });
      return;
    } catch (error) {
      lastError = error;
      console.warn("Homebrew install failed, retrying with another bottle mirror...");
    }
  }
  throw lastError;
}

function ensureRustTargets(targets) {
  if (!commandExists("rustup")) {
    fail("rustup is required. Install Rust from https://rustup.rs");
  }
  const installed = new Set(
    runCapture("rustup", ["target", "list", "--installed"])
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean),
  );
  const missing = targets.filter((target) => !installed.has(target));
  if (missing.length === 0) return;
  log(`Adding Rust targets: ${missing.join(", ")}`);
  run("rustup", ["target", "add", ...missing]);
}

function ensureCargoXwin() {
  if (commandExists("cargo-xwin")) return;
  log("Installing cargo-xwin (Windows cross-compile runner)");
  run("cargo", ["install", "--locked", "cargo-xwin"]);
}

function macTarget() {
  if (universalMac) return "universal-apple-darwin";
  if (process.arch === "arm64") return "aarch64-apple-darwin";
  return "x86_64-apple-darwin";
}

function setupMac() {
  ensureRustTargets(
    universalMac ? ["aarch64-apple-darwin", "x86_64-apple-darwin"] : [macTarget()],
  );
}

function setupWindows() {
  if (isWin) {
    ensureRustTargets(["x86_64-pc-windows-msvc"]);
    return;
  }
  if (!isMac && !isLinux) {
    fail(`Windows packaging is not supported on ${host}`);
  }
  if (isMac) {
    ensureBrewPackages(["nsis", "llvm", "lld"]);
  } else if (!commandExists("makensis") || !commandExists("llvm-rc")) {
    fail("Install nsis, lld, and llvm (for llvm-rc) before packaging Windows from Linux.");
  }
  if (isMac && !llvmBinDir()) {
    fail("llvm-rc was not found after installing LLVM. Check `brew --prefix llvm`.");
  }
  if (isMac && !commandExists("lld-link") && !lldBinDir()) {
    fail("lld-link was not found after installing lld. Check `brew --prefix lld`.");
  }
  if (!commandExists("makensis")) {
    fail("makensis was not found after installing NSIS.");
  }
  ensureRustTargets(["x86_64-pc-windows-msvc"]);
  ensureCargoXwin();
  mkdirSync(xwinCacheDir, { recursive: true });
}

function setup() {
  ensureNodeDeps();
  if (wantMac && isMac) setupMac();
  if (wantWin) setupWindows();
}

function buildMac() {
  const target = macTarget();
  log(`Building macOS (${target})`);
  run(tauriBin(), ["build", "--ci", "--no-sign", "--target", target, "--bundles", "dmg,app"], {
    env: withBuildEnv(),
  });
}

function buildWindows() {
  log("Building Windows (x86_64 NSIS)");
  const extra = isWin
    ? ["--target", "x86_64-pc-windows-msvc", "--bundles", "nsis"]
    : ["--runner", "cargo-xwin", "--target", "x86_64-pc-windows-msvc", "--bundles", "nsis"];
  if (!isWin) {
    console.log(
      "First Windows cross-compile downloads the Windows SDK into src-tauri/.xwin-cache (can take a while).",
    );
  }
  run(tauriBin(), ["build", "--ci", "--no-sign", ...extra], { env: withBuildEnv() });
}

function walk(dir, files = []) {
  if (!existsSync(dir)) return files;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full, files);
    else files.push(full);
  }
  return files;
}

function cargoTargetDirs() {
  const dirs = [];
  const fromEnv = process.env.CARGO_TARGET_DIR;
  if (fromEnv) dirs.push(path.resolve(fromEnv));
  dirs.push(path.join(root, "src-tauri", "target"));
  dirs.push(path.join(root, "target"));
  return [...new Set(dirs)];
}

function collectArtifacts() {
  log("Collecting artifacts into ./release");
  mkdirSync(releaseDir, { recursive: true });
  const artifacts = cargoTargetDirs()
    .flatMap((dir) => walk(dir))
    .filter((file) => {
      const name = file.toLowerCase();
      return name.endsWith(".dmg") || name.endsWith("-setup.exe") || name.endsWith(".msi");
    });

  if (artifacts.length === 0) {
    fail(`Build finished but no installers were found in: ${cargoTargetDirs().join(", ")}`);
  }

  const copied = [];
  const seen = new Set();
  for (const file of artifacts) {
    const name = path.basename(file);
    if (seen.has(name)) continue;
    seen.add(name);
    const dest = path.join(releaseDir, name);
    cpSync(file, dest);
    const relative = path.relative(root, file);
    copied.push({
      file: name,
      bytes: statSync(dest).size,
      source: relative.startsWith("..") ? file : relative,
    });
  }

  const manifest = {
    version: JSON.parse(readFileSync(path.join(root, "src-tauri", "tauri.conf.json"), "utf8")).version,
    builtAt: new Date().toISOString(),
    host: `${host}-${process.arch}`,
    artifacts: copied,
  };
  writeFileSync(path.join(releaseDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);

  console.log("\nDesktop packages:");
  for (const item of copied) {
    const mb = (item.bytes / (1024 * 1024)).toFixed(1);
    console.log(`  ${path.join("release", item.file)}  (${mb} MB)`);
  }
}

if (flags.has("--help") || flags.has("-h")) {
  console.log(`Usage: node scripts/package.mjs [--mac] [--win] [--universal] [--setup] [--skip-setup]

  no flags       package macOS + Windows
  --mac          macOS .dmg only
  --win          Windows NSIS installer only
  --universal    macOS universal (Apple Silicon + Intel)
  --setup        install toolchains only
  --skip-setup   skip toolchain checks`);
  process.exit(0);
}

log("RWKV Agent desktop packaging");
if (!skipSetup) {
  try {
    setup();
  } catch (error) {
    fail(error instanceof Error ? error.message : String(error));
  }
}
if (setupOnly) {
  log("Packaging toolchains are ready.");
  process.exit(0);
}

rmSync(releaseDir, { recursive: true, force: true });
mkdirSync(releaseDir, { recursive: true });

const failed = [];
if (wantMac) {
  if (isMac) {
    try {
      buildMac();
    } catch (error) {
      console.error(error instanceof Error ? error.message : error);
      failed.push("macOS");
    }
  } else {
    console.warn("macOS packages can only be built on a Mac. Windows packaging will continue.");
  }
}

if (wantWin) {
  try {
    buildWindows();
  } catch (error) {
    console.error(error instanceof Error ? error.message : error);
    failed.push("Windows");
  }
}

try {
  collectArtifacts();
} catch (error) {
  if (failed.length === 0) {
    fail(error instanceof Error ? error.message : String(error));
  }
  console.error(error instanceof Error ? error.message : error);
}

if (failed.length > 0) {
  fail(`Packaging finished with errors: ${failed.join(", ")}`);
}

log("Done.");
