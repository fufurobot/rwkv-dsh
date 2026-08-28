import {
  Bot,
  Calculator,
  Check,
  ChevronRight,
  FileText,
  FolderSearch,
  Loader2,
  Search,
  X,
} from "lucide-react";

import { cn } from "../../lib/cn";
import { workersOfTool } from "../../lib/workers";
import type { ToolTrace } from "../../types";

const ICONS: Record<string, typeof FileText> = {
  list_files: FolderSearch,
  read_file: FileText,
  search_text: Search,
  calculator: Calculator,
  run_agents: Bot,
};

type Props = {
  tool: ToolTrace
  onToggle: () => void;
};

export function ToolChip({ tool, onToggle }: Props) {
  const Icon = ICONS[tool.name] ?? FolderSearch;
  const running = tool.ok === undefined;
  const failed = tool.ok === false;
  const workers = workersOfTool(tool);
  const isAgents = tool.name === "run_agents";

  return (
    <div className="min-w-0">
      <button
        type="button"
        onClick={onToggle}
        className="flex max-w-full items-center gap-1.5 rounded-lg py-0.5 text-left text-[13px] text-tool-foreground"
      >
        {!isAgents && (
          <ChevronRight className={cn("h-3.5 w-3.5 shrink-0 transition-transform", tool.open && "rotate-90")} />
        )}
        <Icon className="h-3.5 w-3.5 shrink-0" strokeWidth={1.8} />
        <span className="truncate font-medium">
          {isAgents ? `${workers.length || "多"} 个子任务` : tool.name}
        </span>
        {running ? (
          <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-muted-foreground" />
        ) : failed ? (
          <X className="h-3.5 w-3.5 shrink-0 text-destructive" />
        ) : (
          <Check className="h-3.5 w-3.5 shrink-0 text-success" />
        )}
      </button>
      {!isAgents && tool.open && (
        <pre className="mt-1 max-h-48 overflow-auto rounded-lg bg-code-background px-3 py-2 font-mono text-[12px] leading-5 text-muted-foreground">
          {JSON.stringify({ arguments: tool.arguments, result: tool.payload }, null, 2)}
        </pre>
      )}
    </div>
  );
}
