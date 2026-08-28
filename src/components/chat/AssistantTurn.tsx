import { cjk } from "@streamdown/cjk";
import { code } from "@streamdown/code";
import { Streamdown } from "streamdown";

import { splitRoundSections } from "../../lib/rounds";
import type { ChatMessage } from "../../types";
import { AssistantError } from "./AssistantError";
import { RoundTimeline } from "./RoundTimeline";
import { ToolChip } from "./ToolChip";

type Props = {
  message: ChatMessage;
  status: string;
  onToggleTool: (toolId: string) => void;
  onRetry?: () => void;
};

export function AssistantTurn({ message, status, onToggleTool, onRetry }: Props) {
  const streaming = Boolean(message.streaming);
  const tools = message.tools.filter((tool) => tool.name !== "run_agents");
  const sections = splitRoundSections(message.content);
  const timed = sections.some((section) => section.heading);

  return (
    <div className="min-w-0 space-y-3">
      {tools.map((tool) => (
        <ToolChip key={tool.id} tool={tool} onToggle={() => onToggleTool(tool.id)} />
      ))}
      {timed ? (
        <RoundTimeline sections={sections} streaming={streaming} />
      ) : message.content ? (
        <div className="markdown-content min-w-0 max-w-full text-[15px] leading-7 wrap-anywhere text-foreground">
          <Streamdown
            mode={streaming ? "streaming" : "static"}
            plugins={{ code, cjk }}
            linkSafety={{ enabled: false }}
          >
            {message.content}
          </Streamdown>
        </div>
      ) : streaming ? (
        <p className="text-sm text-muted-foreground">{status === "idle" ? "正在思考…" : status}</p>
      ) : null}
      {message.error && <AssistantError error={message.error} onRetry={onRetry} />}
    </div>
  );
}
