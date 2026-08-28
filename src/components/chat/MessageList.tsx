import { useEffect, useRef } from "react";

import type { ChatMessage } from "../../types";
import { AssistantTurn } from "./AssistantTurn";
import { UserBubble } from "./UserBubble";

type Props = {
  messages: ChatMessage[];
  status: string;
  onToggleTool: (messageId: string, toolId: string) => void;
  onRetry: (messageId: string) => void;
};

export function MessageList({ messages, status, onToggleTool, onRetry }: Props) {
  const scroller = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = scroller.current;
    if (!node) return;
    node.scrollTo({ top: node.scrollHeight });
  }, [messages, status]);

  return (
    <div ref={scroller} className="min-h-0 flex-1 overflow-y-auto px-4 pt-6 pb-36">
      <div className="mx-auto flex max-w-3xl flex-col gap-6">
        {messages.map((message) =>
          message.role === "user" ? (
            <UserBubble key={message.id} content={message.content} />
          ) : (
            <AssistantTurn
              key={message.id}
              message={message}
              status={status}
              onToggleTool={(toolId) => onToggleTool(message.id, toolId)}
              onRetry={message.error ? () => onRetry(message.id) : undefined}
            />
          ),
        )}
      </div>
    </div>
  );
}
