import { cjk } from "@streamdown/cjk";
import { code } from "@streamdown/code";
import { Loader2 } from "lucide-react";
import { Streamdown } from "streamdown";

import { cn } from "../../lib/cn";
import type { RoundSection } from "../../lib/rounds";

type Props = {
  sections: RoundSection[];
  streaming: boolean;
};

export function RoundTimeline({ sections, streaming }: Props) {
  return (
    <ol className="min-w-0">
      {sections.map((section, offset) => {
        const last = offset === sections.length - 1;
        const live = streaming && last;
        return (
          <li key={`${section.index}-${section.heading}-${offset}`} className="relative flex gap-3">
            <div className="flex w-4 shrink-0 flex-col items-center">
              <span
                className={cn(
                  "mt-1.5 size-2.5 rounded-full border-2 bg-background",
                  live
                    ? "border-tool-foreground"
                    : "border-foreground/35",
                )}
              />
              {!last ? <span className="w-px flex-1 bg-border" /> : <span className="w-px flex-1" />}
            </div>
            <article className={cn("min-w-0 flex-1", last ? "pb-1" : "pb-7")}>
              {section.heading ? (
                <header className="mb-2 flex items-baseline gap-2">
                  <h2 className="text-[13px] font-medium tracking-wide text-tool-foreground">
                    {section.heading}
                  </h2>
                  {live ? (
                    <Loader2 className="h-3 w-3 shrink-0 animate-spin text-muted-foreground" />
                  ) : null}
                </header>
              ) : null}
              {section.body ? (
                <div className="markdown-content min-w-0 max-w-full text-[15px] leading-7 wrap-anywhere text-foreground">
                  <Streamdown
                    mode={live ? "streaming" : "static"}
                    plugins={{ code, cjk }}
                    linkSafety={{ enabled: false }}
                  >
                    {section.body}
                  </Streamdown>
                </div>
              ) : live ? (
                <p className="text-sm text-muted-foreground">进行中…</p>
              ) : null}
            </article>
          </li>
        );
      })}
    </ol>
  );
}
