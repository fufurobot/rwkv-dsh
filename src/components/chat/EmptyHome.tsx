import { Bot, Bug, Hammer, ScanSearch, Telescope, type LucideIcon } from "lucide-react";

const SUGGESTIONS: {
  id: string;
  label: string;
  prompt: string;
  icon: LucideIcon;
  iconClass: string;
}[] = [
  {
    id: "explore",
    label: "探索项目",
    prompt: "先列出这个项目的文件结构，再用一两段话说明它是做什么的。",
    icon: Telescope,
    iconClass: "text-primary",
  },
  {
    id: "build",
    label: "怎么跑",
    prompt: "读一下 README，总结这个项目怎么安装和启动。",
    icon: Hammer,
    iconClass: "text-thinking-foreground",
  },
  {
    id: "review",
    label: "审查代码",
    prompt: "在仓库里搜索 TODO 或 FIXME，指出最急的几处。",
    icon: ScanSearch,
    iconClass: "text-success",
  },
  {
    id: "fix",
    label: "试一下工具",
    prompt: "4827×391 等于多少？用计算器算，不要口算。",
    icon: Bug,
    iconClass: "text-tool-foreground",
  },
];

type Props = {
  projectName: string;
  onSelect: (prompt: string) => void;
};

export function EmptyHome({ projectName, onSelect }: Props) {
  const name = projectName.trim() || "这个工作区";

  return (
    <section className="flex h-full min-h-0 w-full items-center justify-center overflow-y-auto px-5 pt-10 pb-36 sm:px-8">
      <div className="flex w-full max-w-3xl flex-col items-center">
        <Bot aria-hidden="true" className="mb-7 size-12 text-muted-foreground/55" strokeWidth={1.5} />
        <h1 className="max-w-2xl text-center text-2xl leading-snug font-normal tracking-tight text-foreground sm:text-[1.75rem]">
          在{" "}
          <span className="underline decoration-foreground/35 decoration-1 underline-offset-4">
            {name}
          </span>{" "}
          里，有什么可以帮忙的？
        </h1>
        <div className="mt-10 grid w-full grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {SUGGESTIONS.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelect(item.prompt)}
                className="group flex min-h-28 flex-col items-start justify-between rounded-2xl border border-suggestion-border bg-transparent p-4 text-left text-foreground transition-[background-color,border-color,transform] duration-150 hover:border-muted-foreground/45 hover:bg-foreground/2.5 active:translate-y-px"
              >
                <Icon className={`size-4.5 ${item.iconClass}`} strokeWidth={1.8} />
                <span className="max-w-[15ch] text-sm leading-5 font-medium text-foreground/90">
                  {item.label}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </section>
  );
}
