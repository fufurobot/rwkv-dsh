import { useMemo, useState } from "react";
import {
  Activity,
  Folder,
  FolderOpen,
  FolderPlus,
  Pin,
  Search,
  Settings,
  SquarePen,
  Trash2,
  X,
} from "lucide-react";

import { cn } from "../lib/cn";
import type { SidebarData } from "../types";

type Props = {
  collapsed: boolean;
  busy: boolean;
  data: SidebarData;
  onNewSession: (projectId?: string) => void;
  onAddProject: () => void;
  onOpenProject: (id: string) => void;
  onOpenSession: (id: string) => void;
  onPinSession: (id: string) => void;
  onDeleteSession: (id: string) => void;
  onOpenSettings: () => void;
  onOpenCallLogs: () => void;
  callLogCount: number;
  callLogsOpen: boolean;
};

export function AppSidebar({
  collapsed,
  busy,
  data,
  onNewSession,
  onAddProject,
  onOpenProject,
  onOpenSession,
  onPinSession,
  onDeleteSession,
  onOpenSettings,
  onOpenCallLogs,
  callLogCount,
  callLogsOpen,
}: Props) {
  const [query, setQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [collapsedProjects, setCollapsedProjects] = useState<Record<string, boolean>>({});

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return data.projects;
    return data.projects
      .map((project) => ({
        ...project,
        sessions: project.sessions.filter(
          (session) =>
            session.title.toLowerCase().includes(needle) ||
            project.name.toLowerCase().includes(needle),
        ),
      }))
      .filter(
        (project) =>
          project.sessions.length > 0 || project.name.toLowerCase().includes(needle),
      );
  }, [data.projects, query]);

  return (
    <aside
      aria-hidden={collapsed || undefined}
      className={cn(
        "h-full shrink-0 overflow-hidden border-r border-sidebar-border/60 bg-sidebar text-sidebar-foreground transition-[width] duration-200 ease-out",
        collapsed ? "w-0 border-r-0" : "w-(--sidebar-width,275px)",
      )}
    >
      <div className="flex h-full w-(--sidebar-width,275px) shrink-0 flex-col">
        <div className="px-2 pb-2 pt-2">
          <div className="flex h-9 items-center justify-between px-1">
            <h1 className="text-lg font-semibold tracking-tight text-foreground">RWKV</h1>
            <button
              type="button"
              onClick={() => {
                setSearchOpen((open) => !open);
                if (searchOpen) setQuery("");
              }}
              className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-sidebar-accent hover:text-foreground"
              aria-label={searchOpen ? "关闭搜索" : "搜索对话"}
            >
              {searchOpen ? <X className="h-4 w-4" /> : <Search className="h-4 w-4" />}
            </button>
          </div>
          {searchOpen && (
            <div className="relative mt-1">
              <Search className="pointer-events-none absolute top-1/2 left-2.5 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
              <input
                type="search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="搜索项目或对话"
                className="h-8 w-full rounded-lg border border-border/60 bg-background/20 pr-2 pl-8 text-sm shadow-none outline-none focus:ring-1 focus:ring-ring"
              />
            </div>
          )}
        </div>

        <nav className="space-y-0.5 px-2 pb-3">
          <button
            type="button"
            disabled={busy}
            onClick={() => onNewSession()}
            className="flex h-8 w-full items-center justify-start gap-2.5 rounded-lg px-1 text-sm font-normal text-foreground hover:bg-sidebar-accent hover:text-foreground disabled:opacity-50"
          >
            <SquarePen className="h-4 w-4" />
            <span>新对话</span>
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={onAddProject}
            className="flex h-8 w-full items-center justify-start gap-2.5 rounded-lg px-1 text-sm font-normal text-muted-foreground hover:bg-sidebar-accent hover:text-foreground disabled:opacity-50"
          >
            <FolderPlus className="h-4 w-4" />
            <span>添加项目</span>
          </button>
        </nav>

        <div className="px-4 pt-1 pb-1">
          <span className="text-xs font-medium text-muted-foreground">项目</span>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto py-1">
          {filtered.length === 0 ? (
            <p className="px-4 py-8 text-center text-sm text-muted-foreground">
              {data.projects.length === 0 ? "添加一个文件夹作为工作区。" : "没有匹配的对话。"}
            </p>
          ) : (
            <div className="space-y-1 px-2 pb-3">
              {filtered.map((project) => {
                const expanded =
                  Boolean(query.trim()) ||
                  collapsedProjects[project.id] !== true;
                return (
                  <section key={project.id}>
                    <div className="group/project relative flex h-8 min-w-0 items-center rounded-lg text-foreground transition-colors hover:bg-sidebar-accent/75 focus-within:bg-sidebar-accent">
                      <button
                        type="button"
                        title={project.path}
                        onClick={() => {
                          setCollapsedProjects((prev) => ({
                            ...prev,
                            [project.id]: expanded,
                          }));
                          onOpenProject(project.id);
                        }}
                        className="flex h-full w-full min-w-0 items-center gap-2 rounded-lg px-1 text-left text-sm"
                      >
                        {expanded ? (
                          <FolderOpen className="h-4 w-4 shrink-0" />
                        ) : (
                          <Folder className="h-4 w-4 shrink-0" />
                        )}
                        <span className="min-w-0 flex-1 truncate font-medium">{project.name}</span>
                      </button>
                      <div className="pointer-events-none absolute top-1/2 right-0.5 flex -translate-y-1/2 items-center rounded-md bg-sidebar-accent pl-1 opacity-0 transition-opacity group-hover/project:pointer-events-auto group-hover/project:opacity-100 group-focus-within/project:pointer-events-auto group-focus-within/project:opacity-100">
                        <button
                          type="button"
                          disabled={busy}
                          title="新建对话"
                          aria-label={`在 ${project.name} 新建对话`}
                          onClick={(event) => {
                            event.stopPropagation();
                            onNewSession(project.id);
                          }}
                          className="flex h-7 w-7 items-center justify-center rounded-md text-foreground hover:bg-foreground/10 disabled:opacity-50"
                        >
                          <SquarePen className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                    {expanded && (
                      <div className="mt-0.5 space-y-0.5">
                        {project.sessions.map((session) => {
                          const selected = session.id === data.activeSessionId;
                          return (
                            <div
                              key={session.id}
                              className={cn(
                                "group/task flex min-w-0 items-center rounded-md transition-colors",
                                selected
                                  ? "bg-sidebar-accent text-foreground"
                                  : "text-muted-foreground hover:bg-sidebar-accent/60 hover:text-foreground",
                              )}
                            >
                              <button
                                type="button"
                                onClick={() => onOpenSession(session.id)}
                                className="min-w-0 flex-1 truncate py-1.5 pr-2 pl-7 text-left text-sm"
                              >
                                {session.title}
                              </button>
                              <div className="flex shrink-0 opacity-0 group-hover/task:opacity-100 group-focus-within/task:opacity-100">
                                <button
                                  type="button"
                                  title={session.pinned ? "取消置顶" : "置顶"}
                                  onClick={() => onPinSession(session.id)}
                                  className="flex h-7 w-7 items-center justify-center text-muted-foreground hover:text-foreground"
                                >
                                  <Pin className={cn("h-3.5 w-3.5", session.pinned && "fill-current")} />
                                </button>
                                <button
                                  type="button"
                                  title="删除对话"
                                  disabled={busy}
                                  onClick={() => onDeleteSession(session.id)}
                                  className="flex h-7 w-7 items-center justify-center text-muted-foreground hover:text-destructive"
                                >
                                  <Trash2 className="h-3.5 w-3.5" />
                                </button>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </section>
                );
              })}
            </div>
          )}
        </div>

        <div className="border-t border-border/50 p-2">
          <button
            type="button"
            onClick={onOpenCallLogs}
            className={cn(
              "flex h-8 w-full items-center gap-2.5 rounded-lg px-1 text-sm transition-colors",
              callLogsOpen
                ? "bg-sidebar-accent text-foreground"
                : "text-muted-foreground hover:bg-sidebar-accent hover:text-foreground",
            )}
          >
            <Activity className="h-4 w-4" />
            <span className="min-w-0 flex-1 text-left">调用日志</span>
            {callLogCount > 0 ? (
              <span className="rounded-md bg-foreground/8 px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                {callLogCount > 999 ? "999+" : callLogCount}
              </span>
            ) : null}
          </button>
          <button
            type="button"
            onClick={onOpenSettings}
            className="flex h-8 w-full items-center gap-2.5 rounded-lg px-1 text-sm text-muted-foreground hover:bg-sidebar-accent hover:text-foreground"
          >
            <Settings className="h-4 w-4" />
            <span>设置</span>
          </button>
        </div>
      </div>
    </aside>
  );
}
