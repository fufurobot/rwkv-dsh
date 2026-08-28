export type ChatErrorPresentation = {
  title: string;
  description: string;
  retryable: boolean;
  stopped: boolean;
};

export function presentChatError(error: string): ChatErrorPresentation {
  const normalized = error.trim().toLowerCase();

  if (normalized === "stopped" || normalized === "cancelled") {
    return {
      title: "已停止生成",
      description: "这次回复已停止，没有继续执行后续操作。",
      retryable: true,
      stopped: true,
    };
  }

  if (
    normalized.includes("g1i function call") ||
    normalized.includes("invalid g1i tool call") ||
    normalized.includes("malformed tool call")
  ) {
    return {
      title: "工具调用格式不完整",
      description:
        "模型返回了无法识别的工具调用，本轮已安全停止，未执行这条无效调用。你可以重新尝试。",
      retryable: true,
      stopped: false,
    };
  }

  if (normalized.includes("工具协议") || normalized.includes("history protocol")) {
    return {
      title: "历史工具协议已隐藏",
      description: "这条旧回答包含未完成的内部工具格式，原始协议不会展示。重新发送即可按新流程分析。",
      retryable: true,
      stopped: false,
    };
  }

  if (
    normalized.includes("response has no choices") ||
    normalized.includes("response contains no text") ||
    normalized.includes("empty response")
  ) {
    return {
      title: "模型没有返回有效内容",
      description: "模型服务返回了空响应。请重试；如果持续发生，请检查模型服务状态。",
      retryable: true,
      stopped: false,
    };
  }

  if (
    normalized.includes("error sending request") ||
    normalized.includes("connection refused") ||
    normalized.includes("timed out") ||
    normalized.includes("timeout")
  ) {
    return {
      title: "无法连接模型服务",
      description: "请求没有成功到达模型服务。请检查服务是否运行以及地址配置是否正确。",
      retryable: true,
      stopped: false,
    };
  }

  if (/^http\s+\d+/.test(normalized)) {
    return {
      title: "模型服务返回错误",
      description: "模型服务暂时无法完成请求。你可以重试，或检查服务端日志。",
      retryable: true,
      stopped: false,
    };
  }

  if (normalized.includes("step limit")) {
    return {
      title: "本轮执行步骤已用尽",
      description: "任务在限定步骤内没有完成。请重试，或把问题拆成更小的范围。",
      retryable: true,
      stopped: false,
    };
  }

  if (normalized.includes("subagent") || normalized.includes("parallel inspection")) {
    return {
      title: "子任务执行失败",
      description: "并行检查没有生成可用结果。请重试，或减少一次要分析的内容。",
      retryable: true,
      stopped: false,
    };
  }

  return {
    title: "本轮回复失败",
    description: "这次请求没有完成。你可以重新尝试，并在需要时查看技术详情。",
    retryable: true,
    stopped: false,
  };
}
