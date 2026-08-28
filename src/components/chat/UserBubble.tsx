type Props = {
  content: string;
};

export function UserBubble({ content }: Props) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[min(36rem,85%)] rounded-2xl bg-user-bubble px-4 py-2.5 text-[15px] leading-6 wrap-anywhere text-user-bubble-foreground whitespace-pre-wrap">
        {content}
      </div>
    </div>
  );
}
