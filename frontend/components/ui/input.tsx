import { cn } from "@/lib/utils";

type InputProps = React.InputHTMLAttributes<HTMLInputElement>;

export function Input({ className, ...props }: InputProps) {
  return (
    <input
      className={cn(
        "w-full rounded-lg border border-[--border] bg-[--muted] px-3 py-2 text-sm",
        "text-[--foreground] placeholder:text-[--muted-fg]",
        "focus:border-[--brand] focus:outline-none focus:ring-1 focus:ring-[--brand]",
        "transition-colors duration-150",
        className,
      )}
      {...props}
    />
  );
}
