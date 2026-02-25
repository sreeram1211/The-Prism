import { cn } from "@/lib/utils";

type BadgeVariant = "brand" | "emerald" | "amber" | "rose" | "cyan" | "muted";

const variantClasses: Record<BadgeVariant, string> = {
  brand:   "bg-[--brand]/15 text-[--brand] border-[--brand]/30",
  emerald: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  amber:   "bg-amber-500/15 text-amber-400 border-amber-500/30",
  rose:    "bg-rose-500/15 text-rose-400 border-rose-500/30",
  cyan:    "bg-cyan-500/15 text-cyan-400 border-cyan-500/30",
  muted:   "bg-[--muted] text-[--muted-fg] border-[--border]",
};

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
}

export function Badge({ className, variant = "muted", ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs font-medium",
        variantClasses[variant],
        className,
      )}
      {...props}
    />
  );
}
