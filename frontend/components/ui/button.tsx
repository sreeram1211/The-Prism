import { cn } from "@/lib/utils";

type ButtonVariant = "primary" | "secondary" | "ghost";

const variantClasses: Record<ButtonVariant, string> = {
  primary:
    "bg-[--brand] hover:bg-[--brand]/90 text-white shadow-lg shadow-[--brand]/20",
  secondary:
    "bg-[--muted] hover:bg-[--card-hover] text-[--foreground] border border-[--border]",
  ghost:
    "hover:bg-[--muted] text-[--muted-fg] hover:text-[--foreground]",
};

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  loading?: boolean;
}

export function Button({
  className,
  variant = "primary",
  loading,
  children,
  disabled,
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium",
        "transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[--brand]",
        "disabled:pointer-events-none disabled:opacity-50",
        variantClasses[variant],
        className,
      )}
      disabled={disabled || loading}
      {...props}
    >
      {loading && (
        <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
      )}
      {children}
    </button>
  );
}
