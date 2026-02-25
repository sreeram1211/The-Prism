"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const navItems = [
  { label: "Resolve", href: "/resolve", icon: "⬡" },
  { label: "Scan", href: "/scan", icon: "◈" },
  { label: "Generate", href: "/generate", icon: "⬟" },
  { label: "Monitor", href: "/monitor", icon: "◉" },
];

export function Nav() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 border-b border-[--border] bg-[--background]/80 backdrop-blur-md">
      <nav className="mx-auto flex max-w-6xl items-center justify-between px-6 py-3">
        {/* Brand */}
        <Link href="/" className="flex items-center gap-2 group">
          <span className="text-lg font-bold text-[--brand] group-hover:text-[--brand]/80 transition-colors">
            ◈ Prism
          </span>
          <span className="hidden text-xs text-[--muted-fg] sm:block">
            AI Behavioral Manifold
          </span>
        </Link>

        {/* Navigation links */}
        <ul className="flex items-center gap-1">
          {navItems.map((item) => {
            const active = pathname.startsWith(item.href);
            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  className={cn(
                    "flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm transition-all duration-150",
                    active
                      ? "bg-[--brand]/10 text-[--brand] border border-[--brand]/20"
                      : "text-[--muted-fg] hover:bg-[--muted] hover:text-[--foreground]",
                  )}
                >
                  <span className="text-xs">{item.icon}</span>
                  {item.label}
                </Link>
              </li>
            );
          })}
        </ul>

        {/* Status dot */}
        <div className="flex items-center gap-2 text-xs text-[--muted-fg]">
          <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="hidden sm:block">Phase 3</span>
        </div>
      </nav>
    </header>
  );
}
