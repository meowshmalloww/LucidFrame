"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  {
    href: "/",
    label: "Create",
    icon: <svg viewBox="0 0 24 24" aria-hidden><path d="M12 4v16M4 12h16" /></svg>,
  },
  {
    href: "/gallery",
    label: "Library",
    icon: <svg viewBox="0 0 24 24" aria-hidden><rect x="4" y="5" width="16" height="14" rx="1" /><path d="m7 16 4-4 3 3 2-2 2 3" /><circle cx="9" cy="9" r="1" /></svg>,
  },
  {
    href: "/settings",
    label: "Settings",
    icon: <svg viewBox="0 0 24 24" aria-hidden><path d="M4 7h10M18 7h2M4 17h2M10 17h10M14 4v6M6 14v6" /></svg>,
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex h-full w-[68px] shrink-0 flex-col border-r border-[#2d2d29] bg-[#1b1b18] text-[#f5f4ed] lg:w-[216px]">
      <Link href="/" className="flex h-[72px] items-center gap-3 border-b border-[#2d2d29] px-5 lg:px-6">
        <span className="grid h-7 w-7 shrink-0 place-items-center rounded-[7px] border border-[#77766e] text-[11px] font-semibold tracking-tight">LF</span>
        <span className="hidden text-sm font-semibold tracking-[-0.02em] lg:block">LucidFrame</span>
      </Link>
      <nav aria-label="Primary navigation" className="flex flex-1 flex-col gap-1 p-3 lg:p-4">
        {navItems.map((item) => {
          const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              aria-current={active ? "page" : undefined}
              title={item.label}
              className={[
                "flex h-11 items-center justify-center gap-3 rounded-[9px] px-3 text-sm transition lg:justify-start",
                active ? "bg-[#f1f0e8] text-[#191916]" : "text-[#a8a79f] hover:bg-[#292925] hover:text-white",
              ].join(" ")}
            >
              <span className="h-[19px] w-[19px] shrink-0 [&>svg]:h-full [&>svg]:w-full [&>svg]:fill-none [&>svg]:stroke-current [&>svg]:stroke-[1.6]">
                {item.icon}
              </span>
              <span className="hidden lg:block">{item.label}</span>
            </Link>
          );
        })}
      </nav>
      <div className="border-t border-[#2d2d29] p-4 text-[11px] leading-relaxed text-[#77766e]">
        <span className="hidden lg:block">Local-first spatial studio</span>
        <span className="block text-center lg:hidden">v1</span>
      </div>
    </aside>
  );
}
