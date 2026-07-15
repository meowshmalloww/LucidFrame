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
    <aside className="flex h-full w-[68px] shrink-0 flex-col border-r border-[#30302c] bg-[#1b1b18] text-[#f5f4ed] lg:w-[204px]">
      <Link href="/" aria-label="LucidFrame home" className="flex h-[68px] items-center gap-3 border-b border-[#30302c] px-[21px] lg:px-5">
        <svg viewBox="0 0 28 28" className="h-[26px] w-[26px] shrink-0 fill-none stroke-current stroke-[1.5]" aria-hidden>
          <path d="M5 8V5h7M16 5h7v7M23 16v7h-7M12 23H5v-7" />
          <path d="m9 18 4-5 3 3 2-2 2 4" />
        </svg>
        <span className="hidden text-[15px] font-medium tracking-[-0.025em] lg:block">LucidFrame</span>
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
    </aside>
  );
}
