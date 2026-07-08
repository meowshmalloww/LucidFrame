import type { Metadata } from "next";
import "@/styles/globals.css";

export const metadata: Metadata = {
  title: "LucidFrame — Step Into the Dream",
  description: "AI art installation that hallucinates a walkable 3D world from a single image.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-void text-gray-200 antialiased">{children}</body>
    </html>
  );
}
