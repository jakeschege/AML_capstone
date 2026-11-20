import type { Metadata } from "next";
import { Inter, Poppins } from "next/font/google";
import "./globals.css";
import { AppShell } from "@/components/AppShell";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-poppins",
  display: "swap",
});

export const metadata: Metadata = {
  title: "FinAccess Coping Notebook",
  description:
    "Weighted patterns of coping among shock-exposed farming households in the FinAccess 2024 analysis sample.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  // Runtime env (Render) — Next bakes NEXT_PUBLIC_* at build time, which Blueprints cannot set via dockerBuildArgs.
  const apiBase =
    process.env.NEXT_PUBLIC_API_BASE?.trim() ||
    process.env.API_BASE?.trim() ||
    "http://localhost:8000";

  return (
    <html lang="en" className={`${inter.variable} ${poppins.variable}`}>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `window.__API_BASE__=${JSON.stringify(apiBase.replace(/\/$/, ""))};`,
          }}
        />
      </head>
      <body className="antialiased">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
