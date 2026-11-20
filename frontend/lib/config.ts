declare global {
  interface Window {
    __API_BASE__?: string;
  }
}

/** API origin for browser + server. Prefer runtime inject (Render) over build-time bake. */
export function getApiBase(): string {
  if (typeof window !== "undefined" && window.__API_BASE__) {
    return window.__API_BASE__.replace(/\/$/, "");
  }
  const fromEnv =
    process.env.NEXT_PUBLIC_API_BASE?.trim() ||
    process.env.API_BASE?.trim() ||
    "http://localhost:8000";
  return fromEnv.replace(/\/$/, "");
}
