import path from "path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Keep tracing inside this app when parent directories also have lockfiles.
  outputFileTracingRoot: path.join(__dirname),
};

export default nextConfig;
