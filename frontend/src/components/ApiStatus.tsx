import { useEffect, useState } from "react";
import { checkHealth } from "@/lib/api";

export function ApiStatus() {
  const [connected, setConnected] = useState<boolean | null>(null);

  useEffect(() => {
    let active = true;

    async function poll() {
      const ok = await checkHealth();
      if (active) setConnected(ok);
    }

    poll();
    const id = setInterval(poll, 30_000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  const apiBase = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground">
      <span className="truncate max-w-[260px]">API: {apiBase}</span>
      <span
        className={`inline-block h-2 w-2 rounded-full ${
          connected === null
            ? "bg-muted-foreground"
            : connected
              ? "bg-green-500"
              : "bg-destructive"
        }`}
      />
      <span>
        {connected === null
          ? "Checking..."
          : connected
            ? "Connected"
            : "Disconnected"}
      </span>
    </div>
  );
}
