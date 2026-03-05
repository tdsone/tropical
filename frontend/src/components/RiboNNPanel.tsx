import { useEffect, useState } from "react";
import { Loader2, FlaskConical } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { predictRiboNN, type RiboNNPrediction } from "@/lib/api";

interface RiboNNPanelProps {
  sequence: string | null;
  proteinSeq: string | null;
}

export function RiboNNPanel({ sequence, proteinSeq }: RiboNNPanelProps) {
  const [prediction, setPrediction] = useState<RiboNNPrediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sequence) {
      setPrediction(null);
      setError(null);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);
    setPrediction(null);

    predictRiboNN(sequence, proteinSeq).then((result) => {
      if (!cancelled) {
        setPrediction(result);
        setLoading(false);
      }
    }).catch((err) => {
      if (!cancelled) {
        setError(err instanceof Error ? err.message : "RiboNN prediction failed");
        setLoading(false);
      }
    });

    return () => { cancelled = true; };
  }, [sequence]);

  if (!sequence && !loading) {
    return null;
  }

  const sorted = prediction
    ? [...prediction.columns.map((col, i) => ({ col, val: prediction.values[i] }))]
        .sort((a, b) => b.val - a.val)
    : [];

  const minVal = sorted.length ? sorted[sorted.length - 1].val : 0;
  const maxVal = sorted.length ? sorted[0].val : 1;
  const range = maxVal - minVal || 1;

  return (
    <div className="space-y-3 mt-4">
      <div className="flex items-center gap-2 flex-wrap">
        <FlaskConical className="h-4 w-4 text-muted-foreground" />
        <h3 className="text-sm font-medium">RiboNN — Predicted Translation Efficiency</h3>
        {loading && <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />}
        {prediction && (
          <span className="text-xs text-muted-foreground">
            5′UTR {prediction.utr5_size} nt · CDS {prediction.cds_size} nt
          </span>
        )}
      </div>

      {error && (
        <p className="text-xs text-destructive">{error}</p>
      )}

      {prediction && (
        <ScrollArea className="h-64 rounded-md border bg-muted/20 p-3">
          <div className="space-y-1">
            {sorted.map(({ col, val }, i) => {
              const barPct = ((val - minVal) / range) * 100;
              const label = col.replace(/^TE_/, "");
              const isTop = i < 5;
              const isBottom = i >= sorted.length - 5;
              return (
                <div key={col} className="flex items-center gap-2 text-xs">
                  <span
                    className={
                      "w-44 shrink-0 truncate " +
                      (isTop ? "font-medium text-foreground" : isBottom ? "text-muted-foreground" : "")
                    }
                    title={label}
                  >
                    {label}
                  </span>
                  <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                    <div
                      className={
                        "h-full rounded-full transition-all " +
                        (isTop ? "bg-primary" : isBottom ? "bg-muted-foreground/40" : "bg-primary/50")
                      }
                      style={{ width: `${barPct}%` }}
                    />
                  </div>
                  <span className="w-12 text-right tabular-nums text-muted-foreground">
                    {val.toFixed(2)}
                  </span>
                  {i === 0 && (
                    <Badge variant="secondary" className="text-[10px] px-1 py-0 h-4">top</Badge>
                  )}
                </div>
              );
            })}
          </div>
        </ScrollArea>
      )}
    </div>
  );
}
