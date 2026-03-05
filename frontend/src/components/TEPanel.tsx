import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Search } from "lucide-react";
import { TE_COLUMNS, displayName, NUM_TE } from "@/lib/constants";

export interface TEState {
  enabled: boolean[];
  values: number[];
}

export function createInitialTEState(): TEState {
  return {
    enabled: Array(NUM_TE).fill(false),
    values: Array(NUM_TE).fill(1.0),
  };
}

interface TEPanelProps {
  state: TEState;
  onChange: (state: TEState) => void;
}

export function TEPanel({ state, onChange }: TEPanelProps) {
  const [search, setSearch] = useState("");

  const filtered = TE_COLUMNS.map((col, i) => ({ col, i })).filter(({ col }) =>
    displayName(col).toLowerCase().includes(search.toLowerCase())
  );

  const enabledCount = state.enabled.filter(Boolean).length;

  function toggleEnabled(index: number) {
    const next = { ...state, enabled: [...state.enabled] };
    next.enabled[index] = !next.enabled[index];
    onChange(next);
  }

  function setValue(index: number, value: number) {
    const next = { ...state, values: [...state.values] };
    next.values[index] = value;
    onChange(next);
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <Label className="text-sm font-medium">TE Targets</Label>
        <span className="text-xs text-muted-foreground">
          {enabledCount} / {NUM_TE} active
        </span>
      </div>

      <div className="relative">
        <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Search cell types..."
          className="pl-9 h-9"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      <ScrollArea className="h-[280px] rounded-md border p-2">
        <div className="space-y-1">
          {filtered.map(({ col, i }) => (
            <div
              key={col}
              className="flex items-center gap-3 rounded-md px-2 py-1.5 hover:bg-muted/50"
            >
              <Switch
                checked={state.enabled[i]}
                onCheckedChange={() => toggleEnabled(i)}
                className="scale-75"
              />
              <span className="text-sm min-w-[140px] truncate" title={displayName(col)}>
                {displayName(col)}
              </span>
              <Slider
                value={[state.values[i]]}
                onValueChange={([v]) => setValue(i, v)}
                min={0}
                max={5}
                step={0.1}
                disabled={!state.enabled[i]}
                className="flex-1"
              />
              <span className="text-xs text-muted-foreground tabular-nums w-8 text-right">
                {state.values[i].toFixed(1)}
              </span>
            </div>
          ))}
          {filtered.length === 0 && (
            <p className="text-sm text-muted-foreground text-center py-4">
              No matching cell types
            </p>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
