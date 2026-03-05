import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";

interface GenerationParamsProps {
  temperature: number;
  topK: number | null;
  maxLength: number;
  onTemperatureChange: (v: number) => void;
  onTopKChange: (v: number | null) => void;
  onMaxLengthChange: (v: number) => void;
}

export function GenerationParams({
  temperature,
  topK,
  maxLength,
  onTemperatureChange,
  onTopKChange,
  onMaxLengthChange,
}: GenerationParamsProps) {
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label className="text-sm font-medium">Temperature</Label>
          <span className="text-sm text-muted-foreground tabular-nums">
            {temperature.toFixed(2)}
          </span>
        </div>
        <Slider
          value={[temperature]}
          onValueChange={([v]) => onTemperatureChange(v)}
          min={0.1}
          max={2.0}
          step={0.05}
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="top-k" className="text-sm font-medium">
          Top-K
        </Label>
        <Input
          id="top-k"
          type="number"
          placeholder="off"
          min={1}
          max={100}
          value={topK ?? ""}
          onChange={(e) => {
            const val = e.target.value;
            onTopKChange(val === "" ? null : parseInt(val, 10));
          }}
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="max-length" className="text-sm font-medium">
          Max Length
        </Label>
        <Input
          id="max-length"
          type="number"
          min={64}
          max={4096}
          step={64}
          value={maxLength}
          onChange={(e) => onMaxLengthChange(parseInt(e.target.value, 10) || 2048)}
        />
      </div>
    </div>
  );
}
