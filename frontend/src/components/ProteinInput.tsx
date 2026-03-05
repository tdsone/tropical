import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { validateProtein } from "@/lib/constants";

interface ProteinInputProps {
  value: string;
  onChange: (value: string) => void;
}

export function ProteinInput({ value, onChange }: ProteinInputProps) {
  const cleaned = value.toUpperCase().replace(/\s/g, "");
  const error = cleaned.length > 0 ? validateProtein(cleaned) : null;

  return (
    <div className="space-y-2">
      <Label htmlFor="protein-seq" className="text-sm font-medium">
        Protein Sequence
      </Label>
      <Textarea
        id="protein-seq"
        placeholder="MVKLTNF..."
        className="font-mono text-sm min-h-[120px] resize-y"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        spellCheck={false}
      />
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{error ? <span className="text-destructive">{error}</span> : "Valid amino acids: A C D E F G H I K L M N P Q R S T V W Y X"}</span>
        <span>{cleaned.length} aa</span>
      </div>
    </div>
  );
}
