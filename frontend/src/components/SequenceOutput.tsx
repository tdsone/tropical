import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Copy, Download, Dna } from "lucide-react";
import { useState } from "react";

interface SequenceOutputProps {
  sequence: string | null;
  loading: boolean;
  error: string | null;
}

export function SequenceOutput({ sequence, loading, error }: SequenceOutputProps) {
  const [copied, setCopied] = useState(false);

  function copyToClipboard() {
    if (!sequence) return;
    navigator.clipboard.writeText(sequence);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  function downloadFasta() {
    if (!sequence) return;
    const content = `>tropical_generated\n${sequence}\n`;
    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "generated_sequence.fasta";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="space-y-3 h-full flex flex-col">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium">Generated RNA Sequence</h3>
        {sequence && (
          <Badge variant="secondary">
            {sequence.length} nt
          </Badge>
        )}
      </div>

      <ScrollArea className="flex-1 min-h-[200px] rounded-md border bg-muted/30 p-4">
        {loading ? (
          <div className="flex items-center justify-center h-full min-h-[180px]">
            <div className="flex flex-col items-center gap-3 text-muted-foreground">
              <Dna className="h-8 w-8 animate-spin" />
              <span className="text-sm">Generating sequence...</span>
            </div>
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-full min-h-[180px]">
            <p className="text-sm text-destructive">{error}</p>
          </div>
        ) : sequence ? (
          <p className="font-mono text-sm break-all leading-relaxed whitespace-pre-wrap">
            {sequence}
          </p>
        ) : (
          <div className="flex items-center justify-center h-full min-h-[180px]">
            <p className="text-sm text-muted-foreground">
              Enter a protein sequence and click Generate to produce an mRNA sequence.
            </p>
          </div>
        )}
      </ScrollArea>

      {sequence && (
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={copyToClipboard}>
            <Copy className="h-4 w-4 mr-1.5" />
            {copied ? "Copied!" : "Copy"}
          </Button>
          <Button variant="outline" size="sm" onClick={downloadFasta}>
            <Download className="h-4 w-4 mr-1.5" />
            Download FASTA
          </Button>
        </div>
      )}
    </div>
  );
}
