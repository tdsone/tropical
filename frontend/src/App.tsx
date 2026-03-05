import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Play, Dna } from "lucide-react";

import { ProteinInput } from "@/components/ProteinInput";
import { GenerationParams } from "@/components/GenerationParams";
import { TEPanel, createInitialTEState } from "@/components/TEPanel";
import type { TEState } from "@/components/TEPanel";
import { SequenceOutput } from "@/components/SequenceOutput";
import { ApiStatus } from "@/components/ApiStatus";
import { RiboNNPanel } from "@/components/RiboNNPanel";
import { generateSequence } from "@/lib/api";
import { validateProtein } from "@/lib/constants";

export default function App() {
  const [proteinSeq, setProteinSeq] = useState("");
  const [temperature, setTemperature] = useState(1.0);
  const [topK, setTopK] = useState<number | null>(null);
  const [maxLength, setMaxLength] = useState(2048);
  const [teState, setTEState] = useState<TEState>(createInitialTEState);

  const [sequence, setSequence] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const cleaned = proteinSeq.toUpperCase().replace(/\s/g, "");
  const validationError = cleaned.length > 0 ? validateProtein(cleaned) : null;
  const canGenerate = !loading && !validationError;

  async function handleGenerate() {
    setLoading(true);
    setError(null);
    setSequence(null);

    const hasTE = teState.enabled.some(Boolean);

    try {
      const result = await generateSequence({
        protein_seq: cleaned.length > 0 ? cleaned : null,
        te_values: hasTE ? teState.values : null,
        te_mask: hasTE
          ? teState.enabled.map((e) => (e ? 1.0 : 0.0))
          : null,
        max_length: maxLength,
        temperature,
        top_k: topK,
      });
      setSequence(result.sequence);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <TooltipProvider>
      <div className="min-h-screen flex flex-col">
        {/* Header */}
        <header className="border-b px-6 py-3 flex items-center gap-3">
          <Dna className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-semibold">Tropical</h1>
          <span className="text-sm text-muted-foreground">
            mRNA Sequence Generator
          </span>
        </header>

        {/* Main content */}
        <main className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-6 p-6 max-w-7xl mx-auto w-full">
          {/* Input panel */}
          <div className="space-y-6">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Input</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <ProteinInput value={proteinSeq} onChange={setProteinSeq} />

                <Separator />

                <div>
                  <h3 className="text-sm font-medium mb-3">
                    Generation Parameters
                  </h3>
                  <GenerationParams
                    temperature={temperature}
                    topK={topK}
                    maxLength={maxLength}
                    onTemperatureChange={setTemperature}
                    onTopKChange={setTopK}
                    onMaxLengthChange={setMaxLength}
                  />
                </div>

                <Separator />

                <TEPanel state={teState} onChange={setTEState} />

                <Button
                  className="w-full"
                  size="lg"
                  disabled={!canGenerate}
                  onClick={handleGenerate}
                >
                  <Play className="h-4 w-4 mr-2" />
                  Generate
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Output panel */}
          <div>
            <Card className="h-full">
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Output</CardTitle>
              </CardHeader>
              <CardContent>
                <SequenceOutput
                  sequence={sequence}
                  loading={loading}
                  error={error}
                />
                <RiboNNPanel sequence={sequence} proteinSeq={cleaned.length > 0 ? cleaned : null} />
              </CardContent>
            </Card>
          </div>
        </main>

        {/* Footer */}
        <footer className="border-t px-6 py-2">
          <ApiStatus />
        </footer>
      </div>
    </TooltipProvider>
  );
}
