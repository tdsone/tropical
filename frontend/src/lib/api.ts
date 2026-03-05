// Modal asgi_app endpoints: {base}-web{suffix}/{route}
const API_BASE =
  import.meta.env.VITE_API_BASE ??
  "https://tom-ellis-lab--tropical-serve-inference";
const API_SUFFIX = import.meta.env.VITE_API_SUFFIX ?? "-dev.modal.run";

const RIBONN_BASE =
  import.meta.env.VITE_RIBONN_API_BASE ??
  "https://tom-ellis-lab--ribonn-serve-ribonn-inference";

function tropicalUrl(route: string): string {
  return `${API_BASE}-web${API_SUFFIX}/${route}`;
}

function ribonnUrl(route: string): string {
  return `${RIBONN_BASE}-web${API_SUFFIX}/${route}`;
}

export interface GenerateRequest {
  protein_seq: string | null;
  te_values: number[] | null;
  te_mask: number[] | null;
  max_length: number;
  temperature: number;
  top_k: number | null;
}

export interface GenerateResponse {
  sequence: string;
  length: number;
}

export async function generateSequence(
  req: GenerateRequest
): Promise<GenerateResponse> {
  const res = await fetch(tropicalUrl("generate"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Generation failed (${res.status}): ${text}`);
  }
  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(tropicalUrl("health"));
    return res.ok;
  } catch {
    return false;
  }
}

export async function fetchTEColumns(): Promise<string[]> {
  const res = await fetch(tropicalUrl("te-columns"));
  if (!res.ok) throw new Error("Failed to fetch TE columns");
  const data = await res.json();
  return data.columns;
}

export interface RiboNNPrediction {
  predictions: Record<string, number>;
  columns: string[];
  values: number[];
  utr5_size: number;
  cds_size: number;
}

export async function predictRiboNN(
  sequence: string,
  protein_seq: string | null = null
): Promise<RiboNNPrediction> {
  const res = await fetch(ribonnUrl("predict"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sequence, protein_seq }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`RiboNN prediction failed (${res.status}): ${text}`);
  }
  return res.json();
}
