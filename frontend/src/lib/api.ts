// Modal gives each endpoint its own URL: {base}-{method}.modal.run
// In dev mode the suffix is "-dev", in deployed mode there's no suffix.
const API_BASE =
  import.meta.env.VITE_API_BASE ??
  "https://tom-ellis-lab--tropical-serve-inference";
const API_SUFFIX = import.meta.env.VITE_API_SUFFIX ?? "-dev.modal.run";

function url(method: string): string {
  return `${API_BASE}-${method}${API_SUFFIX}`;
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
  const res = await fetch(url("generate"), {
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
    const res = await fetch(url("health"));
    return res.ok;
  } catch {
    return false;
  }
}

export async function fetchTEColumns(): Promise<string[]> {
  const res = await fetch(url("te-columns"));
  if (!res.ok) throw new Error("Failed to fetch TE columns");
  const data = await res.json();
  return data.columns;
}
