// app/predict/hooks/usePlanetTypeClassification.ts
"use client";

import { useCallback, useState } from "react";
import { PlanetTypeClassification, Metadata } from "../types";
import { parseCSVFeatures } from "../utils/csvParser";

// Only the 6 features used by your PCA/KNN artifacts
export const CLASSIFICATION_FEATURES = [
  "pl_rade",
  "pl_insol",
  "pl_eqt",
  "pl_orbper",
  "st_teff",
  "st_rad",
] as const;

type SixFeatureRecord = Partial<Record<(typeof CLASSIFICATION_FEATURES)[number], number | null>>;

type APIResponse = {
  chart_base64: string | null;
  classifications: Array<{
    id?: string | number | null;
    PC1?: number | string | null;
    PC2?: number | string | null;
    type_cluster?: number | string | null;
    type_confidence?: number | string | null;
  }>;
  meta?: {
    pca_var_explained?: [number, number];
    kmeans_k?: number;
  };
};

function toNum(v: unknown): number {
  if (v === null || v === undefined) return NaN;
  if (typeof v === "number") return v;
  const n = Number.parseFloat(String(v));
  return Number.isFinite(n) ? n : NaN;
}

/** Keep only the 6 features and coerce to numbers (or null if not finite). */
function sanitizeRow(obj: Record<string, any>): SixFeatureRecord {
  const out: SixFeatureRecord = {};
  for (const key of CLASSIFICATION_FEATURES) {
    const n = toNum(obj?.[key]);
    out[key] = Number.isFinite(n) ? n : null;
  }
  return out;
}

/** Coerce/shape one row of API classifications to the UI type. */
function coerceClassification(r: any): PlanetTypeClassification {
  const pc1 = toNum(r?.PC1);
  const pc2 = toNum(r?.PC2);
  const cluster = toNum(r?.type_cluster);
  const confidence = toNum(r?.type_confidence);

  // Map cluster numbers to type names (consistent with backend)
  const clusterNameMap: Record<number, string> = {
    0: "Rocky Planet",
    1: "Gas Giant",
    2: "Ice Giant",
  };

  return {
    id: r?.id ?? undefined,
    PC1: Number.isFinite(pc1) ? pc1 : undefined,
    PC2: Number.isFinite(pc2) ? pc2 : undefined,
    pca_x: Number.isFinite(pc1) ? pc1 : undefined,
    pca_y: Number.isFinite(pc2) ? pc2 : undefined,
    type_cluster: Number.isFinite(cluster) ? cluster : -1,
    type_pred: Number.isFinite(cluster) ? cluster : -1,
    type_confidence: Number.isFinite(confidence) ? confidence : undefined,
    type_name: Number.isFinite(cluster) ? clusterNameMap[cluster] ?? `Cluster ${cluster}` : "Unknown",
  };
}

export function usePlanetTypeClassification() {
  const [planetTypeChart, setPlanetTypeChart] = useState<string | null>(null);
  const [planetTypeClassifications, setPlanetTypeClassifications] = useState<
    PlanetTypeClassification[]
  >([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pcaExplained, setPcaExplained] = useState<[number, number] | undefined>(undefined);
  const [kmeansK, setKmeansK] = useState<number | undefined>(undefined);

  const fetchPlanetTypeClassifications = useCallback(
    async (
      metadataArray: Metadata[] = [],
      featuresData: string | Array<Record<string, any>>,
      predictedLabels?: number[]  // Add predicted labels parameter
    ) => {
      setLoading(true);
      setError(null);
      try {
        // ---- Build FormData ----
        const formData = new FormData();

        // TOI / row identifiers (optional)
        const toiList = (metadataArray ?? []).map((m) => m?.toi ?? null);
        formData.append("toi_list", JSON.stringify(toiList));

        // Features payload
        let featuresList: SixFeatureRecord[] = [];

        if (typeof featuresData === "string") {
          // CSV path: parse + keep only the 6 columns
          // parseCSVFeatures(csvText, keepCols[]) -> Array<Record<string, number|null>>
          const parsed = parseCSVFeatures(featuresData, [...CLASSIFICATION_FEATURES]);
          featuresList = parsed.map(sanitizeRow);
        } else if (Array.isArray(featuresData)) {
          // Rows path: sanitize each row
          featuresList = featuresData.map((r) => sanitizeRow(r ?? {}));
        } else {
          featuresList = [];
        }

        formData.append("features_json", JSON.stringify(featuresList));

        // Add predicted labels to filter only exoplanet candidates
        if (predictedLabels && Array.isArray(predictedLabels)) {
          formData.append("predicted_labels", JSON.stringify(predictedLabels));
        }

        // ---- Call API ----
        const endpoint = `${process.env.NEXT_PUBLIC_API_ENDPOINT}/classify/planet-types`;
        const res = await fetch(endpoint, {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          const msg = await res.text().catch(() => "");
          throw new Error(`KNN API ${res.status}: ${msg || "request failed"}`);
        }

        const data: APIResponse = await res.json();

        // ---- Normalize response ----
        const rows = Array.isArray(data?.classifications)
          ? data.classifications.map(coerceClassification)
          : [];

        setPlanetTypeChart(typeof data?.chart_base64 === "string" ? data.chart_base64 : null);
        setPlanetTypeClassifications(rows);

        if (Array.isArray(data?.meta?.pca_var_explained)) {
          setPcaExplained(data.meta!.pca_var_explained as [number, number]);
        } else {
          setPcaExplained(undefined);
        }
        setKmeansK(
          typeof data?.meta?.kmeans_k === "number" ? data.meta!.kmeans_k : undefined
        );

        // quick dev signal
        if (!data?.chart_base64) console.warn("Planet types: API returned no chart_base64");
        if (!rows.length) console.warn("Planet types: API returned 0 classifications");
      } catch (e: any) {
        console.error(e);
        setError(e?.message ?? "Unknown error");
        setPlanetTypeChart(null);
        setPlanetTypeClassifications([]);
        setPcaExplained(undefined);
        setKmeansK(undefined);
      } finally {
        setLoading(false);
      }
    },
    []
  );

  return {
    // state
    loading,
    error,
    planetTypeChart,
    planetTypeClassifications,
    pcaExplained,
    kmeansK,
    // action
    fetchPlanetTypeClassifications,
  };
}
