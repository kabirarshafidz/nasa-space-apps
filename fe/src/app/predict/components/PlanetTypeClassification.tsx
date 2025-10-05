"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { BarChart3 } from "lucide-react";
import {
  PlanetTypeClassification as PlanetTypeClassificationType,
  PredictionResults,
  PreTrainedModel,
} from "../types";
import { PlanetChatbot } from "./PlanetChatbot";

// ---- helpers to safely coerce/format numbers ----
function toNum(v: unknown): number {
  if (v === null || v === undefined) return NaN;
  if (typeof v === "number") return v;
  const n = parseFloat(String(v));
  return Number.isFinite(n) ? n : NaN;
}
function fmtNum(v: unknown, digits = 3): string {
  const n = toNum(v);
  return Number.isFinite(n) ? n.toFixed(digits) : "—";
}
function fmtPct01(v: unknown, digits = 1): string {
  const n = toNum(v);
  return Number.isFinite(n) ? `${(n * 100).toFixed(digits)}%` : "—";
}

interface PlanetTypeClassificationProps {
  planetTypeChart: string | null;
  planetTypeClassifications: PlanetTypeClassificationType[];
  planetData?: any[];
  predictionResults?: PredictionResults;
  modelInfo?: PreTrainedModel[];
  pcaExplained?: [number, number];
  kmeansK?: number;
}

export function PlanetTypeClassification({
  planetTypeChart,
  planetTypeClassifications = [],
  planetData = [],
  predictionResults,
  modelInfo = [],
  pcaExplained,
  kmeansK,
}: PlanetTypeClassificationProps) {
  const total = Array.isArray(planetTypeClassifications)
    ? planetTypeClassifications.length
    : 0;

  // safer cluster counts (treat invalid/missing as cluster -1)
  const clusterCounts = (planetTypeClassifications ?? []).reduce<
    Record<number, number>
  >((acc, r) => {
    const kRaw = toNum((r as any).type_cluster);
    const k = Number.isFinite(kRaw) ? (kRaw as number) : -1;
    acc[k] = (acc[k] || 0) + 1;
    return acc;
  }, {});

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="w-5 h-5" />
          Planet Type Classification (PCA → KNN)
        </CardTitle>
        <CardDescription>
          PCA(2D) projection with KNN on (PC1, PC2)
          {typeof kmeansK === "number" ? ` · KMeans(k=${kmeansK}) baseline` : ""}
        </CardDescription>
      </CardHeader>

      <CardContent>
        <div className="grid grid-cols-3 gap-6">
          {/* Chart - 2/3 width */}
          <div className="col-span-2">
            <div className="h-192 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-black rounded-lg border border-gray-300 dark:border-gray-700 overflow-hidden shadow-lg relative">
              {planetTypeChart ? (
                <img
                  src={`data:image/png;base64,${planetTypeChart}`}
                  alt="PCA + KNN Cluster Plot"
                  className="w-full h-full object-contain p-4"
                />
              ) : (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <div className="w-24 h-24 bg-purple-500 rounded-lg mx-auto mb-4 flex items-center justify-center animate-pulse">
                      <BarChart3 className="w-12 h-12 text-white" />
                    </div>
                    <p className="text-gray-700 dark:text-white text-lg font-medium mb-2">
                      Building PCA projection…
                    </p>
                    <p className="text-gray-500 dark:text-gray-400 text-sm">
                      Running preprocessing → PCA(2) → KNN on your input
                    </p>
                  </div>
                </div>
              )}

              {/* Overlay info */}
              <div className="absolute top-4 left-4 flex flex-col gap-2">
                {total > 0 && (
                  <Badge variant="secondary" className="backdrop-blur-sm">
                    {total} object{total === 1 ? "" : "s"} classified
                  </Badge>
                )}
                {Array.isArray(pcaExplained) &&
                  Number.isFinite(pcaExplained[0]) &&
                  Number.isFinite(pcaExplained[1]) && (
                    <Badge variant="outline" className="bg-black/60 text-white">
                      PC1 {Math.round(pcaExplained[0] * 100)}% · PC2{" "}
                      {Math.round(pcaExplained[1] * 100)}%
                    </Badge>
                  )}
              </div>
            </div>

            {/* Cluster summary */}
            {total > 0 && (
              <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-2">
                {Object.entries(clusterCounts).map(([k, v]) => (
                  <div
                    key={k}
                    className="rounded-md border p-3 text-sm bg-white/60 dark:bg-gray-900/40"
                  >
                    <div className="font-medium">
                      Cluster {Number(k) === -1 ? "N/A" : k}
                    </div>
                    <div className="text-muted-foreground">
                      {v} object{Number(v) === 1 ? "" : "s"}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* AI Chatbot - 1/3 width */}
          <div className="col-span-1">
            <div className="h-192">
              <PlanetChatbot
                planetData={planetData}
                predictionResults={predictionResults}
                planetTypeClassifications={planetTypeClassifications}
                modelInfo={modelInfo}
              />
            </div>
          </div>
        </div>

        {/* Table of classifications */}
        {total > 0 && (
          <div className="mt-6 overflow-x-auto">
            <table className="w-full text-sm border rounded-lg overflow-hidden">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th className="px-3 py-2 text-left">ID</th>
                  <th className="px-3 py-2 text-right">PC1</th>
                  <th className="px-3 py-2 text-right">PC2</th>
                  <th className="px-3 py-2 text-right">Cluster</th>
                  <th className="px-3 py-2 text-right">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {planetTypeClassifications.map((r, i) => {
                  const id = (r as any).id ?? `row_${i + 1}`;
                  const pc1 = fmtNum((r as any).PC1);
                  const pc2 = fmtNum((r as any).PC2);
                  const clusterNumRaw = toNum((r as any).type_cluster);
                  const clusterLabel = Number.isFinite(clusterNumRaw)
                    ? String(clusterNumRaw)
                    : "N/A";
                  const conf = fmtPct01((r as any).type_confidence);

                  return (
                    <tr key={`${id}-${pc1}-${pc2}-${clusterLabel}`} className="border-t">
                      <td className="px-3 py-2">{id}</td>
                      <td className="px-3 py-2 text-right">{pc1}</td>
                      <td className="px-3 py-2 text-right">{pc2}</td>
                      <td className="px-3 py-2 text-right">{clusterLabel}</td>
                      <td className="px-3 py-2 text-right">{conf}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
