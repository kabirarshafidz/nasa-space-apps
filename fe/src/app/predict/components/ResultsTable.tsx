"use client";

import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Search, ChevronLeft, ChevronRight } from "lucide-react";
import { PredictionResults } from "../types";

interface ResultsTableProps {
  predictionResults: PredictionResults;
}

export function ResultsTable({ predictionResults }: ResultsTableProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  const positiveCount = predictionResults.predicted_labels.filter(
    (label) => label === 1
  ).length;
  const negativeCount = predictionResults.predicted_labels.filter(
    (label) => label === 0
  ).length;
  const avgConfidence =
    predictionResults.predictions.reduce((a, b) => a + b, 0) /
    predictionResults.predictions.length;

  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950 dark:to-purple-950 p-4 rounded-lg">
        <h3 className="font-semibold mb-2">Prediction Summary</h3>
        <p className="text-sm text-muted-foreground">
          Total Predictions: {predictionResults.feature_count} | Positive:{" "}
          {positiveCount} | Negative: {negativeCount}
        </p>
      </div>

      {/* Search Bar */}
      <div className="mb-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <Input
            type="text"
            placeholder="Search by TOI or Candidate ID..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setCurrentPage(1);
            }}
            className="pl-10 pr-4 py-2"
          />
        </div>
      </div>

      {/* Results Table */}
      <div className="border rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-100 dark:bg-gray-800">
              <tr>
                <th className="px-4 py-3 text-left font-semibold">TOI</th>
                <th className="px-4 py-3 text-left font-semibold">TOIPFX</th>
                <th className="px-4 py-3 text-right font-semibold">
                  Probability
                </th>
                <th className="px-4 py-3 text-center font-semibold">
                  Prediction
                </th>
              </tr>
            </thead>
            <tbody>
              {(() => {
                const filteredResults = predictionResults.predictions
                  .map((prob, idx) => ({
                    toi:
                      predictionResults.metadata?.[idx]?.toi ||
                      `TOI-${idx + 1}`,
                    toipfx: predictionResults.metadata?.[idx]?.toipfx || "N/A",
                    prob,
                    label: predictionResults.predicted_labels[idx],
                    originalIdx: idx,
                  }))
                  .filter(
                    (item) =>
                      item.toi
                        .toLowerCase()
                        .includes(searchQuery.toLowerCase()) ||
                      item.toipfx
                        .toLowerCase()
                        .includes(searchQuery.toLowerCase())
                  );

                const totalPages = Math.ceil(
                  filteredResults.length / itemsPerPage
                );
                const startIdx = (currentPage - 1) * itemsPerPage;
                const endIdx = startIdx + itemsPerPage;
                const paginatedResults = filteredResults.slice(
                  startIdx,
                  endIdx
                );

                if (filteredResults.length === 0) {
                  return (
                    <tr>
                      <td
                        colSpan={4}
                        className="px-4 py-8 text-center text-gray-500 dark:text-gray-400"
                      >
                        No results found for &quot;{searchQuery}&quot;
                      </td>
                    </tr>
                  );
                }

                return paginatedResults.map((item) => (
                  <tr
                    key={item.originalIdx}
                    className="border-t dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800/50"
                  >
                    <td className="px-4 py-3">
                      <span className="font-mono font-semibold text-purple-600 dark:text-purple-400">
                        {item.toi}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span className="font-mono text-gray-700 dark:text-gray-300">
                        {item.toipfx}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right">
                      <div className="flex items-center justify-end gap-3">
                        <div className="flex-1 max-w-[120px] bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full transition-all ${
                              item.label === 1
                                ? "bg-gradient-to-r from-green-400 to-green-600"
                                : "bg-gradient-to-r from-gray-400 to-gray-500"
                            }`}
                            style={{ width: `${item.prob * 100}%` }}
                          ></div>
                        </div>
                        <span className="font-mono text-base font-semibold min-w-[60px]">
                          {(item.prob * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-center">
                      <div className="flex items-center justify-center">
                        {item.label === 1 ? (
                          <div className="flex items-center gap-2 px-3 py-1.5 bg-gradient-to-r from-green-100 to-emerald-100 dark:from-green-900 dark:to-emerald-900 rounded-lg border border-green-200 dark:border-green-700">
                            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                            <span className="font-medium text-green-800 dark:text-green-200">
                              Exoplanet
                            </span>
                          </div>
                        ) : (
                          <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-100 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                            <div className="w-2 h-2 rounded-full bg-gray-400"></div>
                            <span className="font-medium text-gray-700 dark:text-gray-300">
                              Not Exoplanet
                            </span>
                          </div>
                        )}
                      </div>
                    </td>
                  </tr>
                ));
              })()}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {(() => {
          const filteredCount = predictionResults.predictions.filter(
            (_, idx) => {
              const toi =
                predictionResults.metadata?.[idx]?.toi || `TOI-${idx + 1}`;
              const toipfx = predictionResults.metadata?.[idx]?.toipfx || "N/A";
              return (
                toi.toLowerCase().includes(searchQuery.toLowerCase()) ||
                toipfx.toLowerCase().includes(searchQuery.toLowerCase())
              );
            }
          ).length;
          const totalPages = Math.ceil(filteredCount / itemsPerPage);

          if (totalPages <= 1) return null;

          return (
            <div className="flex items-center justify-between px-4 py-3 bg-gray-50 dark:bg-gray-800 border-t dark:border-gray-700">
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Showing {(currentPage - 1) * itemsPerPage + 1} to{" "}
                {Math.min(currentPage * itemsPerPage, filteredCount)} of{" "}
                {filteredCount} results
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                >
                  <ChevronLeft className="w-4 h-4" />
                </Button>
                <div className="flex items-center gap-1">
                  {Array.from({ length: totalPages }, (_, i) => i + 1).map(
                    (page) => {
                      if (
                        page === 1 ||
                        page === totalPages ||
                        (page >= currentPage - 1 && page <= currentPage + 1)
                      ) {
                        return (
                          <Button
                            key={page}
                            variant={
                              page === currentPage ? "default" : "outline"
                            }
                            size="sm"
                            onClick={() => setCurrentPage(page)}
                            className="min-w-[32px]"
                          >
                            {page}
                          </Button>
                        );
                      } else if (
                        page === currentPage - 2 ||
                        page === currentPage + 2
                      ) {
                        return (
                          <span key={page} className="px-2 text-gray-400">
                            ...
                          </span>
                        );
                      }
                      return null;
                    }
                  )}
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    setCurrentPage((p) => Math.min(totalPages, p + 1))
                  }
                  disabled={currentPage === totalPages}
                >
                  <ChevronRight className="w-4 h-4" />
                </Button>
              </div>
            </div>
          );
        })()}
      </div>
    </div>
  );
}
