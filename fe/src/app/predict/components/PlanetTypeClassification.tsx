"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { BarChart3, MessageSquare } from "lucide-react";
import { PlanetTypeClassification as PlanetTypeClassificationType } from "../types";

interface PlanetTypeClassificationProps {
  planetTypeChart: string | null;
  planetTypeClassifications: PlanetTypeClassificationType[];
}

export function PlanetTypeClassification({
  planetTypeChart,
  planetTypeClassifications,
}: PlanetTypeClassificationProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="w-5 h-5" />
          Planet Type Classification (KNN)
        </CardTitle>
        <CardDescription>
          Exoplanet type classification using K-Nearest Neighbors with ground
          truth from type_labels.csv
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-3 gap-6">
          {/* Classification Chart - 2/3 width */}
          <div className="col-span-2">
            <div className="h-192 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-black rounded-lg border border-gray-300 dark:border-gray-700 overflow-hidden shadow-lg relative">
              {planetTypeChart ? (
                <img
                  src={`data:image/png;base64,${planetTypeChart}`}
                  alt="Planet Type Classification"
                  className="w-full h-full object-contain p-4"
                />
              ) : (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <div className="w-24 h-24 bg-purple-500 rounded-lg mx-auto mb-4 flex items-center justify-center animate-pulse">
                      <BarChart3 className="w-12 h-12 text-white" />
                    </div>
                    <p className="text-gray-700 dark:text-white text-lg font-medium mb-2">
                      Loading Classification...
                    </p>
                    <p className="text-gray-500 dark:text-gray-400 text-sm">
                      Analyzing planet types using KNN
                    </p>
                  </div>
                </div>
              )}

              {/* Overlay info */}
              {planetTypeClassifications.length > 0 && (
                <div className="absolute top-4 left-4 bg-black/70 px-3 py-2 rounded-lg backdrop-blur-sm">
                  <p className="text-white text-sm font-medium">
                    {planetTypeClassifications.length} Planets Classified
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* AI Agent - 1/3 width */}
          <div className="col-span-1">
            <div className="bg-black rounded-lg h-192 flex flex-col shadow-lg overflow-hidden">
              <div className="bg-gradient-to-r from-purple-600 to-purple-700 p-4">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 bg-white rounded-lg flex items-center justify-center shadow-md">
                    <MessageSquare className="w-5 h-5 text-purple-600" />
                  </div>
                  <div>
                    <h3 className="text-white font-semibold text-lg">
                      AI Agent
                    </h3>
                    <p className="text-purple-200 text-xs">
                      Planet Type Assistant
                    </p>
                  </div>
                </div>
              </div>

              {/* Classification Summary */}
              <div className="flex-1 space-y-3 mb-4 overflow-y-auto p-4">
                {planetTypeClassifications.length > 0 ? (
                  <>
                    <div className="bg-white/15 backdrop-blur-sm rounded-lg p-3 border border-white/20">
                      <div className="flex items-center gap-2 mb-2">
                        <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                        <span className="text-white text-xs font-medium">
                          AI Classification
                        </span>
                      </div>
                      <p className="text-blue-100 text-xs leading-relaxed mb-2">
                        I&apos;ve classified {planetTypeClassifications.length}{" "}
                        exoplanet(s) using KNN:
                      </p>
                      <div className="space-y-2">
                        {planetTypeClassifications
                          .slice(0, 3)
                          .map((classification, idx) => (
                            <div
                              key={idx}
                              className="bg-black/30 rounded p-2 text-xs"
                            >
                              <div className="flex justify-between items-center">
                                <span className="text-purple-300 font-mono font-semibold">
                                  {classification.toi}
                                </span>
                                <span
                                  className={`px-2 py-0.5 rounded text-xs font-medium ${
                                    classification.type_name === "Sub-Neptune"
                                      ? "bg-blue-500/30 text-blue-200"
                                      : classification.type_name ===
                                        "Ultra-Giant"
                                      ? "bg-purple-500/30 text-purple-200"
                                      : classification.type_name ===
                                        "Super-Earth"
                                      ? "bg-green-500/30 text-green-200"
                                      : "bg-gray-500/30 text-gray-200"
                                  }`}
                                >
                                  {classification.type_name}
                                </span>
                              </div>
                              <div className="text-gray-300 text-xs mt-1">
                                Confidence:{" "}
                                {(classification.type_confidence * 100).toFixed(
                                  1
                                )}
                                %
                              </div>
                            </div>
                          ))}
                        {planetTypeClassifications.length > 3 && (
                          <p className="text-gray-400 text-xs text-center">
                            +{planetTypeClassifications.length - 3} more...
                          </p>
                        )}
                      </div>
                    </div>

                    <div className="bg-white/15 backdrop-blur-sm rounded-lg p-3 border border-white/20">
                      <div className="flex items-center gap-2 mb-2">
                        <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                        <span className="text-white text-xs font-semibold">
                          Type Legend
                        </span>
                      </div>
                      <div className="space-y-1 text-xs text-blue-100">
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 bg-blue-500 rounded"></div>
                          <span>Sub-Neptune: 2-4 R⊕</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 bg-purple-500 rounded"></div>
                          <span>Ultra-Giant: &gt;10 R⊕</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 bg-green-500 rounded"></div>
                          <span>Super-Earth: 1-2 R⊕</span>
                        </div>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="bg-white/15 backdrop-blur-sm rounded-lg p-3 border border-white/20">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
                      <span className="text-white text-xs font-medium">
                        AI Agent
                      </span>
                    </div>
                    <p className="text-blue-100 text-xs leading-relaxed">
                      Analyzing planet types using KNN classification...
                    </p>
                  </div>
                )}
              </div>

              {/* Info Footer */}
              <div className="px-4 pb-4">
                <div className="bg-white/10 rounded-lg p-2 text-center">
                  <p className="text-white/70 text-xs">
                    Classification based on type_labels.csv ground truth
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
