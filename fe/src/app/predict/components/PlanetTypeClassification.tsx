"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { BarChart3, MessageSquare } from "lucide-react";
import { PlanetTypeClassification as PlanetTypeClassificationType, PredictionResults, PreTrainedModel } from "../types";
import { PlanetChatbot } from "./PlanetChatbot";

interface PlanetTypeClassificationProps {
  planetTypeChart: string | null;
  planetTypeClassifications: PlanetTypeClassificationType[];
  planetData?: any[];
  predictionResults?: PredictionResults;
  modelInfo?: PreTrainedModel[];
}

export function PlanetTypeClassification({
  planetTypeChart,
  planetTypeClassifications,
  planetData = [],
  predictionResults,
  modelInfo = [],
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
      </CardContent>
    </Card>
  );
}
