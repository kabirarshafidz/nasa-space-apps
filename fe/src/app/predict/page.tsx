"use client";

import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Stepper,
  StepperIndicator,
  StepperItem,
  StepperSeparator,
  StepperTitle,
  StepperTrigger,
} from "@/components/ui/stepper";
import { ArrowLeft, ArrowRight, BarChart3 } from "lucide-react";
import Link from "next/link";

// Types
import { PreTrainedModel, SingleFeatures, Metadata } from "./types";

// Hooks
import { usePrediction } from "./hooks/usePrediction";
import { usePlanetTypeClassification } from "./hooks/usePlanetTypeClassification";

// Components
import { ModelSelection } from "./components/ModelSelection";
import { DataInput } from "./components/DataInput";
import { ResultsTable } from "./components/ResultsTable";
import { PlanetTypeClassification } from "./components/PlanetTypeClassification";

export default function PredictPage() {
  // Stepper state
  const [currentStep, setCurrentStep] = useState(1);

  // Models
  const [preTrainedModels, setPreTrainedModels] = useState<PreTrainedModel[]>(
    []
  );
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [modelsLoading, setModelsLoading] = useState(true);

  // Input mode & data
  const [predictionType, setPredictionType] = useState<"batch" | "single">(
    "single"
  );
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [singleFeatures, setSingleFeatures] = useState<SingleFeatures>({
    pl_orbper: "",
    pl_trandurh: "",
    pl_trandep: "",
    pl_rade: "",
    pl_insol: "",
    pl_eqt: "",
    st_tmag: "",
    st_dist: "",
    st_teff: "",
    st_logg: "",
    st_rad: "",
    pl_rade_relerr: "",
  });
  const [metadata, setMetadata] = useState<Metadata>({
    toi: "",
    toipfx: "",
  });

  const [planetData, setPlanetData] = useState<any[]>([]);

  // Custom hooks
  const { isLoading, predictionResults, handlePredict } = usePrediction();
  const {
    loading: knnLoading,
    error: knnError,
    planetTypeChart,
    planetTypeClassifications,
    pcaExplained,
    kmeansK,
    fetchPlanetTypeClassifications,
  } = usePlanetTypeClassification();

  const steps = [
    { step: 1, title: "Choose Model", description: "Select a trained model" },
    { step: 2, title: "Upload Data", description: "Provide input features" },
    { step: 3, title: "Results", description: "View predictions" },
  ];

  // Fetch models on mount
  useEffect(() => {
    const getPreTrainedModels = async () => {
      setModelsLoading(true);
      try {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/api/models`
        );
        const data = await response.json();
        setPreTrainedModels(data.models || []);
      } catch (error) {
        console.error("Failed to fetch models:", error);
      } finally {
        setModelsLoading(false);
      }
    };

    getPreTrainedModels();
  }, []);

  // Handlers
  const handleNext = useCallback(async () => {
    if (currentStep === 1 && !selectedModel) {
      alert("Please select a model");
      return;
    }

    if (currentStep === 2) {
      const result = await handlePredict(
        selectedModel,
        predictionType,
        uploadedFile,
        singleFeatures,
        metadata
      );

      if (result) {
        // Prepare planet data for chatbot
        let planetDataArray: any[] = [];
        if (predictionType === "batch" && result.csvText) {
          // Import the CSV parser on demand
          const { parseCSVToObjects } = await import("./utils/csvParser");
          planetDataArray = parseCSVToObjects(result.csvText);
        } else {
          planetDataArray = [{ ...singleFeatures, ...metadata }];
        }
        setPlanetData(planetDataArray);

        // Kick off PCA→KNN type classifications
        await fetchPlanetTypeClassifications(
          result.results?.metadata || [],
          result.csvText || [singleFeatures]
        );

        setCurrentStep(3);
      }
    } else if (currentStep < 3) {
      setCurrentStep(currentStep + 1);
    }
  }, [
    currentStep,
    selectedModel,
    handlePredict,
    predictionType,
    uploadedFile,
    singleFeatures,
    metadata,
    fetchPlanetTypeClassifications,
  ]);

  const handleBack = useCallback(() => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  }, [currentStep]);

  const handleFeatureChange = (featureName: string, value: string) => {
    setSingleFeatures((prev) => ({
      ...prev,
      [featureName]: value,
    }));
  };

  const handleFileUpload = (file: File | null) => {
    setUploadedFile(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    const files = e.dataTransfer.files;
    if (files && files[0] && files[0].name.endsWith(".csv")) {
      setUploadedFile(files[0]);
    } else {
      alert("Please upload a CSV file");
    }
  };

  // Render step content
  const renderStepContent = () => {
    switch (currentStep) {
      case 1:
        return (
          <ModelSelection
            preTrainedModels={preTrainedModels}
            selectedModel={selectedModel}
            onModelSelect={setSelectedModel}
            isLoading={modelsLoading}
          />
        );

      case 2:
        return (
          <DataInput
            predictionType={predictionType}
            setPredictionType={setPredictionType}
            singleFeatures={singleFeatures}
            onFeatureChange={handleFeatureChange}
            metadata={metadata}
            setMetadata={setMetadata}
            uploadedFile={uploadedFile}
            onFileUpload={handleFileUpload}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          />
        );

      case 3:
        return (
          <div className="space-y-6">
            {/* 1) Prediction Results */}
            {predictionResults && (
              <Card>
                <CardHeader>
                  <CardTitle>Prediction Results</CardTitle>
                  <CardDescription>
                    Exoplanet detection predictions for your data
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResultsTable predictionResults={predictionResults} />
                </CardContent>
              </Card>
            )}

            {/* 2) 3D Visualization (placeholder) */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="w-5 h-5" />
                  3D System Visualization
                </CardTitle>
                <CardDescription>
                  Interactive 3D view of the detected exoplanet system
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-160 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-950 dark:to-purple-950 rounded-lg flex items-center justify-center mb-4">
                  <div className="text-center">
                    <div className="w-20 h-20 bg-blue-500 rounded-full mx-auto mb-4 animate-pulse"></div>
                    <p className="text-sm text-muted-foreground">
                      3D Canvas Loading...
                    </p>
                    <p className="text-xs text-muted-foreground mt-2">
                      Star + Planet + Orbit visualization
                    </p>
                  </div>
                </div>

                {/* Info below 3D viz (static placeholders) */}
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                    <div className="text-lg font-semibold text-blue-600 dark:text-blue-400">
                      12.4
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Period (days)
                    </div>
                  </div>
                  <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                    <div className="text-lg font-semibold text-green-600 dark:text-green-400">
                      1.2
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Radius (R⊕)
                    </div>
                  </div>
                  <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                    <div className="text-lg font-semibold text-purple-600 dark:text-purple-400">
                      87.3%
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Confidence
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* 3) Planet Type Classification (PCA→KNN) */}
            <PlanetTypeClassification
              planetTypeChart={planetTypeChart}
              planetTypeClassifications={planetTypeClassifications}
              planetData={planetData}
              predictionResults={predictionResults ?? undefined}
              modelInfo={preTrainedModels}
              pcaExplained={pcaExplained}
              kmeansK={kmeansK}
            />

            {/* Optional: surface KNN errors */}
            {knnError && (
              <p className="text-sm text-red-500">
                PCA/KNN classification error: {knnError}
              </p>
            )}
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="h-[calc(100vh-4rem)] overflow-y-auto ">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">
            Predict Exoplanets from Your Data
          </h1>
          <p className="text-muted-foreground">
            Use machine learning models to predict exoplanet candidates
          </p>
        </div>

        {/* Stepper */}
        <div className="mb-8">
          <Stepper value={currentStep} className="w-full">
            {steps.map((step, index) => (
              <StepperItem
                key={step.step}
                step={step.step}
                className="not-last:flex-1 max-md:items-start"
              >
                <StepperTrigger className="rounded max-md:flex-col">
                  <StepperIndicator />
                  <div className="text-center md:text-left">
                    <StepperTitle>{step.title}</StepperTitle>
                  </div>
                </StepperTrigger>
                {index < steps.length - 1 && (
                  <StepperSeparator className="max-md:mt-3.5 md:mx-4" />
                )}
              </StepperItem>
            ))}
          </Stepper>
        </div>

        {/* Content */}
        <div className="mb-24">{renderStepContent()}</div>

        {/* Nav Buttons */}
        <div className="fixed bottom-8 left-1/2 transform -translate-x-1/2 z-50">
          <div className="flex gap-4 bg-white dark:bg-gray-800 p-4 rounded-lg shadow-lg border">
            <Button
              variant="outline"
              onClick={handleBack}
              disabled={currentStep === 1}
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back
            </Button>
            <Button
              onClick={handleNext}
              disabled={
                isLoading ||
                knnLoading ||
                (currentStep === 2 &&
                  predictionType === "single" &&
                  (Object.values(singleFeatures).some((v) => !v) ||
                    !metadata.toi ||
                    !metadata.toipfx)) ||
                (currentStep === 2 &&
                  predictionType === "batch" &&
                  !uploadedFile) ||
                currentStep === 3
              }
            >
              {isLoading || knnLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                  Processing...
                </>
              ) : currentStep === 3 ? (
                "Finished"
              ) : (
                <>
                  {currentStep === 2 ? "Predict" : "Next"}
                  <ArrowRight className="w-4 h-4 ml-2" />
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
