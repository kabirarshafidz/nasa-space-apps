import { useState } from "react";
import { PredictionResults, SingleFeatures, Metadata } from "../types";
import { parseCSVMetadata, parseCSVFeatures } from "../utils/csvParser";

export function usePrediction() {
  const [isLoading, setIsLoading] = useState(false);
  const [predictionResults, setPredictionResults] = useState<PredictionResults | null>(null);

  const handlePredict = async (
    selectedModel: string | null,
    predictionType: "batch" | "single",
    uploadedFile: File | null,
    singleFeatures: SingleFeatures,
    metadata: Metadata
  ) => {
    if (!selectedModel) {
      alert("Please select a model first");
      return null;
    }

    // Validate metadata fields
    if (predictionType === "single") {
      if (!metadata.toi || !metadata.toipfx) {
        alert("Please enter both TOI and TOIPFX values");
        return null;
      }
    }

    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append(
        "model_name",
        selectedModel
          .replace("models/", "")
          .replace(".pkl", "")
          .replace(".joblib", "")
          .replace("default/", "")
      );

      let metadataArray: Array<{ toi: string; toipfx: string }> = [];
      let csvText: string | null = null;

      if (predictionType === "batch" && uploadedFile) {
        csvText = await uploadedFile.text();

        try {
          metadataArray = parseCSVMetadata(csvText);
        } catch (error) {
          alert(error instanceof Error ? error.message : "Failed to parse CSV");
          setIsLoading(false);
          return null;
        }

        formData.append("file", uploadedFile);
      } else if (predictionType === "single") {
        const featuresArray = [singleFeatures];
        formData.append("features_json", JSON.stringify(featuresArray));
        metadataArray = [metadata];
      } else {
        alert("Please upload a file or enter feature values");
        setIsLoading(false);
        return null;
      }

      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_ENDPOINT}/predict`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Prediction failed");
      }

      const data = await response.json();

      const results = {
        ...data,
        metadata: metadataArray,
      };

      setPredictionResults(results);

      return { results, csvText };
    } catch (error) {
      console.error("Prediction error:", error);
      alert(
        error instanceof Error
          ? error.message
          : "Failed to make prediction. Please try again."
      );
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  return {
    isLoading,
    predictionResults,
    handlePredict,
    setPredictionResults,
  };
}
