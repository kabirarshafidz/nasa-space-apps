"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Stepper,
  StepperIndicator,
  StepperItem,
  StepperSeparator,
  StepperTitle,
  StepperTrigger,
} from "@/components/ui/stepper";
import {
  ArrowLeft,
  ArrowRight,
  Upload,
  Download,
  FileText,
  Database,
  MessageSquare,
  BarChart3,
  Search,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import Link from "next/link";

export default function PredictPage() {
  const [currentStep, setCurrentStep] = useState(1);
  const [preTrainedModels, setPreTrainedModels] = useState<
    {
      name: string;
      key: string;
      size: number;
      last_modified: string;
      url: string;
    }[]
  >([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [predictionType, setPredictionType] = useState<"batch" | "single">(
    "single"
  );
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [predictionResults, setPredictionResults] = useState<{
    predictions: number[];
    predicted_labels: number[];
    feature_count: number;
    metadata?: Array<{ toi: string; toipfx: string }>;
  } | null>(null);
  const [singleFeatures, setSingleFeatures] = useState<Record<string, string>>({
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
  const [metadata, setMetadata] = useState<{ toi: string; toipfx: string }>({
    toi: "",
    toipfx: "",
  });
  const [searchQuery, setSearchQuery] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  const steps = [
    {
      step: 1,
      title: "Choose Model",
    },
    {
      step: 2,
      title: "Upload Data",
    },
    {
      step: 3,
      title: "Result",
    },
  ];

  const requiredFeatures = [
    { name: "pl_orbper", label: "Orbital Period (days)" },
    { name: "pl_trandurh", label: "Transit Duration (hours)" },
    { name: "pl_trandep", label: "Transit Depth (ppm)" },
    { name: "pl_rade", label: "Planet Radius (Earth radii)" },
    { name: "pl_insol", label: "Insolation Flux (Earth flux)" },
    { name: "pl_eqt", label: "Equilibrium Temperature (K)" },
    { name: "st_tmag", label: "TESS Magnitude (mag)" },
    { name: "st_dist", label: "Distance to Star (pc)" },
    { name: "st_teff", label: "Stellar Temperature (K)" },
    { name: "st_logg", label: "Stellar Surface Gravity (log g)" },
    { name: "st_rad", label: "Stellar Radius (Solar radii)" },
    { name: "pl_rade_relerr", label: "Relative Radius Error (log scale)" },
  ];

  const getPreTrainedModels = async () => {
    const preTrainedModels = await fetch(
      `${process.env.NEXT_PUBLIC_API_URL}/api/models`
    );
    const data = await preTrainedModels.json();
    setPreTrainedModels(data.models);
  };

  useEffect(() => {
    getPreTrainedModels();
  }, []);

  const handleNext = async () => {
    if (currentStep === 2) {
      // Make prediction when moving from step 2 to step 3
      await handlePredict();
    } else if (currentStep < 3) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handleBack = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setUploadedFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();

    const files = e.dataTransfer.files;
    if (files && files[0] && files[0].name.endsWith(".csv")) {
      setUploadedFile(files[0]);
    } else {
      alert("Please upload a CSV file");
    }
  };

  const handleFeatureChange = (featureName: string, value: string) => {
    setSingleFeatures((prev) => ({
      ...prev,
      [featureName]: value,
    }));
  };

  const handlePredict = async () => {
    if (!selectedModel) {
      alert("Please select a model first");
      return;
    }

    // Validate metadata fields
    if (predictionType === "single") {
      if (!metadata.toi || !metadata.toipfx) {
        alert("Please enter both TOI and TOIPFX values");
        return;
      }
    }

    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append(
        "model_name",
        selectedModel
          .replace("models/", "")
          .replace(".bks", "")
          .replace(".pkl", "")
          .replace(".joblib", "")
          .replace("default/", "")
      );

      let metadataArray: Array<{ toi: string; toipfx: string }> = [];

      if (predictionType === "batch" && uploadedFile) {
        // Read CSV to extract metadata before sending
        const text = await uploadedFile.text();
        const lines = text.split("\n");
        const headers = lines[0].split(",").map((h) => h.trim());

        const toiIdx = headers.indexOf("toi");
        const toipfxIdx = headers.indexOf("toipfx");

        if (toiIdx === -1 || toipfxIdx === -1) {
          alert("CSV file must contain 'toi' and 'toipfx' columns");
          setIsLoading(false);
          return;
        }

        // Extract metadata from each row
        for (let i = 1; i < lines.length; i++) {
          if (lines[i].trim()) {
            const values = lines[i].split(",");
            metadataArray.push({
              toi: values[toiIdx]?.trim() || "",
              toipfx: values[toipfxIdx]?.trim() || "",
            });
          }
        }

        // Batch prediction with file upload
        formData.append("file", uploadedFile);
      } else if (predictionType === "single") {
        // Single prediction with JSON features
        const featuresArray = [singleFeatures];
        formData.append("features_json", JSON.stringify(featuresArray));

        // Store single metadata
        metadataArray = [metadata];
      } else {
        alert("Please upload a file or enter feature values");
        setIsLoading(false);
        return;
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

      // Add metadata to results
      setPredictionResults({
        ...data,
        metadata: metadataArray,
      });
      setCurrentStep(3);
    } catch (error) {
      console.error("Prediction error:", error);
      alert(
        error instanceof Error
          ? error.message
          : "Failed to make prediction. Please try again."
      );
    } finally {
      setIsLoading(false);
    }
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 1:
        return (
          <Tabs defaultValue="pretrained" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="pretrained">
                Choose Pre-trained Model
              </TabsTrigger>
              <TabsTrigger value="upload">Upload Your Own Model</TabsTrigger>
            </TabsList>

            <TabsContent value="pretrained" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Database className="w-5 h-5" />
                    Choose Pre-trained Model
                  </CardTitle>
                  <CardDescription>
                    Select from our collection of trained models
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="model-select">Model</Label>
                      <Select
                        value={selectedModel || ""}
                        onValueChange={setSelectedModel}
                      >
                        <SelectTrigger id="model-select" className="py-8">
                          <SelectValue placeholder="Select a pre-trained model" />
                        </SelectTrigger>
                        <SelectContent>
                          {preTrainedModels.map((model) => (
                            <SelectItem key={model.key} value={model.key}>
                              <div className="flex flex-col">
                                <span className="font-medium">
                                  {model.name}
                                </span>
                              </div>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="upload" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Upload className="w-5 h-5" />
                    Upload Custom Model
                  </CardTitle>
                  <CardDescription>
                    Upload your own trained model file (.pkl, .joblib, .h5)
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-8 text-center">
                    <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                    <p className="text-gray-600 dark:text-gray-400 mb-4">
                      Drag and drop your model file here, or click to browse
                    </p>
                    <Button variant="outline">Choose File</Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        );

      case 2:
        return (
          <Tabs
            defaultValue="single"
            className="w-full"
            onValueChange={(val) =>
              setPredictionType(val as "batch" | "single")
            }
          >
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="single">Single Prediction</TabsTrigger>
              <TabsTrigger value="batch">Batch Prediction</TabsTrigger>
            </TabsList>

            <TabsContent value="single" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileText className="w-5 h-5" />
                    Candidate Information
                  </CardTitle>
                  <CardDescription>
                    Enter TOI identification information
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="toi" className="text-base font-semibold">
                        TOI <span className="text-red-500">*</span>
                      </Label>
                      <Input
                        id="toi"
                        type="text"
                        placeholder="e.g., 1234"
                        value={metadata.toi}
                        onChange={(e) =>
                          setMetadata({ ...metadata, toi: e.target.value })
                        }
                        className="border-2"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label
                        htmlFor="toipfx"
                        className="text-base font-semibold"
                      >
                        TOIPFX <span className="text-red-500">*</span>
                      </Label>
                      <Input
                        id="toipfx"
                        type="text"
                        placeholder="e.g., 01"
                        value={metadata.toipfx}
                        onChange={(e) =>
                          setMetadata({ ...metadata, toipfx: e.target.value })
                        }
                        className="border-2"
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileText className="w-5 h-5" />
                    Feature Values
                  </CardTitle>
                  <CardDescription>
                    Enter all 12 required features for prediction
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    {requiredFeatures.map((feature) => (
                      <div className="space-y-2" key={feature.name}>
                        <Label htmlFor={feature.name}>
                          {feature.name} [{feature.label}]
                        </Label>
                        <Input
                          id={feature.name}
                          type="number"
                          step="any"
                          placeholder="e.g., 12.4"
                          value={singleFeatures[feature.name]}
                          onChange={(e) =>
                            handleFeatureChange(feature.name, e.target.value)
                          }
                        />
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="batch" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Database className="w-5 h-5" />
                    Batch Data Upload
                  </CardTitle>
                  <CardDescription>
                    Upload a CSV file with exoplanet data for batch predictions.
                    File must include: <strong>toi</strong>,{" "}
                    <strong>toipfx</strong>, and all 12 feature columns.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div
                      className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-8 text-center hover:border-blue-400 dark:hover:border-blue-500 transition-colors cursor-pointer"
                      onClick={() =>
                        document.getElementById("file-upload")?.click()
                      }
                      onDragOver={handleDragOver}
                      onDrop={handleDrop}
                    >
                      <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                      <p className="text-gray-600 dark:text-gray-400 mb-4">
                        {uploadedFile
                          ? `Selected: ${uploadedFile.name}`
                          : "Click to upload or drag and drop a CSV file"}
                      </p>
                      <Button
                        variant="outline"
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          document.getElementById("file-upload")?.click();
                        }}
                      >
                        <Upload className="w-4 h-4 mr-2" />
                        Choose CSV File
                      </Button>
                      <input
                        id="file-upload"
                        type="file"
                        accept=".csv"
                        className="hidden"
                        onChange={handleFileUpload}
                      />
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-4">
                        CSV file with 12 required features
                      </p>
                    </div>
                    {uploadedFile && (
                      <div className="p-4 bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-800">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-sm font-medium text-blue-800 dark:text-blue-200">
                              <strong>File:</strong> {uploadedFile.name}
                            </p>
                            <p className="text-xs text-blue-600 dark:text-blue-300 mt-1">
                              Size: {(uploadedFile.size / 1024).toFixed(2)} KB
                            </p>
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setUploadedFile(null)}
                            className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-200"
                          >
                            <svg
                              className="w-4 h-4"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M6 18L18 6M6 6l12 12"
                              />
                            </svg>
                          </Button>
                        </div>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        );

      case 3:
        if (!predictionResults) {
          return (
            <Card>
              <CardContent className="p-12 text-center">
                <p className="text-muted-foreground">
                  No prediction results available
                </p>
              </CardContent>
            </Card>
          );
        }

        const avgConfidence =
          predictionResults.predictions.reduce((a, b) => a + b, 0) /
          predictionResults.predictions.length;
        const positiveCount = predictionResults.predicted_labels.filter(
          (l) => l === 1
        ).length;
        const negativeCount = predictionResults.feature_count - positiveCount;

        return (
          <div className="space-y-8">
            {/* 1. Results Summary */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Download className="w-5 h-5" />
                  Prediction Results
                </CardTitle>
                <CardDescription>
                  Your prediction analysis is complete
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div
                    className={`p-4 rounded-lg ${
                      positiveCount > 0
                        ? "bg-green-50 dark:bg-green-950"
                        : "bg-blue-50 dark:bg-blue-950"
                    }`}
                  >
                    <h4
                      className={`font-semibold mb-2 ${
                        positiveCount > 0
                          ? "text-green-800 dark:text-green-200"
                          : "text-blue-800 dark:text-blue-200"
                      }`}
                    >
                      {positiveCount > 0
                        ? `${positiveCount} Exoplanet${
                            positiveCount > 1 ? "s" : ""
                          } Detected!`
                        : "No Exoplanets Detected"}
                    </h4>
                    <p
                      className={`text-sm ${
                        positiveCount > 0
                          ? "text-green-700 dark:text-green-300"
                          : "text-blue-700 dark:text-blue-300"
                      }`}
                    >
                      Average Confidence: {(avgConfidence * 100).toFixed(1)}% |
                      Total Predictions: {predictionResults.feature_count} |
                      Positive: {positiveCount} | Negative: {negativeCount}
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
                            <th className="px-4 py-3 text-left font-semibold">
                              TOI
                            </th>
                            <th className="px-4 py-3 text-left font-semibold">
                              TOIPFX
                            </th>
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
                            const filteredResults =
                              predictionResults.predictions
                                .map((prob, idx) => ({
                                  toi:
                                    predictionResults.metadata?.[idx]?.toi ||
                                    `TOI-${idx + 1}`,
                                  toipfx:
                                    predictionResults.metadata?.[idx]?.toipfx ||
                                    "N/A",
                                  prob,
                                  label:
                                    predictionResults.predicted_labels[idx],
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
                                    No results found for "{searchQuery}"
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
                      const filteredCount =
                        predictionResults.predictions.filter((_, idx) => {
                          const toi =
                            predictionResults.metadata?.[idx]?.toi ||
                            `TOI-${idx + 1}`;
                          const toipfx =
                            predictionResults.metadata?.[idx]?.toipfx || "N/A";
                          return (
                            toi
                              .toLowerCase()
                              .includes(searchQuery.toLowerCase()) ||
                            toipfx
                              .toLowerCase()
                              .includes(searchQuery.toLowerCase())
                          );
                        }).length;
                      const totalPages = Math.ceil(
                        filteredCount / itemsPerPage
                      );

                      if (totalPages <= 1) return null;

                      return (
                        <div className="flex items-center justify-between px-4 py-3 bg-gray-50 dark:bg-gray-800 border-t dark:border-gray-700">
                          <div className="text-sm text-gray-600 dark:text-gray-400">
                            Showing {(currentPage - 1) * itemsPerPage + 1} to{" "}
                            {Math.min(
                              currentPage * itemsPerPage,
                              filteredCount
                            )}{" "}
                            of {filteredCount} results
                          </div>
                          <div className="flex items-center gap-2">
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() =>
                                setCurrentPage((p) => Math.max(1, p - 1))
                              }
                              disabled={currentPage === 1}
                            >
                              <ChevronLeft className="w-4 h-4" />
                            </Button>
                            <div className="flex items-center gap-1">
                              {Array.from(
                                { length: totalPages },
                                (_, i) => i + 1
                              ).map((page) => {
                                if (
                                  page === 1 ||
                                  page === totalPages ||
                                  (page >= currentPage - 1 &&
                                    page <= currentPage + 1)
                                ) {
                                  return (
                                    <Button
                                      key={page}
                                      variant={
                                        page === currentPage
                                          ? "default"
                                          : "outline"
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
                                    <span
                                      key={page}
                                      className="px-2 text-gray-400"
                                    >
                                      ...
                                    </span>
                                  );
                                }
                                return null;
                              })}
                            </div>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() =>
                                setCurrentPage((p) =>
                                  Math.min(totalPages, p + 1)
                                )
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
              </CardContent>
            </Card>

            {/* 2. 3D Visualization */}
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

                {/* Info below 3D viz */}
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

            {/* 3. Cluster Analysis */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="w-5 h-5" />
                  Cluster Analysis
                </CardTitle>
                <CardDescription>
                  Data clustering visualization with AI-powered insights
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-6">
                  {/* Cluster Image - 2/3 width */}
                  <div className="col-span-2">
                    <div className="h-192 bg-gradient-to-br from-gray-900 to-black rounded-lg border border-gray-700 overflow-hidden shadow-lg relative">
                      {/* Placeholder for cluster image */}
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center">
                          <div className="w-24 h-24 bg-blue-500 rounded-lg mx-auto mb-4 flex items-center justify-center">
                            <BarChart3 className="w-12 h-12 text-white" />
                          </div>
                          <p className="text-white text-lg font-medium mb-2">
                            Cluster Visualization
                          </p>
                          <p className="text-gray-400 text-sm">
                            Interactive clustering diagram
                          </p>
                        </div>
                      </div>

                      {/* Overlay info */}
                      <div className="absolute top-4 left-4 bg-black/50 px-3 py-2 rounded-lg">
                        <p className="text-white text-sm font-medium">
                          4 Clusters Detected
                        </p>
                      </div>

                      <div className="absolute bottom-4 right-4 bg-black/50 px-3 py-2 rounded-lg">
                        <p className="text-white text-sm">Confidence: 92.3%</p>
                      </div>
                    </div>
                  </div>

                  {/* AI Agent Chatbot - 1/3 width */}
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
                              Cluster Analysis Assistant
                            </p>
                          </div>
                        </div>
                      </div>

                      {/* Chat Messages */}
                      <div className="flex-1 space-y-3 mb-4 overflow-y-auto p-4">
                        <div className="bg-white/15 backdrop-blur-sm rounded-lg p-3 border border-white/20">
                          <div className="flex items-center gap-2 mb-2">
                            <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                            <span className="text-white text-xs font-medium">
                              AI Agent
                            </span>
                          </div>
                          <p className="text-blue-100 text-xs leading-relaxed">
                            I've analyzed your data and identified 4 distinct
                            clusters. Would you like me to explain what each
                            cluster represents?
                          </p>
                        </div>

                        <div className="bg-white/10 backdrop-blur-sm rounded-lg p-3 border border-white/20 ml-6">
                          <div className="flex items-center gap-2 mb-2">
                            <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                            <span className="text-white text-xs font-medium">
                              You
                            </span>
                          </div>
                          <p className="text-blue-100 text-xs leading-relaxed">
                            Yes, please explain the clusters
                          </p>
                        </div>

                        <div className="bg-white/15 backdrop-blur-sm rounded-lg p-3 border border-white/20">
                          <div className="flex items-center gap-2 mb-2">
                            <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                            <span className="text-white text-xs font-medium">
                              AI Agent
                            </span>
                          </div>
                          <p className="text-blue-100 text-xs leading-relaxed">
                            Cluster 1 (Orange): High-mass planets with short
                            orbital periods. Cluster 2 (Purple): Habitable zone
                            candidates. Cluster 3 (Blue): Gas giants. Cluster 4
                            (Gray): Long-period planets.
                          </p>
                        </div>
                      </div>

                      {/* Chat Input */}
                      <div className="space-y-2 px-4 pb-4">
                        <div className="flex gap-2">
                          <Input
                            placeholder="Ask about clusters..."
                            className="flex-1 bg-white/20 border-white/30 text-white placeholder:text-white/70 text-xs"
                          />
                          <Button
                            size="sm"
                            className="bg-white/20 hover:bg-white/30 text-white"
                          >
                            <MessageSquare className="w-4 h-4" />
                          </Button>
                        </div>

                        {/* Quick Actions */}
                        <div className="flex gap-1">
                          <Button
                            size="sm"
                            variant="outline"
                            className="flex-1 text-xs bg-white/10 border-white/30 text-white hover:bg-white/20"
                          >
                            Explain Clusters
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            className="flex-1 text-xs bg-white/10 border-white/30 text-white hover:bg-white/20"
                          >
                            Show Outliers
                          </Button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="h-[calc(100vh-4rem)] overflow-y-auto from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900 p-4 sm:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <header className=" mb-8">
          <h1 className="text-4xl sm:text-5xl font-bold bg-gradient-to-r dark:text-white bg-clip-text text-transparent mb-3">
            Predict Exoplanets from Your Data
          </h1>
          <p className="text-slate-600 dark:text-slate-400 text-lg">
            Upload your data and get exoplanet predictions using our ML models
          </p>
        </header>

        {/* Stepper */}
        <div className="mb-8">
          <Stepper value={currentStep}>
            {steps.map(({ step, title }) => (
              <StepperItem
                key={step}
                step={step}
                className="not-last:flex-1 max-md:items-start"
              >
                <StepperTrigger className="rounded max-md:flex-col">
                  <StepperIndicator />
                  <div className="text-center md:text-left">
                    <StepperTitle>{title}</StepperTitle>
                  </div>
                </StepperTrigger>
                {step < steps.length && (
                  <StepperSeparator className="max-md:mt-3.5 md:mx-4" />
                )}
              </StepperItem>
            ))}
          </Stepper>
        </div>

        {/* Content */}
        <div className="mb-24">{renderStepContent()}</div>

        {/* Floating Footer */}
        <div className="fixed bottom-6 left-1/2 transform -translate-x-1/2 z-50">
          <div className="flex gap-4 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 p-4">
            <Button
              variant="outline"
              onClick={handleBack}
              disabled={currentStep === 1}
              className="min-w-[100px]"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back
            </Button>

            <Button
              onClick={handleNext}
              disabled={
                currentStep === 3 ||
                (currentStep === 1 && !selectedModel) ||
                (currentStep === 2 &&
                  predictionType === "batch" &&
                  !uploadedFile) ||
                isLoading
              }
              className="min-w-[100px]"
            >
              {isLoading ? (
                <>
                  <span className="mr-2">Predicting...</span>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                </>
              ) : currentStep === 3 ? (
                "Complete"
              ) : currentStep === 2 ? (
                "Predict"
              ) : (
                "Next"
              )}
              {!isLoading && currentStep < 3 && (
                <ArrowRight className="w-4 h-4 ml-2" />
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
