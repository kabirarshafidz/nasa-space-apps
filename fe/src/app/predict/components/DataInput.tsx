"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { FileText, Database, Upload } from "lucide-react";
import { SingleFeatures, Metadata, REQUIRED_FEATURES } from "../types";

interface DataInputProps {
  predictionType: "batch" | "single";
  setPredictionType: (type: "batch" | "single") => void;
  singleFeatures: SingleFeatures;
  onFeatureChange: (featureName: string, value: string) => void;
  metadata: Metadata;
  setMetadata: (metadata: Metadata) => void;
  uploadedFile: File | null;
  onFileUpload: (file: File | null) => void;
  onDragOver: (e: React.DragEvent) => void;
  onDrop: (e: React.DragEvent) => void;
  fileError?: string | null;
}

export function DataInput({
  predictionType,
  setPredictionType,
  singleFeatures,
  onFeatureChange,
  metadata,
  setMetadata,
  uploadedFile,
  onFileUpload,
  onDragOver,
  onDrop,
  fileError,
}: DataInputProps) {
  const MAX_FILE_SIZE = 4 * 1024 * 1024; // 4 MB
  const formatBytes = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${Math.round(bytes / Math.pow(k, i))} ${sizes[i]}`;
  };
  return (
    <Tabs
      value={predictionType}
      onValueChange={(value) => setPredictionType(value as "batch" | "single")}
      className="w-full"
    >
      <TabsList className="grid w-full grid-cols-2">
        <TabsTrigger value="single">Single Prediction</TabsTrigger>
        <TabsTrigger value="batch">Batch Prediction</TabsTrigger>
      </TabsList>

      <TabsContent value="single" className="space-y-4">
        <Card className="bg-background/40 backdrop-blur-sm border-primary/30">
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
                <Label htmlFor="toipfx" className="text-base font-semibold">
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

        <Card className="bg-background/40 backdrop-blur-sm border-primary/30">
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
              {REQUIRED_FEATURES.map((feature) => (
                <div className="space-y-2" key={feature.name}>
                  <Label htmlFor={feature.name}>
                    {feature.name} [{feature.label}]
                  </Label>
                  <Input
                    id={feature.name}
                    type="number"
                    step="any"
                    placeholder="e.g., 12.4"
                    value={singleFeatures[feature.name as keyof SingleFeatures]}
                    onChange={(e) =>
                      onFeatureChange(feature.name, e.target.value)
                    }
                  />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="batch" className="space-y-4">
        <Card className="bg-background/40 backdrop-blur-sm border-primary/30">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="w-5 h-5" />
              Batch Data Upload
            </CardTitle>
            <CardDescription>
              Upload a CSV file with exoplanet data for batch predictions. File
              must include: <strong>toi</strong>, <strong>toipfx</strong>, and
              all 12 feature columns.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div
                className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-8 text-center hover:border-blue-400 dark:hover:border-blue-500 transition-colors cursor-pointer"
                onClick={() => document.getElementById("file-upload")?.click()}
                onDragOver={onDragOver}
                onDrop={onDrop}
              >
                <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                <p className="text-sm text-muted-foreground mb-2">
                  {uploadedFile
                    ? `Selected: ${uploadedFile.name}`
                    : "Click to upload or drag and drop a CSV file"}
                </p>
                <p className="text-xs text-muted-foreground mb-4">
                  Max file size: {formatBytes(MAX_FILE_SIZE)}
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
                  onChange={(e) => onFileUpload(e.target.files?.[0] || null)}
                />
              </div>

              {fileError && (
                <div className="flex items-center gap-2 p-3 bg-red-50 dark:bg-red-950 rounded-lg border border-red-200 dark:border-red-800 text-red-700 dark:text-red-400 text-sm">
                  <svg
                    className="w-5 h-5 shrink-0"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  <span>{fileError}</span>
                </div>
              )}

              {uploadedFile && (
                <div className="flex items-center justify-between p-4 bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-800">
                  <div className="flex items-center gap-3">
                    <FileText className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                    <div>
                      <p className="font-medium text-sm">{uploadedFile.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {(uploadedFile.size / 1024).toFixed(2)} KB
                      </p>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => onFileUpload(null)}
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
              )}
            </div>
          </CardContent>
        </Card>
      </TabsContent>
    </Tabs>
  );
}
