"use client";

import { useState } from "react";
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
} from "lucide-react";
import Link from "next/link";

export default function PredictPage() {
  const [currentStep, setCurrentStep] = useState(1);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [predictionType, setPredictionType] = useState<"batch" | "single">(
    "single"
  );

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

  const preTrainedModels = [
    {
      id: "model1",
      name: "TESS Exoplanet Classifier v1.0",
      accuracy: "94.2%",
      description: "Trained on 50,000+ TESS light curves",
    },
    {
      id: "model2",
      name: "Kepler Transit Detector v2.1",
      accuracy: "91.8%",
      description: "Specialized for Kepler mission data",
    },
    {
      id: "model3",
      name: "Multi-Mission Classifier v1.5",
      accuracy: "89.5%",
      description: "Works with TESS, Kepler, and K2 data",
    },
  ];

  const handleNext = () => {
    if (currentStep < 3) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handleBack = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
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
                            <SelectItem key={model.id} value={model.id}>
                              <div className="flex flex-col">
                                <span className="font-medium">
                                  {model.name}
                                </span>
                                <span className="text-sm text-muted-foreground">
                                  {model.description} • Accuracy:{" "}
                                  {model.accuracy}
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
          <Tabs defaultValue="single" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="single">Single Prediction</TabsTrigger>
              <TabsTrigger value="batch">Batch Prediction</TabsTrigger>
            </TabsList>

            <TabsContent value="single" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileText className="w-5 h-5" />
                    Single Prediction Input
                  </CardTitle>
                  <CardDescription>
                    Enter feature values manually for prediction
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="period">Period (days)</Label>
                      <Input id="period" type="number" placeholder="12.4" />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="duration">Duration (hours)</Label>
                      <Input id="duration" type="number" placeholder="2.1" />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="depth">Depth (ppm)</Label>
                      <Input id="depth" type="number" placeholder="1500" />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="snr">Signal-to-Noise Ratio</Label>
                      <Input id="snr" type="number" placeholder="8.5" />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="impact">Impact Parameter</Label>
                      <Input id="impact" type="number" placeholder="0.3" />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="radius">
                        Planet Radius (Earth radii)
                      </Label>
                      <Input id="radius" type="number" placeholder="1.2" />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="temperature">
                        Effective Temperature (K)
                      </Label>
                      <Input
                        id="temperature"
                        type="number"
                        placeholder="5800"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="stellar_radius">
                        Stellar Radius (Solar radii)
                      </Label>
                      <Input
                        id="stellar_radius"
                        type="number"
                        placeholder="1.0"
                      />
                    </div>
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
                    Upload multiple files or a dataset for batch processing
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-8 text-center">
                    <Database className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                    <p className="text-gray-600 dark:text-gray-400 mb-4">
                      Upload multiple CSV files or a ZIP archive
                    </p>
                    <Button variant="outline">Choose Files</Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        );

      case 3:
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
                <div className="p-4 bg-green-50 dark:bg-green-950 rounded-lg">
                  <h4 className="font-semibold text-green-800 dark:text-green-200 mb-2">
                    Exoplanet Detected! 🪐
                  </h4>
                  <p className="text-green-700 dark:text-green-300 text-sm">
                    Confidence: 87.3% | Period: 12.4 days | Radius: 1.2 Earth
                    radii
                  </p>
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
                currentStep === 3 || (currentStep === 1 && !selectedModel)
              }
              className="min-w-[100px]"
            >
              {currentStep === 3 ? "Complete" : "Next"}
              {currentStep < 3 && <ArrowRight className="w-4 h-4 ml-2" />}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
