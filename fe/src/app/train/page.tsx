"use client";

import { useState } from "react";
import {
    Stepper,
    StepperIndicator,
    StepperItem,
    StepperSeparator,
    StepperTitle,
    StepperTrigger,
} from "@/components/ui/stepper";
import { Button } from "@/components/ui/button";
import { CardContent } from "@/components/ui/card";
import { StepCard } from "@/components/StepCard";
import { useFileUpload } from "@/hooks/use-file-upload";
import {
    UploadDataStep,
    PreviewDataStep,
    ConfigureModelStep,
    TrainingResultsStep,
} from "./components";

interface TrainingResult {
    model_name: string;
    model_type: string;
    metrics: {
        auc: number;
        accuracy: number;
        precision: number;
        recall: number;
        f1: number;
        log_loss: number;
    };
    best_iteration?: number;
    feature_importance?: Record<string, number>;
}

interface CsvData {
    [key: string]: string | number;
}

export default function TrainPage() {
    const [currentStep, setCurrentStep] = useState(1);
    const maxSize = 100 * 1024 * 1024; // 100MB

    // Step 1: File upload
    const [
        { files, isDragging, errors },
        {
            handleDragEnter,
            handleDragLeave,
            handleDragOver,
            handleDrop,
            openFileDialog,
            removeFile,
            getInputProps,
        },
    ] = useFileUpload({
        maxSize,
        accept: ".csv,text/csv",
        multiple: false,
    });

    const file = files[0];

    // Step 2: Data preview
    const [csvData, setCsvData] = useState<CsvData[]>([]);
    const [csvHeaders, setCsvHeaders] = useState<string[]>([]);
    const [csvError, setCsvError] = useState<string | null>(null);
    const [currentPage, setCurrentPage] = useState(1);
    const rowsPerPage = 10;

    // Step 3: Model configuration
    const [modelName, setModelName] = useState("");
    const [modelType, setModelType] = useState("xgboost");
    const [testSize, setTestSize] = useState("0.2");
    const [randomState, setRandomState] = useState("42");

    // XGBoost params
    const [xgbEta, setXgbEta] = useState("0.05");
    const [xgbMaxDepth, setXgbMaxDepth] = useState("6");
    const [xgbSubsample, setXgbSubsample] = useState("0.8");
    const [xgbColsampleBytree, setXgbColsampleBytree] = useState("0.8");
    const [xgbNumBoostRound, setXgbNumBoostRound] = useState("2000");
    const [xgbEarlyStoppingRounds, setXgbEarlyStoppingRounds] = useState("50");

    // Step 4: Training & Results
    const [isTraining, setIsTraining] = useState(false);
    const [trainingProgress, setTrainingProgress] = useState(0);
    const [trainingResult, setTrainingResult] = useState<TrainingResult | null>(
        null,
    );
    const [trainingError, setTrainingError] = useState<string | null>(null);
    const [isRetrainDialogOpen, setIsRetrainDialogOpen] = useState(false);

    // Retrain parameters (for dialog)
    const [retrainModelName, setRetrainModelName] = useState("");
    const [retrainModelType, setRetrainModelType] = useState("");
    const [retrainTestSize, setRetrainTestSize] = useState("");

    const steps = [
        {
            step: 1,
            title: "Upload Data",
        },
        {
            step: 2,
            title: "Preview",
        },
        {
            step: 3,
            title: "Configure Model",
        },
        {
            step: 4,
            title: "Result",
        },
    ];

    // Parse CSV file
    const parseCSV = async () => {
        if (!file) {
            setCsvError("No file uploaded");
            return false;
        }

        setCsvError(null);

        try {
            const actualFile = file.file instanceof File ? file.file : null;
            if (!actualFile) throw new Error("Invalid file object");

            const text = await actualFile.text();
            const allLines = text.trim().split("\n");

            // Filter out lines starting with #
            const lines = allLines.filter(line => !line.trim().startsWith("#"));

            if (lines.length === 0) {
                throw new Error("No valid data found in CSV file");
            }

            const headers = lines[0].split(",").map((h) => h.trim());

            const data: CsvData[] = lines.slice(1).map((line) => {
                const values = line.split(",");
                const row: CsvData = {};
                headers.forEach((header, index) => {
                    const value = values[index]?.trim() || "";
                    const numValue = parseFloat(value);
                    row[header] = isNaN(numValue) ? value : numValue;
                });
                return row;
            });

            setCsvHeaders(headers);
            setCsvData(data);
            return true;
        } catch (error) {
            setCsvError(
                error instanceof Error ? error.message : "Failed to parse CSV",
            );
            return false;
        }
    };

    // Handle next step
    const handleNext = async () => {
        if (currentStep === 1) {
            // Validate file upload
            if (!file) {
                setCsvError("Please upload a CSV file");
                return;
            }
            // Parse CSV and move to preview
            const success = await parseCSV();
            if (success) {
                setCurrentStep(2);
            }
        } else if (currentStep === 2) {
            // Move to configuration
            setCurrentStep(3);
        } else if (currentStep === 3) {
            // Validate configuration
            if (!modelName.trim()) {
                setTrainingError("Please enter a model name");
                return;
            }
            // Start training
            await handleTrain();
        }
    };

    // Handle previous step
    const handlePrevious = () => {
        if (currentStep > 1) {
            setCurrentStep(currentStep - 1);
        }
    };

    // Handle training
    const handleTrain = async (params?: {
        modelName?: string;
        modelType?: string;
        testSize?: string;
    }) => {
        const trainModelName = params?.modelName || modelName;
        const trainModelType = params?.modelType || modelType;
        const trainTestSize = params?.testSize || testSize;

        setIsTraining(true);
        setTrainingError(null);
        setTrainingProgress(0);
        setCurrentStep(4);

        try {
            // Simulate progress
            const progressInterval = setInterval(() => {
                setTrainingProgress((prev) => {
                    if (prev >= 90) {
                        clearInterval(progressInterval);
                        return prev;
                    }
                    return prev + 10;
                });
            }, 500);

            const formData = new FormData();
            if (file && file.file instanceof File) {
                formData.append("file", file.file);
            }
            formData.append("model_name", trainModelName);
            formData.append("model_type", trainModelType);
            formData.append("test_size", trainTestSize);
            formData.append("random_state", randomState);

            // XGBoost params
            formData.append("xgb_eta", xgbEta);
            formData.append("xgb_max_depth", xgbMaxDepth);
            formData.append("xgb_subsample", xgbSubsample);
            formData.append("xgb_colsample_bytree", xgbColsampleBytree);
            formData.append("xgb_num_boost_round", xgbNumBoostRound);
            formData.append("xgb_early_stopping_rounds", xgbEarlyStoppingRounds);

            const response = await fetch(
                "http://10.16.146.135:8000/train",
                {
                    method: "POST",
                    body: formData,
                },
            );

            clearInterval(progressInterval);

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || "Training failed");
            }

            const result = await response.json();
            setTrainingResult(result);
            setTrainingProgress(100);
        } catch (error) {
            setTrainingError(
                error instanceof Error ? error.message : "Training failed",
            );
            setTrainingProgress(0);
        } finally {
            setIsTraining(false);
        }
    };

    // Handle retrain with dialog
    const handleRetrainClick = () => {
        setRetrainModelName(modelName);
        setRetrainModelType(modelType);
        setRetrainTestSize(testSize);
        setIsRetrainDialogOpen(true);
    };

    const handleRetrainSubmit = async () => {
        if (!retrainModelName.trim()) {
            setTrainingError("Please enter a model name");
            return;
        }
        setIsRetrainDialogOpen(false);
        setTrainingResult(null);
        await handleTrain({
            modelName: retrainModelName,
            modelType: retrainModelType,
            testSize: retrainTestSize,
        });
    };

    const handleTrainAnother = () => {
        setCurrentStep(1);
        setCsvData([]);
        setCsvHeaders([]);
        setTrainingResult(null);
        setTrainingError(null);
        setModelName("");
    };

    return (
        <div className="flex flex-col h-[calc(100vh-4rem)]">
            <div className="px-10 py-8   w-full flex-1 overflow-y-auto pb-32">
              <div className="max-w-7xl w-full mx-auto">
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-primary-foreground mb-2">
                        Train New Model
                    </h1>
                    <p className="text-primary-foreground/70">
                        Follow the steps to upload your data and train a new model
                    </p>
                </div>

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

                {/* Step Content */}
                <StepCard>
                    <CardContent className="pt-6">
                    {/* Step 1: Upload Data */}
                    {currentStep === 1 && (
                        <UploadDataStep
                            file={file}
                            errors={errors}
                            csvError={csvError}
                            isDragging={isDragging}
                            maxSize={maxSize}
                            handleDragEnter={handleDragEnter as (e: React.DragEvent<Element>) => void}
                            handleDragLeave={handleDragLeave as (e: React.DragEvent<Element>) => void}
                            handleDragOver={handleDragOver as (e: React.DragEvent<Element>) => void}
                            handleDrop={handleDrop as (e: React.DragEvent<Element>) => void}
                            openFileDialog={openFileDialog}
                            removeFile={removeFile}
                            getInputProps={getInputProps}
                        />
                    )}

                    {/* Step 2: Preview Data */}
                    {currentStep === 2 && (
                        <PreviewDataStep
                            csvData={csvData}
                            csvHeaders={csvHeaders}
                            currentPage={currentPage}
                            rowsPerPage={rowsPerPage}
                            setCurrentPage={setCurrentPage}
                        />
                    )}

                    {/* Step 3: Configure Model */}
                    {currentStep === 3 && (
                        <ConfigureModelStep
                            modelName={modelName}
                            setModelName={setModelName}
                            modelType={modelType}
                            setModelType={setModelType}
                            testSize={testSize}
                            setTestSize={setTestSize}
                            xgbEta={xgbEta}
                            setXgbEta={setXgbEta}
                            xgbMaxDepth={xgbMaxDepth}
                            setXgbMaxDepth={setXgbMaxDepth}
                            xgbNumBoostRound={xgbNumBoostRound}
                            setXgbNumBoostRound={setXgbNumBoostRound}
                            trainingError={trainingError}
                        />
                    )}

                    {/* Step 4: Training & Results */}
                    {currentStep === 4 && (
                        <TrainingResultsStep
                            isTraining={isTraining}
                            trainingProgress={trainingProgress}
                            trainingResult={trainingResult}
                            trainingError={trainingError}
                            modelName={modelName}
                            isRetrainDialogOpen={isRetrainDialogOpen}
                            setIsRetrainDialogOpen={setIsRetrainDialogOpen}
                            handleRetrainClick={handleRetrainClick}
                            handleRetrainSubmit={handleRetrainSubmit}
                            handleTrainAnother={handleTrainAnother}
                            retrainModelName={retrainModelName}
                            setRetrainModelName={setRetrainModelName}
                            retrainModelType={retrainModelType}
                            setRetrainModelType={setRetrainModelType}
                            retrainTestSize={retrainTestSize}
                            setRetrainTestSize={setRetrainTestSize}
                        />
                    )}

                    </CardContent>
                </StepCard>
            </div>

            {/* Navigation Buttons - Fixed at Bottom */}
            <div className="fixed bottom-0 left-0 right-0 bg-background border-t border-primary/20 shadow-lg">
                <div className="w-full max-w-7xl mx-auto py-4">
                    <div className="flex justify-between">
                        <Button
                            variant="outline"
                            onClick={handlePrevious}
                            disabled={currentStep === 1 || isTraining}
                            className="border-primary/30 bg-primary/5 hover:bg-primary/20 text-primary-foreground"
                        >
                            Previous
                        </Button>

                        {currentStep < 4 && (
                            <Button
                                onClick={handleNext}
                                disabled={isTraining}
                                className="bg-primary text-primary-foreground hover:bg-primary/90"
                            >
                                {currentStep === 3 ? "Start Training" : "Next"}
                            </Button>
                        )}
                    </div>
                </div>
            </div>
            </div>
        </div>
    );
}
