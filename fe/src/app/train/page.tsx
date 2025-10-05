"use client";

import { useState, useEffect } from "react";
import { useTrainParams } from "./useTrainParams";
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
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import { useFileUpload } from "@/hooks/use-file-upload";
import {
    UploadDataStep,
    PreviewDataStep,
    ConfigureModelStep,
    TrainingResultsStep,
    TrainingSessionList,
} from "./components";
import { TrainingHistory } from "./components/TrainingHistory";
import { getTrainingSessions, getTrainingSession } from "./actions";

interface TrainingSession {
    id: string;
    userId: string;
    createdAt: Date;
    updatedAt: Date | null;
    csvUrl: string | null;
}

interface TrainingResult {
    model_name: string;
    model_type: string;
    oof_metrics: {
        roc_auc: number;
        pr_auc: number;
        precision: number;
        recall: number;
        f1: number;
        logloss: number;
    };
    fold_metrics: Array<{
        roc_auc: number;
        pr_auc: number;
        precision: number;
        recall: number;
        f1: number;
        logloss: number;
    }>;
    confusion: {
        threshold: number;
        counts: {
            TP: number;
            TN: number;
            FP: number;
            FN: number;
            P: number;
            N: number;
        };
        rates: {
            TPR: number;
            TNR: number;
            FPR: number;
            FNR: number;
            PPV: number;
            NPV: number;
            ACC: number;
        };
        matrix: number[][];
    };
    model_url: string;
    charts: {
        roc_curve?: string;
        pr_curve?: string;
        confusion_matrix?: string;
        feature_importance?: string;
        cv_metrics?: string;
        correlation_heatmap?: string;
    };
    timestamp: string;
}

interface CsvData {
    [key: string]: string | number;
}

interface TrainingEntryData {
    id: string;
    result: TrainingResult;
    modelS3Key: string | null;
    createdAt: Date;
    modelId: string | null;
    modelName: string | null;
    modelF1Score: string | null;
}

export default function TrainPage() {
    // URL search params for session and step
    const [params, setParams] = useTrainParams();
    const selectedSessionId = params.session || null;
    const currentStep = params.step;
    const entryId = params.entryId || null;

    // Session management
    const [sessions, setSessions] = useState<TrainingSession[]>([]);
    const [showTrainingSteps, setShowTrainingSteps] = useState(false);
    const [isLoadingSessions, setIsLoadingSessions] = useState(true);
    const [usingExistingCSV, setUsingExistingCSV] = useState(false);
    const maxSize = 100 * 1024 * 1024; // 100MB

    // Load sessions on mount
    useEffect(() => {
        loadSessions();
    }, []);

    // Load training entry result when entryId changes
    useEffect(() => {
        const loadTrainingEntry = async () => {
            if (!entryId || !selectedSessionId) return;

            try {
                const response = await fetch(`/api/training/entries?sessionId=${selectedSessionId}`);
                const data = await response.json();

                if (data.success && data.entries.length > 0) {
                    const entry = data.entries.find((e: TrainingEntryData) => e.id === entryId);
                    if (entry && entry.result) {
                        setTrainingResult(entry.result);
                        setModelName(entry.modelName || "");
                    }
                }

                // Check if session has CSV to set usingExistingCSV flag
                const sessionResult = await getTrainingSession(selectedSessionId);
                if (sessionResult.success && sessionResult.session) {
                    const hasCSV = !!sessionResult.session.csvUrl;
                    console.log("Session has CSV:", hasCSV, "URL:", sessionResult.session.csvUrl);
                    setUsingExistingCSV(hasCSV);
                }
            } catch (error) {
                console.error("Error loading training entry:", error);
            }
        };

        loadTrainingEntry();
    }, [entryId, selectedSessionId]);

    const loadSessions = async () => {
        setIsLoadingSessions(true);
        const result = await getTrainingSessions();
        if (result.success) {
            setSessions(result.sessions);
        }
        setIsLoadingSessions(false);
    };

    const handleSessionSelect = (sessionId: string) => {
        setParams({ session: sessionId, step: 1, entryId: "" });
        setShowTrainingSteps(false); // Keep on session list to show history
    };

    const handleStartTraining = async (sessionId: string, hasCSV: boolean) => {
        // If continuing with existing session, load the latest training entry
        if (hasCSV) {
            try {
                const response = await fetch(`/api/training/entries?sessionId=${sessionId}`);
                const data = await response.json();

                if (data.success && data.entries.length > 0) {
                    const latestEntry = data.entries[0]; // Already sorted by createdAt desc
                    setParams({ session: sessionId, step: 4, entryId: latestEntry.id });
                    setShowTrainingSteps(true);
                    setUsingExistingCSV(hasCSV);
                    return;
                }
            } catch (error) {
                console.error("Error loading latest entry:", error);
            }
        }

        // New session or no entries found
        setParams({ session: sessionId, step: hasCSV ? 3 : 1, entryId: "" });
        setShowTrainingSteps(true);
        setUsingExistingCSV(hasCSV);
        // Skip to step 3 (configure) if CSV already uploaded, otherwise start at step 1 (upload)
    };

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
    const [cvFolds, setCvFolds] = useState("5");
    const [calibrationEnabled, setCalibrationEnabled] = useState(true);
    const [calibrationMethod, setCalibrationMethod] = useState("isotonic");
    const [imputerKind, setImputerKind] = useState("knn");
    const [imputerK, setImputerK] = useState("5");
    const [threshold, setThreshold] = useState("0.5");
    const [modelParams, setModelParams] = useState("{}");

    // Step 4: Training & Results
    const [isTraining, setIsTraining] = useState(false);
    const [trainingProgress, setTrainingProgress] = useState(0);
    const [trainingResult, setTrainingResult] = useState<TrainingResult | null>(
        null,
    );
    const [trainingError, setTrainingError] = useState<string | null>(null);
    const [isFineTuneDialogOpen, setIsFineTuneDialogOpen] = useState(false);

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
                setParams({ step: 2 });
            }
        } else if (currentStep === 2) {
            // Move to configuration
            setParams({ step: 3 });
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
            setParams({ step: currentStep - 1 });
        }
    };

    // Handle training
    const handleTrain = async () => {
        console.log("=== handleTrain called ===");
        console.log("selectedSessionId:", selectedSessionId);
        console.log("file:", file);
        console.log("usingExistingCSV:", usingExistingCSV);
        console.log("modelName:", modelName);
        console.log("modelType:", modelType);

        if (!selectedSessionId) {
            setTrainingError("No training session selected");
            return;
        }

        // Check if we have a file or are using existing CSV
        if (!file && !usingExistingCSV) {
            setTrainingError("No CSV file provided. Please upload a file or use an existing session with data.");
            return;
        }

        const trainModelType = modelType;

        setIsTraining(true);
        setTrainingError(null);
        setTrainingProgress(0);
        setParams({ step: 4 });

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

            // Only append file if we have a new file upload
            if (file && file.file instanceof File) {
                console.log("Appending file to formData");
                formData.append("file", file.file);
            } else {
                console.log("No file to append, using existing CSV from session");
            }

            formData.append("model_name", trainModelType);
            formData.append("model_params", modelParams);
            formData.append("cv_folds", cvFolds);
            formData.append("calibration_enabled", calibrationEnabled.toString());
            formData.append("calibration_method", calibrationMethod);
            formData.append("imputer_kind", imputerKind);
            formData.append("imputer_k", imputerK);
            formData.append("threshold", threshold);
            formData.append("training_session_id", selectedSessionId);
            formData.append("user_model_name", modelName);

            console.log("Sending training request to API...");
            console.log("FormData contents:");
            for (const [key, value] of formData.entries()) {
                if (value instanceof File) {
                    console.log(`  ${key}: File(${value.name}, ${value.size} bytes)`);
                } else {
                    console.log(`  ${key}: ${value}`);
                }
            }

            const response = await fetch("/api/training/save-result", {
                method: "POST",
                body: formData,
            });

            clearInterval(progressInterval);

            if (!response.ok) {
                const error = await response.json();
                console.error("Training API error:", error);
                throw new Error(error.detail || error.error || "Training failed");
            }

            const result = await response.json();
            console.log("Training result:", result);
            setTrainingResult(result);
            setTrainingProgress(100);

            // Update URL params with the new entry ID
            if (result.entryId) {
                setParams({ entryId: result.entryId });
            }
        } catch (error) {
            console.error("Training error:", error);
            setTrainingError(
                error instanceof Error ? error.message : "Training failed",
            );
            setTrainingProgress(0);
        } finally {
            setIsTraining(false);
        }
    };

    // Handle fine-tune - open dialog to modify parameters
    const handleFineTune = () => {
        setIsFineTuneDialogOpen(true);
    };

    // Handle fine-tune submit
    const handleFineTuneSubmit = async () => {
        console.log("=== handleFineTuneSubmit called ===");
        if (!modelName.trim()) {
            setTrainingError("Please enter a model name");
            return;
        }
        setIsFineTuneDialogOpen(false);
        setTrainingResult(null);
        setTrainingError(null);

        // Ensure we're flagged as using existing CSV for fine-tuning
        if (!file && selectedSessionId) {
            console.log("Fine-tuning without new file, verifying session has CSV");
            const sessionResult = await getTrainingSession(selectedSessionId);
            if (sessionResult.success && sessionResult.session?.csvUrl) {
                console.log("Session has CSV URL, setting usingExistingCSV to true");
                setUsingExistingCSV(true);
            } else {
                console.error("Session doesn't have CSV URL!");
                setTrainingError("Session doesn't have a CSV file. Please upload a new file.");
                return;
            }
        }

        await handleTrain();
    };

    const handleTrainAnother = () => {
        setShowTrainingSteps(false);
        setParams({ session: "", step: 1, entryId: "" });
        setUsingExistingCSV(false);
        setCsvData([]);
        setCsvHeaders([]);
        setTrainingResult(null);
        setTrainingError(null);
        setModelName("");
        loadSessions();
    };

    return (
        <div className="flex flex-col h-[calc(100vh-4rem)]">
            <div className="px-10 py-8 w-full flex-1 overflow-y-auto pb-32">
              <div className="max-w-7xl w-full mx-auto">
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-primary-foreground mb-2">
                        Train New Model
                    </h1>
                    <p className="text-primary-foreground/70">
                        {showTrainingSteps
                            ? "Follow the steps to upload your data and train a new model"
                            : "Manage your training sessions or create a new one to get started"}
                    </p>
                </div>

                {/* Training Session List */}
                {!showTrainingSteps && !isLoadingSessions && (
                    <>
                        <TrainingSessionList
                            sessions={sessions}
                            selectedSessionId={selectedSessionId}
                            onSessionSelect={handleSessionSelect}
                            onStartTraining={handleStartTraining}
                            onSessionsChange={loadSessions}
                        />

                        {/* Show training history for selected session */}
                        {selectedSessionId && (
                            <div className="mt-6">
                                <TrainingHistory sessionId={selectedSessionId} />
                            </div>
                        )}
                    </>
                )}

                {isLoadingSessions && !showTrainingSteps && (
                    <div className="text-center py-12">
                        <p className="text-muted-foreground">Loading sessions...</p>
                    </div>
                )}

                {/* Stepper */}
                {showTrainingSteps && (
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
                )}

                {/* Step Content */}
                {showTrainingSteps && (
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
                        <>
                            {usingExistingCSV && (
                                <div className="mb-4 p-4 bg-primary/10 border border-primary/30 rounded-lg">
                                    <p className="text-sm text-primary-foreground">
                                        ℹ️ Using existing dataset from this session. The CSV will be automatically included in training.
                                    </p>
                                </div>
                            )}
                            <ConfigureModelStep
                            modelName={modelName}
                            setModelName={setModelName}
                            modelType={modelType}
                            setModelType={setModelType}
                            cvFolds={cvFolds}
                            setCvFolds={setCvFolds}
                            calibrationEnabled={calibrationEnabled}
                            setCalibrationEnabled={setCalibrationEnabled}
                            calibrationMethod={calibrationMethod}
                            setCalibrationMethod={setCalibrationMethod}
                            imputerKind={imputerKind}
                            setImputerKind={setImputerKind}
                            imputerK={imputerK}
                            setImputerK={setImputerK}
                            threshold={threshold}
                            setThreshold={setThreshold}
                            modelParams={modelParams}
                            setModelParams={setModelParams}
                            trainingError={trainingError}
                        />
                        </>
                    )}

                    {/* Step 4: Training & Results */}
                    {currentStep === 4 && (
                        <TrainingResultsStep
                            isTraining={isTraining}
                            trainingProgress={trainingProgress}
                            trainingResult={trainingResult}
                            trainingError={trainingError}
                            modelName={modelName}
                            sessionId={selectedSessionId}
                        />
                    )}

                    </CardContent>
                </StepCard>
                )}
            </div>

            {/* Navigation Buttons - Fixed at Bottom */}
            {showTrainingSteps && (
            <div className="fixed bottom-0 left-0 right-0 bg-background border-t border-primary/20 shadow-lg">
                <div className="w-full max-w-7xl mx-auto py-4">
                    <div className="flex justify-between">
                        <div className="flex gap-2">
                            {currentStep === 1 && (
                                <Button
                                    variant="outline"
                                    onClick={() => {
                                        setShowTrainingSteps(false);
                                        setParams({ session: "", step: 1, entryId: "" });
                                    }}
                                    disabled={isTraining}
                                    className="border-primary/30 bg-primary/5 hover:bg-primary/20 text-primary-foreground"
                                >
                                    ← Back to Sessions
                                </Button>
                            )}
                            {currentStep > 1 && (
                                <Button
                                    variant="outline"
                                    onClick={handlePrevious}
                                    disabled={isTraining}
                                    className="border-primary/30 bg-primary/5 hover:bg-primary/20 text-primary-foreground"
                                >
                                    Previous
                                </Button>
                            )}
                        </div>

                        {currentStep < 4 && (
                            <Button
                                onClick={handleNext}
                                disabled={isTraining}
                                className="bg-primary text-primary-foreground hover:bg-primary/90"
                            >
                                {currentStep === 3 ? "Start Training" : "Next"}
                            </Button>
                        )}

                        {currentStep === 4 && !isTraining && trainingResult && (
                            <>
                                <Button
                                    variant="outline"
                                    onClick={handleFineTune}
                                    className="border-primary/30 bg-primary/5 hover:bg-primary/20 text-primary-foreground"
                                >
                                    Fine Tune
                                </Button>
                                <Button
                                    onClick={handleTrainAnother}
                                    className="bg-primary text-primary-foreground hover:bg-primary/90"
                                >
                                    Train Another Model
                                </Button>
                            </>
                        )}
                    </div>
                </div>
            </div>
        )}

            {/* Fine-Tune Dialog */}
            <Dialog
                open={isFineTuneDialogOpen}
                onOpenChange={setIsFineTuneDialogOpen}
            >
                <DialogContent className="sm:max-w-5xl bg-background border-primary/30 max-w-4xl max-h-[90vh] flex flex-col">
                    <DialogHeader>
                        <DialogTitle className="text-primary-foreground">
                            Fine-Tune Model Parameters
                        </DialogTitle>
                        <DialogDescription className="text-primary-foreground/70">
                            Adjust all training parameters and retrain the model
                            with the same dataset.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="py-4 overflow-y-auto flex-1">
                        <ConfigureModelStep
                            modelName={modelName}
                            setModelName={setModelName}
                            modelType={modelType}
                            setModelType={setModelType}
                            cvFolds={cvFolds}
                            setCvFolds={setCvFolds}
                            calibrationEnabled={calibrationEnabled}
                            setCalibrationEnabled={setCalibrationEnabled}
                            calibrationMethod={calibrationMethod}
                            setCalibrationMethod={setCalibrationMethod}
                            imputerKind={imputerKind}
                            setImputerKind={setImputerKind}
                            imputerK={imputerK}
                            setImputerK={setImputerK}
                            threshold={threshold}
                            setThreshold={setThreshold}
                            modelParams={modelParams}
                            setModelParams={setModelParams}
                            trainingError={trainingError}
                        />
                    </div>
                    <DialogFooter className="border-t border-primary/30 pt-4 mt-4">
                        <Button
                            variant="outline"
                            onClick={() => setIsFineTuneDialogOpen(false)}
                            className="border-primary/30"
                        >
                            Cancel
                        </Button>
                        <Button
                            onClick={handleFineTuneSubmit}
                            className="bg-primary text-primary-foreground hover:bg-primary/90"
                        >
                            Start Training
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
            </div>
        </div>
    );
}
