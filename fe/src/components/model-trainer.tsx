"use client"

import { useState } from "react"
import { AlertCircleIcon, CheckCircle2, Download, Loader2, PaperclipIcon, Upload, UploadIcon, XIcon } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select"
import { formatBytes, useFileUpload } from "@/hooks/use-file-upload"

type ModelType = "xgboost" | "random_forest" | "logistic_regression"

interface TrainingResult {
    model_name: string
    model_type: string
    metrics: {
        auc: number
        log_loss: number
        accuracy: number
        precision: number
        recall: number
        f1: number
    }
    best_iteration?: number
    feature_importance?: Record<string, number>
}

export default function ModelTrainer() {
    const maxSize = 100 * 1024 * 1024 // 100MB
    const [modelType, setModelType] = useState<ModelType>("xgboost")
    const [modelName, setModelName] = useState("")
    const [isTraining, setIsTraining] = useState(false)
    const [isUploading, setIsUploading] = useState(false)
    const [trainingResult, setTrainingResult] = useState<TrainingResult | null>(null)
    const [trainingError, setTrainingError] = useState<string | null>(null)
    const [uploadSuccess, setUploadSuccess] = useState<string | null>(null)

    // XGBoost params
    const [xgbEta, setXgbEta] = useState("0.05")
    const [xgbMaxDepth, setXgbMaxDepth] = useState("6")
    const [xgbNumRounds, setXgbNumRounds] = useState("2000")

    // Random Forest params
    const [rfNumEstimators, setRfNumEstimators] = useState("600")
    const [rfMaxDepth, setRfMaxDepth] = useState("")

    // Logistic Regression params
    const [lrC, setLrC] = useState("1.0")

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
    })

    const file = files[0]

    const handleTrainModel = async () => {
        if (!file || !modelName) {
            setTrainingError("Please provide a model name and upload a CSV file")
            return
        }

        setIsTraining(true)
        setTrainingError(null)
        setTrainingResult(null)

        try {
            const formData = new FormData()
            // Ensure we're using the actual File object
            const actualFile = file.file instanceof File ? file.file : null
            if (!actualFile) {
                throw new Error("Invalid file object")
            }

            formData.append("file", actualFile)
            formData.append("model_type", modelType)
            formData.append("model_name", modelName)
            formData.append("test_size", "0.2")
            formData.append("random_state", "42")

            // Add model-specific parameters
            if (modelType === "xgboost") {
                formData.append("xgb_eta", xgbEta)
                formData.append("xgb_max_depth", xgbMaxDepth)
                formData.append("xgb_num_boost_round", xgbNumRounds)
            } else if (modelType === "random_forest") {
                formData.append("rf_n_estimators", rfNumEstimators)
                if (rfMaxDepth) {
                    formData.append("rf_max_depth", rfMaxDepth)
                }
            } else if (modelType === "logistic_regression") {
                formData.append("lr_C", lrC)
            }

            const response = await fetch("https://sm1kbhd2-8000.aue.devtunnels.ms/train", {
                method: "POST",
                body: formData,
            })

            if (!response.ok) {
                const error = await response.json()
                throw new Error(error.detail || "Training failed")
            }

            const result: TrainingResult = await response.json()
            setTrainingResult(result)
        } catch (error) {
            setTrainingError(error instanceof Error ? error.message : "An error occurred during training")
        } finally {
            setIsTraining(false)
        }
    }

    const handleDownloadModel = async () => {
        if (!trainingResult) return

        try {
            const response = await fetch(`https://sm1kbhd2-8000.aue.devtunnels.ms/models/${trainingResult.model_name}/download`)

            if (!response.ok) {
                throw new Error("Failed to download model")
            }

            const blob = await response.blob()
            const url = window.URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url

            // Set filename based on model type
            if (trainingResult.model_type === "xgboost") {
                a.download = `${trainingResult.model_name}_xgboost.zip`
            } else {
                a.download = `${trainingResult.model_name}.pkl`
            }

            document.body.appendChild(a)
            a.click()
            window.URL.revokeObjectURL(url)
            document.body.removeChild(a)
        } catch (error) {
            setTrainingError(error instanceof Error ? error.message : "Failed to download model")
        }
    }

    // File upload for model upload
    const [
        { files: modelFiles, isDragging: isModelDragging, errors: modelErrors },
        {
            handleDragEnter: handleModelDragEnter,
            handleDragLeave: handleModelDragLeave,
            handleDragOver: handleModelDragOver,
            handleDrop: handleModelDrop,
            openFileDialog: openModelFileDialog,
            removeFile: removeModelFile,
            getInputProps: getModelInputProps,
        },
    ] = useFileUpload({
        maxSize: 50 * 1024 * 1024, // 50MB for model files
        accept: ".pkl,.zip",
        multiple: false,
    })

    const modelFile = modelFiles[0]
    const [uploadModelName, setUploadModelName] = useState("")
    const [uploadModelType, setUploadModelType] = useState<ModelType>("xgboost")

    const handleUploadModel = async () => {
        if (!modelFile || !uploadModelName) {
            setTrainingError("Please provide a model name and select a model file")
            return
        }

        setIsUploading(true)
        setTrainingError(null)
        setUploadSuccess(null)

        try {
            const formData = new FormData()
            const actualFile = modelFile.file instanceof File ? modelFile.file : null
            if (!actualFile) {
                throw new Error("Invalid file object")
            }

            formData.append("file", actualFile)
            formData.append("model_name", uploadModelName)
            formData.append("model_type", uploadModelType)

            const response = await fetch("https://sm1kbhd2-8000.aue.devtunnels.ms/models/upload", {
                method: "POST",
                body: formData,
            })

            if (!response.ok) {
                const error = await response.json()
                throw new Error(error.detail || "Upload failed")
            }

            const result = await response.json()
            setUploadSuccess(result.message)
            setUploadModelName("")
            removeModelFile(modelFile.id)
        } catch (error) {
            setTrainingError(error instanceof Error ? error.message : "An error occurred during upload")
        } finally {
            setIsUploading(false)
        }
    }

    return (
        <div className="grid gap-6 lg:grid-cols-2">
            {/* Training Configuration */}
            <Card>
                <CardHeader>
                    <CardTitle>Train Model</CardTitle>
                    <CardDescription>
                        Upload your CSV data and configure model parameters
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                    {/* File Upload */}
                    <div className="space-y-2">
                        <Label>Training Data (CSV)</Label>
                        <div
                            role="button"
                            onClick={openFileDialog}
                            onDragEnter={handleDragEnter}
                            onDragLeave={handleDragLeave}
                            onDragOver={handleDragOver}
                            onDrop={handleDrop}
                            data-dragging={isDragging || undefined}
                            className="border-input hover:bg-accent/50 data-[dragging=true]:bg-accent/50 has-[input:focus]:border-ring has-[input:focus]:ring-ring/50 flex min-h-32 flex-col items-center justify-center rounded-xl border border-dashed p-4 transition-colors has-disabled:pointer-events-none has-disabled:opacity-50 has-[input:focus]:ring-[3px] cursor-pointer"
                        >
                            <input
                                {...getInputProps()}
                                className="sr-only"
                                aria-label="Upload CSV file"
                                disabled={Boolean(file) || isTraining}
                            />

                            <div className="flex flex-col items-center justify-center text-center">
                                <div
                                    className="bg-background mb-2 flex size-11 shrink-0 items-center justify-center rounded-full border"
                                    aria-hidden="true"
                                >
                                    <UploadIcon className="size-4 opacity-60" />
                                </div>
                                <p className="mb-1.5 text-sm font-medium">Upload CSV file</p>
                                <p className="text-muted-foreground text-xs">
                                    Drag & drop or click to browse (max. {formatBytes(maxSize)})
                                </p>
                            </div>
                        </div>

                        {errors.length > 0 && (
                            <div
                                className="text-destructive flex items-center gap-1 text-xs"
                                role="alert"
                            >
                                <AlertCircleIcon className="size-3 shrink-0" />
                                <span>{errors[0]}</span>
                            </div>
                        )}

                        {file && (
                            <div className="space-y-2">
                                <div className="flex items-center justify-between gap-2 rounded-xl border px-4 py-2">
                                    <div className="flex items-center gap-3 overflow-hidden">
                                        <PaperclipIcon
                                            className="size-4 shrink-0 opacity-60"
                                            aria-hidden="true"
                                        />
                                        <div className="min-w-0">
                                            <p className="truncate text-[13px] font-medium">
                                                {file.file.name}
                                            </p>
                                            <p className="text-[11px] text-muted-foreground">
                                                {formatBytes(file.file.size)}
                                            </p>
                                        </div>
                                    </div>

                                    <Button
                                        size="icon"
                                        variant="ghost"
                                        className="text-muted-foreground/80 hover:text-foreground -me-2 size-8 hover:bg-transparent"
                                        onClick={() => removeFile(file.id)}
                                        aria-label="Remove file"
                                        disabled={isTraining}
                                    >
                                        <XIcon className="size-4" aria-hidden="true" />
                                    </Button>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Model Name */}
                    <div className="space-y-2">
                        <Label htmlFor="model-name">Model Name</Label>
                        <Input
                            id="model-name"
                            placeholder="e.g., xgb_model_v1"
                            value={modelName}
                            onChange={(e) => setModelName(e.target.value)}
                            disabled={isTraining}
                        />
                    </div>

                    {/* Model Type */}
                    <div className="space-y-2">
                        <Label htmlFor="model-type">Model Type</Label>
                        <Select
                            value={modelType}
                            onValueChange={(value) => setModelType(value as ModelType)}
                            disabled={isTraining}
                        >
                            <SelectTrigger id="model-type">
                                <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="xgboost">XGBoost</SelectItem>
                                <SelectItem value="random_forest">Random Forest</SelectItem>
                                <SelectItem value="logistic_regression">Logistic Regression</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>

                    {/* Model-specific parameters */}
                    {modelType === "xgboost" && (
                        <div className="space-y-4 p-4 rounded-lg bg-muted/50">
                            <h4 className="text-sm font-semibold">XGBoost Parameters</h4>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <Label htmlFor="xgb-eta">Learning Rate (eta)</Label>
                                    <Input
                                        id="xgb-eta"
                                        type="number"
                                        step="0.01"
                                        value={xgbEta}
                                        onChange={(e) => setXgbEta(e.target.value)}
                                        disabled={isTraining}
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="xgb-depth">Max Depth</Label>
                                    <Input
                                        id="xgb-depth"
                                        type="number"
                                        value={xgbMaxDepth}
                                        onChange={(e) => setXgbMaxDepth(e.target.value)}
                                        disabled={isTraining}
                                    />
                                </div>
                                <div className="space-y-2 col-span-2">
                                    <Label htmlFor="xgb-rounds">Num Boost Rounds</Label>
                                    <Input
                                        id="xgb-rounds"
                                        type="number"
                                        value={xgbNumRounds}
                                        onChange={(e) => setXgbNumRounds(e.target.value)}
                                        disabled={isTraining}
                                    />
                                </div>
                            </div>
                        </div>
                    )}

                    {modelType === "random_forest" && (
                        <div className="space-y-4 p-4 rounded-lg bg-muted/50">
                            <h4 className="text-sm font-semibold">Random Forest Parameters</h4>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <Label htmlFor="rf-estimators">Number of Trees</Label>
                                    <Input
                                        id="rf-estimators"
                                        type="number"
                                        value={rfNumEstimators}
                                        onChange={(e) => setRfNumEstimators(e.target.value)}
                                        disabled={isTraining}
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="rf-depth">Max Depth (optional)</Label>
                                    <Input
                                        id="rf-depth"
                                        type="number"
                                        placeholder="None"
                                        value={rfMaxDepth}
                                        onChange={(e) => setRfMaxDepth(e.target.value)}
                                        disabled={isTraining}
                                    />
                                </div>
                            </div>
                        </div>
                    )}

                    {modelType === "logistic_regression" && (
                        <div className="space-y-4 p-4 rounded-lg bg-muted/50">
                            <h4 className="text-sm font-semibold">Logistic Regression Parameters</h4>
                            <div className="space-y-2">
                                <Label htmlFor="lr-c">Regularization Strength (C)</Label>
                                <Input
                                    id="lr-c"
                                    type="number"
                                    step="0.1"
                                    value={lrC}
                                    onChange={(e) => setLrC(e.target.value)}
                                    disabled={isTraining}
                                />
                            </div>
                        </div>
                    )}

                    <Button
                        onClick={handleTrainModel}
                        disabled={!file || !modelName || isTraining}
                        className="w-full"
                    >
                        {isTraining ? (
                            <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Training Model...
                            </>
                        ) : (
                            "Train Model"
                        )}
                    </Button>
                </CardContent>
            </Card>

            {/* Results */}
            <Card>
                <CardHeader>
                    <CardTitle>Training Results</CardTitle>
                    <CardDescription>
                        Model performance metrics and feature importance
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    {trainingError && (
                        <div className="flex items-start gap-2 p-4 rounded-lg bg-destructive/10 border border-destructive/20">
                            <AlertCircleIcon className="h-5 w-5 text-destructive mt-0.5" />
                            <div>
                                <p className="font-semibold text-destructive">Training Error</p>
                                <p className="text-sm text-destructive/80 mt-1">{trainingError}</p>
                            </div>
                        </div>
                    )}

                    {trainingResult && (
                        <div className="space-y-6">
                            <div className="flex items-start gap-2 p-4 rounded-lg bg-green-50 dark:bg-green-950/20 border border-green-200 dark:border-green-900">
                                <CheckCircle2 className="h-5 w-5 text-green-600 dark:text-green-400 mt-0.5" />
                                <div>
                                    <p className="font-semibold text-green-900 dark:text-green-100">
                                        Training Successful!
                                    </p>
                                    <p className="text-sm text-green-700 dark:text-green-300 mt-1">
                                        Model "{trainingResult.model_name}" trained successfully
                                    </p>
                                </div>
                            </div>

                            {/* Metrics */}
                            <div className="space-y-3">
                                <h4 className="font-semibold text-sm">Performance Metrics</h4>
                                <div className="grid grid-cols-2 gap-3">
                                    <MetricCard
                                        label="AUC"
                                        value={(trainingResult.metrics.auc * 100).toFixed(2)}
                                        suffix="%"
                                    />
                                    <MetricCard
                                        label="Accuracy"
                                        value={(trainingResult.metrics.accuracy * 100).toFixed(2)}
                                        suffix="%"
                                    />
                                    <MetricCard
                                        label="Precision"
                                        value={(trainingResult.metrics.precision * 100).toFixed(2)}
                                        suffix="%"
                                    />
                                    <MetricCard
                                        label="Recall"
                                        value={(trainingResult.metrics.recall * 100).toFixed(2)}
                                        suffix="%"
                                    />
                                    <MetricCard
                                        label="F1 Score"
                                        value={(trainingResult.metrics.f1 * 100).toFixed(2)}
                                        suffix="%"
                                    />
                                    <MetricCard
                                        label="Log Loss"
                                        value={trainingResult.metrics.log_loss.toFixed(4)}
                                    />
                                </div>
                            </div>

                            {trainingResult.best_iteration && (
                                <div className="p-3 rounded-lg bg-muted">
                                    <p className="text-sm">
                                        <span className="font-semibold">Best Iteration:</span>{" "}
                                        {trainingResult.best_iteration}
                                    </p>
                                </div>
                            )}

                            {/* Feature Importance */}
                            {trainingResult.feature_importance && (
                                <div className="space-y-3">
                                    <h4 className="font-semibold text-sm">Top Features</h4>
                                    <div className="space-y-2 max-h-64 overflow-y-auto">
                                        {Object.entries(trainingResult.feature_importance)
                                            .slice(0, 10)
                                            .map(([feature, importance]) => (
                                                <div
                                                    key={feature}
                                                    className="flex items-center justify-between p-2 rounded bg-muted/50"
                                                >
                                                    <span className="text-xs font-mono truncate flex-1">
                                                        {feature}
                                                    </span>
                                                    <span className="text-xs font-semibold ml-2">
                                                        {importance.toFixed(4)}
                                                    </span>
                                                </div>
                                            ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {trainingResult && (
                        <Button
                            onClick={handleDownloadModel}
                            variant="outline"
                            className="w-full"
                        >
                            <Download className="mr-2 h-4 w-4" />
                            Download Model
                        </Button>
                    )}

                    {!trainingResult && !trainingError && (
                        <div className="flex flex-col items-center justify-center py-12 text-center text-muted-foreground">
                            <p className="text-sm">No training results yet</p>
                            <p className="text-xs mt-1">Train a model to see results here</p>
                        </div>
                    )}
                </CardContent>
            </Card>

            {/* Upload Model Card */}
            <Card className="lg:col-span-2">
                <CardHeader>
                    <CardTitle>Upload Pre-trained Model</CardTitle>
                    <CardDescription>
                        Upload a previously trained model file (.pkl for RF/LR or .zip for XGBoost)
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                    {uploadSuccess && (
                        <div className="flex items-start gap-2 p-4 rounded-lg bg-green-50 dark:bg-green-950/20 border border-green-200 dark:border-green-900">
                            <CheckCircle2 className="h-5 w-5 text-green-600 dark:text-green-400 mt-0.5" />
                            <div>
                                <p className="font-semibold text-green-900 dark:text-green-100">
                                    Upload Successful!
                                </p>
                                <p className="text-sm text-green-700 dark:text-green-300 mt-1">
                                    {uploadSuccess}
                                </p>
                            </div>
                        </div>
                    )}

                    <div className="grid md:grid-cols-2 gap-6">
                        {/* Model File Upload */}
                        <div className="space-y-2">
                            <Label>Model File</Label>
                            <div
                                role="button"
                                onClick={openModelFileDialog}
                                onDragEnter={handleModelDragEnter}
                                onDragLeave={handleModelDragLeave}
                                onDragOver={handleModelDragOver}
                                onDrop={handleModelDrop}
                                data-dragging={isModelDragging || undefined}
                                className="border-input hover:bg-accent/50 data-[dragging=true]:bg-accent/50 has-[input:focus]:border-ring has-[input:focus]:ring-ring/50 flex min-h-32 flex-col items-center justify-center rounded-xl border border-dashed p-4 transition-colors has-disabled:pointer-events-none has-disabled:opacity-50 has-[input:focus]:ring-[3px] cursor-pointer"
                            >
                                <input
                                    {...getModelInputProps()}
                                    className="sr-only"
                                    aria-label="Upload model file"
                                    disabled={Boolean(modelFile) || isUploading}
                                />

                                <div className="flex flex-col items-center justify-center text-center">
                                    <div
                                        className="bg-background mb-2 flex size-11 shrink-0 items-center justify-center rounded-full border"
                                        aria-hidden="true"
                                    >
                                        <Upload className="size-4 opacity-60" />
                                    </div>
                                    <p className="mb-1.5 text-sm font-medium">Upload model</p>
                                    <p className="text-muted-foreground text-xs">
                                        .pkl or .zip file (max. 50MB)
                                    </p>
                                </div>
                            </div>

                            {modelErrors.length > 0 && (
                                <div
                                    className="text-destructive flex items-center gap-1 text-xs"
                                    role="alert"
                                >
                                    <AlertCircleIcon className="size-3 shrink-0" />
                                    <span>{modelErrors[0]}</span>
                                </div>
                            )}

                            {modelFile && (
                                <div className="space-y-2">
                                    <div className="flex items-center justify-between gap-2 rounded-xl border px-4 py-2">
                                        <div className="flex items-center gap-3 overflow-hidden">
                                            <PaperclipIcon
                                                className="size-4 shrink-0 opacity-60"
                                                aria-hidden="true"
                                            />
                                            <div className="min-w-0">
                                                <p className="truncate text-[13px] font-medium">
                                                    {modelFile.file.name}
                                                </p>
                                                <p className="text-[11px] text-muted-foreground">
                                                    {formatBytes(modelFile.file.size)}
                                                </p>
                                            </div>
                                        </div>

                                        <Button
                                            size="icon"
                                            variant="ghost"
                                            className="text-muted-foreground/80 hover:text-foreground -me-2 size-8 hover:bg-transparent"
                                            onClick={() => removeModelFile(modelFile.id)}
                                            aria-label="Remove file"
                                            disabled={isUploading}
                                        >
                                            <XIcon className="size-4" aria-hidden="true" />
                                        </Button>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Model Configuration */}
                        <div className="space-y-4">
                            <div className="space-y-2">
                                <Label htmlFor="upload-model-name">Model Name</Label>
                                <Input
                                    id="upload-model-name"
                                    placeholder="e.g., my_pretrained_model"
                                    value={uploadModelName}
                                    onChange={(e) => setUploadModelName(e.target.value)}
                                    disabled={isUploading}
                                />
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="upload-model-type">Model Type</Label>
                                <Select
                                    value={uploadModelType}
                                    onValueChange={(value) => setUploadModelType(value as ModelType)}
                                    disabled={isUploading}
                                >
                                    <SelectTrigger id="upload-model-type">
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="xgboost">XGBoost (.zip)</SelectItem>
                                        <SelectItem value="random_forest">Random Forest (.pkl)</SelectItem>
                                        <SelectItem value="logistic_regression">Logistic Regression (.pkl)</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>

                            <Button
                                onClick={handleUploadModel}
                                disabled={!modelFile || !uploadModelName || isUploading}
                                className="w-full"
                            >
                                {isUploading ? (
                                    <>
                                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                        Uploading Model...
                                    </>
                                ) : (
                                    <>
                                        <Upload className="mr-2 h-4 w-4" />
                                        Upload Model
                                    </>
                                )}
                            </Button>
                        </div>
                    </div>
                </CardContent>
            </Card>
        </div>
    )
}

function MetricCard({
    label,
    value,
    suffix = "",
}: {
    label: string
    value: string
    suffix?: string
}) {
    return (
        <div className="p-3 rounded-lg border bg-card">
            <p className="text-xs text-muted-foreground mb-1">{label}</p>
            <p className="text-lg font-bold">
                {value}
                {suffix && <span className="text-sm font-normal ml-0.5">{suffix}</span>}
            </p>
        </div>
    )
}
