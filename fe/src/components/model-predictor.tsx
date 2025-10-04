"use client"

import { useState } from "react"
import { AlertCircleIcon, CheckCircle2, FileUp, Loader2, PaperclipIcon, Send, UploadIcon, XIcon } from "lucide-react"
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
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table"
import {
    Popover,
    PopoverContent,
    PopoverTrigger,
} from "@/components/ui/popover"
import { Badge } from "@/components/ui/badge"
import { formatBytes, useFileUpload } from "@/hooks/use-file-upload"

interface PredictionResult {
    probability: number
    label: number
}

interface RowData {
    [key: string]: any
}

interface CsvPrediction extends RowData {
    _prediction?: PredictionResult
    _isLoading?: boolean
}

export default function ModelPredictor() {
    const maxSize = 100 * 1024 * 1024 // 100MB
    const [selectedModel, setSelectedModel] = useState("")
    const [availableModels, setAvailableModels] = useState<string[]>([])
    const [isLoadingModels, setIsLoadingModels] = useState(false)

    // CSV prediction state
    const [csvData, setCsvData] = useState<CsvPrediction[]>([])
    const [csvHeaders, setCsvHeaders] = useState<string[]>([])
    const [currentPage, setCurrentPage] = useState(1)
    const [rowsPerPage] = useState(10)
    const [csvError, setCsvError] = useState<string | null>(null)

    // Single prediction state
    const featureNames = [
        'log_pl_tranmid', 'log_pl_tranmiderr1', 'log_pl_orbper', 'log_pl_orbpererr1',
        'log_pl_trandurh', 'log_pl_trandurherr1', 'log_pl_trandep', 'log_pl_trandeperr1',
        'log_pl_rade', 'log_pl_radeerr1', 'log_pl_insol', 'log_pl_eqt',
        'log_st_tmag', 'log_st_tmagerr1', 'log_st_dist', 'log_st_disterr1',
        'log_st_teff', 'log_st_tefferr1', 'log_st_logg', 'log_st_loggerr1',
        'log_st_rad', 'log_st_raderr1'
    ]
    const [singleFeatures, setSingleFeatures] = useState<Record<string, string>>(
        Object.fromEntries(featureNames.map(name => [name, '']))
    )
    const [singlePrediction, setSinglePrediction] = useState<PredictionResult | null>(null)
    const [isPredicting, setIsPredicting] = useState(false)
    const [predictionError, setPredictionError] = useState<string | null>(null)

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

    // Load available models
    const loadModels = async () => {
        setIsLoadingModels(true)
        try {
            const response = await fetch("https://sm1kbhd2-8000.aue.devtunnels.ms/models")
            if (!response.ok) throw new Error("Failed to load models")
            const data = await response.json()
            setAvailableModels(data.models || [])
        } catch (error) {
            setCsvError("Failed to load models. Make sure the API is running.")
        } finally {
            setIsLoadingModels(false)
        }
    }

    // Handle CSV upload
    const handleCsvUpload = async () => {
        if (!file || !selectedModel) {
            setCsvError("Please select a model and upload a CSV file")
            return
        }

        setCsvError(null)

        try {
            // Read and parse CSV
            const actualFile = file.file instanceof File ? file.file : null
            if (!actualFile) throw new Error("Invalid file object")

            const text = await actualFile.text()
            const lines = text.trim().split('\n')
            const headers = lines[0].split(',').map(h => h.trim())

            const data: CsvPrediction[] = lines.slice(1).map(line => {
                const values = line.split(',')
                const row: CsvPrediction = {}
                headers.forEach((header, index) => {
                    const value = values[index]?.trim() || ""
                    // Try to convert to number, keep as string if it fails
                    const numValue = parseFloat(value)
                    row[header] = isNaN(numValue) ? value : numValue
                })
                return row
            })

            setCsvHeaders(headers)
            setCsvData(data)
        } catch (error) {
            setCsvError(error instanceof Error ? error.message : "Failed to parse CSV")
        }
    }

    // Predict single row
    const predictRow = async (rowIndex: number) => {
        if (!selectedModel) return

        // Set loading state
        setCsvData(prevData => {
            const updated = [...prevData]
            updated[rowIndex] = { ...updated[rowIndex], _isLoading: true }
            return updated
        })

        try {
            const row = csvData[rowIndex]

            // Create features object, excluding internal fields and ensuring numbers
            const features: Record<string, any> = {}
            Object.entries(row).forEach(([key, value]) => {
                // Skip internal fields
                if (key.startsWith('_')) return

                // Convert to number if it's numeric
                if (typeof value === 'string') {
                    const numValue = parseFloat(value)
                    features[key] = isNaN(numValue) ? value : numValue
                } else {
                    features[key] = value
                }
            })

            const response = await fetch("https://sm1kbhd2-8000.aue.devtunnels.ms/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model_name: selectedModel,
                    features: [features]
                })
            })

            if (!response.ok) {
                const error = await response.json()
                throw new Error(error.detail || "Prediction failed")
            }

            const result = await response.json()

            // Update with prediction result
            setCsvData(prevData => {
                const updated = [...prevData]
                updated[rowIndex] = {
                    ...updated[rowIndex],
                    _prediction: {
                        probability: result.predictions[0],
                        label: result.predicted_labels[0]
                    },
                    _isLoading: false
                }
                return updated
            })
        } catch (error) {
            // Clear loading state on error
            setCsvData(prevData => {
                const updated = [...prevData]
                updated[rowIndex] = { ...updated[rowIndex], _isLoading: false }
                return updated
            })
            setCsvError(error instanceof Error ? error.message : "Prediction failed")
        }
    }

    // Predict single data point
    const handleSinglePredict = async () => {
        if (!selectedModel) {
            setPredictionError("Please select a model")
            return
        }

        if (Object.keys(singleFeatures).length === 0) {
            setPredictionError("Please enter at least one feature")
            return
        }

        setIsPredicting(true)
        setPredictionError(null)
        setSinglePrediction(null)

        try {
            // Convert string values to numbers
            const features: Record<string, any> = {}
            Object.entries(singleFeatures).forEach(([key, value]) => {
                const numValue = parseFloat(value)
                features[key] = isNaN(numValue) ? value : numValue
            })

            const response = await fetch("https://sm1kbhd2-8000.aue.devtunnels.ms/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model_name: selectedModel,
                    features: [features]
                })
            })

            if (!response.ok) {
                const error = await response.json()
                throw new Error(error.detail || "Prediction failed")
            }

            const result = await response.json()
            setSinglePrediction({
                probability: result.predictions[0],
                label: result.predicted_labels[0]
            })
        } catch (error) {
            setPredictionError(error instanceof Error ? error.message : "Prediction failed")
        } finally {
            setIsPredicting(false)
        }
    }

    const updateFeatureValue = (key: string, value: string) => {
        setSingleFeatures({ ...singleFeatures, [key]: value })
    }

    // Pagination
    const totalPages = Math.ceil(csvData.length / rowsPerPage)
    const startIndex = (currentPage - 1) * rowsPerPage
    const endIndex = startIndex + rowsPerPage
    const currentRows = csvData.slice(startIndex, endIndex)

    return (
        <div className="space-y-6">
            {/* Model Selection */}
            <Card>
                <CardHeader>
                    <CardTitle>Select Model</CardTitle>
                    <CardDescription>
                        Choose a trained model to make predictions
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="flex gap-4">
                        <div className="flex-1">
                            <Select
                                value={selectedModel}
                                onValueChange={setSelectedModel}
                            >
                                <SelectTrigger>
                                    <SelectValue placeholder="Select a model..." />
                                </SelectTrigger>
                                <SelectContent>
                                    {availableModels.map((model) => (
                                        <SelectItem key={model} value={model}>
                                            {model}
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </div>
                        <Button onClick={loadModels} disabled={isLoadingModels}>
                            {isLoadingModels ? (
                                <>
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                    Loading...
                                </>
                            ) : (
                                "Load Models"
                            )}
                        </Button>
                    </div>
                    {selectedModel && (
                        <p className="text-sm text-muted-foreground">
                            Selected: <span className="font-semibold">{selectedModel}</span>
                        </p>
                    )}
                </CardContent>
            </Card>

            {/* CSV Batch Prediction */}
            <Card>
                <CardHeader>
                    <CardTitle>Batch Prediction (CSV)</CardTitle>
                    <CardDescription>
                        Upload a CSV file and predict each row individually
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                    {/* CSV Upload */}
                    <div className="space-y-2">
                        <Label>Upload CSV File</Label>
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
                                disabled={Boolean(file)}
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
                                    >
                                        <XIcon className="size-4" aria-hidden="true" />
                                    </Button>
                                </div>
                            </div>
                        )}
                    </div>

                    <Button
                        onClick={handleCsvUpload}
                        disabled={!file || !selectedModel}
                        className="w-full"
                    >
                        <FileUp className="mr-2 h-4 w-4" />
                        Load CSV Data
                    </Button>

                    {csvError && (
                        <div className="flex items-start gap-2 p-4 rounded-lg bg-destructive/10 border border-destructive/20">
                            <AlertCircleIcon className="h-5 w-5 text-destructive mt-0.5" />
                            <div>
                                <p className="font-semibold text-destructive">Error</p>
                                <p className="text-sm text-destructive/80 mt-1">{csvError}</p>
                            </div>
                        </div>
                    )}

                    {/* Data Table */}
                    {csvData.length > 0 && (
                        <div className="space-y-4">
                            <div className="rounded-md border">
                                <Table>
                                    <TableHeader>
                                        <TableRow>
                                            <TableHead className="w-[150px]">Action</TableHead>
                                            {csvHeaders.map((header) => (
                                                <TableHead key={header}>{header}</TableHead>
                                            ))}
                                        </TableRow>
                                    </TableHeader>
                                    <TableBody>
                                        {currentRows.map((row, index) => {
                                            const actualIndex = startIndex + index
                                            return (
                                                <TableRow key={actualIndex}>
                                                    <TableCell>
                                                        <Popover>
                                                            <PopoverTrigger asChild>
                                                                <Button
                                                                    size="sm"
                                                                    onClick={() => !row._prediction && predictRow(actualIndex)}
                                                                    disabled={row._isLoading || !selectedModel}
                                                                    variant={row._prediction ? (row._prediction.label === 1 ? "default" : "secondary") : "outline"}
                                                                >
                                                                    {row._isLoading ? (
                                                                        <>
                                                                            <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                                                                            Predicting...
                                                                        </>
                                                                    ) : row._prediction ? (
                                                                        row._prediction.label === 1 ? "Exoplanet" : "No Exoplanet"
                                                                    ) : (
                                                                        "Predict"
                                                                    )}
                                                                </Button>
                                                            </PopoverTrigger>
                                                            {row._prediction && (
                                                                <PopoverContent className="w-80">
                                                                    <div className="space-y-3">
                                                                        <div className="space-y-2">
                                                                            <h4 className="font-medium leading-none">Prediction Result</h4>
                                                                            <p className="text-sm text-muted-foreground">
                                                                                Classification details for this data point
                                                                            </p>
                                                                        </div>
                                                                        <div className="space-y-2">
                                                                            <div className="flex items-center justify-between">
                                                                                <span className="text-sm font-medium">Classification:</span>
                                                                                <Badge
                                                                                    variant={row._prediction.label === 1 ? "default" : "secondary"}
                                                                                >
                                                                                    {row._prediction.label === 1 ? "Exoplanet" : "No Exoplanet"}
                                                                                </Badge>
                                                                            </div>
                                                                            <div className="flex items-center justify-between">
                                                                                <span className="text-sm font-medium">Confidence:</span>
                                                                                <span className="text-sm">{(row._prediction.probability * 100).toFixed(2)}%</span>
                                                                            </div>
                                                                            <div className="flex items-center justify-between">
                                                                                <span className="text-sm font-medium">Probability:</span>
                                                                                <span className="text-sm font-mono">{row._prediction.probability.toFixed(6)}</span>
                                                                            </div>
                                                                        </div>
                                                                    </div>
                                                                </PopoverContent>
                                                            )}
                                                        </Popover>
                                                    </TableCell>
                                                    {csvHeaders.map((header) => (
                                                        <TableCell key={header} className="font-mono text-xs">
                                                            {row[header]}
                                                        </TableCell>
                                                    ))}
                                                </TableRow>
                                            )
                                        })}
                                    </TableBody>
                                </Table>
                            </div>

                            {/* Pagination */}
                            <div className="flex items-center justify-between">
                                <p className="text-sm text-muted-foreground">
                                    Showing {startIndex + 1} to {Math.min(endIndex, csvData.length)} of {csvData.length} rows
                                </p>
                                <div className="flex gap-2">
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                                        disabled={currentPage === 1}
                                    >
                                        Previous
                                    </Button>
                                    <div className="flex items-center gap-2">
                                        <span className="text-sm">
                                            Page {currentPage} of {totalPages}
                                        </span>
                                    </div>
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                                        disabled={currentPage === totalPages}
                                    >
                                        Next
                                    </Button>
                                </div>
                            </div>
                        </div>
                    )}
                </CardContent>
            </Card>

            {/* Single Data Point Prediction */}
            <Card>
                <CardHeader>
                    <CardTitle>Single Prediction</CardTitle>
                    <CardDescription>
                        Enter feature values manually for a single prediction
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                    {predictionError && (
                        <div className="flex items-start gap-2 p-4 rounded-lg bg-destructive/10 border border-destructive/20">
                            <AlertCircleIcon className="h-5 w-5 text-destructive mt-0.5" />
                            <div>
                                <p className="font-semibold text-destructive">Error</p>
                                <p className="text-sm text-destructive/80 mt-1">{predictionError}</p>
                            </div>
                        </div>
                    )}

                    {/* Feature Inputs */}
                    <div className="space-y-3 max-h-96 overflow-y-auto">
                        {featureNames.map((key) => (
                            <div key={key} className="space-y-1">
                                <Label htmlFor={key} className="text-xs font-mono">{key}</Label>
                                <Input
                                    id={key}
                                    type="number"
                                    step="any"
                                    placeholder="Enter value"
                                    value={singleFeatures[key] || ''}
                                    onChange={(e) => updateFeatureValue(key, e.target.value)}
                                />
                            </div>
                        ))}
                    </div>

                    <Button
                        onClick={handleSinglePredict}
                        disabled={!selectedModel || isPredicting || Object.keys(singleFeatures).length === 0}
                        className="w-full"
                    >
                        {isPredicting ? (
                            <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Predicting...
                            </>
                        ) : (
                            <>
                                <Send className="mr-2 h-4 w-4" />
                                Predict
                            </>
                        )}
                    </Button>

                    {/* Prediction Result */}
                    {singlePrediction && (
                        <div className="p-6 rounded-lg border bg-card space-y-4">
                            <div className="flex items-start gap-2">
                                <CheckCircle2 className="h-5 w-5 text-green-600 dark:text-green-400 mt-0.5" />
                                <div className="flex-1">
                                    <p className="font-semibold text-lg">Prediction Result</p>
                                </div>
                            </div>
                            <div className="space-y-3">
                                <div className="flex items-center justify-between p-4 rounded-lg bg-muted">
                                    <span className="text-sm font-medium">Classification:</span>
                                    <Badge
                                        variant={singlePrediction.label === 1 ? "default" : "secondary"}
                                        className="text-base px-4 py-1"
                                    >
                                        {singlePrediction.label === 1 ? "Exoplanet Detected" : "No Exoplanet"}
                                    </Badge>
                                </div>
                                <div className="flex items-center justify-between p-4 rounded-lg bg-muted">
                                    <span className="text-sm font-medium">Confidence:</span>
                                    <span className="text-lg font-bold">
                                        {(singlePrediction.probability * 100).toFixed(2)}%
                                    </span>
                                </div>
                                <div className="flex items-center justify-between p-4 rounded-lg bg-muted">
                                    <span className="text-sm font-medium">Probability Score:</span>
                                    <span className="text-lg font-mono">
                                        {singlePrediction.probability.toFixed(6)}
                                    </span>
                                </div>
                            </div>
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    )
}
