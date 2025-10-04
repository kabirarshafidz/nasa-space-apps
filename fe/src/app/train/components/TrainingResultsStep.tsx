"use client";

import React from "react";
import {
    AlertCircleIcon,
    CheckCircle2,
    Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from "@/components/ui/dialog";
import {
    ChartConfig,
    ChartContainer,
    ChartTooltip,
    ChartTooltipContent,
} from "@/components/ui/chart";
import {
    Label as RechartsLabel,
    PolarGrid,
    PolarRadiusAxis,
    RadialBar,
    RadialBarChart,
    Bar,
    BarChart,
    XAxis,
    YAxis,
} from "recharts";

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

interface TrainingResultsStepProps {
    isTraining: boolean;
    trainingProgress: number;
    trainingResult: TrainingResult | null;
    trainingError: string | null;
    modelName: string;
    isRetrainDialogOpen: boolean;
    setIsRetrainDialogOpen: (open: boolean) => void;
    handleRetrainClick: () => void;
    handleRetrainSubmit: () => void;
    handleTrainAnother: () => void;
    retrainModelName: string;
    setRetrainModelName: (value: string) => void;
    retrainModelType: string;
    setRetrainModelType: (value: string) => void;
    retrainTestSize: string;
    setRetrainTestSize: (value: string) => void;
}

function MetricCard({ label, value, suffix = "" }: { label: string; value: string; suffix?: string }) {
    return (
        <div className="bg-background border border-primary/30 rounded-lg p-4">
            <p className="text-xs text-primary-foreground/60 mb-1">
                {label}
            </p>
            <p className="text-lg font-bold text-primary-foreground">
                {value}{suffix}
            </p>
        </div>
    );
}

export function TrainingResultsStep({
    isTraining,
    trainingProgress,
    trainingResult,
    trainingError,
    modelName,
    isRetrainDialogOpen,
    setIsRetrainDialogOpen,
    handleRetrainClick,
    handleRetrainSubmit,
    handleTrainAnother,
    retrainModelName,
    setRetrainModelName,
    retrainModelType,
    setRetrainModelType,
    retrainTestSize,
    setRetrainTestSize,
}: TrainingResultsStepProps) {
    // Prepare chart data
    const f1Score = trainingResult?.metrics.f1
        ? Math.round(trainingResult.metrics.f1 * 100)
        : 0;

    const chartData = [
        { metric: "f1", value: f1Score },
    ];

    const chartConfig = {
        value: {
            label: "F1 Score",
        },
        f1: {
            label: "F1",
            color: "#9556E8",
        },
    } satisfies ChartConfig;

    // State for showing all features
    const [showAllFeatures, setShowAllFeatures] = React.useState(false);

    // Prepare feature importance data for chart
    const allFeatureData = trainingResult?.feature_importance
        ? Object.entries(trainingResult.feature_importance)
              .sort(([, a], [, b]) => (b as number) - (a as number))
              .map(([feature, importance]) => ({
                  feature,
                  importance: importance as number,
                  fill: "hsl(var(--primary))",
              }))
        : [];

    const featureData = showAllFeatures
        ? allFeatureData
        : allFeatureData.slice(0, 5);

    const totalFeatures = allFeatureData.length;

    return (
        <div className="space-y-6">
            <div>
                <h2 className="text-xl font-semibold text-primary-foreground mb-2">
                    {isTraining
                        ? "Training in Progress..."
                        : "Training Results"}
                </h2>
                <p className="text-sm text-primary-foreground/70">
                    {isTraining
                        ? "Please wait while your model is being trained"
                        : "Your model has been trained successfully"}
                </p>
            </div>

            {isTraining && (
                <div className="space-y-4">
                    <div className="flex items-center gap-3">
                        <Loader2 className="h-6 w-6 animate-spin text-primary" />
                        <p className="text-sm text-primary-foreground">
                            Training model: {modelName}
                        </p>
                    </div>
                    <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                            <span className="text-primary-foreground/70">
                                Progress
                            </span>
                            <span className="text-primary-foreground font-medium">
                                {trainingProgress}%
                            </span>
                        </div>
                        <Progress value={trainingProgress} className="h-2" />
                    </div>
                </div>
            )}

            {trainingError && (
                <div
                    className="text-destructive flex items-center gap-2 text-sm bg-destructive/10 border border-destructive/30 rounded-lg p-4"
                    role="alert"
                >
                    <AlertCircleIcon className="size-5 shrink-0" />
                    <div>
                        <p className="font-medium">Training Failed</p>
                        <p className="text-xs mt-1">{trainingError}</p>
                    </div>
                </div>
            )}

            {trainingResult && !isTraining && (
                <div className="space-y-6">
                    {/* Success Message */}
                    <div className="flex items-center gap-2 text-green-500 bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                        <CheckCircle2 className="size-5 shrink-0" />
                        <div>
                            <p className="font-medium">
                                Training Successful!
                            </p>
                            <p className="text-xs mt-1">
                                Model &quot;{trainingResult.model_name}&quot; trained successfully
                            </p>
                        </div>
                    </div>

                    {/* Performance Metrics Section */}
                    <div className="space-y-4">
                        <h3 className="text-lg font-semibold text-primary-foreground">
                            Performance Metrics
                        </h3>

                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            {/* Radial Chart - F1 Score */}
                            <div className="flex items-center justify-center p-6">
                                <div className="w-full max-w-[280px]">
                                    <ChartContainer
                                        config={chartConfig}
                                        className="mx-auto aspect-square max-h-[280px]"
                                    >
                                        <RadialBarChart
                                            data={chartData}
                                            startAngle={90}
                                            endAngle={(90-(f1Score * 360 / 100))}
                                            innerRadius={80}
                                            outerRadius={110}
                                        >
                                            <ChartTooltip
                                                cursor={false}
                                                content={<ChartTooltipContent hideLabel nameKey="metric" />}
                                            />
                                            <PolarGrid
                                                gridType="circle"
                                                radialLines={false}
                                                stroke="none"
                                                className="first:fill-muted last:fill-background"
                                                polarRadius={[86, 74]}
                                            />
                                            <RadialBar
                                                dataKey="value"
                                                background
                                                cornerRadius={10}
                                                className="fill-primary"
                                            />
                                            <PolarRadiusAxis tick={false} tickLine={false} axisLine={false}>
                                                <RechartsLabel
                                                    content={({ viewBox }) => {
                                                        if (viewBox && "cx" in viewBox && "cy" in viewBox) {
                                                            return (
                                                                <text
                                                                    x={viewBox.cx}
                                                                    y={viewBox.cy}
                                                                    textAnchor="middle"
                                                                    dominantBaseline="middle"
                                                                >
                                                                    <tspan
                                                                        x={viewBox.cx}
                                                                        y={viewBox.cy}
                                                                        className="fill-foreground text-4xl font-bold"
                                                                    >
                                                                        {f1Score}%
                                                                    </tspan>
                                                                    <tspan
                                                                        x={viewBox.cx}
                                                                        y={(viewBox.cy || 0) + 24}
                                                                        className="fill-muted-foreground"
                                                                    >
                                                                        F1 Score
                                                                    </tspan>
                                                                </text>
                                                            );
                                                        }
                                                    }}
                                                />
                                            </PolarRadiusAxis>
                                        </RadialBarChart>
                                    </ChartContainer>
                                </div>
                            </div>

                            {/* Metric Cards Grid */}
                            <div className="grid grid-cols-2 gap-3 content-start">
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
                                    label="Log Loss"
                                    value={trainingResult.metrics.log_loss.toFixed(4)}
                                />
                            </div>
                        </div>
                    </div>

                    {/* Feature Importance */}
                    {trainingResult.feature_importance && featureData.length > 0 && (
                        <div className="space-y-4">
                            <div className="flex items-center justify-between">
                                <h3 className="text-lg font-semibold text-primary-foreground">
                                    {showAllFeatures ? 'All Features' : 'Top Features'}
                                </h3>
                                {totalFeatures > 5 && (
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        onClick={() => setShowAllFeatures(!showAllFeatures)}
                                        className="text-xs"
                                    >
                                        {showAllFeatures ? 'Show Top 5' : `Show All (${totalFeatures})`}
                                    </Button>
                                )}
                            </div>

                            <div className="bg-primary/5 border border-primary/30 rounded-lg p-4">
                                <ChartContainer
                                    config={{
                                        importance: {
                                            label: "Importance",
                                            color: "hsl(var(--primary))",
                                        },
                                    }}
                                    className={`w-full ${showAllFeatures ? 'h-[600px]' : 'h-[300px]'}`}
                                >
                                    <BarChart
                                        data={featureData}
                                        layout="vertical"
                                        margin={{ top: 5, right: 30, left: 100, bottom: 25 }}
                                    >
                                        <ChartTooltip
                                            cursor={false}
                                            content={
                                                <ChartTooltipContent
                                                    hideLabel
                                                    className="min-w-[12rem]"
                                                    formatter={(value, name, item) => (
                                                        <>
                                                            <div
                                                                className="h-2.5 w-2.5 shrink-0 rounded-[2px] bg-[--color-bg] border-[--color-border]"
                                                                style={{
                                                                    "--color-bg": "hsl(var(--primary))",
                                                                    "--color-border": "hsl(var(--primary))",
                                                                } as React.CSSProperties}
                                                            />
                                                            <div className="flex flex-1 justify-between leading-none items-center">
                                                                <span className="text-muted-foreground">
                                                                    {item.payload.feature}
                                                                </span>
                                                                <span className="text-foreground font-mono font-medium tabular-nums">
                                                                    {value.toLocaleString()}
                                                                </span>
                                                            </div>
                                                        </>
                                                    )}
                                                />
                                            }
                                        />
                                        <XAxis
                                            type="number"
                                            stroke="hsl(var(--foreground))"
                                            tick={{ fill: "hsl(var(--foreground))" }}
                                            label={{
                                                value: "Importance Score",
                                                position: "bottom",
                                                offset: 0,
                                                style: { fill: "#FFF", fontSize: 12 }
                                            }}
                                        />
                                        <YAxis
                                            dataKey="feature"
                                            type="category"
                                            width={90}
                                            tick={{ fontSize: 12, fill: "hsl(var(--foreground))" }}
                                        />
                                        <Bar
                                            dataKey="importance"
                                            className="fill-primary"
                                            radius={[0, 4, 4, 0]}
                                        />
                                    </BarChart>
                                </ChartContainer>
                            </div>
                        </div>
                    )}

                    {/* Best Iteration Info */}
                    {trainingResult.best_iteration && (
                        <div className="bg-primary/5 border border-primary/30 rounded-lg p-4">
                            <p className="text-sm text-primary-foreground/70">
                                <span className="font-semibold">Best Iteration:</span> {trainingResult.best_iteration}
                            </p>
                        </div>
                    )}

                    {/* Action Buttons */}
                    <div className="flex gap-2 justify-end pt-4">
                        <Dialog
                            open={isRetrainDialogOpen}
                            onOpenChange={setIsRetrainDialogOpen}
                        >
                            <DialogTrigger asChild>
                                <Button
                                    onClick={handleRetrainClick}
                                    variant="outline"
                                    className="border-primary/30 bg-primary/5 hover:bg-primary/20 text-primary-foreground"
                                >
                                    Retrain Model
                                </Button>
                            </DialogTrigger>
                            <DialogContent className="bg-background border-primary/30">
                                <DialogHeader>
                                    <DialogTitle className="text-primary-foreground">
                                        Retrain with New Parameters
                                    </DialogTitle>
                                    <DialogDescription className="text-primary-foreground/70">
                                        Adjust the training parameters and
                                        retrain the model with the same dataset.
                                    </DialogDescription>
                                </DialogHeader>
                                <div className="grid grid-cols-1 gap-4 py-4">
                                    <div className="space-y-2">
                                        <Label
                                            htmlFor="retrain-model-name"
                                            className="text-primary-foreground"
                                        >
                                            Model Name{" "}
                                            <span className="text-destructive">
                                                *
                                            </span>
                                        </Label>
                                        <Input
                                            id="retrain-model-name"
                                            placeholder="e.g., planet_classifier_v2"
                                            value={retrainModelName}
                                            onChange={(e) =>
                                                setRetrainModelName(
                                                    e.target.value
                                                )
                                            }
                                            className="bg-primary/5 border-primary/30 text-primary-foreground"
                                        />
                                    </div>

                                    <div className="space-y-2">
                                        <Label
                                            htmlFor="retrain-model-type"
                                            className="text-primary-foreground"
                                        >
                                            Model Type
                                        </Label>
                                        <Input
                                            id="retrain-model-type"
                                            value={retrainModelType}
                                            onChange={(e) =>
                                                setRetrainModelType(e.target.value)
                                            }
                                            className="bg-primary/5 border-primary/30 text-primary-foreground"
                                            readOnly
                                        />
                                    </div>

                                    <div className="space-y-2">
                                        <Label
                                            htmlFor="retrain-test-size"
                                            className="text-primary-foreground"
                                        >
                                            Test Size
                                        </Label>
                                        <Input
                                            id="retrain-test-size"
                                            type="number"
                                            step="0.05"
                                            min="0.1"
                                            max="0.5"
                                            value={retrainTestSize}
                                            onChange={(e) =>
                                                setRetrainTestSize(
                                                    e.target.value
                                                )
                                            }
                                            className="bg-primary/5 border-primary/30 text-primary-foreground"
                                        />
                                    </div>
                                </div>
                                <DialogFooter>
                                    <Button
                                        variant="outline"
                                        onClick={() =>
                                            setIsRetrainDialogOpen(false)
                                        }
                                        className="border-primary/30"
                                    >
                                        Cancel
                                    </Button>
                                    <Button
                                        onClick={handleRetrainSubmit}
                                        className="bg-primary text-primary-foreground hover:bg-primary/90"
                                    >
                                        Start Retraining
                                    </Button>
                                </DialogFooter>
                            </DialogContent>
                        </Dialog>

                        <Button
                            onClick={handleTrainAnother}
                            className="bg-primary text-primary-foreground hover:bg-primary/90"
                        >
                            Train Another Model
                        </Button>
                    </div>
                </div>
            )}
        </div>
    );
}
