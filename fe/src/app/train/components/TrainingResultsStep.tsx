"use client";

import React from "react";
import Image from "next/image";
import { AlertCircleIcon, CheckCircle2, Loader2, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";

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
} from "recharts";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TrainingHistory } from "./TrainingHistory";

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

interface TrainingResultsStepProps {
    isTraining: boolean;
    trainingProgress: number;
    trainingResult: TrainingResult | null;
    trainingError: string | null;
    modelName: string;
    sessionId?: string | null;
}

function MetricCard({
    label,
    value,
    suffix = "",
}: {
    label: string;
    value: string;
    suffix?: string;
}) {
    return (
        <div className="bg-background border border-primary/30 rounded-lg p-4">
            <p className="text-xs text-primary-foreground/60 mb-1">{label}</p>
            <p className="text-lg font-bold text-primary-foreground">
                {value}
                {suffix}
            </p>
        </div>
    );
}

function ConfusionMetricCard({
    label,
    value,
    description,
}: {
    label: string;
    value: string;
    description: string;
}) {
    return (
        <div className="bg-primary/5 border border-primary/30 rounded-lg p-3">
            <p className="text-xs text-primary-foreground/60 mb-1">{label}</p>
            <p className="text-xl font-bold text-primary-foreground mb-1">
                {value}
            </p>
            <p className="text-xs text-primary-foreground/50">{description}</p>
        </div>
    );
}

export function TrainingResultsStep({
    isTraining,
    trainingProgress,
    trainingResult,
    trainingError,
    modelName,
    sessionId,
}: TrainingResultsStepProps) {
    // Prepare chart data
    const f1Score = trainingResult?.oof_metrics.f1
        ? Math.round(trainingResult.oof_metrics.f1 * 100)
        : 0;

    const chartData = [{ metric: "f1", value: f1Score }];

    const chartConfig = {
        value: {
            label: "F1 Score",
        },
        f1: {
            label: "F1",
            color: "#9556E8",
        },
    } satisfies ChartConfig;

    const handleDownloadModel = () => {
        if (trainingResult?.model_url) {
            window.open(trainingResult.model_url, "_blank");
        }
    };

    // Show full width during training
    if (isTraining) {
        return (
            <div className="space-y-6">
                <div>
                    <h2 className="text-xl font-semibold text-primary-foreground mb-2">
                        Training in Progress...
                    </h2>
                    <p className="text-sm text-primary-foreground/70">
                        Please wait while your model is being trained with cross-validation
                    </p>
                </div>

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
            </div>
        );
    }

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Main Results - Left/Center (2 columns) */}
            <div className="lg:col-span-2 space-y-6">
                <div>
                    <h2 className="text-xl font-semibold text-primary-foreground mb-2">
                        Training Results
                    </h2>
                    <p className="text-sm text-primary-foreground/70">
                        Your model has been trained successfully using stratified k-fold CV
                    </p>
                </div>

                {trainingError && (
                <div
                    className="text-destructive flex items-start gap-2 text-sm bg-destructive/10 border border-destructive/30 rounded-lg p-4"
                    role="alert"
                >
                    <AlertCircleIcon className="size-5 shrink-0 mt-0.5" />
                    <div className="flex-1">
                        <p className="font-medium">Training Failed</p>
                        <p className="text-xs mt-1 whitespace-pre-wrap break-words">
                            {trainingError}
                        </p>
                        {trainingError.includes("Missing required columns") && (
                            <div className="mt-3 p-3 bg-background/50 rounded border border-destructive/20">
                                <p className="text-xs font-semibold mb-2">
                                    💡 Tip:
                                </p>
                                <p className="text-xs">
                                    Your CSV file must contain all required
                                    TESS/NASA Exoplanet Archive columns. Please
                                    check the upload step for the complete list
                                    of required column names.
                                </p>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {trainingResult && !isTraining && (
                <div className="space-y-6">
                    {/* Success Message */}
                    <div className="flex items-center gap-2 text-green-500 bg-green-500/10 border border-green-500/30 rounded-lg p-4">
                        <CheckCircle2 className="size-5 shrink-0" />
                        <div className="flex-1">
                            <p className="font-medium">Training Successful!</p>
                            <p className="text-xs mt-1">
                                Model &quot;{trainingResult.model_name}&quot;
                                trained with{" "}
                                {trainingResult.fold_metrics.length}-fold
                                cross-validation
                            </p>
                        </div>
                        <Button
                            onClick={handleDownloadModel}
                            variant="outline"
                            size="sm"
                            className="border-green-500/30 bg-green-500/5 hover:bg-green-500/20 text-green-500"
                        >
                            <Download className="size-4 mr-2" />
                            Download Model
                        </Button>
                    </div>

                    {/* Performance Metrics Section */}
                    <div className="space-y-4">
                        <h3 className="text-lg font-semibold text-primary-foreground">
                            Out-of-Fold Performance Metrics
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
                                            endAngle={
                                                90 - (f1Score * 360) / 100
                                            }
                                            innerRadius={80}
                                            outerRadius={110}
                                        >
                                            <ChartTooltip
                                                cursor={false}
                                                content={
                                                    <ChartTooltipContent
                                                        hideLabel
                                                        nameKey="metric"
                                                    />
                                                }
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
                                            <PolarRadiusAxis
                                                tick={false}
                                                tickLine={false}
                                                axisLine={false}
                                            >
                                                <RechartsLabel
                                                    content={({ viewBox }) => {
                                                        if (
                                                            viewBox &&
                                                            "cx" in viewBox &&
                                                            "cy" in viewBox
                                                        ) {
                                                            return (
                                                                <text
                                                                    x={
                                                                        viewBox.cx
                                                                    }
                                                                    y={
                                                                        viewBox.cy
                                                                    }
                                                                    textAnchor="middle"
                                                                    dominantBaseline="middle"
                                                                >
                                                                    <tspan
                                                                        x={
                                                                            viewBox.cx
                                                                        }
                                                                        y={
                                                                            viewBox.cy
                                                                        }
                                                                        className="fill-foreground text-4xl font-bold"
                                                                    >
                                                                        {
                                                                            f1Score
                                                                        }
                                                                        %
                                                                    </tspan>
                                                                    <tspan
                                                                        x={
                                                                            viewBox.cx
                                                                        }
                                                                        y={
                                                                            (viewBox.cy ||
                                                                                0) +
                                                                            24
                                                                        }
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
                                    label="ROC AUC"
                                    value={(
                                        trainingResult.oof_metrics.roc_auc * 100
                                    ).toFixed(2)}
                                    suffix="%"
                                />
                                <MetricCard
                                    label="PR AUC"
                                    value={(
                                        trainingResult.oof_metrics.pr_auc * 100
                                    ).toFixed(2)}
                                    suffix="%"
                                />
                                <MetricCard
                                    label="Precision"
                                    value={(
                                        trainingResult.oof_metrics.precision *
                                        100
                                    ).toFixed(2)}
                                    suffix="%"
                                />
                                <MetricCard
                                    label="Recall"
                                    value={(
                                        trainingResult.oof_metrics.recall * 100
                                    ).toFixed(2)}
                                    suffix="%"
                                />
                                <MetricCard
                                    label="Log Loss"
                                    value={trainingResult.oof_metrics.logloss.toFixed(
                                        4,
                                    )}
                                />
                            </div>
                        </div>
                    </div>

                    {/* Confusion Matrix Metrics */}
                    <div className="space-y-4">
                        <h3 className="text-lg font-semibold text-primary-foreground">
                            Confusion Matrix Analysis
                        </h3>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                            <ConfusionMetricCard
                                label="True Positives"
                                value={trainingResult.confusion.counts.TP.toString()}
                                description="Correctly predicted positives"
                            />
                            <ConfusionMetricCard
                                label="True Negatives"
                                value={trainingResult.confusion.counts.TN.toString()}
                                description="Correctly predicted negatives"
                            />
                            <ConfusionMetricCard
                                label="False Positives"
                                value={trainingResult.confusion.counts.FP.toString()}
                                description="Incorrectly predicted positives"
                            />
                            <ConfusionMetricCard
                                label="False Negatives"
                                value={trainingResult.confusion.counts.FN.toString()}
                                description="Incorrectly predicted negatives"
                            />
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                            <MetricCard
                                label="TPR (Sensitivity)"
                                value={(
                                    trainingResult.confusion.rates.TPR * 100
                                ).toFixed(2)}
                                suffix="%"
                            />
                            <MetricCard
                                label="TNR (Specificity)"
                                value={(
                                    trainingResult.confusion.rates.TNR * 100
                                ).toFixed(2)}
                                suffix="%"
                            />
                            <MetricCard
                                label="PPV (Precision)"
                                value={(
                                    trainingResult.confusion.rates.PPV * 100
                                ).toFixed(2)}
                                suffix="%"
                            />
                            <MetricCard
                                label="NPV"
                                value={(
                                    trainingResult.confusion.rates.NPV * 100
                                ).toFixed(2)}
                                suffix="%"
                            />
                        </div>
                    </div>

                    {/* Charts Section */}
                    {Object.keys(trainingResult.charts).length > 0 && (
                        <div className="space-y-4">
                            <h3 className="text-lg font-semibold text-primary-foreground">
                                Visualizations
                            </h3>
                            <Tabs defaultValue="roc" className="w-full">
                                <TabsList className="grid w-full grid-cols-3 lg:grid-cols-6">
                                    {trainingResult.charts.roc_curve && (
                                        <TabsTrigger value="roc">
                                            ROC Curve
                                        </TabsTrigger>
                                    )}
                                    {trainingResult.charts.pr_curve && (
                                        <TabsTrigger value="pr">
                                            PR Curve
                                        </TabsTrigger>
                                    )}
                                    {trainingResult.charts.confusion_matrix && (
                                        <TabsTrigger value="confusion">
                                            Confusion
                                        </TabsTrigger>
                                    )}
                                    {trainingResult.charts
                                        .feature_importance && (
                                        <TabsTrigger value="features">
                                            Features
                                        </TabsTrigger>
                                    )}
                                    {trainingResult.charts.cv_metrics && (
                                        <TabsTrigger value="cv">
                                            CV Metrics
                                        </TabsTrigger>
                                    )}
                                    {trainingResult.charts
                                        .correlation_heatmap && (
                                        <TabsTrigger value="correlation">
                                            Correlation
                                        </TabsTrigger>
                                    )}
                                </TabsList>

                                {trainingResult.charts.roc_curve && (
                                    <TabsContent value="roc" className="mt-4">
                                        <div className="bg-background border border-primary/30 rounded-lg p-4">
                                            <h4 className="text-sm font-semibold text-primary-foreground mb-3">
                                                ROC Curve
                                            </h4>
                                            <div className="flex justify-center">
                                                <Image
                                                    src={
                                                        trainingResult.charts
                                                            .roc_curve
                                                    }
                                                    alt="ROC Curve"
                                                    width={600}
                                                    height={450}
                                                    className="max-w-full h-auto rounded"
                                                />
                                            </div>
                                        </div>
                                    </TabsContent>
                                )}

                                {trainingResult.charts.pr_curve && (
                                    <TabsContent value="pr" className="mt-4">
                                        <div className="bg-background border border-primary/30 rounded-lg p-4">
                                            <h4 className="text-sm font-semibold text-primary-foreground mb-3">
                                                Precision-Recall Curve
                                            </h4>
                                            <div className="flex justify-center">
                                                <Image
                                                    src={
                                                        trainingResult.charts
                                                            .pr_curve
                                                    }
                                                    alt="Precision-Recall Curve"
                                                    width={600}
                                                    height={450}
                                                    className="max-w-full h-auto rounded"
                                                />
                                            </div>
                                        </div>
                                    </TabsContent>
                                )}

                                {trainingResult.charts.confusion_matrix && (
                                    <TabsContent
                                        value="confusion"
                                        className="mt-4"
                                    >
                                        <div className="bg-background border border-primary/30 rounded-lg p-4">
                                            <h4 className="text-sm font-semibold text-primary-foreground mb-3">
                                                Confusion Matrix
                                            </h4>
                                            <div className="flex justify-center">
                                                <Image
                                                    src={
                                                        trainingResult.charts
                                                            .confusion_matrix
                                                    }
                                                    alt="Confusion Matrix"
                                                    width={600}
                                                    height={450}
                                                    className="max-w-full h-auto rounded"
                                                />
                                            </div>
                                        </div>
                                    </TabsContent>
                                )}

                                {trainingResult.charts.feature_importance && (
                                    <TabsContent
                                        value="features"
                                        className="mt-4"
                                    >
                                        <div className="bg-background border border-primary/30 rounded-lg p-4">
                                            <h4 className="text-sm font-semibold text-primary-foreground mb-3">
                                                Feature Importance
                                            </h4>
                                            <div className="flex justify-center">
                                                <Image
                                                    src={
                                                        trainingResult.charts
                                                            .feature_importance
                                                    }
                                                    alt="Feature Importance"
                                                    width={600}
                                                    height={450}
                                                    className="max-w-full h-auto rounded"
                                                />
                                            </div>
                                        </div>
                                    </TabsContent>
                                )}

                                {trainingResult.charts.cv_metrics && (
                                    <TabsContent value="cv" className="mt-4">
                                        <div className="bg-background border border-primary/30 rounded-lg p-4">
                                            <h4 className="text-sm font-semibold text-primary-foreground mb-3">
                                                Cross-Validation Metrics by Fold
                                            </h4>
                                            <div className="flex justify-center">
                                                <Image
                                                    src={
                                                        trainingResult.charts
                                                            .cv_metrics
                                                    }
                                                    alt="CV Metrics"
                                                    width={600}
                                                    height={450}
                                                    className="max-w-full h-auto rounded"
                                                />
                                            </div>
                                        </div>
                                    </TabsContent>
                                )}

                                {trainingResult.charts.correlation_heatmap && (
                                    <TabsContent
                                        value="correlation"
                                        className="mt-4"
                                    >
                                        <div className="bg-background border border-primary/30 rounded-lg p-4">
                                            <h4 className="text-sm font-semibold text-primary-foreground mb-3">
                                                Feature Correlation Heatmap
                                            </h4>
                                            <div className="flex justify-center">
                                                <Image
                                                    src={
                                                        trainingResult.charts
                                                            .correlation_heatmap
                                                    }
                                                    alt="Correlation Heatmap"
                                                    width={600}
                                                    height={450}
                                                    className="max-w-full h-auto rounded"
                                                />
                                            </div>
                                        </div>
                                    </TabsContent>
                                )}
                            </Tabs>
                        </div>
                    )}

                    {/* Fold-by-Fold Metrics */}
                    {trainingResult.fold_metrics.length > 0 && (
                        <div className="space-y-4">
                            <h3 className="text-lg font-semibold text-primary-foreground">
                                Fold-by-Fold Metrics
                            </h3>
                            <div className="bg-primary/5 border border-primary/30 rounded-lg p-4 overflow-x-auto">
                                <table className="w-full text-sm">
                                    <thead>
                                        <tr className="border-b border-primary/30">
                                            <th className="text-left py-2 px-3 text-primary-foreground">
                                                Fold
                                            </th>
                                            <th className="text-right py-2 px-3 text-primary-foreground">
                                                ROC AUC
                                            </th>
                                            <th className="text-right py-2 px-3 text-primary-foreground">
                                                PR AUC
                                            </th>
                                            <th className="text-right py-2 px-3 text-primary-foreground">
                                                Precision
                                            </th>
                                            <th className="text-right py-2 px-3 text-primary-foreground">
                                                Recall
                                            </th>
                                            <th className="text-right py-2 px-3 text-primary-foreground">
                                                F1
                                            </th>
                                            <th className="text-right py-2 px-3 text-primary-foreground">
                                                Log Loss
                                            </th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {trainingResult.fold_metrics.map(
                                            (fold, idx) => (
                                                <tr
                                                    key={idx}
                                                    className="border-b border-primary/20"
                                                >
                                                    <td className="py-2 px-3 text-primary-foreground font-medium">
                                                        {idx + 1}
                                                    </td>
                                                    <td className="py-2 px-3 text-right text-primary-foreground">
                                                        {(
                                                            fold.roc_auc * 100
                                                        ).toFixed(2)}
                                                        %
                                                    </td>
                                                    <td className="py-2 px-3 text-right text-primary-foreground">
                                                        {(
                                                            fold.pr_auc * 100
                                                        ).toFixed(2)}
                                                        %
                                                    </td>
                                                    <td className="py-2 px-3 text-right text-primary-foreground">
                                                        {(
                                                            fold.precision * 100
                                                        ).toFixed(2)}
                                                        %
                                                    </td>
                                                    <td className="py-2 px-3 text-right text-primary-foreground">
                                                        {(
                                                            fold.recall * 100
                                                        ).toFixed(2)}
                                                        %
                                                    </td>
                                                    <td className="py-2 px-3 text-right text-primary-foreground">
                                                        {(
                                                            fold.f1 * 100
                                                        ).toFixed(2)}
                                                        %
                                                    </td>
                                                    <td className="py-2 px-3 text-right text-primary-foreground">
                                                        {fold.logloss.toFixed(
                                                            4,
                                                        )}
                                                    </td>
                                                </tr>
                                            ),
                                        )}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    {/* Timestamp */}
                    <div className="bg-primary/5 border border-primary/30 rounded-lg p-4">
                        <p className="text-sm text-primary-foreground/70">
                            <span className="font-semibold">
                                Training Timestamp:
                            </span>{" "}
                            {trainingResult.timestamp}
                        </p>
                        </div>
                    </div>
                )}
            </div>

            {/* Training History - Right (1 column) */}
            {sessionId && (
                <div className="lg:col-span-1">
                    <TrainingHistory sessionId={sessionId} />
                </div>
            )}
        </div>
    );
}
