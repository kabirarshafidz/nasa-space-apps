"use client";

import { useEffect, useState } from "react";
import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Loader2, TrendingUp, Calendar, Award, Cog } from "lucide-react";

interface TrainingEntryData {
    id: string;
    result: {
        model_name?: string;
        model_type?: string;
        oof_metrics?: {
            f1: number;
            roc_auc: number;
            precision: number;
            recall: number;
        };
    };
    modelS3Key: string | null;
    createdAt: Date;
    modelId: string | null;
    modelName: string | null;
    modelF1Score: string | null;
}

interface TrainingHistoryProps {
    sessionId: string;
}

export function TrainingHistory({ sessionId }: TrainingHistoryProps) {
    const [entries, setEntries] = useState<TrainingEntryData[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        loadEntries();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [sessionId]);

    const loadEntries = async () => {
        setIsLoading(true);
        setError(null);

        try {
            const response = await fetch(`/api/training/entries?sessionId=${sessionId}`);
            const data = await response.json();

            console.log(JSON.stringify(data))

            if (data.success) {
                setEntries(data.entries);
            } else {
                setError(data.error || "Failed to load training history");
            }
        } catch (err) {
            setError("Failed to load training history");
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    const formatDate = (date: Date) => {
        return new Date(date).toLocaleDateString("en-US", {
            year: "numeric",
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
        });
    };

    const formatF1Score = (score: string | null) => {
        if (!score) return "N/A";
        const numScore = parseFloat(score);
        return `${numScore.toFixed(2)}%`;
    };

    if (isLoading) {
        return (
            <Card>
                <CardHeader>
                    <CardTitle>Training History</CardTitle>
                    <CardDescription>Loading training runs...</CardDescription>
                </CardHeader>
                <CardContent className="flex items-center justify-center py-8">
                    <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                </CardContent>
            </Card>
        );
    }

    if (error) {
        return (
            <Card>
                <CardHeader>
                    <CardTitle>Training History</CardTitle>
                    <CardDescription className="text-destructive">{error}</CardDescription>
                </CardHeader>
            </Card>
        );
    }

    return (
        <Card className="h-full">
            <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                    <div>
                        <CardTitle className="text-lg">History</CardTitle>
                        <CardDescription className="text-xs">
                            {entries.length} run{entries.length !== 1 ? "s" : ""}
                        </CardDescription>
                    </div>
                    <Button variant="ghost" size="sm" onClick={loadEntries} className="h-7 px-2">
                        Refresh
                    </Button>
                </div>
            </CardHeader>
            <CardContent className="pt-0">
                {entries.length === 0 ? (
                    <div className="text-center py-6 text-muted-foreground">
                        <p className="text-sm">No runs yet.</p>
                    </div>
                ) : (
                    <div className="space-y-2 max-h-[600px] overflow-y-auto">
                        {entries.map((entry, index) => {
                            const result = entry.result;
                            const f1Score = entry.modelF1Score ? parseFloat(entry.modelF1Score) : null;
                            const isTopScore = index === 0;

                            return (
                                <div
                                    key={entry.id}
                                    className="p-3 border rounded-lg hover:bg-accent/50 transition-colors"
                                >
                                    <div className="flex items-start justify-between gap-2 mb-2">
                                        <div className="flex-1 min-w-0">
                                            <div className="flex items-center gap-1 mb-1">
                                              {entry.modelName}
                                              <Badge variant="outline" className="h-4 text-[10px] px-1 gap-0.5">
                                                  <Cog className="h-2.5 w-2.5" />
                                                  {result?.model_type || "Model"}
                                              </Badge>

                                                {isTopScore && f1Score && f1Score > 80 && (
                                                    <Badge variant="default" className="h-4 text-[10px] px-1 gap-0.5">
                                                        <Award className="h-2.5 w-2.5" />
                                                        Best
                                                    </Badge>
                                                )}
                                            </div>
                                        </div>
                                        {f1Score !== null && (
                                            <div className="text-right">
                                                <div className="text-lg font-bold text-primary">
                                                    {formatF1Score(entry.modelF1Score)}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                    <div className="space-y-1 text-xs text-muted-foreground">
                                        <div className="flex items-center gap-1">
                                            <Calendar className="h-3 w-3" />
                                            {formatDate(entry.createdAt)}
                                        </div>
                                        {result?.oof_metrics && (
                                            <div className="space-y-0.5">
                                                <div className="flex items-center gap-1">
                                                    <TrendingUp className="h-3 w-3" />
                                                    F1: {(result.oof_metrics.f1 * 100).toFixed(1)}%
                                                </div>
                                                <div>AUC: {(result.oof_metrics.roc_auc * 100).toFixed(1)}%</div>
                                                <div>Prec: {(result.oof_metrics.precision * 100).toFixed(1)}%</div>
                                                <div>Rec: {(result.oof_metrics.recall * 100).toFixed(1)}%</div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                )}
            </CardContent>
        </Card>
    );
}
