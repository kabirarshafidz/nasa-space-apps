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
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import { createTrainingSession, deleteTrainingSession } from "../actions";
import { Loader2, Plus, Trash2 } from "lucide-react";

interface TrainingSession {
    id: string;
    userId: string;
    createdAt: Date;
    updatedAt: Date | null;
    csvUrl: string | null;
}

interface TrainingSessionListProps {
    sessions: TrainingSession[];
    selectedSessionId: string | null;
    onSessionSelect: (sessionId: string) => void;
    onStartTraining: (sessionId: string, hasCSV: boolean) => void;
    onSessionsChange: () => void;
}

export function TrainingSessionList({
    sessions,
    selectedSessionId,
    onSessionSelect,
    onStartTraining,
    onSessionsChange,
}: TrainingSessionListProps) {
    const [isCreating, setIsCreating] = useState(false);
    const [deletingId, setDeletingId] = useState<string | null>(null);
    const [showDeleteDialog, setShowDeleteDialog] = useState(false);
    const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);

    const handleCreateSession = async () => {
        setIsCreating(true);
        try {
            const result = await createTrainingSession();
            if (result.success && result.session) {
                onSessionsChange();
                // Directly go to training stage for new session
                onStartTraining(result.session.id, false);
            } else {
                alert(result.error || "Failed to create session");
            }
        } catch (error) {
            console.error("Error creating session:", error);
            alert("Failed to create session");
        } finally {
            setIsCreating(false);
        }
    };

    const handleDeleteClick = (sessionId: string) => {
        setSessionToDelete(sessionId);
        setShowDeleteDialog(true);
    };

    const handleDeleteConfirm = async () => {
        if (!sessionToDelete) return;

        setDeletingId(sessionToDelete);
        try {
            const result = await deleteTrainingSession(sessionToDelete);
            if (result.success) {
                onSessionsChange();
            } else {
                alert(result.error || "Failed to delete session");
            }
        } catch (error) {
            console.error("Error deleting session:", error);
            alert("Failed to delete session");
        } finally {
            setDeletingId(null);
            setShowDeleteDialog(false);
            setSessionToDelete(null);
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

    return (
        <>
            <Card className="mb-6 bg-background/40 backdrop-blur-sm border-primary/30">
                <CardHeader>
                    <div className="flex items-center justify-between">
                        <div>
                            <CardTitle>Training Sessions</CardTitle>
                            <CardDescription>
                                Manage your exoplanet detection model training sessions
                            </CardDescription>
                        </div>
                        <Button
                            onClick={handleCreateSession}
                            disabled={isCreating}
                            className="gap-2"
                        >
                            {isCreating ? (
                                <>
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                    Creating...
                                </>
                            ) : (
                                <>
                                    <Plus className="h-4 w-4" />
                                    New Training Session
                                </>
                            )}
                        </Button>
                    </div>
                </CardHeader>
                <CardContent>
                    {sessions.length === 0 ? (
                        <div className="text-center py-8 text-muted-foreground">
                            <p>No training sessions yet.</p>
                            <p className="text-sm mt-2">
                                Create a new session to start training your model.
                            </p>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {sessions.map((session) => {
                                const isSelected = selectedSessionId === session.id;
                                return (
                                    <div
                                        key={session.id}
                                        className={`flex items-center justify-between p-4 border rounded-lg hover:bg-accent/50 transition-colors ${isSelected ? "border-primary bg-primary/5" : ""
                                            }`}
                                    >
                                        <div className="flex-1">
                                            <div className="font-medium">
                                                Session {session.id.slice(0, 8)}...
                                            </div>
                                            <div className="text-sm text-muted-foreground">
                                                Created: {formatDate(session.createdAt)}
                                            </div>
                                            {session.csvUrl && (
                                                <div className="text-xs text-muted-foreground mt-1">
                                                    Dataset uploaded
                                                </div>
                                            )}
                                        </div>
                                        <div className="flex gap-2">
                                            {isSelected ? (
                                                <Button
                                                    size="sm"
                                                    onClick={() => onStartTraining(session.id, !!session.csvUrl)}
                                                >
                                                    {session.csvUrl ? "Continue Training" : "Start Training"}
                                                </Button>
                                            ) : (
                                                <Button
                                                    variant="outline"
                                                    size="sm"
                                                    onClick={() => onSessionSelect(session.id)}
                                                >
                                                    Select
                                                </Button>
                                            )}
                                            <Button
                                                variant="ghost"
                                                size="sm"
                                                onClick={() => handleDeleteClick(session.id)}
                                                disabled={deletingId === session.id}
                                            >
                                                {deletingId === session.id ? (
                                                    <Loader2 className="h-4 w-4 animate-spin" />
                                                ) : (
                                                    <Trash2 className="h-4 w-4 text-destructive" />
                                                )}
                                            </Button>
                                        </div>
                                    </div>
                                )
                            })}
                        </div>
                    )}
                </CardContent>
            </Card>

            <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Delete Training Session</DialogTitle>
                        <DialogDescription>
                            Are you sure you want to delete this training session? This
                            action cannot be undone and will also delete all associated
                            training entries and results.
                        </DialogDescription>
                    </DialogHeader>
                    <DialogFooter>
                        <Button
                            variant="outline"
                            onClick={() => setShowDeleteDialog(false)}
                        >
                            Cancel
                        </Button>
                        <Button
                            variant="destructive"
                            onClick={handleDeleteConfirm}
                            disabled={deletingId !== null}
                        >
                            {deletingId !== null ? (
                                <>
                                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                                    Deleting...
                                </>
                            ) : (
                                "Delete"
                            )}
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </>
    );
}
