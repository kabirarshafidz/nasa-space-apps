import {
    AlertCircleIcon,
    PaperclipIcon,
    UploadIcon,
    XIcon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { formatBytes } from "@/hooks/use-file-upload";

interface FileMetadata {
    name: string;
    size: number;
    type: string;
}

interface FileWithPreview {
    file: File | FileMetadata;
    id: string;
    preview?: string;
}

interface UploadDataStepProps {
    file: FileWithPreview | undefined;
    errors: string[];
    csvError: string | null;
    isDragging: boolean;
    maxSize: number;
    handleDragEnter: (e: React.DragEvent<Element>) => void;
    handleDragLeave: (e: React.DragEvent<Element>) => void;
    handleDragOver: (e: React.DragEvent<Element>) => void;
    handleDrop: (e: React.DragEvent<Element>) => void;
    openFileDialog: () => void;
    removeFile: (id: string) => void;
    getInputProps: () => React.InputHTMLAttributes<HTMLInputElement>;
}

export function UploadDataStep({
    file,
    errors,
    csvError,
    isDragging,
    maxSize,
    handleDragEnter,
    handleDragLeave,
    handleDragOver,
    handleDrop,
    openFileDialog,
    removeFile,
    getInputProps,
}: UploadDataStepProps) {
    return (
        <div className="space-y-6">
            <div>
                <h2 className="text-xl font-semibold text-primary-foreground mb-2">
                    Upload Training Data
                </h2>
                <p className="text-sm text-primary-foreground/70">
                    Upload a CSV file containing your training dataset
                </p>
            </div>

            <div className="space-y-2">
                <div
                    role="button"
                    onClick={openFileDialog}
                    onDragEnter={handleDragEnter}
                    onDragLeave={handleDragLeave}
                    onDragOver={handleDragOver}
                    onDrop={handleDrop}
                    data-dragging={isDragging || undefined}
                    className="border-primary/30 hover:bg-primary/5 data-[dragging=true]:bg-primary/10 has-[input:focus]:border-primary has-[input:focus]:ring-primary/50 flex min-h-72 flex-col items-center justify-center rounded-xl border border-dashed p-4 transition-colors has-disabled:pointer-events-none has-disabled:opacity-50 has-[input:focus]:ring-[3px] cursor-pointer"
                >
                    <input
                        {...getInputProps()}
                        className="sr-only"
                        aria-label="Upload file"
                        disabled={Boolean(file)}
                    />

                    <div className="flex flex-col items-center justify-center text-center">
                        <div
                            className="bg-primary/20 mb-2 flex size-11 shrink-0 items-center justify-center rounded-full border border-primary/30"
                            aria-hidden="true"
                        >
                            <UploadIcon className="size-4 text-primary-foreground" />
                        </div>
                        <p className="mb-1.5 text-sm font-medium text-primary-foreground">
                            Upload CSV file
                        </p>
                        <p className="text-primary-foreground/60 text-xs">
                            Drag & drop or click to browse (max.{" "}
                            {formatBytes(maxSize)})
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

                {csvError && (
                    <div
                        className="text-destructive flex items-center gap-1 text-xs"
                        role="alert"
                    >
                        <AlertCircleIcon className="size-3 shrink-0" />
                        <span>{csvError}</span>
                    </div>
                )}

                {file && (
                    <div className="space-y-2">
                        <div className="flex items-center justify-between gap-2 rounded-xl border border-primary/30 bg-primary/5 px-4 py-2">
                            <div className="flex items-center gap-3 overflow-hidden">
                                <PaperclipIcon
                                    className="size-4 shrink-0 text-primary-foreground/60"
                                    aria-hidden="true"
                                />
                                <div className="min-w-0">
                                    <p className="truncate text-[13px] font-medium text-primary-foreground">
                                        {file.file instanceof File ? file.file.name : file.file.name}
                                    </p>
                                    <p className="text-[11px] text-primary-foreground/60">
                                        {formatBytes(file.file instanceof File ? file.file.size : file.file.size)}
                                    </p>
                                </div>
                            </div>

                            <Button
                                size="icon"
                                variant="ghost"
                                className="text-primary-foreground/80 hover:text-primary-foreground -me-2 size-8 hover:bg-primary/10"
                                onClick={() => removeFile(file.id)}
                                aria-label="Remove file"
                            >
                                <XIcon className="size-4" aria-hidden="true" />
                            </Button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
