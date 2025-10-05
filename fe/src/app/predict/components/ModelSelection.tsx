"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Database,
  Upload,
  CheckCircle2,
  User,
  Sparkles,
  Loader2
} from "lucide-react";
import { PreTrainedModel } from "../types";
import { cn } from "@/lib/utils";

interface ModelSelectionProps {
  preTrainedModels: PreTrainedModel[];
  selectedModel: string | null;
  onModelSelect: (model: string) => void;
  isLoading?: boolean;
}

export function ModelSelection({
  preTrainedModels,
  selectedModel,
  onModelSelect,
  isLoading = false,
}: ModelSelectionProps) {
  // Separate models by type
  const userModels = preTrainedModels.filter((model) => !model.isDefault);
  const defaultModels = preTrainedModels.filter((model) => model.isDefault);

  const formatDate = (date: Date) => {
    return new Date(date).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  };

  const formatSize = (bytes: number) => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${Math.round((bytes / Math.pow(k, i)) * 100) / 100} ${sizes[i]}`;
  };

  const formatAccuracy = (f1Score: number) => {
    if (f1Score === 0) return "N/A";
    return `${(f1Score).toFixed(2)}%`;
  };

  const LoadingState = () => (
    <div className="flex flex-col items-center justify-center py-12">
      <Loader2 className="w-8 h-8 text-primary animate-spin mb-4" />
      <p className="text-sm text-muted-foreground">Loading models...</p>
    </div>
  );

  const EmptyState = ({ message }: { message: string }) => (
    <div className="text-center py-8">
      <Database className="w-12 h-12 mx-auto text-muted-foreground mb-3 opacity-50" />
      <p className="text-sm text-muted-foreground">{message}</p>
    </div>
  );

  const ModelTable = ({
    models,
    showMetadata = true
  }: {
    models: PreTrainedModel[];
    showMetadata?: boolean;
  }) => (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[40px]"></TableHead>
            <TableHead>Model Name</TableHead>
            <TableHead>Type</TableHead>
            {showMetadata && (
              <>
                <TableHead className="text-right">Accuracy</TableHead>
              </>
            )}
          </TableRow>
        </TableHeader>
        <TableBody>
          {models.map((model) => {
            const isSelected = selectedModel === model.key;
            return (
              <TableRow
                key={model.id}
                className={cn(
                  "cursor-pointer",
                  isSelected && "bg-muted/50"
                )}
                onClick={() => onModelSelect(model.key)}
              >
                <TableCell className="text-center">
                  {isSelected && (
                    <CheckCircle2 className="w-4 h-4 text-primary" />
                  )}
                </TableCell>
                <TableCell>
                  <div>
                    <div className="font-medium">{model.name}</div>
                    <div className="text-xs text-muted-foreground">{model.key}</div>
                  </div>
                </TableCell>
                <TableCell>
                  {model.isDefault ? (
                    <Badge variant="secondary" className="text-xs">
                      <Sparkles className="w-3 h-3 mr-1" />
                      Default
                    </Badge>
                  ) : (
                    <Badge variant="outline" className="text-xs">
                      <User className="w-3 h-3 mr-1" />
                      Custom
                    </Badge>
                  )}
                </TableCell>
                {showMetadata && (
                  <>
                    <TableCell className="text-right">
                      <span
                        className={cn(
                          "font-medium",
                          model.f1Score > 0
                            ? "text-green-600 dark:text-green-400"
                            : "text-muted-foreground"
                        )}
                      >
                        {formatAccuracy(model.f1Score)}
                      </span>
                    </TableCell>
                  </>
                )}
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );

  return (
    <Tabs defaultValue="pretrained" className="w-full">
      <TabsList className="grid w-full grid-cols-2">
        <TabsTrigger value="pretrained">
          <Database className="w-4 h-4 mr-2" />
          Pre-trained Models
        </TabsTrigger>
        <TabsTrigger value="upload">
          <Upload className="w-4 h-4 mr-2" />
          Upload Model
        </TabsTrigger>
      </TabsList>

      <TabsContent value="pretrained" className="space-y-6">
        {isLoading ? (
          <Card>
            <CardContent>
              <LoadingState />
            </CardContent>
          </Card>
        ) : (
          <>
            {/* User Models Section */}
            {userModels.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <User className="w-5 h-5" />
                    Your Models
                  </CardTitle>
                  <CardDescription>
                    Models you've trained ({userModels.length})
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ModelTable models={userModels} />
                </CardContent>
              </Card>
            )}

            {/* Default Models Section */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Sparkles className="w-5 h-5" />
                  Default Models
                </CardTitle>
                <CardDescription>
                  Pre-trained system models ({defaultModels.length})
                </CardDescription>
              </CardHeader>
              <CardContent>
                {defaultModels.length > 0 ? (
                  <ModelTable models={defaultModels} showMetadata={false} />
                ) : (
                  <EmptyState message="No default models available" />
                )}
              </CardContent>
            </Card>

            {/* Empty state when no models at all */}
            {userModels.length === 0 && defaultModels.length === 0 && (
              <Card>
                <CardContent>
                  <EmptyState message="No models found. Train a model first or upload one." />
                </CardContent>
              </Card>
            )}
          </>
        )}
      </TabsContent>

      <TabsContent value="upload" className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="w-5 h-5" />
              Upload Custom Model
            </CardTitle>
            <CardDescription>
              Upload your own trained model file (coming soon)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="border-2 border-dashed rounded-lg p-8 text-center bg-muted/30">
              <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground opacity-50" />
              <p className="text-sm text-muted-foreground mb-4">
                Upload your model file (.pkl, .joblib, .bks)
              </p>
              <Button variant="outline" disabled>
                Choose File
              </Button>
              <p className="text-xs text-muted-foreground mt-3">
                This feature will be available soon
              </p>
            </div>
          </CardContent>
        </Card>
      </TabsContent>
    </Tabs>
  );
}
