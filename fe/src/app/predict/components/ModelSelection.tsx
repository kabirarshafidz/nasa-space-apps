"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Database, Upload } from "lucide-react";
import { PreTrainedModel } from "../types";

interface ModelSelectionProps {
  preTrainedModels: PreTrainedModel[];
  selectedModel: string | null;
  onModelSelect: (model: string) => void;
}

export function ModelSelection({
  preTrainedModels,
  selectedModel,
  onModelSelect,
}: ModelSelectionProps) {
  return (
    <Tabs defaultValue="pretrained" className="w-full">
      <TabsList className="grid w-full grid-cols-2">
        <TabsTrigger value="pretrained">Choose a Pre-trained Model</TabsTrigger>
        <TabsTrigger value="upload">Upload Your Own Model</TabsTrigger>
      </TabsList>

      <TabsContent value="pretrained" className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="w-5 h-5" />
              Select Pre-trained Model
            </CardTitle>
            <CardDescription>
              Choose from available trained models
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Select value={selectedModel || ""} onValueChange={onModelSelect}>
              <SelectTrigger className="p-6">
                <SelectValue placeholder="Select a model" />
              </SelectTrigger>
              <SelectContent>
                {preTrainedModels.map((model) => (
                  <SelectItem key={model.key} value={model.key}>
                    {model.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="upload" className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="w-5 h-5" />
              Upload Custom Model
            </CardTitle>
            <CardDescription>
              Upload your own trained model file
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="border-2 border-dashed rounded-lg p-8 text-center">
              <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
              <p className="text-sm text-muted-foreground mb-4">
                Upload your model file (.pkl, .json)
              </p>
              <Button variant="outline">Choose File</Button>
            </div>
          </CardContent>
        </Card>
      </TabsContent>
    </Tabs>
  );
}
