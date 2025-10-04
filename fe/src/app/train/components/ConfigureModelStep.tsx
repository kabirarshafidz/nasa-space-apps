import { AlertCircleIcon } from "lucide-react";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";

interface ConfigureModelStepProps {
    modelName: string;
    setModelName: (value: string) => void;
    modelType: string;
    setModelType: (value: string) => void;
    testSize: string;
    setTestSize: (value: string) => void;
    xgbEta: string;
    setXgbEta: (value: string) => void;
    xgbMaxDepth: string;
    setXgbMaxDepth: (value: string) => void;
    xgbNumBoostRound: string;
    setXgbNumBoostRound: (value: string) => void;
    trainingError: string | null;
}

export function ConfigureModelStep({
    modelName,
    setModelName,
    modelType,
    setModelType,
    testSize,
    setTestSize,
    xgbEta,
    setXgbEta,
    xgbMaxDepth,
    setXgbMaxDepth,
    xgbNumBoostRound,
    setXgbNumBoostRound,
    trainingError,
}: ConfigureModelStepProps) {
    return (
        <div className="space-y-6">
            <div>
                <h2 className="text-xl font-semibold text-primary-foreground mb-2">
                    Configure Training Parameters
                </h2>
                <p className="text-sm text-primary-foreground/70">
                    Set up your model configuration and training parameters
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                    <Label
                        htmlFor="modelName"
                        className="text-primary-foreground"
                    >
                        Model Name{" "}
                        <span className="text-destructive">*</span>
                    </Label>
                    <Input
                        id="modelName"
                        placeholder="e.g., exoplanet_classifier_v1"
                        value={modelName}
                        onChange={(e) => setModelName(e.target.value)}
                        className="bg-primary/5 border-primary/30 text-primary-foreground placeholder:text-primary-foreground/40"
                    />
                    <p className="text-xs text-primary-foreground/60">
                        A unique name for your trained model
                    </p>
                </div>

                <div className="space-y-2">
                    <Label
                        htmlFor="modelType"
                        className="text-primary-foreground"
                    >
                        Model Type
                    </Label>
                    <Select value={modelType} onValueChange={setModelType}>
                        <SelectTrigger className="bg-primary/5 border-primary/30 text-primary-foreground">
                            <SelectValue placeholder="Select model type" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="xgboost">XGBoost</SelectItem>
                            <SelectItem value="random_forest">
                                Random Forest
                            </SelectItem>
                            <SelectItem value="logistic_regression">
                                Logistic Regression
                            </SelectItem>
                        </SelectContent>
                    </Select>
                    <p className="text-xs text-primary-foreground/60">
                        Choose the algorithm for training
                    </p>
                </div>

                <div className="space-y-2">
                    <Label
                        htmlFor="testSize"
                        className="text-primary-foreground"
                    >
                        Test Size
                    </Label>
                    <div className="flex items-center gap-3">
                        <Slider
                            id="testSize"
                            min={0.1}
                            max={0.5}
                            step={0.05}
                            value={[parseFloat(testSize) || 0.2]}
                            onValueChange={(value) => setTestSize(value[0].toString())}
                            className="flex-1"
                        />
                        <Input
                            type="number"
                            step="0.05"
                            min="0.1"
                            max="0.5"
                            value={testSize}
                            onChange={(e) => setTestSize(e.target.value)}
                            className="w-20 bg-primary/5 border-primary/30 text-primary-foreground text-center"
                        />
                    </div>
                    <p className="text-xs text-primary-foreground/60">
                        Proportion of data for validation (0.1-0.5)
                    </p>
                </div>
            </div>

            {/* XGBoost Specific Parameters */}
            {modelType === "xgboost" && (
                <div className="space-y-4">
                    <h3 className="text-md font-semibold text-primary-foreground">
                        XGBoost Parameters
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div className="space-y-2">
                            <Label
                                htmlFor="xgbEta"
                                className="text-primary-foreground"
                            >
                                Learning Rate (eta)
                            </Label>
                            <div className="flex items-center gap-3">
                                <Slider
                                    id="xgbEta"
                                    min={0.01}
                                    max={0.3}
                                    step={0.01}
                                    value={[parseFloat(xgbEta) || 0.05]}
                                    onValueChange={(value) => setXgbEta(value[0].toString())}
                                    className="flex-1"
                                />
                                <Input
                                    type="number"
                                    step="0.01"
                                    min="0.01"
                                    max="0.3"
                                    value={xgbEta}
                                    onChange={(e) => setXgbEta(e.target.value)}
                                    className="w-20 bg-primary/5 border-primary/30 text-primary-foreground text-center"
                                />
                            </div>
                            <p className="text-xs text-primary-foreground/60">
                                Step size shrinkage (0.01-0.3)
                            </p>
                        </div>

                        <div className="space-y-2">
                            <Label
                                htmlFor="xgbMaxDepth"
                                className="text-primary-foreground"
                            >
                                Max Depth
                            </Label>
                            <div className="flex items-center gap-3">
                                <Slider
                                    id="xgbMaxDepth"
                                    min={3}
                                    max={10}
                                    step={1}
                                    value={[parseInt(xgbMaxDepth) || 6]}
                                    onValueChange={(value) => setXgbMaxDepth(value[0].toString())}
                                    className="flex-1"
                                />
                                <Input
                                    type="number"
                                    step="1"
                                    min="3"
                                    max="10"
                                    value={xgbMaxDepth}
                                    onChange={(e) => setXgbMaxDepth(e.target.value)}
                                    className="w-20 bg-primary/5 border-primary/30 text-primary-foreground text-center"
                                />
                            </div>
                            <p className="text-xs text-primary-foreground/60">
                                Maximum tree depth (3-10)
                            </p>
                        </div>

                        <div className="space-y-2">
                            <Label
                                htmlFor="xgbNumBoostRound"
                                className="text-primary-foreground"
                            >
                                Boost Rounds
                            </Label>
                            <div className="flex items-center gap-3">
                                <Slider
                                    id="xgbNumBoostRound"
                                    min={100}
                                    max={5000}
                                    step={100}
                                    value={[parseInt(xgbNumBoostRound) || 2000]}
                                    onValueChange={(value) => setXgbNumBoostRound(value[0].toString())}
                                    className="flex-1"
                                />
                                <Input
                                    type="number"
                                    step="100"
                                    min="100"
                                    max="5000"
                                    value={xgbNumBoostRound}
                                    onChange={(e) => setXgbNumBoostRound(e.target.value)}
                                    className="w-20 bg-primary/5 border-primary/30 text-primary-foreground text-center"
                                />
                            </div>
                            <p className="text-xs text-primary-foreground/60">
                                Number of boosting iterations (100-5000)
                            </p>
                        </div>
                    </div>
                </div>
            )}

            {/* Custom Actions */}
            <div className="flex gap-3 pt-4">
                <Button
                    type="button"
                    variant="outline"
                    onClick={() => {
                        setTestSize("0.2");
                        setXgbEta("0.05");
                        setXgbMaxDepth("6");
                        setXgbNumBoostRound("2000");
                    }}
                    className="border-primary/30 text-primary-foreground hover:bg-primary/10"
                >
                    Reset to Defaults
                </Button>
            </div>

            {trainingError && (
                <div
                    className="text-destructive flex items-center gap-1 text-sm bg-destructive/10 border border-destructive/30 rounded-lg p-3"
                    role="alert"
                >
                    <AlertCircleIcon className="size-4 shrink-0" />
                    <span>{trainingError}</span>
                </div>
            )}
        </div>
    );
}
