import { AlertCircleIcon } from "lucide-react";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
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
    cvFolds: string;
    setCvFolds: (value: string) => void;
    calibrationEnabled: boolean;
    setCalibrationEnabled: (value: boolean) => void;
    calibrationMethod: string;
    setCalibrationMethod: (value: string) => void;
    imputerKind: string;
    setImputerKind: (value: string) => void;
    imputerK: string;
    setImputerK: (value: string) => void;
    threshold: string;
    setThreshold: (value: string) => void;
    modelParams: string;
    setModelParams: (value: string) => void;
    trainingError: string | null;
}

export function ConfigureModelStep({
    modelName,
    setModelName,
    modelType,
    setModelType,
    cvFolds,
    setCvFolds,
    calibrationEnabled,
    setCalibrationEnabled,
    calibrationMethod,
    setCalibrationMethod,
    imputerKind,
    setImputerKind,
    imputerK,
    setImputerK,
    threshold,
    setThreshold,
    modelParams,
    setModelParams,
    trainingError,
}: ConfigureModelStepProps) {
    return (
        <div className="space-y-6">
            <div>
                <h2 className="text-xl font-semibold text-primary-foreground mb-2">
                    Configure Training Parameters
                </h2>
                <p className="text-sm text-primary-foreground/70">
                    Set up your model configuration and cross-validation parameters
                </p>
            </div>

            {/* Basic Configuration */}
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
                            <SelectItem value="rf">Random Forest</SelectItem>
                            <SelectItem value="logreg">Logistic Regression</SelectItem>
                        </SelectContent>
                    </Select>
                    <p className="text-xs text-primary-foreground/60">
                        Choose the algorithm for training
                    </p>
                </div>
            </div>

            {/* Cross-Validation Settings */}
            <div className="space-y-4 border-t border-primary/20 pt-6">
                <h3 className="text-md font-semibold text-primary-foreground">
                    Cross-Validation Settings
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                        <Label
                            htmlFor="cvFolds"
                            className="text-primary-foreground"
                        >
                            CV Folds
                        </Label>
                        <div className="flex items-center gap-3">
                            <Slider
                                id="cvFolds"
                                min={3}
                                max={10}
                                step={1}
                                value={[parseInt(cvFolds) || 5]}
                                onValueChange={(value) => setCvFolds(value[0].toString())}
                                className="flex-1"
                            />
                            <Input
                                type="number"
                                step="1"
                                min="3"
                                max="10"
                                value={cvFolds}
                                onChange={(e) => setCvFolds(e.target.value)}
                                className="w-20 bg-primary/5 border-primary/30 text-primary-foreground text-center"
                            />
                        </div>
                        <p className="text-xs text-primary-foreground/60">
                            Number of cross-validation folds (3-10)
                        </p>
                    </div>

                    <div className="space-y-2">
                        <Label
                            htmlFor="threshold"
                            className="text-primary-foreground"
                        >
                            Classification Threshold
                        </Label>
                        <div className="flex items-center gap-3">
                            <Slider
                                id="threshold"
                                min={0.1}
                                max={0.9}
                                step={0.05}
                                value={[parseFloat(threshold) || 0.5]}
                                onValueChange={(value) => setThreshold(value[0].toString())}
                                className="flex-1"
                            />
                            <Input
                                type="number"
                                step="0.05"
                                min="0.1"
                                max="0.9"
                                value={threshold}
                                onChange={(e) => setThreshold(e.target.value)}
                                className="w-20 bg-primary/5 border-primary/30 text-primary-foreground text-center"
                            />
                        </div>
                        <p className="text-xs text-primary-foreground/60">
                            Probability threshold for classification (0.1-0.9)
                        </p>
                    </div>
                </div>
            </div>

            {/* Preprocessing Settings */}
            <div className="space-y-4 border-t border-primary/20 pt-6">
                <h3 className="text-md font-semibold text-primary-foreground">
                    Preprocessing Settings
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                        <Label
                            htmlFor="imputerKind"
                            className="text-primary-foreground"
                        >
                            Imputer Method
                        </Label>
                        <Select value={imputerKind} onValueChange={setImputerKind}>
                            <SelectTrigger className="bg-primary/5 border-primary/30 text-primary-foreground">
                                <SelectValue placeholder="Select imputer" />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="knn">K-Nearest Neighbors</SelectItem>
                                <SelectItem value="median">Median</SelectItem>
                            </SelectContent>
                        </Select>
                        <p className="text-xs text-primary-foreground/60">
                            Method for handling missing values
                        </p>
                    </div>

                    {imputerKind === "knn" && (
                        <div className="space-y-2">
                            <Label
                                htmlFor="imputerK"
                                className="text-primary-foreground"
                            >
                                KNN Neighbors (k)
                            </Label>
                            <div className="flex items-center gap-3">
                                <Slider
                                    id="imputerK"
                                    min={3}
                                    max={15}
                                    step={1}
                                    value={[parseInt(imputerK) || 5]}
                                    onValueChange={(value) => setImputerK(value[0].toString())}
                                    className="flex-1"
                                />
                                <Input
                                    type="number"
                                    step="1"
                                    min="3"
                                    max="15"
                                    value={imputerK}
                                    onChange={(e) => setImputerK(e.target.value)}
                                    className="w-20 bg-primary/5 border-primary/30 text-primary-foreground text-center"
                                />
                            </div>
                            <p className="text-xs text-primary-foreground/60">
                                Number of neighbors for KNN imputation (3-15)
                            </p>
                        </div>
                    )}
                </div>
            </div>

            {/* Calibration Settings */}
            <div className="space-y-4 border-t border-primary/20 pt-6">
                <h3 className="text-md font-semibold text-primary-foreground">
                    Probability Calibration
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="flex items-center justify-between space-x-2 bg-primary/5 border border-primary/30 rounded-lg p-4">
                        <div className="space-y-0.5">
                            <Label
                                htmlFor="calibrationEnabled"
                                className="text-primary-foreground cursor-pointer"
                            >
                                Enable Calibration
                            </Label>
                            <p className="text-xs text-primary-foreground/60">
                                Improve probability estimates
                            </p>
                        </div>
                        <Switch
                            id="calibrationEnabled"
                            checked={calibrationEnabled}
                            onCheckedChange={setCalibrationEnabled}
                        />
                    </div>

                    {calibrationEnabled && (
                        <div className="space-y-2">
                            <Label
                                htmlFor="calibrationMethod"
                                className="text-primary-foreground"
                            >
                                Calibration Method
                            </Label>
                            <Select value={calibrationMethod} onValueChange={setCalibrationMethod}>
                                <SelectTrigger className="bg-primary/5 border-primary/30 text-primary-foreground">
                                    <SelectValue placeholder="Select method" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="isotonic">Isotonic</SelectItem>
                                    <SelectItem value="sigmoid">Sigmoid (Platt)</SelectItem>
                                </SelectContent>
                            </Select>
                            <p className="text-xs text-primary-foreground/60">
                                Calibration algorithm to use
                            </p>
                        </div>
                    )}
                </div>
            </div>

            {/* Advanced Model Parameters */}
            <div className="space-y-4 border-t border-primary/20 pt-6">
                <h3 className="text-md font-semibold text-primary-foreground">
                    Advanced Model Parameters (JSON)
                </h3>
                <div className="space-y-2">
                    <Label
                        htmlFor="modelParams"
                        className="text-primary-foreground"
                    >
                        Model-Specific Parameters
                    </Label>
                    <textarea
                        id="modelParams"
                        placeholder='{"max_depth": 6, "learning_rate": 0.05}'
                        value={modelParams}
                        onChange={(e) => setModelParams(e.target.value)}
                        className="w-full min-h-[100px] bg-primary/5 border border-primary/30 rounded-md px-3 py-2 text-primary-foreground font-mono text-sm placeholder:text-primary-foreground/40 focus:outline-none focus:ring-2 focus:ring-primary"
                    />
                    <p className="text-xs text-primary-foreground/60">
                        Optional JSON object with model-specific parameters. Leave as {"{}"} for defaults.
                    </p>
                </div>
            </div>

            {/* Reset Button */}
            <div className="flex gap-3 pt-4">
                <Button
                    type="button"
                    variant="outline"
                    onClick={() => {
                        setCvFolds("5");
                        setThreshold("0.5");
                        setImputerKind("knn");
                        setImputerK("5");
                        setCalibrationEnabled(true);
                        setCalibrationMethod("isotonic");
                        setModelParams("{}");
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
