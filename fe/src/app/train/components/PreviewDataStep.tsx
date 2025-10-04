import { Button } from "@/components/ui/button";

interface CsvData {
    [key: string]: string | number;
}

interface PreviewDataStepProps {
    csvData: CsvData[];
    csvHeaders: string[];
    currentPage: number;
    rowsPerPage: number;
    setCurrentPage: (page: number | ((prev: number) => number)) => void;
}

export function PreviewDataStep({
    csvData,
    csvHeaders,
    currentPage,
    rowsPerPage,
    setCurrentPage,
}: PreviewDataStepProps) {
    const totalPages = Math.ceil(csvData.length / rowsPerPage);
    const startIndex = (currentPage - 1) * rowsPerPage;
    const endIndex = startIndex + rowsPerPage;
    const currentRows = csvData.slice(startIndex, endIndex);

    return (
        <div className="space-y-6">
            <div>
                <h2 className="text-xl font-semibold text-primary-foreground mb-2">
                    Preview Your Data
                </h2>
                <p className="text-sm text-primary-foreground/70">
                    Review the uploaded data before training ({csvData.length}{" "}
                    rows, {csvHeaders.length} columns)
                </p>
            </div>

            <div className="rounded-lg border border-primary/30 bg-primary/5 overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead className="bg-primary/20 border-b border-primary/30">
                            <tr>
                                <th className="px-4 py-3 text-left text-xs font-medium text-primary-foreground">
                                    #
                                </th>
                                {csvHeaders.map((header) => (
                                    <th
                                        key={header}
                                        className="px-4 py-3 text-left text-xs font-medium text-primary-foreground"
                                    >
                                        {header}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-primary/20">
                            {currentRows.map((row, index) => (
                                <tr
                                    key={startIndex + index}
                                    className="hover:bg-primary/10 transition-colors"
                                >
                                    <td className="px-4 py-3 text-sm text-primary-foreground/80 font-medium">
                                        {startIndex + index + 1}
                                    </td>
                                    {csvHeaders.map((header) => (
                                        <td
                                            key={header}
                                            className="px-4 py-3 text-sm text-primary-foreground/70"
                                        >
                                            {typeof row[header] === "number"
                                                ? row[header].toFixed(4)
                                                : row[header]}
                                        </td>
                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                {/* Pagination */}
                {totalPages > 1 && (
                    <div className="flex items-center justify-between px-4 py-3 border-t border-primary/30 bg-primary/10">
                        <p className="text-xs text-primary-foreground/70">
                            Showing {startIndex + 1} to{" "}
                            {Math.min(endIndex, csvData.length)} of{" "}
                            {csvData.length} rows
                        </p>
                        <div className="flex gap-2">
                            <Button
                                size="sm"
                                variant="outline"
                                onClick={() =>
                                    setCurrentPage((p) => Math.max(1, p - 1))
                                }
                                disabled={currentPage === 1}
                                className="border-primary/30 bg-primary/5 hover:bg-primary/20 text-primary-foreground"
                            >
                                Previous
                            </Button>
                            <Button
                                size="sm"
                                variant="outline"
                                onClick={() =>
                                    setCurrentPage((p) =>
                                        Math.min(totalPages, p + 1)
                                    )
                                }
                                disabled={currentPage === totalPages}
                                className="border-primary/30 bg-primary/5 hover:bg-primary/20 text-primary-foreground"
                            >
                                Next
                            </Button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
