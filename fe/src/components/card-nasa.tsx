import { cn } from "@/lib/utils";

export default function NasaCard({
    children,
    className,
}: {
    children: React.ReactNode;
    className?: string;
}) {
    return (
        <div className={cn("relative w-full", className)}>
            <div className="absolute inset-0 h-full w-full scale-[0.80] transform rounded-full bg-gradient-to-r from-blue-500 to-teal-500 blur-3xl opacity-50" />
            <div className="relative flex h-full flex-col items-start justify-between overflow-hidden rounded-2xl border border-gray-700 bg-black/40 backdrop-blur-sm px-6 py-6">
                {children}
            </div>
        </div>
    );
}
