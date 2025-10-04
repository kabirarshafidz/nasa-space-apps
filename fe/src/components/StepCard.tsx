import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface StepCardProps {
  className?: string;
  children: React.ReactNode;
}

export function StepCard({ className, children }: StepCardProps) {
  return (
    <Card className={cn("bg-black/[.00] border-primary/20 backdrop-blur-sm", className)}>
      {children}
    </Card>
  );
}
