"use client";
import { cn } from "@/lib/utils";
import { motion } from "motion/react";
import React, { useMemo } from "react";

export const Meteors = ({
  number,
  className,
}: {
  number?: number;
  className?: string;
}) => {
  const meteorCount = number || 20;

  // Generate stable animation values to prevent hydration mismatch
  const meteorStyles = useMemo(() => {
    return Array.from({ length: meteorCount }, (_, idx) => {
      const position = idx * (800 / meteorCount) - 400;

      // Use deterministic "random" values based on index
      const seed = idx * 0.618033988749895; // Golden ratio for good distribution
      const delay = (seed % 5).toFixed(1); // 0-5s delay
      const duration = (5 + ((seed * 7) % 5)).toFixed(0); // 5-10s duration

      return {
        position,
        delay,
        duration,
      };
    });
  }, [meteorCount]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      {meteorStyles.map((style, idx) => {
        return (
          <span
            key={"meteor" + idx}
            className={cn(
              "animate-meteor-effect absolute h-0.5 w-0.5 rotate-[45deg] rounded-[9999px] bg-slate-500 shadow-[0_0_0_1px_#ffffff10]",
              "before:absolute before:top-1/2 before:h-[1px] before:w-[50px] before:-translate-y-[50%] before:transform before:bg-gradient-to-r before:from-[#64748b] before:to-transparent before:content-['']",
              className
            )}
            style={{
              top: "-40px",
              left: style.position + "px",
              animationDelay: style.delay + "s",
              animationDuration: style.duration + "s",
            }}
          ></span>
        );
      })}
    </motion.div>
  );
};
