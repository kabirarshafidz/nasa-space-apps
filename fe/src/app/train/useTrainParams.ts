"use client";

import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";

export function useTrainParams() {
  return useQueryStates(
    {
      session: parseAsString.withDefault(""),
      step: parseAsInteger.withDefault(1),
      entryId: parseAsString.withDefault(""),
    },
    {
      history: "push",
      shallow: true,
    }
  );
}
