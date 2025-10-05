// CSV parsing utilities

export function findHeaderRow(lines: string[]): { headers: string[]; headerIdx: number } {
  let headerIdx = 0;
  let headers: string[] = [];
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line && !line.startsWith("#")) {
      headers = line.split(",").map((h) => h.trim());
      headerIdx = i;
      break;
    }
  }
  
  return { headers, headerIdx };
}

export function parseCSVMetadata(
  text: string
): Array<{ toi: string; toipfx: string }> {
  const lines = text.split("\n");
  const { headers, headerIdx } = findHeaderRow(lines);
  
  if (headers.length === 0) {
    throw new Error("Could not find valid CSV header");
  }
  
  const toiIdx = headers.indexOf("toi");
  const toipfxIdx = headers.indexOf("toipfx");
  
  if (toiIdx === -1 || toipfxIdx === -1) {
    throw new Error("CSV file must contain 'toi' and 'toipfx' columns");
  }
  
  const metadataArray: Array<{ toi: string; toipfx: string }> = [];
  
  for (let i = headerIdx + 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line && !line.startsWith("#")) {
      const values = line.split(",");
      metadataArray.push({
        toi: values[toiIdx]?.trim() || "",
        toipfx: values[toipfxIdx]?.trim() || "",
      });
    }
  }
  
  return metadataArray;
}

export function parseCSVFeatures(
  text: string,
  featureNames: string[]
): Array<Record<string, number | null>> {
  const lines = text.split("\n");
  const { headers, headerIdx } = findHeaderRow(lines);
  
  if (headers.length === 0) {
    throw new Error("Could not find valid CSV header");
  }
  
  const featuresList: Array<Record<string, number | null>> = [];
  
  for (let i = headerIdx + 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line && !line.startsWith("#")) {
      const values = line.split(",");
      const row: Record<string, number | null> = {};
      
      headers.forEach((header, idx) => {
        if (featureNames.includes(header)) {
          row[header] = parseFloat(values[idx]) || null;
        }
      });
      
      featuresList.push(row);
    }
  }
  
  return featuresList;
}
