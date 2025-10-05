import { NextRequest } from 'next/server';
import { openai } from '@ai-sdk/openai';
import { streamText, tool, CoreMessage } from 'ai';
import { z } from 'zod';

/**
 * Enhanced /api/chat route with AI SDK tool calling integration.
 *
 * Goals:
 * 1. Avoid wasting model tokens by letting the AI decide when to query the full dataset
 *    using proper tool calls.
 * 2. Use the AI SDK's built-in tool calling mechanism for reliable, type-safe tool execution.
 * 3. Preserve streaming for normal LLM responses.
 *
 * How it works:
 * - We receive { messages, planetData, predictionResults, planetTypeClassifications, modelInfo }.
 * - We define three tools for the AI model:
 *     a) dataset_stats              → comprehensive statistics about the dataset
 *     b) filter_radius              → filter planets by radius with comparison operators
 *     c) top_confidence_predictions → get highest confidence predictions
 * - The model receives a compact data snapshot and decides when to use tools for detailed queries.
 * - Tools execute server-side with access to the full dataset, returning results to the model.
 * - The model then formulates a response based on tool results and streams it to the client.
 */

/* ---------------------------------- Types ---------------------------------- */

type BasicMessage = { role: 'user' | 'assistant' | 'system'; content: string };

interface PlanetRecord {
  toi?: string;
  toipfx?: string;
  pl_rade?: number;        // Planet radius (Earth radii)
  pl_orbper?: number;      // Orbital period (days)
  pl_eqt?: number;         // Equilibrium temperature (K)
  type_name?: string;
  type_confidence?: number;
  [key: string]: unknown;
}

interface PredictionResult {
  id?: string;
  toi?: string;
  predicted_type?: string;
  confidence?: number;
  [key: string]: unknown;
}

interface PredictionResults {
  predictions: PredictionResult[];
  [key: string]: unknown;
}

interface PlanetTypeClassification {
  toi?: string;
  type_name?: string;
  confidence?: number;
  [key: string]: unknown;
}

/* ------------------------------ Tool Executors ------------------------------ */

function toolDatasetStats(
  planetData: PlanetRecord[],
  classifications: PlanetTypeClassification[],
  predictionResults?: PredictionResults
): string {
  console.log('[Tool] toolDatasetStats called with', {
    planetDataCount: planetData.length,
    classificationsCount: classifications.length,
    hasPredictions: !!predictionResults
  });
  const total = planetData.length;

  const radii = planetData
    .map(p => typeof p.pl_rade === 'number' ? p.pl_rade : null)
    .filter((v): v is number => v !== null);

  const eqTemps = planetData
    .map(p => typeof p.pl_eqt === 'number' ? p.pl_eqt : null)
    .filter((v): v is number => v !== null);

  const radiusStats = summarizeNumeric(radii);
  const tempStats = summarizeNumeric(eqTemps);

  // Classification distribution
  const typeCounts = new Map<string, number>();
  classifications.forEach(c => {
    if (c.type_name) {
      typeCounts.set(c.type_name, (typeCounts.get(c.type_name) || 0) + 1);
    }
  });

  // Predicted type distribution (if available)
  const predictedTypeCounts = new Map<string, number>();
  if (predictionResults?.predictions) {
    predictionResults.predictions.forEach(p => {
      if (p.predicted_type) {
        predictedTypeCounts.set(
          p.predicted_type,
          (predictedTypeCounts.get(p.predicted_type) || 0) + 1
        );
      }
    });
  }

  const formatDist = (m: Map<string, number>) =>
    [...m.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([k, v]) => `${k}: ${v}`)
      .join(', ') || 'N/A';

  return [
    'Dataset Statistics',
    '------------------',
    `Total planets: ${total}`,
    `Classified planets: ${classifications.length}`,
    ``,
    `Radius (R⊕) stats: ${formatStats(radiusStats)}`,
    `Equilibrium Temp (K) stats: ${formatStats(tempStats)}`,
    ``,
    `Classification distribution: ${formatDist(typeCounts)}`,
    predictionResults?.predictions
      ? `Predicted type distribution: ${formatDist(predictedTypeCounts)}`
      : null,
    '',
    'Note: These figures are computed directly from the provided in‑memory datasets (no model tokens used).'
  ]
    .filter(Boolean)
    .join('\n');
}

function toolFilterRadius(
  planetData: PlanetRecord[],
  op: '>' | '>=' | '<' | '<=',
  value: number
): string {
  console.log('[Tool] toolFilterRadius called with', { op, value, planetDataCount: planetData.length });
  const compare = (r: number) => {
    switch (op) {
      case '>': return r > value;
      case '>=': return r >= value;
      case '<': return r < value;
      case '<=': return r <= value;
    }
  };

  const filtered = planetData.filter(p => typeof p.pl_rade === 'number' && compare(p.pl_rade));
  const limited = filtered.slice(0, 25);

  const rows = limited.map(p => {
    const rid = (p.toi || p.toipfx || 'unknown');
    const r = p.pl_rade;
    const t = p.type_name || '—';
    const conf = p.type_confidence != null ? p.type_confidence.toFixed(2) : '—';
    return `${rid}\tR=${r}\tType=${t}\tConf=${conf}`;
  });

  return [
    `Filtered Planets (radius ${op} ${value} R⊕)`,
    '-------------------------------------------',
    `Matches: ${filtered.length} (showing up to 25)`,
    '',
    'ID\tRadius\tType\tConfidence',
    ...rows,
    '',
    'Direct dataset filter performed without calling the model.'
  ].join('\n');
}

function toolTopConfidencePredictions(
  predictionResults: PredictionResults | undefined,
  classifications: PlanetTypeClassification[],
  limit: number
): string {
  console.log('[Tool] toolTopConfidencePredictions called with', {
    limit,
    hasPredictions: !!predictionResults?.predictions,
    predictionCount: predictionResults?.predictions?.length || 0
  });
  if (!predictionResults?.predictions?.length) {
    return 'No prediction results available to compute top confidence planets.';
  }

  const preds = [...predictionResults.predictions]
    .filter(p => typeof p.confidence === 'number')
    .sort((a, b) => (b.confidence! - a.confidence!))
    .slice(0, Math.max(1, Math.min(limit, 25)));

  const classMap = new Map<string, PlanetTypeClassification>();
  classifications.forEach(c => {
    if (c.toi) classMap.set(c.toi, c);
  });

  const lines = preds.map(p => {
    const id = p.toi || p.id || 'unknown';
    const predicted = p.predicted_type || '—';
    const conf = p.confidence != null ? p.confidence.toFixed(4) : '—';
    const classified = classMap.get(id);
    const actualType = classified?.type_name || 'N/A';
    const actualConf =
      classified?.confidence != null ? classified.confidence.toFixed(2) : '—';
    return `${id}\tPred=${predicted} (${conf})\tClassified=${actualType} (${actualConf})`;
  });

  return [
    `Top ${preds.length} High‑Confidence Predictions`,
    '---------------------------------------',
    'ID\tPredicted (confidence)\tClassified (confidence)',
    ...lines,
    '',
    'Served from local prediction data (no model tokens used).'
  ].join('\n');
}

function toolCountByType(
  planetData: PlanetRecord[],
  classifications: PlanetTypeClassification[]
): string {
  console.log('[Tool] toolCountByType called with', {
    planetDataCount: planetData.length,
    classificationsCount: classifications.length
  });
  const typeCounts = new Map<string, number>();
  const unclassifiedCount = planetData.length - classifications.length;

  // Count by type_name
  classifications.forEach(c => {
    if (c.type_name) {
      typeCounts.set(c.type_name, (typeCounts.get(c.type_name) || 0) + 1);
    }
  });

  const sortedCounts = [...typeCounts.entries()].sort((a, b) => b[1] - a[1]);

  const lines = sortedCounts.map(([type, count]) => {
    const percentage = ((count / planetData.length) * 100).toFixed(1);
    return `${type}: ${count} (${percentage}%)`;
  });

  if (unclassifiedCount > 0) {
    const percentage = ((unclassifiedCount / planetData.length) * 100).toFixed(1);
    lines.push(`Unclassified: ${unclassifiedCount} (${percentage}%)`);
  }

  return [
    'Exoplanet Count by Type',
    '----------------------',
    `Total planets: ${planetData.length}`,
    `Classified: ${classifications.length}`,
    `Unclassified: ${unclassifiedCount}`,
    '',
    'Breakdown by type:',
    ...lines,
    '',
    'Direct count from dataset (no model tokens used).'
  ].join('\n');
}

function toolFilterByRange(
  planetData: PlanetRecord[],
  field: string,
  min?: number,
  max?: number
): string {
  console.log('[Tool] toolFilterByRange called with', { field, min, max, planetDataCount: planetData.length });
  // Map common field names to actual property keys
  const fieldMap: Record<string, string> = {
    'radius': 'pl_rade',
    'pl_rade': 'pl_rade',
    'period': 'pl_orbper',
    'pl_orbper': 'pl_orbper',
    'orbital_period': 'pl_orbper',
    'temperature': 'pl_eqt',
    'pl_eqt': 'pl_eqt',
    'equilibrium_temperature': 'pl_eqt',
    'confidence': 'type_confidence',
    'type_confidence': 'type_confidence'
  };

  const actualField = fieldMap[field.toLowerCase()] || field;

  const filtered = planetData.filter(p => {
    const value = p[actualField as keyof PlanetRecord];
    if (typeof value !== 'number') return false;
    if (min !== undefined && value < min) return false;
    if (max !== undefined && value > max) return false;
    return true;
  });

  const limited = filtered.slice(0, 25);

  const rows = limited.map(p => {
    const id = p.toi || p.toipfx || 'unknown';
    const value = p[actualField as keyof PlanetRecord];
    const type = p.type_name || '—';
    const conf = p.type_confidence != null ? p.type_confidence.toFixed(2) : '—';
    return `${id}\t${actualField}=${value}\tType=${type}\tConf=${conf}`;
  });

  const rangeDesc = min !== undefined && max !== undefined
    ? `${min} to ${max}`
    : min !== undefined
      ? `>= ${min}`
      : max !== undefined
        ? `<= ${max}`
        : 'all values';

  return [
    `Planets with ${actualField} in range ${rangeDesc}`,
    '-------------------------------------------',
    `Matches: ${filtered.length} (showing up to 25)`,
    '',
    `ID\t${actualField}\tType\tConfidence`,
    ...rows,
    '',
    'Direct dataset filter performed without calling the model.'
  ].join('\n');
}

function toolQueryPlanets(
  planetData: PlanetRecord[],
  classifications: PlanetTypeClassification[],
  params: {
    type_name?: string;
    min_confidence?: number;
    min_radius?: number;
    max_radius?: number;
    min_period?: number;
    max_period?: number;
    limit?: number;
  }
): string {
  console.log('[Tool] toolQueryPlanets called with', {
    params,
    planetDataCount: planetData.length,
    classificationsCount: classifications.length
  });
  const {
    type_name,
    min_confidence,
    min_radius,
    max_radius,
    min_period,
    max_period,
    limit = 25
  } = params;

  // Build classification map
  const classMap = new Map<string, PlanetTypeClassification>();
  classifications.forEach(c => {
    if (c.toi) classMap.set(c.toi, c);
  });

  let filtered = planetData.filter(p => {
    // Filter by type
    if (type_name && p.type_name !== type_name) return false;

    // Filter by confidence
    if (min_confidence !== undefined) {
      const conf = p.type_confidence;
      if (typeof conf !== 'number' || conf < min_confidence) return false;
    }

    // Filter by radius
    if (min_radius !== undefined && (typeof p.pl_rade !== 'number' || p.pl_rade < min_radius)) return false;
    if (max_radius !== undefined && (typeof p.pl_rade !== 'number' || p.pl_rade > max_radius)) return false;

    // Filter by period
    if (min_period !== undefined && (typeof p.pl_orbper !== 'number' || p.pl_orbper < min_period)) return false;
    if (max_period !== undefined && (typeof p.pl_orbper !== 'number' || p.pl_orbper > max_period)) return false;

    return true;
  });

  const limited = filtered.slice(0, Math.min(limit, 50));

  const rows = limited.map(p => {
    const id = p.toi || p.toipfx || 'unknown';
    const r = p.pl_rade != null ? p.pl_rade.toFixed(2) : '—';
    const period = p.pl_orbper != null ? p.pl_orbper.toFixed(2) : '—';
    const temp = p.pl_eqt != null ? p.pl_eqt.toFixed(0) : '—';
    const type = p.type_name || '—';
    const conf = p.type_confidence != null ? p.type_confidence.toFixed(2) : '—';
    return `${id}\tR=${r}\tP=${period}d\tT=${temp}K\t${type}\t${conf}`;
  });

  const filters: string[] = [];
  if (type_name) filters.push(`type=${type_name}`);
  if (min_confidence) filters.push(`min_conf=${min_confidence}`);
  if (min_radius) filters.push(`min_radius=${min_radius}`);
  if (max_radius) filters.push(`max_radius=${max_radius}`);
  if (min_period) filters.push(`min_period=${min_period}`);
  if (max_period) filters.push(`max_period=${max_period}`);

  return [
    `Query Results: ${filters.length > 0 ? filters.join(', ') : 'all planets'}`,
    '-------------------------------------------',
    `Total matches: ${filtered.length} (showing up to ${Math.min(limit, 50)})`,
    '',
    'ID\tRadius\tPeriod\tTemp\tType\tConf',
    ...rows,
    '',
    'Direct dataset query performed without calling the model.'
  ].join('\n');
}

function toolSearchByField(
  planetData: PlanetRecord[],
  fieldName: string,
  fieldValue: string | number,
  operator: 'equals' | 'greater_than' | 'less_than' | 'contains' = 'equals',
  limit: number = 25
): string {
  console.log('[Tool] toolSearchByField called with', {
    fieldName,
    fieldValue,
    operator,
    limit,
    planetDataCount: planetData.length
  });

  const filtered = planetData.filter(p => {
    const value = p[fieldName as keyof PlanetRecord];

    if (value === undefined || value === null) return false;

    switch (operator) {
      case 'equals':
        return value === fieldValue;
      case 'greater_than':
        return typeof value === 'number' && typeof fieldValue === 'number' && value > fieldValue;
      case 'less_than':
        return typeof value === 'number' && typeof fieldValue === 'number' && value < fieldValue;
      case 'contains':
        return String(value).toLowerCase().includes(String(fieldValue).toLowerCase());
      default:
        return false;
    }
  });

  const limited = filtered.slice(0, Math.min(limit, 50));

  const rows = limited.map(p => {
    const id = p.toi || p.toipfx || 'unknown';
    const fieldVal = p[fieldName as keyof PlanetRecord];
    const r = p.pl_rade != null ? p.pl_rade.toFixed(2) : '—';
    const type = p.type_name || '—';
    const conf = p.type_confidence != null ? p.type_confidence.toFixed(2) : '—';
    return `${id}\t${fieldName}=${fieldVal}\tR=${r}\t${type}\t${conf}`;
  });

  return [
    `Search Results: ${fieldName} ${operator} ${fieldValue}`,
    '-------------------------------------------',
    `Total matches: ${filtered.length} (showing up to ${Math.min(limit, 50)})`,
    '',
    `ID\t${fieldName}\tRadius\tType\tConf`,
    ...rows,
    '',
    'Direct dataset search performed without calling the model.',
    filtered.length === 0 ? `\nNote: No planets found with ${fieldName} ${operator} ${fieldValue}. This field may not exist in the dataset or no matches were found.` : ''
  ].join('\n');
}

/* ------------------------------- Util Helpers ------------------------------- */

interface NumericStats {
  count: number;
  min: number;
  max: number;
  mean: number;
  p25: number;
  p50: number;
  p75: number;
}

function summarizeNumeric(values: number[]): NumericStats | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const sum = sorted.reduce((a, b) => a + b, 0);
  const q = (p: number) => {
    if (sorted.length === 1) return sorted[0];
    const pos = (sorted.length - 1) * p;
    const base = Math.floor(pos);
    const rest = pos - base;
    return sorted[base] + (sorted[base + 1] !== undefined ? rest * (sorted[base + 1] - sorted[base]) : 0);
  };
  return {
    count: sorted.length,
    min: sorted[0],
    max: sorted[sorted.length - 1],
    mean: sum / sorted.length,
    p25: q(0.25),
    p50: q(0.5),
    p75: q(0.75),
  };
}

function formatStats(st: NumericStats | null): string {
  if (!st) return 'N/A';
  return `n=${st.count}, min=${round(st.min)}, p25=${round(st.p25)}, median=${round(st.p50)}, p75=${round(st.p75)}, max=${round(st.max)}, mean=${round(st.mean)}`;
}

function round(n: number, d = 2): number {
  return parseFloat(n.toFixed(d));
}

/**
 * Create a compact textual context summary (few planets + distributions) to reduce tokens.
 */
function buildCompactContext(
  planetData: PlanetRecord[],
  classifications: PlanetTypeClassification[],
  predictionResults?: PredictionResults
): string {
  const sample = planetData.slice(0, 5).map(p => ({
    id: p.toi || p.toipfx,
    r: p.pl_rade,
    period: p.pl_orbper,
    eqt: p.pl_eqt,
    type: p.type_name,
    tconf: p.type_confidence
  }));

  const typeCounts = new Map<string, number>();
  classifications.forEach(c => {
    if (c.type_name) typeCounts.set(c.type_name, (typeCounts.get(c.type_name) || 0) + 1);
  });

  const predictionSummary = predictionResults?.predictions
    ? `Predictions: ${predictionResults.predictions.length}`
    : 'Predictions: 0';

  return [
    'DATA SNAPSHOT (truncated to preserve tokens)',
    `Planets total: ${planetData.length}, Classified: ${classifications.length}, ${predictionSummary}`,
    `Type distribution (count): ${[...typeCounts.entries()].map(([k, v]) => `${k}=${v}`).join(', ') || 'N/A'}`,
    'Sample planets (id,radius,period,eqt,type,conf):',
    sample.map(s => `${s.id}|R=${s.r}|P=${s.period}|T=${s.eqt}|${s.type}|${s.tconf}`).join('; ')
  ].join('\n');
}

/* ------------------------------- Route Handler ------------------------------ */

export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(() => ({}));
    const {
      messages = [],
      planetData = [],
      predictionResults,
      planetTypeClassifications = [],
      modelInfo = []
    } = body as {
      messages?: BasicMessage[];
      planetData?: PlanetRecord[];
      predictionResults?: PredictionResults;
      planetTypeClassifications?: PlanetTypeClassification[];
      modelInfo?: unknown[];
    };

    if (!Array.isArray(messages)) {
      return jsonError('Invalid request: messages must be an array', 400);
    }

    // Ensure planetData is an array
    const safePlanetData = Array.isArray(planetData) ? planetData : [];
    const safePlanetTypeClassifications = Array.isArray(planetTypeClassifications) ? planetTypeClassifications : [];

    // Build minimal context summary & call model with tools
    const contextSummary = buildCompactContext(
      safePlanetData,
      safePlanetTypeClassifications,
      predictionResults
    );

    const system = `
You are the AI Planet Assistant for an exoplanet prediction application.

You receive:
- A compact dataset snapshot (NOT the full raw data).
- User messages.

You have access to tools that can query the full dataset:
- dataset_stats: Get comprehensive statistics about the dataset
- filter_radius: Filter planets by radius with comparison operators
- top_confidence_predictions: Get the highest confidence predictions
- count_by_type: Count exoplanets by their classification type
- filter_by_range: Filter planets by a range of values for any numeric field
- query_planets: Advanced query with multiple filters (type, confidence, radius, period)
- search_by_field: Search for planets by any field name (e.g., sy_snum for number of stars, sy_pnum for number of planets in system)

Rules:
1. Use the available tools when users ask for specific data queries, statistics, or filtering.
2. After calling a tool and receiving results, ALWAYS provide a natural language summary and explanation of the results.
3. Never fabricate exact counts beyond what the snapshot provides - use tools instead.
4. Provide concise, domain-relevant explanations.
5. Planet type categories by radius (Earth radii):
   - Super-Earth (1–2)
   - Sub-Neptune (2–4)
   - Neptune-like (4–10)
   - Ultra-Giant (>10)

Compact snapshot:
${contextSummary}
`.trim();

    const modelMessages: BasicMessage[] = [
      { role: 'system', content: system },
      ...messages.map(m => ({
        role: (m.role === 'user' || m.role === 'assistant' || m.role === 'system') ? m.role : 'user',
        content: String(m.content ?? '')
      }))
    ];

    console.log('[Chat API] Starting streamText with', {
      messageCount: modelMessages.length,
      planetDataCount: safePlanetData.length,
      classificationsCount: safePlanetTypeClassifications.length,
      hasPredictions: !!predictionResults
    });

    const result = streamText({
      model: openai('gpt-4o-mini'),
      messages: modelMessages,
      onStepFinish: ({ text, toolCalls, toolResults, finishReason, usage }) => {
        console.log('[Chat API] Step finished:', {
          textLength: text.length,
          toolCallsCount: toolCalls.length,
          toolResultsCount: toolResults.length,
          finishReason,
          toolCallNames: toolCalls.map(tc => tc.toolName)
        });
      },
      tools: {
        dataset_stats: tool({
          description: 'Get comprehensive statistics about the exoplanet dataset including radius stats, temperature stats, and type distributions',
          inputSchema: z.object({}),
          execute: async () => {
            return toolDatasetStats(
              safePlanetData,
              safePlanetTypeClassifications,
              predictionResults
            );
          }
        }),
        filter_radius: tool({
          description: 'Filter planets by radius with a comparison operator (>, >=, <, <=)',
          inputSchema: z.object({
            operator: z.enum(['>', '>=', '<', '<=']).describe('The comparison operator'),
            value: z.number().describe('The radius value in Earth radii to compare against')
          }),
          execute: async ({ operator, value }) => {
            return toolFilterRadius(safePlanetData, operator, value);
          }
        }),
        top_confidence_predictions: tool({
          description: 'Get the planets with the highest confidence predictions',
          inputSchema: z.object({
            limit: z.number().optional().default(5).describe('Number of top predictions to return (max 25)')
          }),
          execute: async ({ limit }) => {
            return toolTopConfidencePredictions(
              predictionResults,
              safePlanetTypeClassifications,
              limit || 5
            );
          }
        }),
        count_by_type: tool({
          description: 'Count exoplanets grouped by their classification type (Super-Earth, Sub-Neptune, Neptune-like, Ultra-Giant, etc.) and show the distribution',
          inputSchema: z.object({}),
          execute: async () => {
            return toolCountByType(
              safePlanetData,
              safePlanetTypeClassifications
            );
          }
        }),
        filter_by_range: tool({
          description: 'Filter planets by a range of values for any numeric field. Supports fields: radius (pl_rade), period (pl_orbper), temperature (pl_eqt), confidence (type_confidence)',
          inputSchema: z.object({
            field: z.string().describe('The field name to filter on (e.g., "radius", "period", "temperature", "confidence")'),
            min: z.number().optional().describe('Minimum value (inclusive)'),
            max: z.number().optional().describe('Maximum value (inclusive)')
          }),
          execute: async ({ field, min, max }) => {
            return toolFilterByRange(safePlanetData, field, min, max);
          }
        }),
        query_planets: tool({
          description: 'Advanced query to filter planets with multiple criteria including type, confidence, radius range, and period range',
          inputSchema: z.object({
            type_name: z.string().optional().describe('Filter by planet type (e.g., "Super-Earth", "Sub-Neptune", "Neptune-like", "Ultra-Giant")'),
            min_confidence: z.number().optional().describe('Minimum classification confidence (0-1)'),
            min_radius: z.number().optional().describe('Minimum radius in Earth radii'),
            max_radius: z.number().optional().describe('Maximum radius in Earth radii'),
            min_period: z.number().optional().describe('Minimum orbital period in days'),
            max_period: z.number().optional().describe('Maximum orbital period in days'),
            limit: z.number().optional().default(25).describe('Maximum number of results to return (max 50)')
          }),
          execute: async (params) => {
            return toolQueryPlanets(
              safePlanetData,
              safePlanetTypeClassifications,
              params
            );
          }
        }),
        search_by_field: tool({
          description: 'Search for planets by any field name and value. Common fields: sy_snum (number of stars in system), sy_pnum (number of planets in system), st_mass (stellar mass), st_rad (stellar radius), hostname, etc.',
          inputSchema: z.object({
            field_name: z.string().describe('The field name to search (e.g., "sy_snum", "sy_pnum", "hostname")'),
            field_value: z.union([z.string(), z.number()]).describe('The value to search for'),
            operator: z.enum(['equals', 'greater_than', 'less_than', 'contains']).optional().default('equals').describe('Comparison operator'),
            limit: z.number().optional().default(25).describe('Maximum number of results to return (max 50)')
          }),
          execute: async ({ field_name, field_value, operator, limit }) => {
            return toolSearchByField(
              safePlanetData,
              field_name,
              field_value,
              operator || 'equals',
              limit || 25
            );
          }
        })
      },
    });

    const stream = new ReadableStream({
      async start(controller) {
        try {
          let hasToolResults = false;
          const toolResults: Array<{ toolName: string; output: string }> = [];

          // Use fullStream to get all events including tool calls and results
          for await (const chunk of result.fullStream) {
            console.log('[Chat API] Full stream chunk:', chunk.type);

            if (chunk.type === 'text-delta') {
              console.log('[Chat API] Text delta:', chunk.text);
              controller.enqueue(`data: ${JSON.stringify(chunk.text)}\n\n`);
            } else if (chunk.type === 'tool-call') {
              console.log('[Chat API] Tool call:', chunk.toolName);
              // Optionally send tool call notification to UI
            } else if (chunk.type === 'tool-result') {
              const output = (chunk as any).output || (chunk as any).result || '';
              console.log('[Chat API] Tool result for:', chunk.toolName, 'output length:', output.length);
              hasToolResults = true;
              toolResults.push({ toolName: chunk.toolName, output });
            } else if (chunk.type === 'finish') {
              console.log('[Chat API] Finish reason:', chunk.finishReason);
              // If we have tool results but no text was generated, send the tool results directly
              if (hasToolResults && toolResults.length > 0) {
                console.log('[Chat API] Sending tool results directly to UI');
                for (const { output } of toolResults) {
                  // Stream the tool output as chunks
                  const lines = output.split('\n');
                  for (const line of lines) {
                    controller.enqueue(`data: ${JSON.stringify(line + '\n')}\n\n`);
                  }
                }
              }
            }
          }
          console.log('[Chat API] Stream complete');
          controller.enqueue('data: [DONE]\n\n');
        } catch (err) {
          console.error('[Chat API] Stream error:', err);
          controller.enqueue(
            `data: ${JSON.stringify({
              error: err instanceof Error ? err.message : 'Unknown streaming error'
            })}\n\n`
          );
        } finally {
          controller.close();
        }
      }
    });

    return new Response(stream, {
      headers: sseHeaders()
    });

  } catch (error) {
    console.error('Chat API error (enhanced route):', error);
    return jsonError('Internal Server Error', 500);
  }
}

/* ------------------------------- HTTP Helpers ------------------------------- */

function sseHeaders(): Record<string, string> {
  return {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive'
  };
}

function jsonError(message: string, status: number): Response {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { 'Content-Type': 'application/json' }
  });
}
