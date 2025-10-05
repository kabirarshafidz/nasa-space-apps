// Type definitions for the prediction page

export interface PreTrainedModel {
  name: string;
  key: string;
  size: number;
  last_modified: string;
  url: string;
}

export interface PredictionResults {
  predictions: number[];
  predicted_labels: number[];
  feature_count: number;
  metadata?: Array<{ toi: string; toipfx: string }>;
}

export interface PlanetTypeClassification {
  toi: string;
  type_pred: number;
  type_confidence: number;
  type_name: string;
}

export interface SingleFeatures {
  pl_orbper: string;
  pl_trandurh: string;
  pl_trandep: string;
  pl_rade: string;
  pl_insol: string;
  pl_eqt: string;
  st_tmag: string;
  st_dist: string;
  st_teff: string;
  st_logg: string;
  st_rad: string;
  pl_rade_relerr: string;
}

export interface Metadata {
  toi: string;
  toipfx: string;
}

export interface RequiredFeature {
  name: string;
  label: string;
}

export const REQUIRED_FEATURES: RequiredFeature[] = [
  { name: "pl_orbper", label: "Orbital Period (days)" },
  { name: "pl_trandurh", label: "Transit Duration (hours)" },
  { name: "pl_trandep", label: "Transit Depth (ppm)" },
  { name: "pl_rade", label: "Planet Radius (Earth radii)" },
  { name: "pl_insol", label: "Insolation Flux (Earth flux)" },
  { name: "pl_eqt", label: "Equilibrium Temperature (K)" },
  { name: "st_tmag", label: "TESS Magnitude" },
  { name: "st_dist", label: "Distance (pc)" },
  { name: "st_teff", label: "Stellar Effective Temp (K)" },
  { name: "st_logg", label: "Stellar Surface Gravity (log g)" },
  { name: "st_rad", label: "Stellar Radius (Solar radii)" },
  { name: "pl_rade_relerr", label: "Planet Radius Relative Error" },
];
