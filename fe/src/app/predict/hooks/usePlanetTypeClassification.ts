import { useState } from "react";
import { PlanetTypeClassification, Metadata } from "../types";
import { parseCSVFeatures } from "../utils/csvParser";

const CLASSIFICATION_FEATURES = [
  "pl_rade",
  "pl_insol",
  "pl_eqt",
  "pl_orbper",
  "st_teff",
  "st_rad",
];

export function usePlanetTypeClassification() {
  const [planetTypeChart, setPlanetTypeChart] = useState<string | null>(null);
  const [planetTypeClassifications, setPlanetTypeClassifications] = useState<
    PlanetTypeClassification[]
  >([]);

  const fetchPlanetTypeClassifications = async (
    metadataArray: Metadata[],
    featuresData: any
  ) => {
    try {
      const formData = new FormData();

      // Extract TOI list
      const toiList = metadataArray.map((m) => m.toi);
      formData.append("toi_list", JSON.stringify(toiList));

      // Prepare features JSON
      let featuresList: Array<Record<string, number | null>> = [];
      
      if (typeof featuresData === "string") {
        try {
          featuresList = parseCSVFeatures(featuresData, CLASSIFICATION_FEATURES);
        } catch (error) {
          console.error("Could not parse CSV for planet type classification:", error);
          return;
        }
      } else {
        // Single prediction
        featuresList = featuresData;
      }

      formData.append("features_json", JSON.stringify(featuresList));

      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_ENDPOINT}/classify/planet-types`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (response.ok) {
        const data = await response.json();
        setPlanetTypeChart(data.chart_base64);
        setPlanetTypeClassifications(data.classifications || []);
      }
    } catch (error) {
      console.error("Planet type classification error:", error);
      // Don't fail the whole prediction if classification fails
    }
  };

  return {
    planetTypeChart,
    planetTypeClassifications,
    fetchPlanetTypeClassifications,
  };
}
