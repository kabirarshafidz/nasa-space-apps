import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET() {
  try {
    // Read the TESS CSV file from the data directory
    const filePath = path.join(process.cwd(), "..", "data", "tess.csv");
    
    if (!fs.existsSync(filePath)) {
      return NextResponse.json(
        { error: "TESS data file not found" },
        { status: 404 }
      );
    }
    
    const data = fs.readFileSync(filePath, "utf-8");
    
    return new NextResponse(data, {
      status: 200,
      headers: {
        "Content-Type": "text/csv",
        "Cache-Control": "public, max-age=3600", // Cache for 1 hour
      },
    });
  } catch (error) {
    console.error("Error reading TESS data:", error);
    return NextResponse.json(
      { error: "Failed to load TESS data" },
      { status: 500 }
    );
  }
}
