import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { trainingEntry, model, trainingSession } from "@/lib/schema";
import { auth } from "@/lib/auth";
import { headers } from "next/headers";
import { eq } from "drizzle-orm";
import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";

// S3/R2 Configuration
const s3Client = new S3Client({
    region: "auto",
    endpoint: "https://26966518ccbce9889c6f3ca4b63214d8.r2.cloudflarestorage.com",
    credentials: {
        accessKeyId: "e4be7a0c006e11055ae1b3083995f6e6",
        secretAccessKey: "86c9e2d2eed11ed7f548f4dfee38e538872c832257d6c397e05f8a0c653df0bc",
    },
});

const BUCKET_NAME = "nasa";
const PUBLIC_URL_BASE = "https://pub-000e8ab9810a4b32ae818ab4c4881da5.r2.dev";

export async function POST(req: NextRequest) {
    try {
        // Validate session
        const session = await auth.api.getSession({
            headers: await headers(),
        });

        if (!session?.user) {
            return NextResponse.json(
                { success: false, error: "Unauthorized" },
                { status: 401 }
            );
        }

        // Get form data from request
        const formData = await req.formData();
        const trainingSessionId = formData.get("training_session_id") as string;
        const csvFile = formData.get("file") as File | null;
        const userModelName = formData.get("user_model_name") as string;

        if (!trainingSessionId) {
            return NextResponse.json(
                { success: false, error: "Training session ID is required" },
                { status: 400 }
            );
        }

        // Verify the training session belongs to the user
        const ts = await db.query.trainingSession.findFirst({
            where: (table, { eq }) => eq(table.id, trainingSessionId),
        });

        if (!ts) {
            return NextResponse.json(
                { success: false, error: "Training session not found" },
                { status: 404 }
            );
        }

        if (ts.userId !== session.user.id) {
            return NextResponse.json(
                { success: false, error: "Unauthorized access to training session" },
                { status: 403 }
            );
        }

        // Upload CSV to S3 if provided and not already uploaded
        let csvS3Key = ts.csvS3Key;
        let csvUrl = ts.csvUrl;

        if (csvFile && !csvS3Key) {
            const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
            csvS3Key = `datasets/session_${trainingSessionId}_${timestamp}.csv`;

            const arrayBuffer = await csvFile.arrayBuffer();
            const buffer = Buffer.from(arrayBuffer);

            await s3Client.send(
                new PutObjectCommand({
                    Bucket: BUCKET_NAME,
                    Key: csvS3Key,
                    Body: buffer,
                    ContentType: "text/csv",
                })
            );

            csvUrl = `${PUBLIC_URL_BASE}/${csvS3Key}`;

            // Update training session with CSV info
            await db
                .update(trainingSession)
                .set({
                    csvS3Key,
                    csvUrl,
                })
                .where(eq(trainingSession.id, trainingSessionId));
        }

        // Forward request to Python API
        const pythonApiUrl = `${process.env.NEXT_PUBLIC_API_ENDPOINT}/train/cv`;

        const pythonResponse = await fetch(pythonApiUrl, {
            method: "POST",
            body: formData,
        });

        if (!pythonResponse.ok) {
            const error = await pythonResponse.json();
            return NextResponse.json(
                { success: false, error: error.detail || "Training failed" },
                { status: pythonResponse.status }
            );
        }

        const result = await pythonResponse.json();

        // Extract model S3 key from model_url
        const modelS3Key = result.model_url.split("/").pop() || result.model_url;

        // Save training entry to database
        const [newEntry] = await db
            .insert(trainingEntry)
            .values({
                trainingSessionId: trainingSessionId,
                result,
                modelS3Key,
            })
            .returning();

        // Calculate F1 score as string (multiply by 100 to preserve 2 decimal places)
        const f1ScoreStr = (result.oof_metrics.f1 * 100).toFixed(2);

        // Create model entry - use user's custom name instead of algorithm name
        const [newModel] = await db
            .insert(model)
            .values({
                name: userModelName || result.model_name,
                key: modelS3Key,
                size: 0,
                f1Score: f1ScoreStr,
                trainingEntryId: newEntry.id,
            })
            .returning();

        // Return the training result along with DB IDs
        return NextResponse.json({
            ...result,
            entryId: newEntry.id,
            modelId: newModel.id,
        });

    } catch (error) {
        console.error("Error in training proxy:", error);
        return NextResponse.json(
            {
                success: false,
                error: error instanceof Error ? error.message : "Failed to process training request",
            },
            { status: 500 }
        );
    }
}
