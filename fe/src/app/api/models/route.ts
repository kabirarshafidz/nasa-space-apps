import { NextResponse } from "next/server";
import { db } from "@/lib/db";
import { auth } from "@/lib/auth";
import { headers } from "next/headers";
import { sql } from "drizzle-orm";
import { S3Client, ListObjectsV2Command } from "@aws-sdk/client-s3";

// S3/R2 Configuration
const s3Client = new S3Client({
    region: "auto",
    endpoint: "https://26966518ccbce9889c6f3ca4b63214d8.r2.cloudflarestorage.com",
    credentials: {
        accessKeyId: process.env.S3_ACCESS_KEY_ID!,
        secretAccessKey: process.env.S3_SECRET_ACCESS_KEY!,
    },
});

const BUCKET_NAME = "nasa";
const PUBLIC_URL_BASE = "https://pub-000e8ab9810a4b32ae818ab4c4881da5.r2.dev";

export async function GET() {
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

        // Get all user's models with their training entry details using SQL
        const userModels = await db.execute(sql`
            SELECT
                m.id,
                m.name,
                m.key,
                m.size,
                m.f1_score,
                m.created_at,
                m.training_entry_id,
                te.training_session_id
            FROM model m
            INNER JOIN training_entry te ON m.training_entry_id = te.id
            INNER JOIN training_session ts ON te.training_session_id = ts.id
            WHERE ts.user_id = ${session.user.id}
            ORDER BY m.created_at DESC
        `);

        // Format user models
        const formattedUserModels = (userModels.rows as Array<{
            id: string;
            name: string;
            key: string;
            size: number;
            f1_score: string;
            created_at: Date;
            training_entry_id: string;
            training_session_id: string;
        }>).map((m) => ({
            id: m.id,
            name: m.name,
            key: m.key,
            size: m.size,
            f1Score: parseFloat(m.f1_score),
            createdAt: new Date(m.created_at),
            url: `https://pub-000e8ab9810a4b32ae818ab4c4881da5.r2.dev/${m.key}`,
            isDefault: false,
            owner: "user",
        }));

        // Get default models from S3
        const defaultModels: Array<{
            id: string;
            name: string;
            key: string;
            size: number;
            f1Score: number;
            createdAt: Date;
            url: string;
            isDefault: boolean;
            owner: string;
        }> = [];

        try {
            const s3Response = await s3Client.send(
                new ListObjectsV2Command({
                    Bucket: BUCKET_NAME,
                    Prefix: "default/",
                })
            );

            if (s3Response.Contents) {
                for (const obj of s3Response.Contents) {
                    if (obj.Key && (obj.Key.endsWith(".joblib") || obj.Key.endsWith(".pkl"))) {
                        const modelName = obj.Key.replace("default/", "")
                            .replace(".joblib", "")
                            .replace(".pkl", "");

                        defaultModels.push({
                            id: `default-${modelName}`,
                            name: modelName,
                            key: obj.Key,
                            size: obj.Size || 0,
                            f1Score: 0, // Default models don't have F1 scores
                            createdAt: obj.LastModified || new Date(),
                            url: `${PUBLIC_URL_BASE}/${obj.Key}`,
                            isDefault: true,
                            owner: "system",
                        });
                    }
                }
            }
        } catch (s3Error) {
            console.error("Error fetching default models from S3:", s3Error);
            // Continue even if S3 fails
        }

        // Combine user models and default models
        const allModels = [...formattedUserModels, ...defaultModels];

        return NextResponse.json({
            success: true,
            models: allModels,
            count: allModels.length,
            userModelCount: formattedUserModels.length,
            defaultModelCount: defaultModels.length,
        });
    } catch (error) {
        console.error("Error fetching models:", error);
        return NextResponse.json(
            {
                success: false,
                error: error instanceof Error ? error.message : "Failed to fetch models",
            },
            { status: 500 }
        );
    }
}
