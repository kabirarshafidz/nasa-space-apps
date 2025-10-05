import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { trainingEntry, model } from "@/lib/schema";
import { auth } from "@/lib/auth";
import { headers } from "next/headers";
import { eq, desc } from "drizzle-orm";

export async function GET(req: NextRequest) {
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

        // Get session ID from query params
        const { searchParams } = new URL(req.url);
        const trainingSessionId = searchParams.get("sessionId");

        if (!trainingSessionId) {
            return NextResponse.json(
                { success: false, error: "Session ID is required" },
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

        // Get all training entries for this session with their models
        const entries = await db
            .select({
                id: trainingEntry.id,
                result: trainingEntry.result,
                modelS3Key: trainingEntry.modelS3Key,
                createdAt: trainingEntry.createdAt,
                modelId: model.id,
                modelName: model.name,
                modelF1Score: model.f1Score,
            })
            .from(trainingEntry)
            .leftJoin(model, eq(model.trainingEntryId, trainingEntry.id))
            .where(eq(trainingEntry.trainingSessionId, trainingSessionId))
            .orderBy(desc(trainingEntry.createdAt));

        return NextResponse.json({
            success: true,
            entries,
        });
    } catch (error) {
        console.error("Error fetching training entries:", error);
        return NextResponse.json(
            {
                success: false,
                error: error instanceof Error ? error.message : "Failed to fetch training entries",
            },
            { status: 500 }
        );
    }
}
