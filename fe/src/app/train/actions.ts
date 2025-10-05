"use server";

import { db } from "@/lib/db";
import { trainingSession } from "@/lib/schema";
import { eq } from "drizzle-orm";
import { auth } from "@/lib/auth";
import { headers } from "next/headers";
import { randomUUID } from "crypto";

export async function getTrainingSessions() {
    const session = await auth.api.getSession({
        headers: await headers(),
    });

    if (!session?.user) {
        return { success: false, error: "Unauthorized", sessions: [] };
    }

    try {
        const sessions = await db
            .select()
            .from(trainingSession)
            .where(eq(trainingSession.userId, session.user.id))
            .orderBy(trainingSession.createdAt);

        return { success: true, sessions };
    } catch (error) {
        console.error("Error fetching training sessions:", error);
        return { success: false, error: "Failed to fetch sessions", sessions: [] };
    }
}

export async function createTrainingSession() {
    const session = await auth.api.getSession({
        headers: await headers(),
    });

    if (!session?.user) {
        return { success: false, error: "Unauthorized" };
    }

    try {
        const newSession = await db
            .insert(trainingSession)
            .values({
                id: randomUUID(),
                userId: session.user.id,
            })
            .returning();

        return { success: true, session: newSession[0] };
    } catch (error) {
        console.error("Error creating training session:", error);
        return { success: false, error: "Failed to create session" };
    }
}

export async function deleteTrainingSession(sessionId: string) {
    const session = await auth.api.getSession({
        headers: await headers(),
    });

    if (!session?.user) {
        return { success: false, error: "Unauthorized" };
    }

    try {
        // Verify the session belongs to the user
        const existingSession = await db
            .select()
            .from(trainingSession)
            .where(eq(trainingSession.id, sessionId))
            .limit(1);

        if (existingSession.length === 0) {
            return { success: false, error: "Session not found" };
        }

        if (existingSession[0].userId !== session.user.id) {
            return { success: false, error: "Unauthorized" };
        }

        await db
            .delete(trainingSession)
            .where(eq(trainingSession.id, sessionId));

        return { success: true };
    } catch (error) {
        console.error("Error deleting training session:", error);
        return { success: false, error: "Failed to delete session" };
    }
}

export async function getTrainingSession(sessionId: string) {
    const session = await auth.api.getSession({
        headers: await headers(),
    });

    if (!session?.user) {
        return { success: false, error: "Unauthorized", session: null };
    }

    try {
        const trainingSessionData = await db.query.trainingSession.findFirst({
            where: (ts, { eq }) => eq(ts.id, sessionId),
        });

        if (!trainingSessionData) {
            return { success: false, error: "Session not found", session: null };
        }

        if (trainingSessionData.userId !== session.user.id) {
            return { success: false, error: "Unauthorized", session: null };
        }

        return { success: true, session: trainingSessionData };
    } catch (error) {
        console.error("Error fetching training session:", error);
        return { success: false, error: "Failed to fetch session", session: null };
    }
}
