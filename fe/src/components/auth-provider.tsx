"use client";

import { useEffect } from "react";
import { authClient } from "@/lib/auth-client";

export function AuthProvider({ children }: { children: React.ReactNode }) {
    useEffect(() => {
        const initializeAuth = async () => {
            const session = await authClient.getSession();

            if (!session.data) {
                await authClient.signIn.anonymous();
            }
        };

        initializeAuth();
    }, []);

    return <>{children}</>;
}
