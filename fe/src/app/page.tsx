"use client";

import { useRouter } from "next/navigation";
import { Meteors } from "@/components/ui/meteors";

export default function HomePage() {
    const router = useRouter();

    return (
        <div className="h-[calc(100vh-4rem)] overflow-hidden flex flex-col justify-center pl-24 px-40">
            <div className="mb-12">
                <h1 className="text-8xl font-bold mb-4">Discover New Worlds</h1>
                <p className="text-2xl max-w-4xl">
                    Use machine learning to predict exoplanets or train your own
                    model with scientific data. Upload CSVs, visualize
                    predictions, and explore the cosmos through data.
                </p>
            </div>

            {/* Cards Section */}
            <div className="flex gap-6 justify-start items-stretch max-w-4xl">
                {/* Train Card */}
                <div
                    onClick={() => router.push("/train")}
                    className="relative w-full cursor-pointer transition-transform hover:scale-105"
                >
                    <div className="absolute inset-0 h-full w-full scale-[0.80] transform rounded-full bg-gradient-to-r from-blue-500 to-teal-500 blur-3xl opacity-50" />
                    <div className="relative flex h-full flex-col items-start justify-between overflow-hidden rounded-2xl border border-gray-700 bg-black/40 backdrop-blur-sm px-6 py-6">
                        <div className="flex items-center gap-3 mb-2">
                            <div className="flex h-6 w-6 items-center justify-center rounded-full border border-gray-600">
                                <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                    strokeWidth="1.5"
                                    stroke="currentColor"
                                    className="h-3 w-3 text-gray-400"
                                >
                                    <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        d="M4.26 10.147a60.436 60.436 0 00-.491 6.347A48.627 48.627 0 0112 20.904a48.627 48.627 0 018.232-4.41 60.46 60.46 0 00-.491-6.347m-15.482 0a50.57 50.57 0 00-2.658-.813A59.905 59.905 0 0112 3.493a59.902 59.902 0 0110.399 5.84c-.896.248-1.783.52-2.658.814m-15.482 0A50.697 50.697 0 0112 13.489a50.702 50.702 0 017.74-3.342M6.75 15a.75.75 0 100-1.5.75.75 0 000 1.5zm0 0v-3.675A55.378 55.378 0 0112 8.443m-7.007 11.55A5.981 5.981 0 006.75 15.75v-1.5"
                                    />
                                </svg>
                            </div>
                            <h2 className="relative z-50 text-2xl font-bold text-white">
                                Train Model
                            </h2>
                        </div>

                        <p className="relative z-50 text-sm font-normal text-slate-400">
                            Upload your dataset and train a custom ML model to
                            discover exoplanets.
                        </p>

                        <Meteors number={15} />
                    </div>
                </div>

                {/* Predict Card */}
                <div
                    onClick={() => router.push("/predict")}
                    className="relative w-full cursor-pointer transition-transform hover:scale-105"
                >
                    <div className="absolute inset-0 h-full w-full scale-[0.80] transform rounded-full bg-gradient-to-r from-purple-500 to-pink-500 blur-3xl opacity-50" />
                    <div className="relative flex h-full flex-col items-start justify-between overflow-hidden rounded-2xl border border-gray-700 bg-black/40 backdrop-blur-sm px-6 py-6">
                        <div className="flex items-center gap-3 mb-2">
                            <div className="flex h-6 w-6 items-center justify-center rounded-full border border-gray-600">
                                <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                    strokeWidth="1.5"
                                    stroke="currentColor"
                                    className="h-3 w-3 text-gray-400"
                                >
                                    <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z"
                                    />
                                </svg>
                            </div>
                            <h2 className="relative z-50 text-2xl font-bold text-white">
                                Predict Exoplanets
                            </h2>
                        </div>

                        <p className="relative z-50 text-sm font-normal text-slate-400">
                            Use pre-trained models to predict exoplanets from
                            observation data.
                        </p>

                        <Meteors number={15} />
                    </div>
                </div>
            </div>
        </div>
    );
}
