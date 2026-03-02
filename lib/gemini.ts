import { GoogleGenerativeAI } from "@google/generative-ai";

if (!process.env.GEMINI_API_KEY) {
    throw new Error("Missing GEMINI_API_KEY in environment variables");
}

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Concurrency control to prevent hitting rate limits during orchestration
let activeRequests = 0;
const MAX_CONCURRENT_REQUESTS = 2; // Gemini Flash Tier allows ~15 RPM, let's play safe
const requestQueue: (() => void)[] = [];

async function acquireSlot() {
    if (activeRequests < MAX_CONCURRENT_REQUESTS) {
        activeRequests++;
        return;
    }
    return new Promise<void>(resolve => {
        requestQueue.push(resolve);
    });
}

function releaseSlot() {
    activeRequests--;
    if (requestQueue.length > 0) {
        const next = requestQueue.shift();
        if (next) {
            activeRequests++;
            next();
        }
    }
}

/**
 * Helper to wrap functions with exponential backoff retry logic and jitter.
 */
async function withRetry<T>(fn: () => Promise<T>, maxRetries = 10, baseDelay = 2000): Promise<T> {
    let lastError: any;

    // Always wait for a slot before making the request
    await acquireSlot();

    try {
        for (let i = 0; i <= maxRetries; i++) {
            try {
                return await fn();
            } catch (error: any) {
                lastError = error;
                const isServiceError =
                    error.message?.includes("503") ||
                    error.message?.includes("Service Unavailable") ||
                    error.message?.includes("429") ||
                    error.message?.includes("Too Many Requests") ||
                    error.message?.includes("Deadline Exceeded");

                if (!isServiceError || i === maxRetries) {
                    break;
                }

                // Exponential backoff with jitter
                const backoff = baseDelay * Math.pow(2, i);
                const jitter = Math.random() * 1000;
                const delay = backoff + jitter;

                console.warn(`[GEMINI] rate/service error: ${error.message.substring(0, 100)}. Retrying in ${Math.round(delay)}ms... (Attempt ${i + 1}/${maxRetries})`);
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
    } finally {
        // Release slot regardless of success or final failure
        releaseSlot();
    }
    throw lastError;
}

/**
 * Generates an embedding for a given text chunk.
 * Using gemini-embedding-001 with outputDimensionality: 768 to match database.
 */
export async function generateEmbedding(text: string) {
    return withRetry(async () => {
        const modelName = process.env.GEMINI_EMBEDDING_MODEL || "gemini-embedding-001";
        console.log(`[GEMINI] Using embedding model: ${modelName}`);
        const model = genAI.getGenerativeModel({ model: modelName });
        const result = await model.embedContent({
            content: { role: "user", parts: [{ text }] },
            outputDimensionality: 768,
        } as any);
        return result.embedding.values;
    });
}

/**
 * Interacts with Gemini Pro for structured extraction or general reasoning.
 */
export async function generateContent(prompt: string, modelName = process.env.GEMINI_MODEL || "gemini-2.0-flash") {
    return withRetry(async () => {
        console.log(`[GEMINI] Using content model: ${modelName}`);
        const model = genAI.getGenerativeModel({ model: modelName });
        const result = await model.generateContent(prompt);
        return result.response.text();
    });
}

/**
 * Generates structured JSON output from Gemini using a schema-following prompt.
 */
export async function generateStructuredOutput<T>(prompt: string, schemaDescription: string): Promise<T> {
    console.log("[GEMINI] Generating structured output...");

    return withRetry(async () => {
        const modelName = process.env.GEMINI_MODEL || "gemini-2.0-flash";
        console.log(`[GEMINI] Generating structured output with model: ${modelName}...`);
        const model = genAI.getGenerativeModel({
            model: modelName,
            generationConfig: {
                responseMimeType: "application/json",
            }
        });

        const fullPrompt = `
        ${prompt}
        
        You MUST return the output as a JSON object following this schema:
        ${schemaDescription}
      `;

        const result = await model.generateContent(fullPrompt);
        const text = result.response.text();
        console.log("[GEMINI] Response received");
        return JSON.parse(text) as T;
    }).catch((error: any) => {
        console.error("[GEMINI] Final failure:", error.message);
        throw new Error(`Gemini failed after retries: ${error.message}`);
    });
}
