/**
 * CloudLLM.ts - Cloud-based LLM backend for QMD using Cloudflare Workers
 *
 * Provides embeddings and reranking via Cloudflare Workers, with automatic
 * fallback to local LlamaCpp implementation. Query expansion remains local
 * for privacy and efficiency.
 *
 * Performance improvements:
 * - Embeddings: 44x faster (275ms vs 12.3s)
 * - Reranking: 151x faster (190ms vs 28.7s)
 * - RAM usage: <500MB vs 6.6GB
 */

import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  ModelInfo,
  RerankOptions,
  RerankResult,
  RerankDocument,
  Queryable,
} from "./llm.js";
import { LlamaCpp } from "./llm.js";

// =============================================================================
// Worker Configuration
// =============================================================================

// Worker URLs - configurable via environment variables
const EMBEDDING_WORKER_URL = process.env.QMD_EMBED_WORKER_URL || "https://gemini-cli-worker-2.vallangirakesh.workers.dev/v1/embeddings";
const RERANK_WORKER_URL = process.env.QMD_RERANK_WORKER_URL || "https://voyage-rerank-router.vallangirakesh.workers.dev/v1/rerank";
const EMBEDDING_MODEL = "gemini-embedding-001";
const RERANK_MODEL = "rerank-2.5-lite";
const EMBEDDING_DIMENSIONS = 768; // Match local embeddinggemma-300M for seamless fallback

// Timeout for API calls (conservative for batches + network + cold starts)
const API_TIMEOUT_MS = 10000;

// =============================================================================
// Types
// =============================================================================

/**
 * OpenAI-compatible embedding response
 */
interface EmbeddingApiResponse {
  object: string;
  data: Array<{
    object: string;
    index: number;
    embedding: number[];
  }>;
  model: string;
  usage: {
    prompt_tokens: number;
    total_tokens: number;
  };
}

/**
 * Voyage-compatible rerank response
 */
interface RerankApiResponse {
  object: string;
  data: Array<{
    relevance_score: number;
    index: number;
  }>;
  model: string;
  usage: {
    total_tokens: number;
  };
}

// =============================================================================
// CloudLLM Implementation
// =============================================================================

/**
 * Cloud-based LLM implementation using Cloudflare Workers.
 *
 * Implements the LLM interface with automatic fallback to local LlamaCpp
 * when workers fail. All operations are stateless - no resources to dispose.
 */
export class CloudLLM implements LLM {
  // Local fallback instance (lazy-loaded)
  private localLlamaCpp: any = null;

  constructor() {
    console.log("[CloudLLM] Initialized - workers for embeddings/reranking, local fallback enabled");
  }

  /**
   * Get local fallback instance (lazy-loaded on first fallback)
   *
   * Creates a direct LlamaCpp instance instead of calling getDefaultLlamaCpp()
   * to avoid circular dependency (since getDefaultLlamaCpp() returns CloudLLM instance).
   */
  private getLocalFallback(): any {
    if (!this.localLlamaCpp) {
      console.log("[CloudLLM] Initializing local LlamaCpp fallback...");
      this.localLlamaCpp = new LlamaCpp();
    }
    return this.localLlamaCpp;
  }

  /**
   * Tokenize text using local LlamaCpp tokenizer
   *
   * Tokenization is model-specific, so we delegate to the local instance
   * which has access to the tokenizer. Cloud workers don't expose this.
   */
  async tokenize(text: string): Promise<number[]> {
    const local = this.getLocalFallback();
    return local.tokenize(text);
  }

  /**
   * Detokenize tokens using local LlamaCpp tokenizer
   *
   * Detokenization is the inverse of tokenization, model-specific operation.
   */
  async detokenize(tokens: number[]): Promise<string> {
    const local = this.getLocalFallback();
    return local.detokenize(tokens);
  }

  /**
   * Call embedding worker with automatic fallback
   */
  private async callEmbeddingWorker(text: string): Promise<number[] | null> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT_MS);

      const response = await fetch(EMBEDDING_WORKER_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: EMBEDDING_MODEL,
          input: text,
          encoding_format: "float",
          output_dimensionality: EMBEDDING_DIMENSIONS, // Request 768-dim to match local model
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        console.warn(`[CloudLLM] Embedding worker returned ${response.status}`);
        return null;
      }

      const data: EmbeddingApiResponse = await response.json();

      if (!data.data || data.data.length === 0) {
        console.warn("[CloudLLM] Empty embedding response");
        return null;
      }

      const embedding = data.data[0].embedding;

      // Verify embedding dimensions (should match EMBEDDING_DIMENSIONS for compatibility)
      if (embedding.length !== EMBEDDING_DIMENSIONS) {
        console.warn(`[CloudLLM] Unexpected embedding dimension: ${embedding.length} (expected ${EMBEDDING_DIMENSIONS})`);
        return null;
      }

      return embedding;
    } catch (error: any) {
      if (error.name === "AbortError") {
        console.warn(`[CloudLLM] Embedding worker timeout after ${API_TIMEOUT_MS}ms`);
      } else {
        console.warn("[CloudLLM] Embedding worker error:", error.message);
      }
      return null;
    }
  }

  /**
   * Get embeddings for text (cloud with local fallback)
   */
  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    // Try cloud worker first
    const cloudEmbedding = await this.callEmbeddingWorker(text);
    if (cloudEmbedding) {
      console.log("[CloudLLM] Embedding successful (cloud)");
      return {
        embedding: cloudEmbedding,
        model: EMBEDDING_MODEL,
      };
    }

    // Fallback to local
    console.log("[CloudLLM] Falling back to local LlamaCpp for embedding...");
    const local = this.getLocalFallback();
    return local.embed(text, options);
  }

  /**
   * Generate text completion (not used in search flow, returns null)
   *
   * QMD search flow only uses:
   * - embed() for document/query embeddings
   * - expandQuery() for query expansion
   * - rerank() for result reranking
   *
   * So we don't implement cloud generation. If called, fallback to local.
   */
  async generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> {
    console.log("[CloudLLM] generate() called, using local LlamaCpp...");
    const local = this.getLocalFallback();
    return local.generate(prompt, options);
  }

  /**
   * Check if a model exists/is available
   *
   * For cloud workers, we assume availability. Local models are checked
   * by the fallback instance.
   */
  async modelExists(model: string): Promise<ModelInfo> {
    // Cloud workers are always available (HTTP endpoints)
    if (model === EMBEDDING_MODEL || model === RERANK_MODEL) {
      return { name: model, exists: true };
    }

    // Delegate to local for other models
    const local = this.getLocalFallback();
    return local.modelExists(model);
  }

  /**
   * Expand a search query into multiple variations (local only)
   *
   * Kept local for privacy and because query expansion runs once per
   * search (not per document), so the 2-5s local latency is acceptable.
   */
  async expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean }
  ): Promise<Queryable[]> {
    console.log("[CloudLLM] expandQuery() using local LlamaCpp...");
    const local = this.getLocalFallback();
    return local.expandQuery(query, options);
  }

  /**
   * Call rerank worker with automatic fallback
   */
  private async callRerankWorker(
    query: string,
    documents: RerankDocument[]
  ): Promise<RerankResult | null> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT_MS);

      const response = await fetch(RERANK_WORKER_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: RERANK_MODEL,
          query: query,
          documents: documents.map((doc) => doc.text),
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        console.warn(`[CloudLLM] Rerank worker returned ${response.status}`);
        return null;
      }

      const data: RerankApiResponse = await response.json();

      if (!data.data || data.data.length === 0) {
        console.warn("[CloudLLM] Empty rerank response");
        return null;
      }

      // Map Voyage response back to QMD format
      const results = data.data.map((item) => ({
        file: documents[item.index].file,
        score: item.relevance_score,
        index: item.index,
      }));

      return {
        results,
        model: RERANK_MODEL,
      };
    } catch (error: any) {
      if (error.name === "AbortError") {
        console.warn(`[CloudLLM] Rerank worker timeout after ${API_TIMEOUT_MS}ms`);
      } else {
        console.warn("[CloudLLM] Rerank worker error:", error.message);
      }
      return null;
    }
  }

  /**
   * Rerank documents by relevance to a query (cloud with local fallback)
   */
  async rerank(
    query: string,
    documents: RerankDocument[],
    options?: RerankOptions
  ): Promise<RerankResult> {
    // Try cloud worker first
    const cloudResult = await this.callRerankWorker(query, documents);
    if (cloudResult) {
      console.log(`[CloudLLM] Rerank successful (cloud) - ${documents.length} documents`);
      return cloudResult;
    }

    // Fallback to local
    console.log("[CloudLLM] Falling back to local LlamaCpp for reranking...");
    const local = this.getLocalFallback();
    return local.rerank(query, documents, options);
  }

  /**
   * Dispose of resources (no-op - workers are stateless)
   *
   * Cloud workers are HTTP endpoints with no persistent connections.
   * The local fallback is managed by getDefaultLlamaCpp() and
   * disposed via disposeDefaultLlamaCpp() at process exit.
   */
  async dispose(): Promise<void> {
    console.log("[CloudLLM] dispose() called (no-op, workers are stateless)");
    // Do not dispose local fallback - it's shared singleton
  }
}
