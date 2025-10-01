/**
 * RSE Configuration - User-Controllable Settings
 * Centralized configuration to avoid brittle hardcoded values
 */

export interface EmbeddingBackendConfig {
  type: 'python-service' | 'ollama';

  // Python FastAPI service
  pythonServiceUrl?: string;
  pythonServiceTimeout?: number;

  // Ollama
  ollamaUrl?: string;
  ollamaModel?: string;
  ollamaTimeout?: number;
}

export interface RSEConfig {
  // Embedding backend configuration
  embeddingBackend: EmbeddingBackendConfig;

  // RSE algorithm parameters
  threshold: number;
  windowSize: number;
  manifoldDimension: number;
  useManifoldMetrics: boolean;

  // Performance tuning
  batchSize: number;
  maxConcurrentRequests: number;
}

// Default configuration - user can override via environment or constructor
export const DEFAULT_RSE_CONFIG: RSEConfig = {
  embeddingBackend: {
    type: process.env.EMBEDDING_BACKEND as 'python-service' | 'ollama' || 'ollama',

    // Python service defaults
    pythonServiceUrl: process.env.PYTHON_SERVICE_URL || 'http://127.0.0.1:8001',
    pythonServiceTimeout: parseInt(process.env.PYTHON_SERVICE_TIMEOUT || '30000'),

    // Ollama defaults
    ollamaUrl: process.env.OLLAMA_URL || 'http://127.0.0.1:11434',
    ollamaModel: process.env.OLLAMA_EMBEDDING_MODEL || 'embeddinggemma',
    ollamaTimeout: parseInt(process.env.OLLAMA_TIMEOUT || '30000'),
  },

  // RSE algorithm parameters
  threshold: parseFloat(process.env.RSE_THRESHOLD || '0.1'),
  windowSize: parseInt(process.env.RSE_WINDOW_SIZE || '3'),
  manifoldDimension: parseInt(process.env.RSE_MANIFOLD_DIM || '3'),
  useManifoldMetrics: process.env.RSE_USE_MANIFOLD !== 'false',

  // Performance
  batchSize: parseInt(process.env.RSE_BATCH_SIZE || '10'),
  maxConcurrentRequests: parseInt(process.env.RSE_MAX_CONCURRENT || '5'),
};

/**
 * Embedding backend implementations
 */
export class EmbeddingBackend {
  private config: EmbeddingBackendConfig;

  constructor(config: EmbeddingBackendConfig) {
    this.config = config;
  }

  /**
   * Generate embedding for a single text
   */
  async generateEmbedding(text: string): Promise<Float64Array> {
    switch (this.config.type) {
      case 'python-service':
        return this.generatePythonServiceEmbedding(text);
      case 'ollama':
        return this.generateOllamaEmbedding(text);
      default:
        throw new Error(`Unknown embedding backend type: ${this.config.type}`);
    }
  }

  /**
   * Generate embeddings for multiple texts (batch)
   */
  async generateEmbeddings(texts: string[]): Promise<Float64Array[]> {
    switch (this.config.type) {
      case 'python-service':
        return this.generatePythonServiceEmbeddings(texts);
      case 'ollama':
        // Use native batch API endpoint /api/embed
        return this.generateOllamaEmbeddings(texts);
      default:
        throw new Error(`Unknown embedding backend type: ${this.config.type}`);
    }
  }

  /**
   * Python FastAPI embedding service
   */
  private async generatePythonServiceEmbedding(text: string): Promise<Float64Array> {
    const url = `${this.config.pythonServiceUrl}/embed`;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.config.pythonServiceTimeout);

      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`Python service error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return new Float64Array(data.embedding);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      throw new Error(`Python embedding service failed: ${errorMessage}`);
    }
  }

  /**
   * Python FastAPI batch embedding service
   */
  private async generatePythonServiceEmbeddings(texts: string[]): Promise<Float64Array[]> {
    const url = `${this.config.pythonServiceUrl}/embed/batch`;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.config.pythonServiceTimeout);

      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`Python service error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return data.map((item: { embedding: number[] }) => new Float64Array(item.embedding));

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      throw new Error(`Python batch embedding service failed: ${errorMessage}`);
    }
  }

  /**
   * Ollama embedding generation using NEW /api/embed endpoint
   * This endpoint supports batch embedding and new parameters
   */
  private async generateOllamaEmbedding(text: string): Promise<Float64Array> {
    const url = `${this.config.ollamaUrl}/api/embed`;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.config.ollamaTimeout);

      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: this.config.ollamaModel,
          input: text,  // NEW: 'input' instead of 'prompt'
          truncate: true,  // NEW: handle context length overflow
          keep_alive: '5m',  // NEW: model memory management
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`Ollama error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();

      // NEW: Response format is { embeddings: [[...]] } - array of arrays
      if (!data.embeddings || !Array.isArray(data.embeddings) || data.embeddings.length === 0) {
        throw new Error('Invalid embedding response from Ollama');
      }

      return new Float64Array(data.embeddings[0]);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      throw new Error(`Ollama embedding failed: ${errorMessage}`);
    }
  }

  /**
   * Ollama batch embedding generation using NEW /api/embed endpoint
   * This is the proper way to do batch embeddings with Ollama
   */
  private async generateOllamaEmbeddings(texts: string[]): Promise<Float64Array[]> {
    const url = `${this.config.ollamaUrl}/api/embed`;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.config.ollamaTimeout);

      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: this.config.ollamaModel,
          input: texts,  // Pass array directly for batch processing
          truncate: true,
          keep_alive: '5m',
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`Ollama error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();

      if (!data.embeddings || !Array.isArray(data.embeddings)) {
        throw new Error('Invalid batch embedding response from Ollama');
      }

      return data.embeddings.map((embedding: number[]) => new Float64Array(embedding));

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      throw new Error(`Ollama batch embedding failed: ${errorMessage}`);
    }
  }

  /**
   * Health check for the embedding backend
   */
  async healthCheck(): Promise<{ status: string; details: any }> {
    try {
      switch (this.config.type) {
        case 'python-service': {
          const url = `${this.config.pythonServiceUrl}/health`;
          const response = await fetch(url);
          const data = await response.json();
          return {
            status: response.ok ? 'healthy' : 'unhealthy',
            details: data,
          };
        }
        case 'ollama': {
          const url = `${this.config.ollamaUrl}/api/tags`;
          const response = await fetch(url);
          const data = await response.json();
          return {
            status: response.ok ? 'healthy' : 'unhealthy',
            details: { models: data.models, ollamaModel: this.config.ollamaModel },
          };
        }
        default:
          return {
            status: 'unknown',
            details: { error: 'Unknown backend type' },
          };
      }
    } catch (error) {
      return {
        status: 'unhealthy',
        details: { error: error instanceof Error ? error.message : String(error) },
      };
    }
  }
}
