/**
 * MCP Server for Resonant Semantic Embedding
 * Exposes RSE functionality through Model Context Protocol
 * 
 * ASYNC REFACTOR: Removed mock embeddings, using real embedding backends
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { ResonantSemanticEmbedding } from './index.js';
import { DEFAULT_RSE_CONFIG, EmbeddingBackend, RSEConfig } from './config.js';

/**
 * Initialize embedding backend and RSE engine
 */
const config: RSEConfig = DEFAULT_RSE_CONFIG;
const embeddingBackend = new EmbeddingBackend(config.embeddingBackend);

// Create async embedding function for RSE
const embeddingFunction = async (text: string): Promise<Float64Array> => {
  return embeddingBackend.generateEmbedding(text);
};

// Initialize RSE engine with REAL embedding function
const rseEngine = new ResonantSemanticEmbedding(
  embeddingFunction,
  config.threshold,
  config.windowSize,
  config.manifoldDimension,
  config.useManifoldMetrics
);

// Document cache for analysis
const documentCache = new Map<string, string>();

// Create MCP server instance
const server = new Server(
  {
    name: "resonant-semantic-embedding",
    version: "2.0.0",  // Version bump for async refactor
  },
  {
    capabilities: {
      tools: {},
      resources: {},
    },
  }
);

// Logging utility
function log(level: 'info' | 'warn' | 'error', message: string, data?: any): void {
  const timestamp = new Date().toISOString();
  console.error(`${timestamp} [RSE Server] [${level}] ${message}`, data ? JSON.stringify(data) : '');
}

// Error handling utility
function handleError(error: unknown, operation: string): string {
  const errorMessage = error instanceof Error ? error.message : String(error);
  log('error', `Error in ${operation}`, { error: errorMessage });
  return `Error in ${operation}: ${errorMessage}`;
}

/**
 * List available tools
 */
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "analyze_document_rse",
        description: "Analyze a document using Resonant Semantic Embedding (RSE) to extract frequency-domain semantic features with REAL embeddings",
        inputSchema: {
          type: "object",
          properties: {
            document: {
              type: "string",
              description: "The text document to analyze using RSE"
            },
            use_manifold: {
              type: "boolean",
              description: "Whether to use manifold learning for enhanced semantic analysis",
              default: true
            }
          },
          required: ["document"]
        }
      },
      {
        name: "compare_documents_rse",
        description: "Compare semantic similarity between two documents using RSE distance metrics with REAL embeddings",
        inputSchema: {
          type: "object",
          properties: {
            document1: {
              type: "string",
              description: "First document for comparison"
            },
            document2: {
              type: "string",
              description: "Second document for comparison"
            },
            use_geometric_similarity: {
              type: "boolean",
              description: "Whether to use geometric (manifold-based) similarity",
              default: true
            }
          },
          required: ["document1", "document2"]
        }
      },
      {
        name: "semantic_hierarchy_analysis",
        description: "Analyze the semantic hierarchy of a document by frequency importance using REAL embeddings",
        inputSchema: {
          type: "object",
          properties: {
            document: {
              type: "string",
              description: "Document to analyze for semantic hierarchy"
            }
          },
          required: ["document"]
        }
      },
      {
        name: "semantic_complexity_analysis",
        description: "Analyze semantic complexity using manifold curvature metrics with REAL embeddings",
        inputSchema: {
          type: "object",
          properties: {
            document: {
              type: "string",
              description: "Document to analyze for semantic complexity"
            }
          },
          required: ["document"]
        }
      },
      {
        name: "store_document",
        description: "Store a document for later analysis and comparison",
        inputSchema: {
          type: "object",
          properties: {
            document_id: {
              type: "string",
              description: "Unique identifier for the document"
            },
            document: {
              type: "string",
              description: "Document content to store"
            }
          },
          required: ["document_id", "document"]
        }
      },
      {
        name: "embedding_backend_health",
        description: "Check the health status of the embedding backend (Python service or Ollama)",
        inputSchema: {
          type: "object",
          properties: {},
          required: []
        }
      }
    ]
  };
});

/**
 * Handle tool calls
 * ASYNC REFACTOR: All handlers now properly await async operations
 */
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  
  try {
    log('info', `Tool called: ${name}`, { args });

    switch (name) {
      case "analyze_document_rse": {
        const { document, use_manifold = true } = args as { document: string; use_manifold?: boolean };
        
        if (use_manifold) {
          // ASYNC: Now properly awaits
          const manifoldRSE = await rseEngine.generateManifoldRSE(document);
          
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({
                  type: "manifold_rse_analysis",
                  embedding_backend: config.embeddingBackend.type,
                  total_energy: manifoldRSE.totalEnergy,
                  compression_ratio: manifoldRSE.compressionRatio,
                  resonant_components: manifoldRSE.components.length,
                  manifold_dimension: manifoldRSE.manifoldDimension,
                  average_curvature: manifoldRSE.averageCurvature,
                  semantic_complexity_indicator: manifoldRSE.averageCurvature > 0.5 ? "high" : "moderate",
                  top_frequency_components: manifoldRSE.geodesicComponents.slice(0, 5).map(comp => ({
                    frequency: comp.frequency,
                    amplitude: comp.amplitude,
                    phase_vector_norm: Math.sqrt(comp.phase.reduce((sum, p) => sum + p*p, 0))
                  })),
                  interpretation: {
                    energy: `Total semantic energy: ${manifoldRSE.totalEnergy.toFixed(3)}`,
                    compression: `Retained ${(manifoldRSE.compressionRatio * 100).toFixed(1)}% of frequency components`,
                    complexity: `Average manifold curvature: ${manifoldRSE.averageCurvature.toFixed(4)} (higher = more complex semantics)`,
                    dimensions: `Semantic manifold has ${manifoldRSE.manifoldDimension} intrinsic dimensions`
                  }
                }, null, 2)
              }
            ]
          };
        } else {
          // ASYNC: Now properly awaits
          const rse = await rseEngine.generateRSE(document);
          
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({
                  type: "basic_rse_analysis",
                  embedding_backend: config.embeddingBackend.type,
                  total_energy: rse.totalEnergy,
                  compression_ratio: rse.compressionRatio,
                  resonant_components: rse.components.length,
                  threshold: rse.threshold,
                  top_frequency_components: rse.components.slice(0, 5).map(comp => ({
                    frequency: comp.frequency,
                    amplitude: comp.amplitude,
                    phase_vector_norm: Math.sqrt(comp.phase.reduce((sum, p) => sum + p*p, 0))
                  })),
                  interpretation: {
                    energy: `Total semantic energy: ${rse.totalEnergy.toFixed(3)}`,
                    compression: `Retained ${(rse.compressionRatio * 100).toFixed(1)}% of frequency components`,
                    components: `${rse.components.length} resonant frequency components identified`
                  }
                }, null, 2)
              }
            ]
          };
        }
      }

      case "compare_documents_rse": {
        const { document1, document2, use_geometric_similarity = true } = args as { 
          document1: string; 
          document2: string; 
          use_geometric_similarity?: boolean 
        };
        
        // ASYNC: Now properly awaits
        const similarity = use_geometric_similarity 
          ? await rseEngine.geometricSimilarity(document1, document2)
          : await rseEngine.similarity(document1, document2);
        
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                type: "rse_similarity_comparison",
                embedding_backend: config.embeddingBackend.type,
                method: use_geometric_similarity ? "geometric (manifold-based)" : "standard RSE distance",
                similarity_score: similarity,
                distance_score: -Math.log(similarity),
                interpretation: similarity > 0.8 
                  ? "Very similar semantic content"
                  : similarity > 0.6
                  ? "Moderately similar semantic content"
                  : similarity > 0.4
                  ? "Somewhat similar semantic content"
                  : "Dissimilar semantic content",
                details: {
                  score_explanation: "Score of 1.0 = identical, 0.0 = completely different",
                  method_note: use_geometric_similarity 
                    ? "Geometric similarity uses manifold geodesic distances for enhanced semantic understanding"
                    : "Standard similarity uses frequency-domain RSE distance metric"
                }
              }, null, 2)
            }
          ]
        };
      }

      case "semantic_hierarchy_analysis": {
        const { document } = args as { document: string };
        
        // ASYNC: Now properly awaits
        const hierarchy = await rseEngine.analyzeSemanticHierarchy(document);
        
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                type: "semantic_hierarchy",
                embedding_backend: config.embeddingBackend.type,
                total_components: hierarchy.length,
                frequency_components: hierarchy.map((comp, index) => ({
                  rank: index + 1,
                  frequency: comp.frequency,
                  amplitude: comp.amplitude,
                  importance: index < 3 ? "high" : index < 6 ? "medium" : "low",
                  phase_complexity: Math.sqrt(comp.phase.reduce((sum, p) => sum + p*p, 0))
                })),
                interpretation: {
                  top_3: "The top 3 frequency components represent the most semantically important patterns",
                  amplitude_meaning: "Higher amplitude indicates stronger semantic resonance at that frequency",
                  phase_meaning: "Phase complexity measures the directional distribution of semantic content"
                }
              }, null, 2)
            }
          ]
        };
      }

      case "semantic_complexity_analysis": {
        const { document } = args as { document: string };
        
        // ASYNC: Now properly awaits
        const complexity = await rseEngine.analyzeSemanticComplexity(document);
        
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                type: "semantic_complexity",
                embedding_backend: config.embeddingBackend.type,
                average_curvature: complexity.averageCurvature,
                max_curvature: complexity.maxCurvature,
                curvature_variance: complexity.curvatureVariance,
                complexity_level: complexity.averageCurvature > 0.5 
                  ? "high" 
                  : complexity.averageCurvature > 0.2 
                  ? "moderate" 
                  : "low",
                most_complex_regions: complexity.complexityRegions.slice(0, 5).map((region, index) => ({
                  rank: index + 1,
                  sentence: region.sentence.substring(0, 100) + (region.sentence.length > 100 ? "..." : ""),
                  curvature: region.curvature,
                  complexity: region.curvature > 0.5 ? "high" : region.curvature > 0.2 ? "moderate" : "low"
                })),
                interpretation: {
                  curvature_meaning: "Manifold curvature measures semantic density - higher values indicate more complex, interconnected concepts",
                  variance_meaning: `Curvature variance of ${complexity.curvatureVariance.toFixed(4)} indicates ${complexity.curvatureVariance > 0.1 ? "highly variable" : "relatively uniform"} semantic complexity across the document`,
                  application: "High curvature regions may benefit from more detailed analysis or entity resolution"
                }
              }, null, 2)
            }
          ]
        };
      }

      case "store_document": {
        const { document_id, document } = args as { document_id: string; document: string };
        
        documentCache.set(document_id, document);
        
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                type: "document_stored",
                document_id,
                document_length: document.length,
                cache_size: documentCache.size,
                message: `Document "${document_id}" stored successfully`
              }, null, 2)
            }
          ]
        };
      }

      case "embedding_backend_health": {
        // Check backend health
        const health = await embeddingBackend.healthCheck();
        
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                type: "embedding_backend_health",
                backend_type: config.embeddingBackend.type,
                status: health.status,
                details: health.details,
                configuration: {
                  type: config.embeddingBackend.type,
                  python_service_url: config.embeddingBackend.pythonServiceUrl,
                  ollama_url: config.embeddingBackend.ollamaUrl,
                  ollama_model: config.embeddingBackend.ollamaModel
                }
              }, null, 2)
            }
          ]
        };
      }

      default:
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ error: `Unknown tool: ${name}` }, null, 2)
            }
          ],
          isError: true
        };
    }
  } catch (error) {
    const errorMessage = handleError(error, name);
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({ 
            error: errorMessage,
            tool: name,
            embedding_backend: config.embeddingBackend.type
          }, null, 2)
        }
      ],
      isError: true
    };
  }
});

/**
 * List available resources
 */
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  return {
    resources: [
      {
        uri: "rse://stored-documents",
        name: "Stored Documents",
        description: "Cache of documents stored for analysis",
        mimeType: "application/json"
      },
      {
        uri: "rse://algorithm-info",
        name: "RSE Algorithm Information",
        description: "Details about the Resonant Semantic Embedding algorithm",
        mimeType: "application/json"
      },
      {
        uri: "rse://config",
        name: "RSE Configuration",
        description: "Current RSE and embedding backend configuration",
        mimeType: "application/json"
      }
    ]
  };
});

/**
 * Handle resource reads
 */
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const { uri } = request.params;
  
  try {
    switch (uri) {
      case "rse://stored-documents":
        const documents = Array.from(documentCache.entries()).map(([id, doc]) => ({
          id,
          length: doc.length,
          preview: doc.substring(0, 100) + (doc.length > 100 ? "..." : "")
        }));
        
        return {
          contents: [
            {
              uri,
              mimeType: "application/json",
              text: JSON.stringify({ documents, total: documentCache.size }, null, 2)
            }
          ]
        };
        
      case "rse://algorithm-info":
        return {
          contents: [
            {
              uri,
              mimeType: "application/json",
              text: JSON.stringify({
                name: "Resonant Semantic Embedding (RSE)",
                version: "2.0.0",
                description: "Frequency-domain semantic analysis using manifold learning and production embeddings",
                features: [
                  "Semantic Fourier Transform for frequency decomposition",
                  "Manifold learning with geodesic distance metrics",
                  "Curvature-based semantic complexity analysis",
                  "Real embedding generation via Python service or Ollama"
                ],
                parameters: {
                  threshold: config.threshold,
                  window_size: config.windowSize,
                  manifold_dimension: config.manifoldDimension,
                  use_manifold_metrics: config.useManifoldMetrics
                },
                mathematical_foundations: {
                  fourier_transform: "Decomposes semantic signals into frequency components",
                  manifold_learning: "Projects embeddings onto lower-dimensional semantic manifolds",
                  curvature_analysis: "Measures local semantic complexity via manifold geometry",
                  geodesic_distance: "Computes similarity using curved manifold paths"
                }
              }, null, 2)
            }
          ]
        };
        
      case "rse://config":
        return {
          contents: [
            {
              uri,
              mimeType: "application/json",
              text: JSON.stringify({
                embedding_backend: {
                  type: config.embeddingBackend.type,
                  python_service_url: config.embeddingBackend.pythonServiceUrl,
                  python_service_timeout: config.embeddingBackend.pythonServiceTimeout,
                  ollama_url: config.embeddingBackend.ollamaUrl,
                  ollama_model: config.embeddingBackend.ollamaModel,
                  ollama_timeout: config.embeddingBackend.ollamaTimeout
                },
                rse_parameters: {
                  threshold: config.threshold,
                  window_size: config.windowSize,
                  manifold_dimension: config.manifoldDimension,
                  use_manifold_metrics: config.useManifoldMetrics
                },
                performance: {
                  batch_size: config.batchSize,
                  max_concurrent_requests: config.maxConcurrentRequests
                },
                environment_variables: {
                  EMBEDDING_BACKEND: "python-service or ollama",
                  PYTHON_SERVICE_URL: "http://127.0.0.1:8001",
                  OLLAMA_URL: "http://127.0.0.1:11434",
                  OLLAMA_EMBEDDING_MODEL: "nomic-embed-text or mxbai-embed-large",
                  RSE_THRESHOLD: "0.1",
                  RSE_WINDOW_SIZE: "3",
                  RSE_MANIFOLD_DIM: "3"
                }
              }, null, 2)
            }
          ]
        };
        
      default:
        return {
          contents: [
            {
              uri,
              mimeType: "text/plain",
              text: `Unknown resource: ${uri}`
            }
          ]
        };
    }
  } catch (error) {
    const errorMessage = handleError(error, `read resource ${uri}`);
    return {
      contents: [
        {
          uri,
          mimeType: "text/plain",
          text: errorMessage
        }
      ]
    };
  }
});

/**
 * Start the server
 */
async function main() {
  log('info', 'Starting Resonant Semantic Embedding MCP Server v2.0.0');
  log('info', 'Configuration', {
    embedding_backend: config.embeddingBackend.type,
    threshold: config.threshold,
    window_size: config.windowSize,
    manifold_dimension: config.manifoldDimension
  });
  
  // Health check on startup
  try {
    const health = await embeddingBackend.healthCheck();
    log('info', 'Embedding backend health check', { status: health.status, details: health.details });
    
    if (health.status !== 'healthy') {
      log('warn', 'Embedding backend is not healthy - tools may fail until backend is available');
    }
  } catch (error) {
    log('error', 'Failed to perform initial health check', { error: String(error) });
  }
  
  const transport = new StdioServerTransport();
  await server.connect(transport);
  log('info', 'RSE MCP Server connected and ready');
}

main().catch((error) => {
  log('error', 'Fatal error in main', { error: String(error) });
  process.exit(1);
});
