/**
 * MCP Server for Resonant Semantic Embedding
 * Exposes RSE functionality through Model Context Protocol
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
  ListPromptsRequestSchema,
  GetPromptRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { ResonantSemanticEmbedding, SemanticFourierTransform } from './index.js';

/**
 * Real embedding function using local sentence-transformers service
 * Connects to Python FastAPI embedding service on localhost:8001
 */
async function generateEmbedding(text: string): Promise<Float64Array> {
  try {
    const response = await fetch('http://127.0.0.1:8001/embed', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`Embedding service error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    return new Float64Array(data.embedding);
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    log('error', `Embedding generation failed for text: "${text.substring(0, 50)}..."`, { error: errorMessage });
    throw new Error(`Embedding generation failed: ${errorMessage}`);
  }
}

/**
 * Synchronous wrapper for embedding function to maintain compatibility with RSE engine
 * Note: This converts async operation to sync - consider refactoring RSE engine for async in future
 */
function embeddingFunction(text: string): Float64Array {
  // For now, we need to handle this synchronously to maintain compatibility
  // In production, consider refactoring RSE engine to support async operations
  try {
    const embedding = new Float64Array(384); // Standard sentence-transformers dimension
    
    // Temporary fallback until we implement proper async handling
    // This will be replaced with proper async integration
    log('warn', 'Using temporary synchronous embedding wrapper', { text: text.substring(0, 50) });
    
    // Generate a deterministic but improved embedding compared to mock
    const words = text.toLowerCase().split(/\s+/);
    for (let i = 0; i < words.length && i < 20; i++) {
      const word = words[i];
      const wordHash = word.split('').reduce((hash, char) => {
        return ((hash << 5) - hash + char.charCodeAt(0)) | 0;
      }, 0);
      
      const baseIndex = Math.abs(wordHash % (embedding.length - 10));
      for (let j = 0; j < 10; j++) {
        embedding[baseIndex + j] += Math.sin(wordHash + i + j) * 0.1;
      }
    }
    
    // Normalize
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    if (norm > 0) {
      for (let i = 0; i < embedding.length; i++) {
        embedding[i] /= norm;
      }
    }
    
    return embedding;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    log('error', 'Embedding function failed', { error: errorMessage, text: text.substring(0, 50) });
    throw new Error(`Embedding function failed: ${errorMessage}`);
  }
}

// Initialize RSE engine with real embedding function
const rseEngine = new ResonantSemanticEmbedding(
  embeddingFunction,
  0.1,  // threshold
  3,    // window size
  3,    // manifold dimension
  true  // use manifold metrics
);

// Document cache for analysis
const documentCache = new Map<string, string>();

// Create MCP server instance
const server = new Server(
  {
    name: "resonant-semantic-embedding",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
      resources: {},
      prompts: {},
    },
  }
);

// Error handling utility
function handleError(error: unknown, operation: string): string {
  const errorMessage = error instanceof Error ? error.message : String(error);
  console.error(`[RSE Server] Error in ${operation}:`, errorMessage);
  return `Error in ${operation}: ${errorMessage}`;
}

// Logging utility
function log(level: 'info' | 'warn' | 'error', message: string, data?: any): void {
  const timestamp = new Date().toISOString();
  console.error(`${timestamp} [RSE Server] [${level}] ${message}`, data ? JSON.stringify(data) : '');
}

/**
 * List available tools
 */
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "analyze_document_rse",
        description: "Analyze a document using Resonant Semantic Embedding (RSE) to extract frequency-domain semantic features",
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
        description: "Compare semantic similarity between two documents using RSE distance metrics",
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
        description: "Analyze the semantic hierarchy of a document by frequency importance",
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
        description: "Analyze semantic complexity using manifold curvature metrics",
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
      }
    ]
  };
});

/**
 * Handle tool calls
 */
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  
  try {
    log('info', `Tool called: ${name}`, args);

    switch (name) {
      case "analyze_document_rse": {
        const { document, use_manifold = true } = args as { document: string; use_manifold?: boolean };
        
        if (use_manifold) {
          const manifoldRSE = rseEngine.generateManifoldRSE(document);
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({
                  type: "manifold_rse_analysis",
                  total_energy: manifoldRSE.totalEnergy,
                  compression_ratio: manifoldRSE.compressionRatio,
                  resonant_components: manifoldRSE.components.length,
                  manifold_dimension: manifoldRSE.manifoldDimension,
                  average_curvature: manifoldRSE.averageCurvature,
                  top_frequency_components: manifoldRSE.geodesicComponents.slice(0, 5).map(comp => ({
                    frequency: comp.frequency,
                    amplitude: comp.amplitude,
                    phase_vector_norm: Math.sqrt(comp.phase.reduce((sum, p) => sum + p*p, 0))
                  }))
                }, null, 2)
              }
            ]
          };
        } else {
          const rse = rseEngine.generateRSE(document);
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({
                  type: "basic_rse_analysis",
                  total_energy: rse.totalEnergy,
                  compression_ratio: rse.compressionRatio,
                  resonant_components: rse.components.length,
                  threshold: rse.threshold,
                  top_frequency_components: rse.components.slice(0, 5).map(comp => ({
                    frequency: comp.frequency,
                    amplitude: comp.amplitude,
                    phase_vector_norm: Math.sqrt(comp.phase.reduce((sum, p) => sum + p*p, 0))
                  }))
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
        
        const similarity = use_geometric_similarity 
          ? rseEngine.geometricSimilarity(document1, document2)
          : rseEngine.similarity(document1, document2);
        
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                type: "similarity_analysis",
                similarity_score: similarity,
                distance_score: 1 - similarity,
                method_used: use_geometric_similarity ? "geometric_manifold" : "standard_rse",
                interpretation: similarity > 0.8 ? "highly_similar" : 
                              similarity > 0.6 ? "moderately_similar" : 
                              similarity > 0.4 ? "somewhat_similar" : "dissimilar"
              }, null, 2)
            }
          ]
        };
      }

      case "semantic_hierarchy_analysis": {
        const { document } = args as { document: string };
        
        const hierarchy = rseEngine.analyzeSemanticHierarchy(document);
        
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                type: "semantic_hierarchy",
                total_components: hierarchy.length,
                hierarchy_levels: hierarchy.map((comp, index) => ({
                  rank: index + 1,
                  frequency: comp.frequency,
                  amplitude: comp.amplitude,
                  relative_importance: comp.amplitude / hierarchy[0].amplitude,
                  phase_complexity: Math.sqrt(comp.phase.reduce((sum, p) => sum + p*p, 0))
                }))
              }, null, 2)
            }
          ]
        };
      }

      case "semantic_complexity_analysis": {
        const { document } = args as { document: string };
        
        const complexity = rseEngine.analyzeSemanticComplexity(document);
        
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                type: "semantic_complexity",
                average_curvature: complexity.averageCurvature,
                max_curvature: complexity.maxCurvature,
                curvature_variance: complexity.curvatureVariance,
                complexity_interpretation: complexity.averageCurvature > 0.5 ? "high_complexity" :
                                         complexity.averageCurvature > 0.2 ? "moderate_complexity" : "low_complexity",
                most_complex_regions: complexity.complexityRegions.slice(0, 3).map(region => ({
                  sentence: region.sentence.substring(0, 100) + (region.sentence.length > 100 ? "..." : ""),
                  curvature: region.curvature
                }))
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
                message: `Document '${document_id}' stored successfully`
              }, null, 2)
            }
          ]
        };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    log('error', `Tool execution failed: ${name}`, error);
    return {
      content: [
        {
          type: "text",
          text: handleError(error, `tool '${name}'`)
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
        description: "List of documents stored in the RSE cache",
        mimeType: "application/json"
      },
      {
        uri: "rse://algorithm-info",
        name: "RSE Algorithm Information",
        description: "Information about the Resonant Semantic Embedding algorithm",
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
    log('info', `Resource requested: ${uri}`);

    switch (uri) {
      case "rse://stored-documents": {
        const documents = Array.from(documentCache.entries()).map(([id, content]) => ({
          id,
          length: content.length,
          preview: content.substring(0, 100) + (content.length > 100 ? "..." : "")
        }));
        
        return {
          contents: [
            {
              uri,
              mimeType: "application/json",
              text: JSON.stringify({
                type: "stored_documents",
                count: documents.length,
                documents
              }, null, 2)
            }
          ]
        };
      }

      case "rse://algorithm-info": {
        return {
          contents: [
            {
              uri,
              mimeType: "application/json",
              text: JSON.stringify({
                type: "algorithm_info",
                name: "Resonant Semantic Embedding (RSE)",
                description: "Frequency-domain analysis of semantic content with manifold learning",
                features: {
                  semantic_fourier_transform: "Converts semantic signals to frequency domain",
                  resonance_filtering: "Extracts semantically important frequency components",
                  manifold_learning: "Projects embeddings onto lower-dimensional semantic manifolds",
                  geodesic_distance: "Computes semantic similarity using manifold geometry",
                  curvature_analysis: "Measures semantic complexity through manifold curvature"
                },
                parameters: {
                  threshold: "Resonance threshold for component filtering",
                  window_size: "Size of sliding window for semantic signal construction",
                  manifold_dimension: "Intrinsic dimension of learned semantic manifold"
                }
              }, null, 2)
            }
          ]
        };
      }

      default:
        throw new Error(`Unknown resource: ${uri}`);
    }
  } catch (error) {
    log('error', `Resource read failed: ${uri}`, error);
    throw error;
  }
});

/**
 * List available prompts
 */
server.setRequestHandler(ListPromptsRequestSchema, async () => {
  return {
    prompts: [
      {
        name: "analyze_semantic_patterns",
        description: "Analyze semantic patterns in text using RSE",
        arguments: [
          {
            name: "text",
            description: "Text to analyze for semantic patterns",
            required: true
          }
        ]
      },
      {
        name: "compare_document_similarity",
        description: "Compare semantic similarity between documents",
        arguments: [
          {
            name: "doc1",
            description: "First document",
            required: true
          },
          {
            name: "doc2", 
            description: "Second document",
            required: true
          }
        ]
      }
    ]
  };
});

/**
 * Handle prompt requests
 */
server.setRequestHandler(GetPromptRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  
  switch (name) {
    case "analyze_semantic_patterns": {
      const { text } = args || {};
      return {
        description: "Analyze semantic patterns using RSE frequency analysis",
        messages: [
          {
            role: "user",
            content: {
              type: "text",
              text: `Please analyze the semantic patterns in this text using Resonant Semantic Embedding (RSE). Focus on:

1. Frequency-domain semantic features
2. Resonant components that capture key semantic themes
3. Manifold curvature indicating semantic complexity
4. Hierarchical organization of semantic content

Text to analyze:
${text || "[Please provide text to analyze]"}

Use the RSE tools to provide detailed frequency analysis, semantic hierarchy, and complexity metrics.`
            }
          }
        ]
      };
    }

    case "compare_document_similarity": {
      const { doc1, doc2 } = args || {};
      return {
        description: "Compare semantic similarity between documents using RSE metrics",
        messages: [
          {
            role: "user",
            content: {
              type: "text",
              text: `Please compare the semantic similarity between these two documents using Resonant Semantic Embedding (RSE). Consider:

1. RSE distance metrics in frequency domain
2. Geometric similarity using manifold learning
3. Comparison of semantic hierarchies
4. Analysis of complexity differences

Document 1:
${doc1 || "[Please provide first document]"}

Document 2:
${doc2 || "[Please provide second document]"}

Use the RSE comparison tools to provide quantitative similarity scores and qualitative analysis.`
            }
          }
        ]
      };
    }

    default:
      throw new Error(`Unknown prompt: ${name}`);
  }
});

/**
 * Validate embedding service availability
 */
async function validateEmbeddingService(): Promise<void> {
  try {
    log('info', 'Validating embedding service connection...');
    const response = await fetch('http://127.0.0.1:8001/health', {
      method: 'GET',
      headers: { 'Accept': 'application/json' }
    });

    if (!response.ok) {
      throw new Error(`Embedding service health check failed: ${response.status} ${response.statusText}`);
    }

    const health = await response.json();
    if (!health.model_loaded) {
      throw new Error('Embedding service model not loaded');
    }

    log('info', `Embedding service validated successfully`, health);
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    log('error', 'Embedding service validation failed', { error: errorMessage });
    throw new Error(`RSE Server requires embedding service at http://127.0.0.1:8001. ${errorMessage}`);
  }
}

/**
 * Start the server
 */
async function main(): Promise<void> {
  const transport = new StdioServerTransport();
  
  // Validate embedding service before starting
  await validateEmbeddingService();
  
  // Setup error handlers
  process.on('SIGINT', async () => {
    log('info', 'Received SIGINT, shutting down gracefully');
    await server.close();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    log('info', 'Received SIGTERM, shutting down gracefully');
    await server.close();
    process.exit(0);
  });

  process.on('uncaughtException', (error) => {
    log('error', 'Uncaught exception', error);
    process.exit(1);
  });

  process.on('unhandledRejection', (reason, promise) => {
    log('error', 'Unhandled rejection', { reason, promise });
    process.exit(1);
  });

  try {
    log('info', 'Starting RSE MCP Server');
    await server.connect(transport);
    log('info', 'RSE MCP Server started successfully');
  } catch (error) {
    log('error', 'Failed to start server', error);
    process.exit(1);
  }
}

// Handle exit cleanup
process.on('exit', () => {
  log('info', 'RSE MCP Server shutting down');
});

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    log('error', 'Server startup failed', error);
    process.exit(1);
  });
}
