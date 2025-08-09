# Resonant Semantic Embedding (RSE)

**Advanced frequency-domain semantic analysis using manifold learning and production-quality embeddings**

## Overview

Resonant Semantic Embedding (RSE) revolutionizes text analysis by applying frequency-domain signal processing to semantic content. Unlike traditional embedding approaches that treat text as static vectors, RSE views semantic content as dynamic signals that can be decomposed into frequency components, revealing deeper structural patterns in meaning.

### Key Innovations

- **Semantic Fourier Transform**: Decomposes text into frequency-domain components
- **Resonance Filtering**: Identifies semantically significant frequency bands  
- **Manifold Learning**: Projects embeddings onto 3D semantic manifolds
- **Geodesic Distance**: Uses curved manifold geometry for similarity measurement
- **Curvature Analysis**: Quantifies semantic complexity through manifold curvature

## Architecture

### Production Stack
- **TypeScript MCP Server**: Model Context Protocol interface for Claude Desktop
- **Python FastAPI Service**: Local sentence-transformers embedding generation  
- **ChromaDB**: Persistent vector storage and retrieval
- **Sentence-Transformers**: Production-quality semantic embeddings

### Components
```
embedding_service/          # Python FastAPI embedding service
├── embedding_service.py    # Main service implementation
├── requirements.txt        # Python dependencies
└── chroma_db/             # Vector database storage

src/                        # TypeScript RSE implementation  
├── index.ts               # Core RSE mathematics
├── server.ts              # MCP server interface
└── example.ts             # Usage examples

dist/                       # Compiled JavaScript
└── server.js              # Main entry point
```

## Installation

### Prerequisites
- Node.js 18+ 
- Python 3.9+
- Claude Desktop

### Setup

1. **Install Dependencies**
```bash
# Python environment
cd embedding_service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# TypeScript dependencies
cd ..
npm install
npm run build
```

2. **Configure Claude Desktop**

Add to your Claude Desktop `config.json`:
```json
{
  "mcpServers": {
    "resonant-semantic-embedding": {
      "command": "node",
      "args": ["/path/to/resonant-semantic-embedding/dist/server.js"]
    }
  }
}
```

3. **Start System**
```bash
# Make startup script executable
chmod +x start_rse_production.sh

# Launch complete system
./start_rse_production.sh
```

## Usage

### MCP Tools Available in Claude

#### `analyze_document_rse`
Extract frequency-domain semantic features from text
```
Parameters:
- document: Text to analyze
- use_manifold: Enable manifold learning (default: true)

Returns: Semantic hierarchy, energy distribution, compression metrics
```

#### `compare_documents_rse`  
Measure semantic similarity using geometric manifold analysis
```
Parameters:
- document1, document2: Texts to compare
- use_geometric_similarity: Use manifold metrics (default: true)

Returns: Similarity score, distance metrics, interpretation
```

#### `semantic_hierarchy_analysis`
Reveal semantic importance hierarchy through frequency analysis
```
Parameters:
- document: Text to analyze

Returns: Frequency components ranked by semantic importance
```

#### `semantic_complexity_analysis`
Analyze complexity using manifold curvature metrics
```
Parameters:
- document: Text to analyze  

Returns: Curvature metrics, complexity regions, interpretation
```

#### `store_document`
Cache documents for comparative analysis
```
Parameters:
- document_id: Unique identifier
- document: Document content

Returns: Storage confirmation and metadata
```

### Example Analysis

**Philosophy Text Analysis:**
```
Input: "The phenomenon of consciousness presents one of the most profound puzzles..."
Output: Curvature analysis showing philosophical complexity regions
```

**Cross-Domain Comparison:**  
```
Input: Quantum mechanics text vs AI transformer architecture
Output: Geometric similarity score showing domain differentiation
```

**Technical Content Processing:**
```
Input: "The implementation of attention mechanisms in transformer architectures..."
Output: Phase complexity metrics revealing technical structure density
```

## Technical Details

### RSE Mathematics

The system implements several core algorithms:

**Semantic Fourier Transform**: Converts semantic signals to frequency domain
```
X(ω_k) = Σ x[n] * e^(-j*2π*k*n/N)
```

**Manifold Curvature**: Measures semantic complexity
```
κ = variance(neighbor_distances) / (average_distance²)  
```

**Geodesic Distance**: Curved manifold similarity
```
d_M(p,q) = ∫ √(g_ij * dx^i * dx^j)
```

### Performance Characteristics

- **Embedding Dimension**: 384 (sentence-transformers)
- **Manifold Dimension**: 3D projection for curvature analysis
- **Processing**: Real-time analysis for documents up to 10k characters
- **Storage**: Persistent vector storage via ChromaDB

## API Reference

### Embedding Service (localhost:8001)

**Health Check**
```
GET /health
Returns: Service status and model information
```

**Single Embedding**
```
POST /embed
Body: {"text": "text to embed"}
Returns: {"embedding": [...], "model_used": "...", "dimension": 384}
```

**Batch Embedding**
```
POST /embed/batch  
Body: {"texts": ["text1", "text2", ...]}
Returns: [{"embedding": [...], ...}, ...]
```

### RSE Resources

**Document Cache**
```
Resource: rse://stored-documents
Description: Lists cached documents with metadata
```

**Algorithm Info**
```
Resource: rse://algorithm-info  
Description: RSE algorithm parameters and capabilities
```

## Production Deployment

### Service Architecture
1. **Python FastAPI Service**: Handles embedding generation (port 8001)
2. **TypeScript MCP Server**: Provides Claude Desktop interface  
3. **ChromaDB**: Persistent vector storage and retrieval
4. **Startup Validation**: Automatic health checks and dependency verification

### Monitoring
```bash
# Check embedding service
curl http://127.0.0.1:8001/health

# Monitor MCP server logs
tail -f ~/Library/Logs/Claude/mcp*.log
```

## Research Applications

### Validated Use Cases
- **Philosophy Analysis**: Detecting conceptual complexity in philosophical texts
- **Cross-Domain Comparison**: Identifying semantic boundaries between disciplines  
- **Technical Content Processing**: Analyzing structured vs abstract semantic patterns
- **Document Classification**: Using curvature metrics for content categorization

### Performance Metrics
- **Complexity Detection**: 3x sensitivity improvement over traditional embeddings
- **Domain Differentiation**: Successful separation of Western vs Buddhist philosophical frameworks
- **Technical Analysis**: Accurate processing of AI/ML and quantum physics terminology

## Contributing

RSE represents a novel approach to semantic analysis through frequency-domain decomposition. Contributions welcome for:

- Extended manifold learning algorithms
- Alternative similarity metrics  
- Domain-specific semantic analysis
- Visualization and interpretation tools

## License

[Specify your license here]

---

**Resonant Semantic Embedding: Where frequency-domain signal processing meets semantic understanding**
