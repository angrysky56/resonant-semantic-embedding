# AI Guidance: Resonant Semantic Embedding (RSE) MCP Server

## 🧠 Understanding RSE Technology

### Core Concept
Resonant Semantic Embedding (RSE) transforms text analysis by applying frequency-domain analysis to semantic content. Instead of treating text as static embeddings, RSE views semantic content as signals that can be decomposed into frequency components, revealing deeper structural patterns in meaning.

### Key Technical Innovations

1. **Semantic Fourier Transform**: Converts semantic signals (sequences of embeddings) into frequency domain representations
2. **Resonance Filtering**: Identifies the most semantically significant frequency components using energy thresholds
3. **Manifold Learning**: Projects high-dimensional embeddings onto lower-dimensional semantic manifolds
4. **Geodesic Distance**: Computes semantic similarity using curved manifold geometry rather than flat Euclidean space
5. **Curvature Analysis**: Measures semantic complexity through manifold curvature properties

## 🛠 Available MCP Tools

### Primary Analysis Tools

#### `analyze_document_rse`
**Purpose**: Extract frequency-domain semantic features from text
**When to Use**: 
- Initial analysis of document structure
- Understanding semantic hierarchy
- Identifying dominant themes
- Comparing semantic complexity across documents

**Parameters**:
- `document` (required): The text to analyze
- `use_manifold` (optional, default: true): Enable manifold learning for enhanced analysis

**Output Interpretation**:
- `total_energy`: Overall semantic richness of the document
- `compression_ratio`: Efficiency of semantic representation (lower = more compressed)
- `resonant_components`: Number of significant semantic frequencies
- `manifold_dimension`: Intrinsic dimensionality of semantic space
- `average_curvature`: Semantic complexity measure (higher = more complex)

#### `compare_documents_rse`
**Purpose**: Measure semantic similarity using RSE distance metrics
**When to Use**:
- Document clustering and classification
- Plagiarism detection
- Content recommendation
- Semantic search and retrieval

**Parameters**:
- `document1` (required): First document for comparison
- `document2` (required): Second document for comparison  
- `use_geometric_similarity` (optional, default: true): Use manifold-based similarity

**Output Interpretation**:
- `similarity_score`: 0-1 scale (higher = more similar)
- `distance_score`: Inverse of similarity (lower = more similar)
- `method_used`: "geometric_manifold" or "standard_rse"
- `interpretation`: Categorical similarity assessment

#### `semantic_hierarchy_analysis`
**Purpose**: Reveal the semantic importance hierarchy through frequency analysis
**When to Use**:
- Understanding document structure
- Extracting key themes in order of importance
- Content summarization guidance
- Topic modeling enhancement

**Output Interpretation**:
- Components ranked by semantic importance (amplitude)
- `relative_importance`: Comparison to most dominant component
- `phase_complexity`: Measure of semantic nuance

#### `semantic_complexity_analysis`
**Purpose**: Analyze semantic complexity using manifold curvature
**When to Use**:
- Assessing text difficulty
- Identifying semantically dense regions
- Content complexity scoring
- Educational material grading

**Output Interpretation**:
- `average_curvature`: Overall semantic complexity
- `curvature_variance`: Consistency of complexity across text
- `most_complex_regions`: Sentences with highest semantic density

### Utility Tools

#### `store_document`
**Purpose**: Cache documents for comparative analysis
**When to Use**:
- Building document collections for analysis
- Preparing for batch comparisons
- Creating semantic knowledge bases

## 📊 Available Resources

### `rse://stored-documents`
Lists all cached documents with metadata and previews

### `rse://algorithm-info`
Provides comprehensive information about the RSE algorithm and its parameters

## 💡 AI Guidance Prompts

### `analyze_semantic_patterns`
**Purpose**: Comprehensive semantic pattern analysis workflow
**Use Case**: When you need to understand the deep semantic structure of text

**Workflow**:
1. Extract frequency-domain features
2. Identify resonant semantic components
3. Analyze manifold curvature for complexity
4. Interpret semantic hierarchy

### `compare_document_similarity`
**Purpose**: Guided document comparison workflow
**Use Case**: When you need to quantify and understand semantic relationships between texts

**Workflow**:
1. Generate RSE representations for both documents
2. Compute geometric similarity using manifold metrics
3. Analyze semantic hierarchies for comparison
4. Provide interpretive context for similarity scores

## 🎯 Best Practices

### When to Use RSE vs Traditional Methods

**Use RSE When**:
- You need to understand semantic structure, not just content
- Working with complex, nuanced texts
- Comparing documents with different surface features but similar meaning
- Analyzing semantic complexity and richness
- Building semantic hierarchies

**Use Traditional Methods When**:
- Simple keyword matching is sufficient
- Working with short, simple texts
- Performance is critical and sophistication isn't needed
- You need exact string matching

### Optimization Tips

1. **Manifold Learning**: Enable for complex texts, disable for simple analysis
2. **Document Caching**: Store frequently compared documents using `store_document`
3. **Batch Analysis**: Process multiple documents in sequence for efficiency
4. **Threshold Tuning**: The default 0.1 threshold works well; lower values capture more detail

### Interpreting Results

#### Similarity Scores
- `> 0.8`: Highly similar (potential duplicates or close variants)
- `0.6-0.8`: Moderately similar (related topics, similar themes)
- `0.4-0.6`: Somewhat similar (some shared concepts)
- `< 0.4`: Dissimilar (different topics or approaches)

#### Complexity Scores
- `> 0.5`: High complexity (technical, nuanced, multi-layered)
- `0.2-0.5`: Moderate complexity (standard written content)
- `< 0.2`: Low complexity (simple, straightforward text)

#### Compression Ratios
- `< 0.3`: Highly compressed (clear semantic structure)
- `0.3-0.7`: Moderately compressed (typical content)
- `> 0.7`: Poorly compressed (semantic noise, low structure)

## 🔬 Advanced Use Cases

### Academic Research
- Analyze semantic evolution across document versions
- Measure conceptual similarity between research papers
- Identify semantic gaps in literature reviews

### Content Management
- Automatically categorize documents by semantic complexity
- Detect duplicate content with different surface expressions
- Build semantic recommendation systems

### Educational Applications
- Grade text complexity for appropriate reading levels
- Identify key concepts through frequency analysis
- Measure semantic comprehension in student writing

### Creative Writing
- Analyze narrative complexity and structure
- Compare writing styles across authors
- Optimize semantic flow and coherence

## 🚨 Common Pitfalls

1. **Over-interpretation**: RSE provides semantic structure analysis, not semantic meaning interpretation
2. **Scale Sensitivity**: Very short texts may not have sufficient semantic structure for analysis
3. **Domain Specificity**: Results are most meaningful within similar document domains
4. **Embedding Quality**: RSE quality depends on the underlying embedding function quality

## 🔧 Technical Notes

- Default embedding function is a mock implementation for demonstration
- In production, replace with quality embeddings (OpenAI, Sentence-BERT, etc.)
- Manifold dimension of 3 is optimal for most text analysis tasks
- Window size of 3 sentences balances local and global semantic context

## 📈 Performance Considerations

- Manifold learning adds computational overhead but provides richer analysis
- Document caching improves performance for repeated comparisons
- Batch processing multiple documents is more efficient than individual calls
- Consider disabling manifold learning for real-time applications requiring speed
