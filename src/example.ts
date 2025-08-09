import { ResonantSemanticEmbedding } from './index.js';

/**
 * Example usage of enhanced RSE with manifold learning capabilities
 * Demonstrates both classical RSE and manifold-enhanced geometric similarity
 */

// Mock embedding function for demonstration
// In practice, you'd use a real embedding model like OpenAI, Sentence-BERT, etc.
function mockEmbeddingFunction(text: string): Float64Array {
  // Simple hash-based embedding for demonstration
  const words = text.toLowerCase().split(/\s+/);
  const embedding = new Float64Array(128);
  
  for (let i = 0; i < words.length; i++) {
    const word = words[i];
    for (let j = 0; j < word.length; j++) {
      const charCode = word.charCodeAt(j);
      embedding[charCode % 128] += Math.sin(i + j + charCode) * 0.1;
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
}

// Sample documents for analysis
const documents: Record<string, string> = {
  physics: `
    Quantum mechanics describes the behavior of matter and energy at the atomic scale.
    The wave-particle duality reveals that particles exhibit both wave and particle properties.
    Heisenberg's uncertainty principle limits our ability to simultaneously measure position and momentum.
    Schrödinger's equation governs the evolution of quantum systems over time.
    Quantum entanglement creates correlations between particles that cannot be explained classically.
  `,
  
  biology: `
    Cellular biology studies the fundamental units of life and their processes.
    DNA contains the genetic instructions for all living organisms.
    Protein synthesis involves transcription and translation of genetic information.
    Mitochondria serve as the powerhouses of eukaryotic cells.
    Evolution through natural selection shapes the diversity of life on Earth.
  `,
  
  economics: `
    Market dynamics reflect the interaction between supply and demand forces.
    Economic equilibrium occurs when market supply equals market demand.
    Inflation represents the general increase in prices over time.
    Monetary policy tools help central banks control economic stability.
    Comparative advantage explains patterns of international trade.
  `,
  
  philosophy: `
    Epistemology examines the nature and sources of human knowledge.
    The problem of induction questions whether past observations predict future events.
    Consciousness remains one of the deepest mysteries in philosophy of mind.
    Ethical frameworks provide guidelines for moral decision-making.
    The relationship between free will and determinism continues to puzzle philosophers.
  `
};

async function demonstrateRSE() {
  console.log('🔬 Resonant Semantic Embedding Demonstration\n');
  
  // Initialize RSE with both classical and manifold capabilities
  const rse = new ResonantSemanticEmbedding(
    mockEmbeddingFunction,
    0.15,  // threshold
    3,     // window size
    4,     // manifold dimension
    true   // use manifold metrics
  );
  
  console.log('📊 Document Analysis:\n');
  
  // Analyze each document's semantic complexity
  for (const [topic, text] of Object.entries(documents)) {
    console.log(`--- ${topic.toUpperCase()} ---`);
    
    // Classical RSE analysis
    const classicalRSE = rse.generateRSE(text);
    console.log(`Resonant components: ${classicalRSE.components.length}`);
    console.log(`Compression ratio: ${(classicalRSE.compressionRatio * 100).toFixed(1)}%`);
    console.log(`Total energy: ${classicalRSE.totalEnergy.toFixed(3)}`);
    
    // Manifold-enhanced analysis
    const manifoldRSE = rse.generateManifoldRSE(text);
    console.log(`Manifold dimension: ${manifoldRSE.manifoldDimension}`);
    console.log(`Average curvature: ${manifoldRSE.averageCurvature.toFixed(4)}`);
    console.log(`Geodesic components: ${manifoldRSE.geodesicComponents.length}`);
    
    // Semantic complexity analysis
    const complexity = rse.analyzeSemanticComplexity(text);
    console.log(`Max curvature: ${complexity.maxCurvature.toFixed(4)}`);
    console.log(`Curvature variance: ${complexity.curvatureVariance.toFixed(4)}`);
    
    // Show most complex sentences
    console.log('Most semantically complex sentences:');
    complexity.complexityRegions.slice(0, 2).forEach((region, index) => {
      console.log(`  ${index + 1}. [${region.curvature.toFixed(3)}] ${region.sentence.trim()}`);
    });
    
    console.log();
  }
  
  console.log('🔗 Cross-Domain Similarity Analysis:\n');
  
  // Compare similarities using both classical and manifold approaches
  const topics = Object.keys(documents);
  
  console.log('Classical RSE Similarities:');
  for (let i = 0; i < topics.length; i++) {
    for (let j = i + 1; j < topics.length; j++) {
      const similarity = rse.similarity(documents[topics[i]], documents[topics[j]]);
      console.log(`${topics[i]} ↔ ${topics[j]}: ${similarity.toFixed(4)}`);
    }
  }
  
  console.log('\nManifold-Enhanced Geometric Similarities:');
  for (let i = 0; i < topics.length; i++) {
    for (let j = i + 1; j < topics.length; j++) {
      const similarity = rse.geometricSimilarity(documents[topics[i]], documents[topics[j]]);
      console.log(`${topics[i]} ↔ ${topics[j]}: ${similarity.toFixed(4)}`);
    }
  }
  
  console.log('\n🧠 Semantic Hierarchy Analysis:\n');
  
  // Analyze semantic hierarchy for one document
  const physicsHierarchy = rse.analyzeSemanticHierarchy(documents.physics);
  console.log('Physics document - Top semantic frequencies:');
  physicsHierarchy.slice(0, 5).forEach((component, index) => {
    console.log(`  ${index + 1}. Freq: ${component.frequency.toFixed(3)}, Amplitude: ${component.amplitude.toFixed(4)}`);
  });
  
  console.log('\n✨ Key Insights:');
  console.log('- Higher curvature values indicate more semantically dense content');
  console.log('- Manifold-enhanced similarities capture geometric relationships');
  console.log('- Frequency analysis reveals hierarchical semantic structure');
  console.log('- Geodesic distances provide more nuanced similarity measures');
}

// Run the demonstration
if (import.meta.url === `file://${process.argv[1]}`) {
  demonstrateRSE().catch(console.error);
}

export { demonstrateRSE };
