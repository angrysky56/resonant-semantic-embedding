import { Matrix, EigenvalueDecomposition } from 'ml-matrix';

/**
 * Core mathematical types for RSE implementation
 * ASYNC REFACTOR: Embedding function signature updated to Promise<Float64Array>
 */

export interface SemanticSignal {
  readonly signal: Float64Array[];  // Vector-valued signal over [0,1]
  readonly windowSize: number;
  readonly dimension: number;
}

/**
 * Manifold-based extensions for geometric semantic analysis
 * Implements concepts from RSE manifold paper
 */
export interface ManifoldPoint {
  readonly coordinates: Float64Array;  // Point on learned manifold
  readonly tangentVector?: Float64Array;  // Tangent space representation
  readonly curvature?: number;  // Local semantic curvature
}

export interface GeometricSemanticSignal extends SemanticSignal {
  readonly manifoldPoints: ManifoldPoint[];  // Signal as curve on manifold
  readonly geodesicLength: number;  // Total path length along manifold
}

export interface ManifoldRSE extends RSERepresentation {
  readonly geodesicComponents: FrequencyComponent[];  // Frequency analysis on manifold
  readonly manifoldDimension: number;  // Intrinsic manifold dimension
  readonly averageCurvature: number;  // Mean semantic complexity measure
}

export interface FrequencyComponent {
  readonly frequency: number;       // ω_k
  readonly amplitude: number;       // |X(ω_k)|
  readonly phase: Float64Array;     // arg(X(ω_k)) - complex vector
}

export interface RSERepresentation {
  readonly components: FrequencyComponent[];
  readonly threshold: number;
  readonly totalEnergy: number;
  readonly compressionRatio: number;
}

/**
 * Semantic Fourier Transform implementation
 * Following mathematical framework from technical paper
 * 
 * ASYNC REFACTOR: Made async to support real embedding backends
 * PRESERVED: All mathematical algorithms unchanged
 */
export class SemanticFourierTransform {
  
  /**
   * Construct semantic signal from document
   * Implements Construction 1.1 from paper
   * 
   * ASYNC CHANGE: Now async to await embedding generation
   */
  static async constructSemanticSignal(
    sentences: string[],
    embeddingFunction: (text: string) => Promise<Float64Array>,
    windowSize: number = 3
  ): Promise<SemanticSignal> {
    
    const windows: string[][] = [];
    
    // Create overlapping windows (PRESERVED: same algorithm)
    for (let i = 0; i <= sentences.length - windowSize; i++) {
      windows.push(sentences.slice(i, i + windowSize));
    }
    
    // Generate embeddings for each window (ASYNC: now await each embedding)
    const signal: Float64Array[] = [];
    for (const window of windows) {
      const embedding = await embeddingFunction(window.join(' '));
      signal.push(embedding);
    }
    
    return {
      signal,
      windowSize,
      dimension: signal[0]?.length || 0
    };
  }
  
  /**
   * Compute Semantic Fourier Transform
   * Implements Definition 2.1 from paper
   * 
   * PRESERVED: Pure mathematical operation, no async needed
   */
  static computeSFT(semanticSignal: SemanticSignal): FrequencyComponent[] {
    const { signal, dimension } = semanticSignal;
    const N = signal.length;
    const components: FrequencyComponent[] = [];
    
    // For each frequency bin (PRESERVED: exact same math)
    for (let k = 0; k < N; k++) {
      const frequency = (2 * Math.PI * k) / N;
      
      // Compute complex-valued transform for each dimension
      const real = new Float64Array(dimension);
      const imag = new Float64Array(dimension);
      
      for (let n = 0; n < N; n++) {
        const phase = -2 * Math.PI * k * n / N;
        const cosPhase = Math.cos(phase);
        const sinPhase = Math.sin(phase);
        
        for (let d = 0; d < dimension; d++) {
          real[d] += signal[n][d] * cosPhase;
          imag[d] += signal[n][d] * sinPhase;
        }
      }
      
      // Compute amplitude and phase (PRESERVED: same calculation)
      const amplitude = Math.sqrt(
        real.reduce((sum, r, i) => sum + r*r + imag[i]*imag[i], 0)
      );
      
      const phase = new Float64Array(dimension);
      for (let d = 0; d < dimension; d++) {
        phase[d] = Math.atan2(imag[d], real[d]);
      }
      
      components.push({ frequency, amplitude, phase });
    }
    
    return components;
  }
  
  /**
   * Extract resonant components
   * Implements Definition 2.3 from paper
   * 
   * PRESERVED: Pure mathematical operation, no async needed
   */
  static extractResonantComponents(
    components: FrequencyComponent[],
    threshold: number = 0.1
  ): RSERepresentation {
    
    // Calculate total energy (PRESERVED)
    const totalEnergy = Math.sqrt(
      components.reduce((sum, comp) => sum + comp.amplitude * comp.amplitude, 0)
    );
    
    // Apply resonance threshold (PRESERVED)
    const resonantThreshold = threshold * totalEnergy;
    const resonantComponents = components
      .filter(comp => comp.amplitude > resonantThreshold)
      .sort((a, b) => b.amplitude - a.amplitude);
    
    const compressionRatio = resonantComponents.length / components.length;
    
    return {
      components: resonantComponents,
      threshold: resonantThreshold,
      totalEnergy,
      compressionRatio
    };
  }
  
  /**
   * Compute RSE distance metric
   * Implements Definition 3.3 from paper
   * 
   * PRESERVED: Pure mathematical operation
   */
  static computeRSEDistance(
    rse1: RSERepresentation,
    rse2: RSERepresentation,
    frequencyWeighting: number = 0.5
  ): number {
    
    const maxComponents = Math.min(rse1.components.length, rse2.components.length);
    let distance = 0;
    
    for (let k = 0; k < maxComponents; k++) {
      const comp1 = rse1.components[k];
      const comp2 = rse2.components[k];
      
      // Frequency-importance weighting w_k = k^(-β) (PRESERVED)
      const weight = Math.pow(k + 1, -frequencyWeighting);
      
      // Compute weighted phase vector difference (PRESERVED)
      let phaseDiff = 0;
      for (let d = 0; d < comp1.phase.length && d < comp2.phase.length; d++) {
        const diff = comp1.amplitude * comp1.phase[d] - comp2.amplitude * comp2.phase[d];
        phaseDiff += diff * diff;
      }
      
      distance += weight * Math.sqrt(phaseDiff);
    }
    
    return distance;
  }

  /**
   * Compute geodesic distance on learned manifold
   * Implements manifold-based distance from manifold paper
   * 
   * PRESERVED: Pure mathematical operation
   */
  static computeGeodesicDistance(
    point1: ManifoldPoint,
    point2: ManifoldPoint,
    manifoldMetric?: Matrix
  ): number {
    
    if (!manifoldMetric) {
      // Fallback to Euclidean distance (PRESERVED)
      return SemanticFourierTransform.euclideanDistance(
        point1.coordinates, 
        point2.coordinates
      );
    }
    
    // Approximate geodesic distance using Riemannian metric (PRESERVED)
    const diff = new Float64Array(point1.coordinates.length);
    for (let i = 0; i < diff.length; i++) {
      diff[i] = point2.coordinates[i] - point1.coordinates[i];
    }
    
    const diffVector = Matrix.columnVector(Array.from(diff));
    const metricDistance = diffVector.transpose().mmul(manifoldMetric).mmul(diffVector);
    
    return Math.sqrt(metricDistance.get(0, 0));
  }

  /**
   * Simple Euclidean distance helper
   * PRESERVED: Pure mathematical operation
   */
  private static euclideanDistance(v1: Float64Array, v2: Float64Array): number {
    let sum = 0;
    for (let i = 0; i < Math.min(v1.length, v2.length); i++) {
      const diff = v1[i] - v2[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }

  /**
   * Learn manifold structure from embedding data
   * Simplified manifold learning using local PCA
   * 
   * PRESERVED: All mathematical algorithms unchanged
   */
  static learnManifoldStructure(
    embeddings: Float64Array[],
    intrinsicDimension: number = 3
  ): { 
    manifoldPoints: ManifoldPoint[],
    metric: Matrix,
    avgCurvature: number 
  } {
    
    const embeddingMatrix = new Matrix(embeddings.map(emb => Array.from(emb)));
    
    // Compute covariance matrix manually (PRESERVED)
    const covarianceMatrix = SemanticFourierTransform.computeCovarianceMatrix(embeddingMatrix);
    
    // Perform PCA using eigenvalue decomposition (PRESERVED)
    const eigenDecomposition = new EigenvalueDecomposition(covarianceMatrix);
    const eigenvectors = eigenDecomposition.eigenvectorMatrix;
    
    // Project embeddings onto manifold (PRESERVED)
    const manifoldBasis = eigenvectors.subMatrix(0, embeddingMatrix.columns - 1, 0, intrinsicDimension - 1);
    const projectedPoints = embeddingMatrix.mmul(manifoldBasis);
    
    // Create manifold points with curvature estimates (PRESERVED)
    const manifoldPoints: ManifoldPoint[] = [];
    const curvatures: number[] = [];
    
    for (let i = 0; i < projectedPoints.rows; i++) {
      const coordinates = new Float64Array(projectedPoints.getRow(i));
      
      const curvature = SemanticFourierTransform.estimateLocalCurvature(
        coordinates, 
        projectedPoints, 
        i
      );
      curvatures.push(curvature);
      
      manifoldPoints.push({ coordinates, curvature });
    }
    
    // Create Riemannian metric tensor (PRESERVED)
    const metric = Matrix.eye(intrinsicDimension);
    const avgCurvature = curvatures.reduce((sum, c) => sum + c, 0) / curvatures.length;
    
    return { manifoldPoints, metric, avgCurvature };
  }

  /**
   * Compute covariance matrix manually
   * Implements: Cov(X) = E[(X - μ)(X - μ)^T]
   * 
   * PRESERVED: Pure mathematical operation
   */
  private static computeCovarianceMatrix(matrix: Matrix): Matrix {
    // Compute mean for each column (PRESERVED)
    const means = new Float64Array(matrix.columns);
    for (let col = 0; col < matrix.columns; col++) {
      let sum = 0;
      for (let row = 0; row < matrix.rows; row++) {
        sum += matrix.get(row, col);
      }
      means[col] = sum / matrix.rows;
    }
    
    // Center the data (PRESERVED)
    const centeredMatrix = new Matrix(matrix.rows, matrix.columns);
    for (let row = 0; row < matrix.rows; row++) {
      for (let col = 0; col < matrix.columns; col++) {
        centeredMatrix.set(row, col, matrix.get(row, col) - means[col]);
      }
    }
    
    // Compute covariance (PRESERVED)
    const transpose = centeredMatrix.transpose();
    const covariance = transpose.mmul(centeredMatrix);
    
    const n = matrix.rows;
    return covariance.div(n - 1);
  }

  /**
   * Estimate local curvature at a point
   * Higher curvature indicates semantic complexity
   * 
   * PRESERVED: Pure mathematical operation
   */
  private static estimateLocalCurvature(
    point: Float64Array,
    allPoints: Matrix,
    pointIndex: number,
    neighborhoodSize: number = 5
  ): number {
    
    // Find nearest neighbors (PRESERVED)
    const distances: { index: number, distance: number }[] = [];
    
    for (let i = 0; i < allPoints.rows; i++) {
      if (i === pointIndex) continue;
      
      const neighbor = new Float64Array(allPoints.getRow(i));
      const distance = SemanticFourierTransform.euclideanDistance(point, neighbor);
      distances.push({ index: i, distance });
    }
    
    distances.sort((a, b) => a.distance - b.distance);
    const neighbors = distances.slice(0, neighborhoodSize);
    
    // Estimate curvature from neighbor distribution (PRESERVED)
    const avgDistance = neighbors.reduce((sum, n) => sum + n.distance, 0) / neighbors.length;
    const variance = neighbors.reduce((sum, n) => sum + Math.pow(n.distance - avgDistance, 2), 0) / neighbors.length;
    
    return variance / (avgDistance * avgDistance + 1e-8);
  }
}

/**
 * Enhanced RSE class with manifold learning capabilities
 * Integrates concepts from both RSE technical and manifold papers
 * 
 * ASYNC REFACTOR: Main API methods now async to support real embeddings
 * PRESERVED: All mathematical sophistication maintained
 */
export class ResonantSemanticEmbedding {
  private threshold: number;
  private windowSize: number;
  private embeddingFunction: (text: string) => Promise<Float64Array>;
  private manifoldDimension: number;
  private useManifoldMetrics: boolean;
  
  constructor(
    embeddingFunction: (text: string) => Promise<Float64Array>,
    threshold: number = 0.1,
    windowSize: number = 3,
    manifoldDimension: number = 3,
    useManifoldMetrics: boolean = true
  ) {
    this.embeddingFunction = embeddingFunction;
    this.threshold = threshold;
    this.windowSize = windowSize;
    this.manifoldDimension = manifoldDimension;
    this.useManifoldMetrics = useManifoldMetrics;
  }
  
  /**
   * Generate RSE representation for a document
   * Implements full pipeline from paper
   * 
   * ASYNC CHANGE: Now async to support real embedding generation
   * PRESERVED: All mathematical pipeline steps unchanged
   */
  async generateRSE(document: string): Promise<RSERepresentation> {
    // Split into sentences (PRESERVED)
    const sentences = document.split(/[.!?]+/).filter(s => s.trim().length > 0);
    
    // Construct semantic signal (ASYNC: now awaits embeddings)
    const semanticSignal = await SemanticFourierTransform.constructSemanticSignal(
      sentences,
      this.embeddingFunction,
      this.windowSize
    );
    
    // Compute frequency spectrum (PRESERVED)
    const spectrum = SemanticFourierTransform.computeSFT(semanticSignal);
    
    // Extract resonant components (PRESERVED)
    const rse = SemanticFourierTransform.extractResonantComponents(
      spectrum,
      this.threshold
    );
    
    return rse;
  }

  /**
   * Generate enhanced manifold-based RSE representation
   * Incorporates geodesic analysis from manifold paper
   * 
   * ASYNC CHANGE: Now async to support real embedding generation
   * PRESERVED: All mathematical algorithms unchanged
   */
  async generateManifoldRSE(document: string): Promise<ManifoldRSE> {
    const baseRSE = await this.generateRSE(document);
    
    // Extract embeddings for manifold learning (ASYNC: now awaits)
    const sentences = document.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const embeddings: Float64Array[] = [];
    for (const sentence of sentences) {
      const embedding = await this.embeddingFunction(sentence);
      embeddings.push(embedding);
    }
    
    // Learn manifold structure (PRESERVED: same algorithm)
    const manifoldData = SemanticFourierTransform.learnManifoldStructure(
      embeddings,
      this.manifoldDimension
    );
    
    // Create geometric semantic signal (PRESERVED)
    const geometricSignal: GeometricSemanticSignal = {
      signal: embeddings,
      windowSize: this.windowSize,
      dimension: embeddings[0]?.length || 0,
      manifoldPoints: manifoldData.manifoldPoints,
      geodesicLength: this.computeGeodesicPathLength(manifoldData.manifoldPoints)
    };
    
    // Perform frequency analysis on manifold (PRESERVED)
    const manifoldSpectrum = this.computeManifoldFrequencyAnalysis(geometricSignal);
    
    return {
      ...baseRSE,
      geodesicComponents: manifoldSpectrum,
      manifoldDimension: this.manifoldDimension,
      averageCurvature: manifoldData.avgCurvature
    };
  }

  /**
   * Compute total geodesic path length along manifold
   * PRESERVED: Pure mathematical operation
   */
  private computeGeodesicPathLength(manifoldPoints: ManifoldPoint[]): number {
    let totalLength = 0;
    
    for (let i = 1; i < manifoldPoints.length; i++) {
      const distance = SemanticFourierTransform.computeGeodesicDistance(
        manifoldPoints[i-1],
        manifoldPoints[i]
      );
      totalLength += distance;
    }
    
    return totalLength;
  }

  /**
   * Perform frequency analysis considering manifold curvature
   * PRESERVED: Pure mathematical operation
   */
  private computeManifoldFrequencyAnalysis(signal: GeometricSemanticSignal): FrequencyComponent[] {
    // Weight frequency components by local curvature (PRESERVED)
    const baseSpectrum = SemanticFourierTransform.computeSFT(signal);
    
    return baseSpectrum.map((component, index) => {
      const curvatureWeight = signal.manifoldPoints[index % signal.manifoldPoints.length]?.curvature || 1;
      
      return {
        ...component,
        amplitude: component.amplitude * (1 + curvatureWeight)
      };
    });
  }
  
  /**
   * Enhanced similarity computation using geodesic distances
   * Implements manifold-aware similarity from manifold paper
   * 
   * ASYNC CHANGE: Now async due to generateManifoldRSE
   * PRESERVED: Mathematical comparison logic unchanged
   */
  async geometricSimilarity(doc1: string, doc2: string): Promise<number> {
    if (!this.useManifoldMetrics) {
      return this.similarity(doc1, doc2);
    }

    const manifoldRSE1 = await this.generateManifoldRSE(doc1);
    const manifoldRSE2 = await this.generateManifoldRSE(doc2);
    
    // Compute geodesic-based distance (PRESERVED)
    const geodesicDistance = this.computeGeodesicRSEDistance(manifoldRSE1, manifoldRSE2);
    
    // Convert distance to similarity (PRESERVED)
    return Math.exp(-geodesicDistance);
  }

  /**
   * Compute distance between ManifoldRSE representations using geodesic metrics
   * PRESERVED: Pure mathematical operation
   */
  private computeGeodesicRSEDistance(rse1: ManifoldRSE, rse2: ManifoldRSE): number {
    const maxComponents = Math.min(rse1.geodesicComponents.length, rse2.geodesicComponents.length);
    let distance = 0;
    
    for (let k = 0; k < maxComponents; k++) {
      const comp1 = rse1.geodesicComponents[k];
      const comp2 = rse2.geodesicComponents[k];
      
      // Frequency-importance weighting enhanced with curvature (PRESERVED)
      const curvatureWeight = (rse1.averageCurvature + rse2.averageCurvature) / 2;
      const weight = Math.pow(k + 1, -0.5) * (1 + curvatureWeight);
      
      // Compute weighted phase vector difference (PRESERVED)
      let phaseDiff = 0;
      for (let d = 0; d < comp1.phase.length && d < comp2.phase.length; d++) {
        const diff = comp1.amplitude * comp1.phase[d] - comp2.amplitude * comp2.phase[d];
        phaseDiff += diff * diff;
      }
      
      distance += weight * Math.sqrt(phaseDiff);
    }
    
    // Add manifold complexity penalty (PRESERVED)
    const complexityDifference = Math.abs(rse1.averageCurvature - rse2.averageCurvature);
    distance += 0.1 * complexityDifference;
    
    return distance;
  }
  
  /**
   * Compute similarity between documents using RSE (original method)
   * 
   * ASYNC CHANGE: Now async due to generateRSE
   * PRESERVED: Mathematical comparison unchanged
   */
  async similarity(doc1: string, doc2: string): Promise<number> {
    const rse1 = await this.generateRSE(doc1);
    const rse2 = await this.generateRSE(doc2);
    
    const distance = SemanticFourierTransform.computeRSEDistance(rse1, rse2);
    
    return Math.exp(-distance);
  }
  
  /**
   * Analyze semantic hierarchy of document
   * Returns frequency components ordered by semantic importance
   * 
   * ASYNC CHANGE: Now async due to generateRSE
   * PRESERVED: Same analysis logic
   */
  async analyzeSemanticHierarchy(document: string): Promise<FrequencyComponent[]> {
    const rse = await this.generateRSE(document);
    return rse.components;
  }

  /**
   * Analyze semantic complexity using manifold curvature
   * Higher curvature indicates more semantically dense regions
   * 
   * ASYNC CHANGE: Now async to await embeddings
   * PRESERVED: All mathematical analysis unchanged
   */
  async analyzeSemanticComplexity(document: string): Promise<{
    averageCurvature: number,
    maxCurvature: number,
    curvatureVariance: number,
    complexityRegions: { sentence: string, curvature: number }[]
  }> {
    const sentences = document.split(/[.!?]+/).filter(s => s.trim().length > 0);
    
    // Generate embeddings (ASYNC)
    const embeddings: Float64Array[] = [];
    for (const sentence of sentences) {
      const embedding = await this.embeddingFunction(sentence);
      embeddings.push(embedding);
    }
    
    // Learn manifold and compute curvatures (PRESERVED)
    const manifoldData = SemanticFourierTransform.learnManifoldStructure(
      embeddings,
      this.manifoldDimension
    );
    
    const curvatures = manifoldData.manifoldPoints.map(p => p.curvature || 0);
    const averageCurvature = curvatures.reduce((sum, c) => sum + c, 0) / curvatures.length;
    const maxCurvature = Math.max(...curvatures);
    const curvatureVariance = curvatures.reduce((sum, c) => sum + Math.pow(c - averageCurvature, 2), 0) / curvatures.length;
    
    const complexityRegions = sentences.map((sentence, index) => ({
      sentence,
      curvature: curvatures[index] || 0
    })).sort((a, b) => b.curvature - a.curvature);
    
    return {
      averageCurvature,
      maxCurvature,
      curvatureVariance,
      complexityRegions
    };
  }
}
