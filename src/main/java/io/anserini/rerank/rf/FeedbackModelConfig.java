package io.anserini.rerank.rf;

public class FeedbackModelConfig {


  public final int fbTerms;
  public final int fbDocs;
  public final float originalQueryWeight;
  public final boolean outputQuery;
  public final boolean removeStopWord;
  public final boolean prunDocTerm;
  public final boolean pruneModel;
  public final boolean normalize;

  public FeedbackModelConfig(int fbTerms, int fbDocs,
      float originalQueryWeight, boolean outputQuery, boolean removeStopWord,
      boolean prunDocTerm, boolean pruneModel, boolean normalize) {
    this.fbTerms = fbTerms;
    this.fbDocs = fbDocs;
    this.originalQueryWeight = originalQueryWeight;
    this.outputQuery = outputQuery;
    this.removeStopWord = removeStopWord;
    this.prunDocTerm = prunDocTerm;
    this.pruneModel = pruneModel;
    this.normalize = normalize;
  }

  public int getFbTerms() {
    return fbTerms;
  }

  public int getFbDocs() {
    return fbDocs;
  }

  public float getOriginalQueryWeight() {
    return originalQueryWeight;
  }

  public boolean isOutputQuery() {
    return outputQuery;
  }

  public boolean isRemoveStopWord() {
    return removeStopWord;
  }

  public boolean isPrunDocTerm() {
    return prunDocTerm;
  }

  public boolean isPruneModel() {
    return pruneModel;
  }

  public boolean isNormalize() {
    return normalize;
  }
}