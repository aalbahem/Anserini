package io.anserini.rerank;

import io.anserini.search.SearchArgs;
import org.apache.lucene.analysis.Analyzer;

public abstract class RerankerFactory {

  public abstract Reranker create(Analyzer analyzer, String field);
}
