package io.anserini.rerank.factories;

import io.anserini.rerank.Reranker;
import io.anserini.rerank.RerankerFactory;
import io.anserini.rerank.lib.RFReranker;
import io.anserini.search.SearchArgs;
import org.apache.lucene.analysis.Analyzer;

public class RFRerankerkFactory extends RerankerFactory {

  @Override
  public Reranker create(Analyzer analyzer, String field) {
      RFReranker reranker = new RFReranker(analyzer,field);
      return reranker;
  }
}
