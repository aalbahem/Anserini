package io.anserini.rerank.rf;

import io.anserini.rerank.rf.FeedbackModel;
import io.anserini.search.SearchArgs;
import org.apache.lucene.analysis.Analyzer;

public abstract class FeedbackModelFactory {

  public abstract FeedbackModel create(Analyzer analyzer, String field,SearchArgs args);
}
