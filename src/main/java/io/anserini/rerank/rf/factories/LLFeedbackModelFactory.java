package io.anserini.rerank.rf.factories;

import io.anserini.rerank.rf.FeedbackModel;
import io.anserini.rerank.rf.FeedbackModelConfig;
import io.anserini.rerank.rf.FeedbackModelFactory;
import io.anserini.rerank.rf.models.LogLogisticRelevanceModel;
import io.anserini.search.SearchArgs;
import org.apache.lucene.analysis.Analyzer;

public class LLFeedbackModelFactory extends FeedbackModelFactory {
  @Override
  public FeedbackModel create(Analyzer analyzer, String field, SearchArgs args) {

    FeedbackModelConfig config = new FeedbackModelConfig(
      Integer.parseInt(args.rf__fbDocs[0]),
      Integer.parseInt(args.rf_fbTerms[0]),
      Float.parseFloat(args.rf__originalQueryWeight[0]),
        args.rf__outputQuery,
        args.rf__stopword,
        args.rf__pruneDocTerm,
        args.rf__pruneModel,
        args.rf__normalize
    );

    LogLogisticRelevanceModel logLogisticRelevanceModel = new LogLogisticRelevanceModel(analyzer,field,Float.parseFloat(args.ll_c[0]),config);
    return logLogisticRelevanceModel;
  }
}
