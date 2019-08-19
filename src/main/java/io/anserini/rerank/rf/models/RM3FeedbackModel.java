package io.anserini.rerank.rf.models;

import io.anserini.rerank.rf.FeedbackModel;
import io.anserini.rerank.rf.FeedbackModelConfig;
import io.anserini.util.FeatureVector;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.search.CollectionStatistics;
import org.apache.lucene.search.TermStatistics;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class RM3FeedbackModel extends FeedbackModel {

  private static final Logger LOG = LogManager.getLogger(LogLogisticRelevanceModel.class);

  public RM3FeedbackModel(Analyzer analyzer, String field, FeedbackModelConfig config) {
    super(analyzer, field,config);
  }


  public FeatureVector estimate(String query, FeatureVector[] docvectors, float[] scores) {
    FeatureVector qfv = getQfv(query);

    Set<String> vocab = getVocab(docvectors);
    // Precompute the norms once and cache results.
    float[] norms = computeNorms(docvectors);

    FeatureVector f = new FeatureVector();
    for (String term : vocab) {
      float fbWeight = 0.0f;
      for (int i = 0; i < docvectors.length; i++) {
        // Avoids zero-length feedback documents, which causes division by zero when computing term weights.
        // Zero-length feedback documents occur (e.g., with CAR17) when a document has only terms
        // that accents (which are indexed, but not selected for feedback).
        if (norms[i] > 0.001f) {
          fbWeight += (docvectors[i].getFeatureWeight(term) / norms[i]) * scores[i];
        }
      }
      f.addFeatureWeight(term, fbWeight);
    }

    f.pruneToSize(getFbTerms());
    f.scaleToUnitL1Norm();

    FeatureVector rm = FeatureVector.interpolate(qfv, f, getConfig().originalQueryWeight);
    return rm;
  }
}
