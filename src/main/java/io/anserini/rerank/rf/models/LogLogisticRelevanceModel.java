package io.anserini.rerank.rf.models;

import io.anserini.rerank.rf.FeedbackModel;
import io.anserini.rerank.rf.FeedbackModelConfig;
import io.anserini.util.FeatureVector;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermStates;
import org.apache.lucene.search.CollectionStatistics;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TermStatistics;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class LogLogisticRelevanceModel extends FeedbackModel {

  private static final Logger LOG = LogManager.getLogger(LogLogisticRelevanceModel.class);

  private float c = 0.2f;

  public LogLogisticRelevanceModel(Analyzer analyzer, String field, float c,
      FeedbackModelConfig config) {
    super(analyzer, field,config);
    this.c = c;
  }


  public FeatureVector estimate(String query, FeatureVector[] docvectors, float[] scores) {
    Set<String> vocab = getVocab(docvectors);
    // Precompute the norms once and cache templates.
    float[] norms = computeNorms(docvectors);
    FeatureVector f = new FeatureVector();


    Map<String,TermStatistics> termStatisticsList;
    CollectionStatistics collectionStatistics;

    try {
      collectionStatistics = getIndexSearcher().collectionStatistics(getField());
      termStatisticsList = new HashMap<>();
      collectTermStatitics(vocab,termStatisticsList);

      for (String term : vocab) {
        LLFeedbakWegith llFeedbakWegith = new LLFeedbakWegith((float)c,collectionStatistics,termStatisticsList.get(term));
        f.addFeatureWeight(term,(float)llFeedbakWegith.computeFW(term,norms,docvectors));
      }


      if (getConfig().pruneModel){
        f.pruneToSize(getFbTerms());
      }

      if (getConfig().normalize){
        f.scaleToUnitL1Norm();
      }

    } catch (IOException e) {
      e.printStackTrace();
      LOG.warn("Failed to extract LLmodel");
    }

    return f;
  }

}

class LLFeedbakWegith{
  private static final Logger LOG = LogManager.getLogger(LLFeedbakWegith.class);

  // Collection statisticsts for a certain term
  private CollectionStatistics collectionStats;
  private TermStatistics termStats;
  // the free parameter
  private double c = 0.5f;



  public LLFeedbakWegith(float c,CollectionStatistics collectionStatistics, TermStatistics termStatistics){
    this.c = c;
    this.collectionStats = collectionStatistics;
    this.termStats = termStatistics;
  }

  public double computeFW(String term, float[] norms, FeatureVector[] docvectors){

    double fbWeight = 0.0;
    for (int i = 0; i < docvectors.length; i++) {
      // Avoids zero-length feedback documents, which causes division by zero when computing term weights.
      // Zero-length feedback documents occur (e.g., with CAR17) when a document has only terms
      // that accents (which are indexed, but not selected for feedback).
      if (norms[i] > 0.001f) {
        try {
          fbWeight += computeFW(term,docvectors[i].getFeatureWeight(term),norms[i]);
        } catch (IOException e) {
          LOG.warn("Not able to calculate FW for " + term + " " + i);
        }
      }
    }
    return fbWeight/docvectors.length;
  }


  public double computeFW(String term, double tf, double dLenght) throws IOException {
    double yW = computeLambdaW(term);

    double fw = Math.log( (t(tf,dLenght) + yW) / yW);
    return fw;

  }

  public double t(double tf,double dLenght) {
//    T F(w, D) log(1+ c avgl/|D|)

    double avgL = collectionStats.sumTotalTermFreq() / (double)(collectionStats.docCount());

    return  tf * Math.log(1 + (c * avgL / dLenght));
  }

  public double computeLambdaW(String term) throws IOException {
    double N_w  = termStats.docFreq();
    double N    = collectionStats.docCount();
    return N_w/N;
  }
}
