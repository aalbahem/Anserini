package io.anserini.rerank.lib;

import io.anserini.rerank.RerankerContext;
import io.anserini.rerank.ScoredDocuments;
import io.anserini.util.FeatureVector;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermContext;
import org.apache.lucene.search.*;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class LogLogisticRelevanceModelReranker extends Rm3Reranker {

  private static final Logger LOG = LogManager.getLogger(LogLogisticRelevanceModelReranker.class);

  private double c;

  private IndexSearcher indexSearcher ;

  private Map<String,TermStatistics> termStatisticsList;
  private CollectionStatistics collectionStatistics;

  public LogLogisticRelevanceModelReranker(Analyzer analyzer, String field, int fbTerms, int fbDocs, float originalQueryWeight, boolean outputQuery) {
    super(analyzer, field, fbTerms, fbDocs, originalQueryWeight, outputQuery);
    c = originalQueryWeight;
  }

  public LogLogisticRelevanceModelReranker(Analyzer analyzer, String field, int fbTerms, int fbDocs, float originalQueryWeight, boolean outputQuery, boolean removeStopwords) {
    super(analyzer, field, fbTerms, fbDocs, originalQueryWeight, outputQuery,removeStopwords);
    c = originalQueryWeight;
  }

  protected FeatureVector estimateRelevanceModel(float[] scores, FeatureVector[] docvectors) {
    Set<String> vocab = getVocab(docvectors);
    // Precompute the norms once and cache templates.
    float[] norms = computeNorms(docvectors);
    FeatureVector f = new FeatureVector();
    try {
      collectionStatistics = indexSearcher.collectionStatistics(getField());
      termStatisticsList = new HashMap<>();
      for (String term : vocab){
        TermContext termContext =TermContext.build(indexSearcher.getIndexReader().getContext(),new Term(getField(),term));
        termStatisticsList.put(term, indexSearcher.termStatistics(new Term(getField(),new BytesRef(term)),termContext));
      }

      for (String term : vocab) {
        LLFeedbakWegith llFeedbakWegith = new LLFeedbakWegith((float)c,collectionStatistics,termStatisticsList.get(term));
        f.addFeatureWeight(term,(float)llFeedbakWegith.computeFW(term,norms,docvectors));
      }

      f.pruneToSize(getFbTerms());
//      f.scaleToUnitL1Norm();
    } catch (IOException e) {
      e.printStackTrace();
      LOG.warn("Failed to extract LLmodel");
    }

    return f;
  }

  @Override
  public ScoredDocuments rerank(ScoredDocuments docs, RerankerContext context) {
    this.indexSearcher = context.getIndexSearcher();
    return super.rerank(docs,context);
  }


  public IndexSearcher getIndexSearcher() {
    return indexSearcher;
  }

  public void setIndexSearcher(IndexSearcher indexSearcher) {
    this.indexSearcher = indexSearcher;
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
