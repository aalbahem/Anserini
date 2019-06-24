package io.anserini.rerank.lib;

import io.anserini.rerank.RerankerContext;
import io.anserini.rerank.ScoredDocuments;
import io.anserini.util.AnalyzerUtils;
import io.anserini.util.FeatureVector;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermContext;
import org.apache.lucene.search.*;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;


public class RocchioReranker extends Rm3Reranker {

  private static final Logger LOG = LogManager.getLogger(LogLogisticRelevanceModelReranker.class);

  private double c;

  private IndexSearcher indexSearcher ;

  private Map<String,TermStatistics> termStatisticsList = new HashMap<>();
  private CollectionStatistics collectionStatistics;

  private int[] relDocs;
  private int[] nonRelDocs;

  private float alpha = 1.f;
  private float beta = 0.85f;
  private float gamma = 0.00f;



  public void setRelDocs(int[] relDocs) {
    this.relDocs = relDocs;
  }

  public void setNonRelDocs(int[] nonRelDocs) {
    this.nonRelDocs = nonRelDocs;
  }

  public RocchioReranker(Analyzer analyzer, String field, int fbTerms, int fbDocs, float originalQueryWeight, boolean outputQuery) {
    super(analyzer, field, fbTerms, fbDocs, originalQueryWeight, outputQuery);
    c = originalQueryWeight;
  }

  public RocchioReranker(Analyzer analyzer, String field, int fbTerms, int fbDocs, float originalQueryWeight, boolean outputQuery, boolean removeStopwords) {
    super(analyzer, field, fbTerms, fbDocs, originalQueryWeight, outputQuery,removeStopwords);
    c = originalQueryWeight;
  }



  @Override
  public ScoredDocuments rerank(ScoredDocuments docs, RerankerContext context) {
    assert(docs.documents.length == docs.scores.length);

    this.indexSearcher = context.getIndexSearcher();
    return super.rerank(docs,context);
  }

  public FeatureVector estimateRocchioModel(String[] contents, float[] scores, RerankerContext context) {
    return estimateRocchioModel(buildDocVectors(contents,context.getIndexSearcher().getIndexReader(),context.getSearchArgs().searchtweets),context);
  }

  protected FeatureVector getQfv(RerankerContext context) {
    FeatureVector model = FeatureVector.fromTerms(AnalyzerUtils.tokenize(getAnalyzer(), context.getQueryText()));

    float norm = (float) model.computeMaxNorm();
    addTermStatistic(model.getFeatures());
    FeatureVector qfv = new FeatureVector();

    for (String term : model.getFeatures()){
      long tf = termStatisticsList.get(term).docFreq();
      if (tf >0){
        float idf = (float)Math.log(collectionStatistics.docCount()/termStatisticsList.get(term).docFreq());
        qfv.addFeatureWeight(term,(0.5f + 0.5f * tf/norm) * idf);
      }
    }

    return qfv;
  }

  protected FeatureVector estimateRocchioModel(FeatureVector[] docvectors, RerankerContext context) {

    try {

      collectionStatistics = indexSearcher.collectionStatistics(getField());
    }catch (Exception e){
      e.printStackTrace();
      LOG.warn("Failed to get stats for field " +getField());
      return new FeatureVector();
    }
    FeatureVector qfv = getQfv(context).dropNaN();

    FeatureVector[] relVectors = new FeatureVector[relDocs.length];
    for (int i = 0; i < relVectors.length; i++){
       relVectors[i] = docvectors[relDocs[i]];
    }


    FeatureVector[] norelVectors = new FeatureVector[nonRelDocs.length];
    for (int i = 0; i < norelVectors.length; i++){
      norelVectors[i] = docvectors[nonRelDocs[i]];
    }

    FeatureVector relModel = estimateModel(relVectors).dropNaN();
    FeatureVector norelModel = estimateModel(norelVectors).dropNaN();

    FeatureVector model = FeatureVector.linearCombineation(qfv, relModel,alpha,beta);
    model = FeatureVector.linearCombineation(model,norelModel,1,-1 * gamma);
    model = model.pruneToSize(getFbTerms());
    model.pruneToThreshold(0.0001);
    model = model.scaleToUnitL1Norm().dropNaN();

    return model;
  }

  protected FeatureVector estimateModel(FeatureVector[] docvectors) {
    Set<String> vocab = getVocab(docvectors);
    // Precompute the norms once and cache templates.
    float[] norms = computeNorms(docvectors);
    FeatureVector f = new FeatureVector();
    try {
      collectionStatistics = indexSearcher.collectionStatistics(getField());
      addTermStatistic(vocab);

      for (String term : vocab) {
        RocchioFeedbakWegith rocchioFeedbakWegith = new RocchioFeedbakWegith(collectionStatistics,termStatisticsList.get(term));
        f.addFeatureWeight(term,(float)rocchioFeedbakWegith.computeFW(term,norms,docvectors));
      }
    } catch (IOException e) {
      e.printStackTrace();
      LOG.warn("Failed to extract rocchio model");
    }

    return f;
  }

  private void addTermStatistic(Set<String> vocab)  {
    for (String term : vocab) {
      if (termStatisticsList.containsKey(term)) continue;

      TermContext termContext = null;
      try {
        termContext = TermContext.build(indexSearcher.getIndexReader().getContext(), new Term(getField(), term));
        termStatisticsList.put(term, indexSearcher.termStatistics(new Term(getField(), new BytesRef(term)), termContext));
      } catch (IOException e) {
        e.printStackTrace();
        LOG.warn("Failed to get stats for " + term);
      }

    }
  }


  public IndexSearcher getIndexSearcher() {
    return indexSearcher;
  }

  public void setIndexSearcher(IndexSearcher indexSearcher) {
    this.indexSearcher = indexSearcher;
  }
}

class RocchioFeedbakWegith{
  private static final Logger LOG = LogManager.getLogger(RocchioFeedbakWegith.class);

  // Collection statisticsts for a certain term
  private CollectionStatistics collectionStats;
  private TermStatistics termStats;


  public RocchioFeedbakWegith(CollectionStatistics collectionStatistics, TermStatistics termStatistics){
    this.collectionStats = collectionStatistics;
    this.termStats = termStatistics;
  }

  public double computeFW(String term, float[] norms, FeatureVector[] docvectors){

    if (docvectors.length == 0) return 0;

    double fbWeight = 0.0;
    for (int i = 0; i < docvectors.length; i++) {
      // Avoids zero-length feedback documents, which causes division by zero when computing term weights.
      // Zero-length feedback documents occur (e.g., with CAR17) when a document has only terms
      // that accents (which are indexed, but not selected for feedback).
      if (norms[i] > 0.001f) {
        try {
          fbWeight += computeFW(docvectors[i].getFeatureWeight(term));
        } catch (IOException e) {
          LOG.warn("Not able to calculate FW for " + term + " " + i);
        }
      }
    }

    if (Double.isInfinite(fbWeight)){
      LOG.warn("FBWeight can not be infinity " + term + " " + fbWeight);
    }
    return fbWeight/docvectors.length;
  }


  public double computeFW(double tf) throws IOException {
    double fw = tf>0? tf * idf():0;
    if (Double.isInfinite(fw)){
      LOG.warn("Too big value " + tf);
    }
    return fw;

  }

  public double idf() {
    double idf = Math.log((collectionStats.docCount()/ (double)(termStats.docFreq())));
    return idf;
  }
}
