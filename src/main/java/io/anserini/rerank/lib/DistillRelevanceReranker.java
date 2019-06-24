/**
 * Anserini: A toolkit for reproducible information retrieval research built on Lucene
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.anserini.rerank.lib;

import io.anserini.index.generator.LuceneDocumentGenerator;
import io.anserini.rerank.RerankerContext;
import io.anserini.rerank.ScoredDocuments;
import io.anserini.search.query.FilterQueryBuilder;
import io.anserini.util.FeatureVector;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermContext;
import org.apache.lucene.search.*;
import org.apache.lucene.util.BytesRef;
import sun.util.logging.resources.logging;

import java.io.IOException;
import java.util.*;

import static io.anserini.search.SearchCollection.BREAK_SCORE_TIES_BY_DOCID;
import static io.anserini.search.SearchCollection.BREAK_SCORE_TIES_BY_TWEETID;

public class DistillRelevanceReranker extends Rm3Reranker {
  private static final Logger LOG = LogManager.getLogger(DistillRelevanceReranker.class);

  private Map<String, TermStatistics> termStatisticsList = new HashMap<>();

  private IndexSearcher indexSearcher ;

  private CollectionStatistics collectionStatistics;

  private int[] relDocs;
  private int[] nonRelDocs;

  private float nrY = 0.2f;
  private float cY = 0.1f;



  public void setRelDocs(int[] relDocs) {
    this.relDocs = relDocs;
  }

  public void setNonRelDocs(int[] nonRelDocs) {
    this.nonRelDocs = nonRelDocs;
  }

  public void setIndexSearcher(IndexSearcher indexSearcher) {
    this.indexSearcher = indexSearcher;
  }


  public DistillRelevanceReranker(Analyzer analyzer, String field, int fbTerms, int fbDocs, float originalQueryWeight, boolean outputQuery) {
    super(analyzer, field, fbTerms, fbDocs, originalQueryWeight, outputQuery);
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

  protected FeatureVector estimateModel(FeatureVector[] docvectors) {
    Set<String> vocab = getVocab(docvectors);
    // Precompute the norms once and cache templates.
    float[] norms = computeNorms(docvectors);
    float norm = sum(norms);

    FeatureVector f = new FeatureVector();
    try {
      collectionStatistics = indexSearcher.collectionStatistics(getField());
      addTermStatistic(vocab);

      for (String term : vocab) {
        MLEWegith mleWegith = new MLEWegith(collectionStatistics,termStatisticsList.get(term),norm);
        f.addFeatureWeight(term,(float)mleWegith.computeFW(term,norms,docvectors));
      }
    } catch (IOException e) {
      e.printStackTrace();
      LOG.warn("Failed to extract rocchio model");
    }

    return f;
  }

  public FeatureVector estimateDistillModel(String[] contents, RerankerContext context) {
    return estimateDistillModel(buildDocVectors(contents,context.getIndexSearcher().getIndexReader(),context.getSearchArgs().searchtweets),context);
  }


  public FeatureVector estimateDistillModel(FeatureVector[] docvectors, RerankerContext context) {

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

    Set<String> voccab = new HashSet<>();
    voccab.addAll(relModel.getFeatures());
    voccab.addAll(norelModel.getFeatures());

    FeatureVector collectionModel = estimateCollectionModel(voccab);

    FeatureVector distModel = EM(docvectors,relModel,norelModel,collectionModel);
    return distModel;
  }

  public FeatureVector estimateCollectionModel(Set<String> voccab){
    FeatureVector model = new FeatureVector();
    for (String term : voccab){
       model.addFeatureWeight(term,termStatisticsList.get(term).totalTermFreq() / (float)collectionStatistics.sumTotalTermFreq());
    }

    return model;
  }

  public float logLikelihood(FeatureVector[] relDocTFVectors, FeatureVector relModel, FeatureVector nonRelModel, FeatureVector collectionModel){
    float log  = 0.0f;
    for (FeatureVector doc : relDocTFVectors){
      for (String w : doc.getFeatures()){
         float fbWeight = (1-nrY-cY)* relModel.getFeatureWeight(w) + nrY * nonRelModel.getFeatureWeight(w) + cY * collectionModel.getFeatureWeight(w);
         log += (doc.getFeatureWeight(w) * Math.log(fbWeight));
      }
    }
    return log;
  }

  public FeatureVector EM(FeatureVector[] relDocTFVectors, FeatureVector relModel, FeatureVector nonRelModel, FeatureVector collectionModel){

    int iterations = 100;
    FeatureVector pRelNext = new FeatureVector();

    for (String w: relModel.getFeatures()){
      pRelNext.addFeatureWeight(w,relModel.getFeatureWeight(w));
    }

    float[] logs = new float[iterations];
    for (int i = 0; i < iterations; i++) {
      for (String w : relModel.getFeatures()) {
        float wTn = computeTN(nrY, cY, pRelNext.getFeatureWeight(w), nonRelModel.getFeatureWeight(w), collectionModel.getFeatureWeight(w));
        float wPrelTNPlus = computeNextPrel(w, wTn, relDocTFVectors);
        pRelNext.updateFeatureWeight(w, wPrelTNPlus);
      }

      pRelNext.scaleToUnitL1Norm();
      float logSum = logLikelihood(relDocTFVectors,pRelNext,nonRelModel,collectionModel);
      logs[i] = logSum;
    }

//    LOG.debug("Log sum is : " + Arrays.toString(logs) );
    return pRelNext;
  }

  public float computeTN(float nrY, float cY, float pRel, float pNR, float pC){
    float tn = (1-nrY-cY) * pRel;
    float norm = (1-nrY-cY) * pRel + nrY * pNR + pC * cY;
    tn = tn/norm;
    return tn;
  }

  public float computeNextPrel(String term, float tn, FeatureVector[] tfVecs){
    float wPrel = 0.0f;

    for (FeatureVector tfVec: tfVecs){
      wPrel += (tfVec.getFeatureWeight(term)) * tn;
    }
    return wPrel;
  }

  @Override
  public ScoredDocuments rerank(ScoredDocuments docs, RerankerContext context) {

    assert (docs.documents.length == docs.scores.length);

    IndexSearcher searcher = context.getIndexSearcher();
    Query finalQuery = reformulateQuery(docs, context);


//    QueryRescorer rm3Rescoer = new RM3QueryRescorer(finalQuery);
//    TopDocs rs;
//    try {
//      rs = rm3Rescoer.rescore(searcher,docs.topDocs,context.getSearchArgs().hits);
//    } catch (IOException e) {
//      e.printStackTrace();
//      return docs;
//    }

    Query docNumFilter = FilterQueryBuilder.buildSetQuery(LuceneDocumentGenerator.FIELD_ID, getFieldValues(docs, LuceneDocumentGenerator.FIELD_ID, searcher));
    finalQuery = FilterQueryBuilder.addFilterQuery(finalQuery, docNumFilter);
    TopDocs rs;
    try {
      // Figure out how to break the scoring ties.
      if (context.getSearchArgs().arbitraryScoreTieBreak) {
        rs = searcher.search(finalQuery, context.getSearchArgs().hits);
      } else if (context.getSearchArgs().searchtweets) {
        rs = searcher.search(finalQuery, context.getSearchArgs().hits, BREAK_SCORE_TIES_BY_TWEETID, true, true);
      } else {
        rs = searcher.search(finalQuery, context.getSearchArgs().hits, BREAK_SCORE_TIES_BY_DOCID, true, true);
      }
    } catch (IOException e) {
      e.printStackTrace();
      return docs;
    }

    return ScoredDocuments.fromTopDocs(rs, searcher);

  }


  private Set<String> getFieldValues(ScoredDocuments docs, String fld, IndexSearcher searcher) {
    Set<String> values = new HashSet<>();

    for (int id : docs.ids) {
      try {
        values.add(searcher.doc(id).getField(fld).stringValue());
      } catch (IOException e) {
        LOG.warn(String.format("Failed to extract %s from document with lucene id %s", fld, id));
        e.printStackTrace();
      }
    }
    return values;
  }

  public Query reformulateQuery(ScoredDocuments docs, RerankerContext context) {
    FeatureVector rm3 = estimateRM3Model(docs, context);

    return buildFeedbackQuery(context, rm3);
  }


  @Override
  public String tag() {
    return "CLRm3(fbDocs=" + getFbDocs() + ",fbTerms=" + getFbTerms() + ",originalQueryWeight:" + getOriginalQueryWeight() + ")";
  }


}

class DistillQueryRescorer extends QueryRescorer {
  public DistillQueryRescorer(Query query) {
    super(query);
  }

  @Override
  protected float combine(float firstPassScore, boolean secondPassMatches, float secondPassScore) {
    return secondPassScore;
  }
}


class MLEWegith{
  private static final Logger LOG = LogManager.getLogger(MLEWegith.class);

  // Collection statisticsts for a certain term
  private CollectionStatistics collectionStats;
  private TermStatistics termStats;
  private float norm;

  public MLEWegith(CollectionStatistics collectionStatistics, TermStatistics termStatistics, float norm){
    this.collectionStats = collectionStatistics;
    this.termStats = termStatistics;
    this.norm = norm;
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
    return fbWeight/norm;
  }


  public double computeFW(double tf) throws IOException {
    double fw = tf>0? tf:0;
    if (Double.isInfinite(fw)){
      LOG.warn("Too big value " + tf);
    }
    return fw;

  }

}

