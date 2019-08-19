/**
 * Anserini: A Lucene toolkit for replicable information retrieval research
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.anserini.rerank.rf;

import io.anserini.rerank.RerankerContext;
import io.anserini.rerank.ScoredDocuments;
import io.anserini.util.AnalyzerUtils;
import io.anserini.util.FeatureVector;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * An abstract class that provides the base for major relevacne feedbacl models opertions and options.
 * In particular, it provides the follows:
 *  Operations:
 *  <ul>
 *    <li>Tokenize and build basic document vectors with tf frequency as weight.</li>
 *    <li>Tokenize and build basic query fector with tf frequency as the default weight</li>
 *    <li>Apply stop word removal if requested</li>
 *    <li></li>
 *    <li></li>
 *  </ul>
 */
public abstract class FeedbackModel {
  private static final Logger LOG = LogManager.getLogger(FeedbackModel.class);

  private final Analyzer analyzer;
  private final String field;


  private final float originalQueryWeight;
  private final int fbTerms;
  private final int fbDocs;
  private final boolean outputQuery;
  private final boolean removeStopWord;
  private final boolean normalize;
  private final boolean prunDocTerm;
  private final boolean pruneModel;

  private final FeedbackModelConfig config;
  protected IndexSearcher indexSearcher ;

  public boolean isRemoveStopWord() {
    return removeStopWord;
  }

  public Analyzer getAnalyzer() {
    return analyzer;
  }

  public String getField() {
    return field;
  }

  public int getFbTerms() {
    return fbTerms;
  }

  public int getFbDocs() {
    return fbDocs;
  }

  public boolean isOutputQuery() {
    return outputQuery;
  }

  public FeedbackModelConfig getConfig(){
    return config;
  }

  public FeedbackModel(Analyzer analyzer, String field, FeedbackModelConfig config) {
    this.analyzer = analyzer;
    this.field = field;
    this.fbTerms = config.getFbTerms();
    this.fbDocs = config.fbDocs;
    this.outputQuery = config.outputQuery;
    this.removeStopWord = config.removeStopWord;
    this.originalQueryWeight = config.originalQueryWeight;
    this.normalize = config.normalize;
    this.prunDocTerm = config.prunDocTerm;
    this.pruneModel = config.pruneModel;
    this.config = config;
  }


  public FeatureVector getQfv(RerankerContext context) {
    return getQfv(context.getQueryText());
  }

  public FeatureVector getQfv (String query){
    return FeatureVector.fromTerms(AnalyzerUtils.tokenize(analyzer, query)).scaleToUnitL1Norm();
  }

  public Query buildFeedbackQuery(RerankerContext context, FeatureVector model) {
    BooleanQuery.Builder feedbackQueryBuilder = new BooleanQuery.Builder();

    Iterator<String> terms = model.iterator();
    while (terms.hasNext()) {
      String term = terms.next();
      float prob = model.getFeatureWeight(term);
      feedbackQueryBuilder.add(new BoostQuery(new TermQuery(new Term(this.field, term)), prob), BooleanClause.Occur.SHOULD);
    }



    Query feedbackQuery = feedbackQueryBuilder.build();

    if (this.outputQuery) {
      LOG.info("QID: " + context.getQueryId());
      LOG.info("Original Query: " + context.getQuery().toString(this.field));
      LOG.info("Running new query: " + feedbackQuery.toString(this.field));
    }

    Query finalQuery = feedbackQuery;
    // If there's a filter condition, we need to add in the constraint.
    // Otherwise, just use the feedback query.
    if (context.getFilter() != null) {
      BooleanQuery.Builder bqBuilder = new BooleanQuery.Builder();
      bqBuilder.add(context.getFilter(), BooleanClause.Occur.FILTER);
      bqBuilder.add(feedbackQuery, BooleanClause.Occur.MUST);
      finalQuery = bqBuilder.build();
    }
    return finalQuery;
  }

  public Query buildFeedbackQuery(FeatureVector model){
    BooleanQuery.Builder feedbackQueryBuilder = new BooleanQuery.Builder();

    Iterator<String> terms = model.iterator();
    while (terms.hasNext()) {
      String term = terms.next();
      float prob = model.getFeatureWeight(term);
      feedbackQueryBuilder.add(new BoostQuery(new TermQuery(new Term(this.field, term)), prob), BooleanClause.Occur.SHOULD);
    }



    Query feedbackQuery = feedbackQueryBuilder.build();
    return feedbackQuery;
  }

  public FeatureVector estimate(String query, ScoredDocuments docs, IndexReader reader, boolean tweetsearch) {
    FeatureVector[] docvectors = buildDocVectors(docs,reader,tweetsearch);
    return estimate(query,docvectors,docs.scores);
  }


  public abstract FeatureVector estimate(String query,FeatureVector[] docvectors,  float[] scores);


  public FeatureVector estimate(String query, String[] contents, float[] scores, IndexReader reader, boolean tweetsearch) {
    return estimate(query,buildDocVectors(contents,reader,tweetsearch),scores);
  }

  protected float[] computeNorms(FeatureVector[] docvectors) {
    float[] norms = new float[docvectors.length];
    for (int i = 0; i < docvectors.length; i++) {
      norms[i] = (float) docvectors[i].computeL1Norm();
    }
    return norms;
  }

  protected FeatureVector createdFeatureVector(Terms terms, IndexReader reader, boolean tweetsearch) {
    FeatureVector f = new FeatureVector();

    try {
      int numDocs = reader.numDocs();
      TermsEnum termsEnum = terms.iterator();

      BytesRef text;
      while ((text = termsEnum.next()) != null) {
        String term = text.utf8ToString();

        if (term.length() < 2 || term.length() > 20) continue;
        if (!term.matches("[a-z0-9]+")) continue;
        if (removeStopWord && isStopword(term,reader, tweetsearch, numDocs)) continue;

        int freq = (int) termsEnum.totalTermFreq();
        f.addFeatureWeight(term, (float) freq);
      }

    } catch (Exception e) {
      e.printStackTrace();
      // Return empty feature vector
      return f;
    }

    return f;
  }


  private List<String> filterStopwords(List<String> terms, IndexReader reader, boolean tweetsearch, int numDocs) throws IOException {
    return terms.stream().filter(term -> {
          try {
            return !removeStopWord || !isStopword(term, reader, tweetsearch, numDocs);
          } catch (IOException e) {
            e.printStackTrace();
            LOG.warn("Fail to test " + term );
            return false;
          }
        }
    ).collect(Collectors.toList());
  }

  private boolean isStopword(String term, IndexReader reader, boolean tweetsearch, int numDocs) throws IOException {
    // This seemingly arbitrary logic needs some explanation. See following PR for details:
    //   https://github.com/castorini/Anserini/pull/289
    //
    // We have long known that stopwords have a big impact in RM3. If we include stopwords
    // in feedback, effectiveness is affected negatively. In the previous implementation, we
    // built custom stopwords lists by selecting top k terms from the collection. We only
    // had two stopwords lists, for gov2 and for Twitter. The gov2 list is used on all
    // collections other than Twitter.
    //
    // The logic below instead uses a df threshold: If a term appears in more than n percent
    // of the documents, then it is discarded as a feedback term. This heuristic has the
    // advantage of getting rid of collection-specific stopwords lists, but at the cost of
    // introducing an additional tuning parameter.
    //
    // Cognizant of the dangers of (essentially) tuning on test data, here's what I
    // (@lintool) did:
    //
    // + For newswire collections, I picked a number, 10%, that seemed right. This value
    //   actually increased effectiveness in most conditions across all newswire collections.
    //
    // + This 10% value worked fine on web collections; effectiveness didn't change much.
    //
    // Since this was the first and only heuristic value I selected, we're not really tuning
    // parameters.
    //
    // The 10% threshold, however, doesn't work well on tweets because tweets are much
    // shorter. Based on a list terms in the collection by df: For the Tweets2011 collection,
    // I found a threshold close to a nice round number that approximated the length of the
    // current stopwords list, by eyeballing the df values. This turned out to be 1%. I did
    // this again for the Tweets2013 collection, using the same approach, and obtained a value
    // of 0.7%.
    //
    // With both values, we obtained effectiveness pretty close to the old values with the
    // custom stopwords list.
    int df = reader.docFreq(new Term(field, term));
    float ratio = (float) df / numDocs;
    if (tweetsearch) {
      if (numDocs > 100000000) { // Probably Tweets2013
        if (ratio > 0.007f) return true;
      } else {
        if (ratio > 0.01f) return true;
      }
    } else if (ratio > 0.1f) return true;
    return false;
  }


  protected FeatureVector[] buildDocVectors(ScoredDocuments docs, IndexReader reader, boolean tweetsearch){
    int numdocs = docs.documents.length < fbDocs ? docs.documents.length : fbDocs;
    FeatureVector[] docvectors = new FeatureVector[numdocs];

    for (int i = 0; i < numdocs; i++) {
      try {
        Terms termVector = reader.getTermVector(docs.ids[i], field);
        FeatureVector docVector = null;
        if (termVector==null){
          String text = docs.documents[i].get(field);
          List<String> filteredWords = filterStopwords(
              AnalyzerUtils.tokenize(analyzer,text),reader,tweetsearch,numdocs);
          docVector = FeatureVector.fromTerms(filteredWords);
        }else{
          docVector = createdFeatureVector(termVector , reader, tweetsearch);
        }

        docvectors[i] = docVector;
      } catch (IOException e) {
        e.printStackTrace();
        // Just return empty feature vector.
        docvectors[i] = null;
      }
    }

    return docvectors;
  }

  public static Set<String> getVocab(FeatureVector[] docvectors){
    Set<String> vocab = new HashSet<>();

    for (FeatureVector docVector: docvectors){
      vocab.addAll(docVector.getFeatures());
    }
    return vocab;
  }

  protected FeatureVector[] buildDocVectors(String[] contents, IndexReader reader, boolean tweetsearch){
    int numdocs = contents.length < fbDocs ? contents.length : fbDocs;
    FeatureVector[] docvectors = new FeatureVector[numdocs];

    for (int i = 0; i < numdocs; i++) {
      try {
        FeatureVector docVector = null;

        List<String> filteredWords = filterStopwords(
            AnalyzerUtils.tokenize(analyzer,contents[i]),reader,tweetsearch,reader.numDocs());
//        List<String> filteredWords = AnalyzerUtils.tokenize(analyzer,contents[i]);
        if (false){
          throw new IOException();
        }
        docVector = FeatureVector.fromTerms(filteredWords);

        docvectors[i] = docVector;
      } catch (IOException e) {
        e.printStackTrace();
        // Just return empty feature vector.
        docvectors[i] = null;
      }
    }
    return docvectors;
  }

  public void collectTermStatitics(Set<String> vocab, Map<String,TermStatistics> termStatisticsList)
       {
    for (String term : vocab){
      TermStates termContext = null;
      try {
        termContext = TermStates.build(
            getIndexSearcher().getIndexReader().getContext(),new Term(getField(),term),true);
        termStatisticsList.put(term, getIndexSearcher()
            .termStatistics(new Term(getField(),new BytesRef(term)),termContext));
      } catch (IOException e) {
        e.printStackTrace();
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


