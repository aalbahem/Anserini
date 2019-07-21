package io.anserini.rerank.lib;

import io.anserini.rerank.Reranker;
import io.anserini.rerank.RerankerContext;
import io.anserini.rerank.ScoredDocuments;
import io.anserini.rerank.rf.FeedbackModel;
import io.anserini.rerank.rf.FeedbackModelFactory;
import io.anserini.rerank.rf.factories.LLFeedbackModelFactory;
import io.anserini.search.SearchArgs;
import io.anserini.util.FeatureVector;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;

import java.io.IOException;

import static io.anserini.search.SearchCollection.BREAK_SCORE_TIES_BY_DOCID;
import static io.anserini.search.SearchCollection.BREAK_SCORE_TIES_BY_TWEETID;

public class RFReranker implements Reranker {


  private Analyzer analyzer;
  private String field;

  public RFReranker (Analyzer analyzer, String field){
    this.analyzer = analyzer;
    this.field = field;
  }


  public FeedbackModelFactory getFactory(RerankerContext context){

    SearchArgs args = context.getSearchArgs();
    if (args.ll){
      return new LLFeedbackModelFactory();
    }

    return null;
  }

  @Override
  public ScoredDocuments rerank(ScoredDocuments docs, RerankerContext context) {

    assert(docs.documents.length == docs.scores.length);

    FeedbackModel method = getFactory(context).create(analyzer,field,context.getSearchArgs());
    method.setIndexSearcher(context.getIndexSearcher());

    IndexSearcher searcher = context.getIndexSearcher();
    FeatureVector model = method.estimate(context.getQueryText(),docs,searcher.getIndexReader(),context.getSearchArgs().searchtweets);
    Query finalQuery = method.buildFeedbackQuery(context,model);

    TopDocs rs;
    try {
      // Figure out how to break the scoring ties.
      if (context.getSearchArgs().arbitraryScoreTieBreak) {
        rs = searcher.search(finalQuery, context.getSearchArgs().hits);
      } else if (context.getSearchArgs().searchtweets) {
        rs = searcher.search(finalQuery, context.getSearchArgs().hits, BREAK_SCORE_TIES_BY_TWEETID, true);
      } else {
        rs = searcher.search(finalQuery, context.getSearchArgs().hits, BREAK_SCORE_TIES_BY_DOCID, true);
      }
    } catch (IOException e) {
      e.printStackTrace();
      return docs;
    }

    return ScoredDocuments.fromTopDocs(rs, searcher);
  }

  @Override
  public String tag() {
    return "RF: ";
  }
}
